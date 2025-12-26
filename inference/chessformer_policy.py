from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch

import chess

from inference.encoding import ContextOptions, canonicalize, make_model_batch
from inference.sampling import sample_from_logits, select_promo


@dataclass
class PolicyOutput:
    move: Optional[chess.Move]
    policy_prob: float
    is_resign: bool = False
    is_flag: bool = False
    stats: Optional[dict] = None


def _mirror_square(sq: int) -> int:
    # Same as `sq ^ 56` used in `process_pgn.py`.
    return sq ^ 56


def _canonical_to_real_move(move: chess.Move, real_turn: chess.Color) -> chess.Move:
    if real_turn == chess.WHITE:
        return move
    return chess.Move(
        _mirror_square(move.from_square),
        _mirror_square(move.to_square),
        promotion=move.promotion,
    )


def choose_move(
    *,
    model: torch.nn.Module,
    board: chess.Board,
    board_history,
    repetition_flags,
    ctx: ContextOptions,
    time_history_s: Optional[list[float]] = None,
    temperature: float,
    top_p: float,
    device: torch.device,
    rng: Optional[np.random.Generator] = None,
) -> PolicyOutput:
    """Runs a single forward pass and samples a legal move.

    - Uses canonical board (white-to-move) for model inputs.
    - Applies legal-move mask; resign/flag are excluded.
    - Handles promotions via the separate promo head.
    """

    real_turn = board.turn

    batch = make_model_batch(
        board=board,
        board_history=board_history,
        repetition_flags=repetition_flags,
        time_history_s=time_history_s,
        ctx=ctx,
        device=device,
    )

    # Ask for promotion logits.
    with torch.inference_mode():
        with torch.autocast(device_type=device.type, enabled=(device.type == "cuda")):
            outputs = model(batch, return_promo=True)

    move_logits = outputs[0].squeeze(0)  # (4098,)
    promo_logits = outputs[-1].squeeze(0)  # (8,4)

    # Sample from the masked logits. (Masking happens inside the model.)
    sample = sample_from_logits(move_logits, temperature=temperature, top_p=top_p, rng=rng)

    if sample.move_index == 4096:
        return PolicyOutput(move=None, policy_prob=sample.prob, is_resign=True)
    if sample.move_index == 4097:
        return PolicyOutput(move=None, policy_prob=sample.prob, is_flag=True)

    from_sq_c = sample.move_index // 64
    to_sq_c = sample.move_index % 64

    # Promotion handling: if any legal move from->to is a promotion, pick piece with promo head.
    canonical = canonicalize(board)
    candidates = [mv for mv in canonical.legal_moves if mv.from_square == from_sq_c and mv.to_square == to_sq_c]
    if not candidates:
        # Shouldn't happen if legal_mask is correct; fall back to first legal.
        mv = next(iter(canonical.legal_moves))
    else:
        # If promotions exist for this (from,to), pick one using promo head.
        promo_moves = [mv for mv in candidates if mv.promotion is not None]
        if promo_moves:
            if 56 <= to_sq_c <= 63:
                file_idx = to_sq_c - 56
                promo_char = select_promo(promo_logits[file_idx], temperature=temperature, top_p=top_p, rng=rng)
            else:
                promo_char = "q"

            promo_piece = {"q": chess.QUEEN, "r": chess.ROOK, "b": chess.BISHOP, "n": chess.KNIGHT}[promo_char]
            mv = chess.Move(from_sq_c, to_sq_c, promotion=promo_piece)
            if mv not in canonical.legal_moves:
                mv = promo_moves[0]
        else:
            mv = candidates[0]

    real_move = _canonical_to_real_move(mv, real_turn)
    return PolicyOutput(move=real_move, policy_prob=sample.prob)
