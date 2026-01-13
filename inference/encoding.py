from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass
from typing import Deque, Iterable, List, Optional, Tuple

import numpy as np
import torch

import chess


HISTORY_LEN = 8
NUM_SQUARES = 64
NUM_POLICY_OUTPUTS = 4098

# Must match `process_pgn.py`
PIECE_MAP = {
    chess.PAWN: 1,
    chess.KNIGHT: 2,
    chess.BISHOP: 3,
    chess.ROOK: 4,
    chess.QUEEN: 5,
    chess.KING: 6,
}

TC_BLITZ = 0
TC_RAPID = 1
TC_CLASSICAL = 2

# Known increment values the model was trained on (in seconds).
# Rare increments are clamped to the nearest known value to improve generalization.
KNOWN_INCREMENTS = [0, 2, 3, 5, 10]


def clamp_to_known_increment(inc_s: float) -> float:
    """Clamp an increment to the nearest known value for better model generalization."""
    return float(min(KNOWN_INCREMENTS, key=lambda x: abs(x - inc_s)))


def get_tc_category(base_seconds: float, inc_seconds: float) -> int:
    # Same rule as `process_pgn.get_tc_category`.
    # <600 = Blitz, <1800 = Rapid, >=1800 = Classical
    duration = float(base_seconds) + 40.0 * float(inc_seconds)
    if duration < 600.0:
        return TC_BLITZ
    if duration < 1800.0:
        return TC_RAPID
    return TC_CLASSICAL


def canonicalize(board: chess.Board) -> chess.Board:
    # Same as `process_pgn_v2.canonicalize(board, board.turn)`.
    return board if board.turn == chess.WHITE else board.mirror()


def encode_board(board: chess.Board) -> List[int]:
    tokens = [0] * NUM_SQUARES
    for square in range(NUM_SQUARES):
        piece = board.piece_at(square)
        if piece is None:
            continue
        val = PIECE_MAP[piece.piece_type]
        tokens[square] = val if piece.color == chess.WHITE else val + 6
    return tokens


def encode_legal_moves(board: chess.Board) -> List[Tuple[int, int, int]]:
    # promo: 0 none, 1 q, 2 r, 3 b, 4 n (matches process_pgn_v2.PROMOTION_MAP)
    promo_map = {None: 0, chess.QUEEN: 1, chess.ROOK: 2, chess.BISHOP: 3, chess.KNIGHT: 4}
    out: List[Tuple[int, int, int]] = []
    for mv in board.legal_moves:
        out.append((mv.from_square, mv.to_square, promo_map.get(mv.promotion, 0)))
    return out


def build_history_from_position(
    start_board: chess.Board,
    moves_uci: Iterable[str],
    *,
    history_len: int = HISTORY_LEN,
) -> Tuple[chess.Board, List[List[int]], List[int]]:
    """Replays moves and returns (final_board, board_history, repetition_flags).

    `board_history` is a list of `history_len` encoded canonical boards, newest first,
    matching `process_pgn_v2`.
    """

    board = start_board.copy(stack=False)

    hist: Deque[List[int]] = deque(maxlen=history_len)
    raw: Deque[chess.Board] = deque(maxlen=history_len)

    # Initial position before any moves: include it (like training does per-ply).
    canonical = canonicalize(board)
    hist.appendleft(encode_board(canonical))
    raw.appendleft(canonical.copy(stack=False))

    for uci in moves_uci:
        mv = chess.Move.from_uci(uci)
        if mv not in board.legal_moves:
            raise ValueError(f"Illegal move in position list: {uci}")
        board.push(mv)
        canonical = canonicalize(board)
        hist.appendleft(encode_board(canonical))
        raw.appendleft(canonical.copy(stack=False))

    board_history = list(hist)
    while len(board_history) < history_len:
        board_history.append([0] * NUM_SQUARES)
    board_history = board_history[:history_len]

    repetition = _repetition_flags(list(raw), history_len=history_len)
    return board, board_history, repetition


def _repetition_flags(history: List[chess.Board], *, history_len: int) -> List[int]:
    # Same idea as `process_pgn_v2.repetition_flags`.
    seen: dict[str, int] = {}
    flags: List[int] = []
    for b in history:
        key = b.board_fen()
        seen[key] = seen.get(key, 0) + 1
        flags.append(1 if seen[key] >= 2 else 0)
    while len(flags) < history_len:
        flags.insert(0, 0)
    return flags[:history_len]


@dataclass
class ContextOptions:
    active_elo: int = 1900
    opponent_elo: int = 1900
    active_clock_s: float = 300.0
    opponent_clock_s: float = 300.0
    active_inc_s: float = 0.0
    opponent_inc_s: float = 0.0
    # Optional: game base time (seconds) for time-control categorization.
    # Training uses the *initial* base time from PGN TimeControl, not the remaining clock.
    tc_base_s: float | None = None
    halfmove_clock: int = 0


def make_model_batch(
    *,
    board: chess.Board,
    board_history: List[List[int]],
    repetition_flags: List[int],
    time_history_s: Optional[List[float]] = None,
    ctx: ContextOptions,
    tc_cat: Optional[int] = None,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    """Builds the exact tensor batch expected by `ChessformerV2.forward`.

    Notes:
    - Mirrors training: board_history + legal_moves are canonicalized, but castling/ep/scalars
      are taken from the *actual* board (see `process_pgn_v2`).
    - `legal_mask` excludes resign/flag for UCI play.
    """

    if time_history_s is None:
        time_history_s = [0.0] * HISTORY_LEN
    if len(time_history_s) != HISTORY_LEN:
        raise ValueError(f"time_history_s must have len {HISTORY_LEN}")

    # Canonical board for legal-move encoding.
    canonical = canonicalize(board)
    legal_moves = encode_legal_moves(canonical)

    legal_mask = torch.zeros(NUM_POLICY_OUTPUTS, dtype=torch.bool)
    for from_sq, to_sq, _promo in legal_moves:
        legal_mask[from_sq * 64 + to_sq] = True
    
    # Do NOT unmask resign (4096) or flag (4097) during inference.
    # In training, these are treated as policy targets (only unmasked when they ARE the target).
    # During inference, we mask them out so the model only picks legal chess moves.
    # The raw logits at indices 4096/4097 can still be read for analysis/display.

    # --- Scalars (must match dataset.py normalization) ---
    active_elo_norm = (ctx.active_elo - 1900) / 700.0
    opp_elo_norm = (ctx.opponent_elo - 1900) / 700.0
    ply_norm = board.fullmove_number * 2 - (0 if board.turn == chess.WHITE else 1)
    ply_norm = ply_norm / 100.0

    # Use actual clock times directly - no scaling for bullet games.
    # The model will see the real time values even for bullet (<180s).
    active_clock_norm = math.log1p(max(0.0, ctx.active_clock_s)) / 10.0
    opp_clock_norm = math.log1p(max(0.0, ctx.opponent_clock_s)) / 10.0
    
    # Clamp increments to known values for better time prediction generalization.
    # The model struggles with rare increment values it wasn't trained on.
    clamped_active_inc = clamp_to_known_increment(ctx.active_inc_s)
    clamped_opp_inc = clamp_to_known_increment(ctx.opponent_inc_s)
    active_inc_norm = clamped_active_inc / 30.0
    opp_inc_norm = clamped_opp_inc / 30.0
    hmc_norm = float(ctx.halfmove_clock) / 100.0

    scalars = torch.tensor(
        [
            active_elo_norm,
            opp_elo_norm,
            ply_norm,
            active_clock_norm,
            opp_clock_norm,
            active_inc_norm,
            opp_inc_norm,
            hmc_norm,
        ],
        dtype=torch.float32,
    )

    if tc_cat is None:
        # Match `process_pgn_v2.get_tc_category`: duration = base + 40*inc.
        # Use the game's base time when known; otherwise fall back to the largest remaining clock.
        base_s = float(ctx.tc_base_s) if ctx.tc_base_s is not None else float(max(ctx.active_clock_s, ctx.opponent_clock_s))
        tc_cat = get_tc_category(base_s, ctx.active_inc_s)

    castling = torch.tensor(
        [
            int(board.has_kingside_castling_rights(chess.WHITE)),
            int(board.has_queenside_castling_rights(chess.WHITE)),
            int(board.has_kingside_castling_rights(chess.BLACK)),
            int(board.has_queenside_castling_rights(chess.BLACK)),
        ],
        dtype=torch.float32,
    )

    ep_mask = torch.zeros(64, dtype=torch.float32)
    if board.ep_square is not None:
        ep_mask[int(board.ep_square)] = 1.0

    batch = {
        "board_history": torch.tensor(board_history, dtype=torch.long),
        "time_history": torch.tensor(np.array(time_history_s, dtype=np.float32) / 60.0),
        "rep_flags": torch.tensor(repetition_flags, dtype=torch.float32),
        "castling": castling,
        "ep_mask": ep_mask,
        "scalars": scalars,
        "tc_cat": torch.tensor(int(tc_cat), dtype=torch.long),
        "legal_mask": legal_mask,
    }

    # Add batch dimension.
    batch = {k: v.unsqueeze(0).to(device) for k, v in batch.items()}
    return batch
