"""compare_ACPL.py

Compute Average Centipawn Loss (ACPL) and CPL variance for a model
versus a dataset of human games (e.g. `data_v2/val_balanced.parquet`).

The script is intentionally defensive: it probes the parquet to find
useful columns, can operate per-game or per-position, and uses Stockfish
via `python-chess` for centipawn evaluations. If no engine path is
provided the script will attempt to use `stockfish` on PATH.

Usage examples:
  python compare_ACPL.py --parquet data_v2/val_balanced.parquet --num-games 2000 \
      --engine-path /usr/bin/stockfish --model-py model.py --checkpoint inference/chessformer_v2_smolgen_best.pt

Notes:
- The script picks the model's top policy move (argmax over legal moves)
  rather than sampling.
- Engine parameters (time or depth) are configurable; longer evals are
  slower but more accurate.
"""

from __future__ import annotations

import argparse
import math
import random
import statistics
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import pyarrow.parquet as pq
import pyarrow as pa
import numpy as np
import torch
import chess
import chess.engine

from inference import model_loader
from inference.encoding import PIECE_MAP, make_model_batch, ContextOptions


# Reverse PIECE_MAP: numeric -> (piece_type, color)
_REVERSE_PIECE = {}
for k, v in PIECE_MAP.items():
    _REVERSE_PIECE[v] = (k, chess.WHITE)
    _REVERSE_PIECE[v + 6] = (k, chess.BLACK)


def decode_board_from_encoded(encoded: List[int]) -> chess.Board:
    """Decode a single canonical encoded board (length 64) into a Board.

    The dataset stores canonicalized boards (white-to-move). We return
    a board with `board.turn == chess.WHITE`.
    """
    board = chess.Board.empty()
    for sq, val in enumerate(encoded):
        if val == 0:
            continue
        piece_type, color = _REVERSE_PIECE.get(val, (None, None))
        if piece_type is None:
            continue
        board.set_piece_at(sq, chess.Piece(piece_type, color))
    # Ensure white to move (encoded board is canonical)
    board.turn = chess.WHITE
    return board


def get_model_move(
    model,
    device: torch.device,
    board: chess.Board,
    board_history: List[List[int]],
    repetition_flags: List[int],
    ctx: ContextOptions,
    temperature: float = 0.0,
    top_p: float = 1.0,
    rng: random.Random = None,
) -> chess.Move:
    """Get a move from the model.
    
    Args:
        temperature: 0.0 = argmax (deterministic), >0 = sample with temperature
        top_p: nucleus sampling threshold (1.0 = no filtering, <1.0 = keep top cumulative prob)
        rng: random.Random instance for reproducibility
    """
    batch = make_model_batch(
        board=board,
        board_history=board_history,
        repetition_flags=repetition_flags,
        time_history_s=None,
        ctx=ctx,
        device=device,
    )
    with torch.inference_mode():
        outputs = model(batch, return_promo=True)
    move_logits = outputs[0].squeeze(0).cpu().numpy()  # (4098,)
    promo_logits = outputs[-1].squeeze(0).cpu().numpy()  # (8,4)

    # Mask illegal moves
    legal_mask = batch["legal_mask"].squeeze(0).cpu().numpy().astype(bool)
    masked = np.where(legal_mask, move_logits, -1e9)
    
    # Select move index based on temperature
    if temperature <= 0.0:
        # Argmax (deterministic)
        idx = int(np.argmax(masked))
    else:
        # Apply temperature and convert to probabilities
        scaled = masked / temperature
        scaled = scaled - np.max(scaled)  # for numerical stability
        probs = np.exp(scaled)
        probs = probs / probs.sum()
        
        # Apply top-p (nucleus) sampling
        if top_p < 1.0:
            sorted_indices = np.argsort(probs)[::-1]
            sorted_probs = probs[sorted_indices]
            cumsum = np.cumsum(sorted_probs)
            # Find cutoff index
            cutoff_idx = np.searchsorted(cumsum, top_p) + 1
            # Zero out low-probability tokens
            keep_indices = sorted_indices[:cutoff_idx]
            mask = np.zeros_like(probs, dtype=bool)
            mask[keep_indices] = True
            probs = np.where(mask, probs, 0.0)
            probs = probs / probs.sum()  # renormalize
        
        # Sample
        if rng is not None:
            idx = rng.choices(range(len(probs)), weights=probs, k=1)[0]
        else:
            idx = np.random.choice(len(probs), p=probs)
    
    if idx >= 4096:
        # resign/flag - treat as no move
        return None
    from_sq = idx // 64
    to_sq = idx % 64

    # Handle promotions: if any legal move from->to has promotion, choose promo by promo head
    candidates = [mv for mv in board.legal_moves if mv.from_square == from_sq and mv.to_square == to_sq]
    if not candidates:
        # fallback: pick a random legal move (shouldn't happen)
        return next(iter(board.legal_moves))
    promo_moves = [mv for mv in candidates if mv.promotion is not None]
    if promo_moves:
        # promo file index (if to_sq in last rank)
        if 56 <= to_sq <= 63:
            file_idx = to_sq - 56
            # choose max logit for promo (could also apply temperature here)
            promo_choice = int(np.argmax(promo_logits[file_idx]))
            promo_map = {0: None, 1: chess.QUEEN, 2: chess.ROOK, 3: chess.BISHOP, 4: chess.KNIGHT}
            promo_piece = promo_map.get(promo_choice, chess.QUEEN)
            mv = chess.Move(from_sq, to_sq, promotion=promo_piece)
            if mv in board.legal_moves:
                return mv
        # otherwise pick first promo move
        return promo_moves[0]
    return candidates[0]


def score_to_cp(score: chess.engine.PovScore, *, mate_cp: int = 10000) -> int:
    """Convert a Score or PovScore to centipawns from the perspective of the side to move.
    
    Mate scores are converted to a large but bounded value (mate_cp) to avoid
    extreme outliers distorting ACPL calculations.
    """
    rel_score = score.relative if hasattr(score, 'relative') else score
    if rel_score.is_mate():
        mate_in = rel_score.mate()
        if mate_in is None:
            return mate_cp
        # positive mate -> winning, negative -> losing
        sign = 1 if mate_in > 0 else -1
        return sign * mate_cp
    cp = rel_score.score()
    if cp is None:
        return 0
    return int(cp)


def evaluate_position_with_engine(engine: chess.engine.SimpleEngine, board: chess.Board, *, limit) -> Tuple[int, chess.engine.PovScore]:
    # engine returns an Analysis or dict; use engine.play or engine.analyse
    info = engine.analyse(board, limit)
    score = info.get("score")
    if isinstance(score, chess.engine.PovScore):
        cp = score_to_cp(score)
        return cp, score
    # fallback
    return score_to_cp(chess.engine.PovScore(None)), score


def probe_parquet_columns(pq_path: Path) -> List[str]:
    pf = pq.ParquetFile(str(pq_path))
    return pf.schema.names


def iterate_games_or_positions(pq_path: Path, *, min_elo: int, max_elo: int) -> Iterable[Dict[str, Any]]:
    """Yield rows or grouped games from the parquet.

    Yields dict rows that include at minimum:
      - 'board_history' (list of lists)
      - 'repetition_flags' (list)
      - 'policy_move' (tuple)
      - elo fields (for filtering)

    If a 'game_id' column is present the function yields grouped games
    (dict with 'rows' list). Otherwise yields individual rows (positions).
    """
    pf = pq.ParquetFile(str(pq_path))
    cols = pf.schema.names

    # Candidate game id columns
    gid_candidates = [c for c in ["game_id", "game_uuid", "game_hash", "game_index", "pgn_id"] if c in cols]

    # Read all rows into table (val set should be manageable). If huge, user can sample externally.
    table = pf.read()
    df = table.to_pandas()

    # Filter by rating if possible
    if "white_elo" in df.columns and "black_elo" in df.columns:
        mask = (df["white_elo"] >= min_elo) & (df["white_elo"] <= max_elo) & (df["black_elo"] >= min_elo) & (df["black_elo"] <= max_elo)
        df = df[mask]
    elif "active_elo" in df.columns:
        mask = (df["active_elo"] >= min_elo) & (df["active_elo"] <= max_elo)
        df = df[mask]

    if gid_candidates:
        gid = gid_candidates[0]
        grouped = df.groupby(gid)
        for game_id, group in grouped:
            yield {"game_id": game_id, "rows": group}
    else:
        for _, row in df.iterrows():
            yield {"row": row}


def main():
    p = argparse.ArgumentParser(description="Compare ACPL between dataset and model")
    p.add_argument("--parquet", required=True, help="Path to val_parquet (e.g. data_v2/val_balanced.parquet)")
    p.add_argument("--num-games", type=int, default=4000, help="Number of games (or positions) to sample")
    p.add_argument("--min-elo", type=int, default=2400)
    p.add_argument("--max-elo", type=int, default=2500)
    p.add_argument("--engine-path", type=str, default="stockfish", help="Path to stockfish binary (or 'stockfish' on PATH)")
    p.add_argument("--engine-time", type=float, default=0.05, help="Engine time per position in seconds (or use --engine-depth)")
    p.add_argument("--engine-depth", type=int, default=None, help="Engine depth (overrides engine-time if set)")
    p.add_argument("--model-py", default="model.py")
    p.add_argument("--checkpoint", default="inference/chessformer_v2_smolgen_best.pt")
    p.add_argument("--config-name", default="smolgen")
    p.add_argument("--device", default="cpu")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--temperature", type=float, default=0.0, help="Sampling temperature (0=argmax, >0=sample)")
    p.add_argument("--top-p", type=float, default=0.95, help="Nucleus sampling threshold (1.0=no filtering)")
    args = p.parse_args()
    
    print(f"Temperature: {args.temperature}, Top-p: {args.top_p}")

    pq_path = Path(args.parquet)
    if not pq_path.exists():
        raise SystemExit(f"Parquet not found: {pq_path}")

    print(f"Probing parquet columns...")
    cols = probe_parquet_columns(pq_path)
    print("Columns:", cols)

    print("Loading model... This may take a few seconds.")
    loaded = model_loader.load_chessformer_v2(
        model_py_path=args.model_py,
        config_name=args.config_name,
        checkpoint_path=args.checkpoint,
        device=args.device,
    )

    engine = None
    try:
        print(f"Starting engine: {args.engine_path}")
        engine = chess.engine.SimpleEngine.popen_uci(args.engine_path)
    except Exception as e:
        print("Warning: failed to start engine:", e)
        # Try a common fallback: `stockfish` on PATH
        if args.engine_path != "stockfish":
            try:
                print("Trying 'stockfish' on PATH...")
                engine = chess.engine.SimpleEngine.popen_uci("stockfish")
                print("Started 'stockfish' from PATH")
            except Exception as e2:
                print("Failed to start 'stockfish' on PATH:", e2)

        if engine is None:
            print("Could not start any engine. Please install Stockfish or provide a valid UCI engine path.")
            print("On Debian/Ubuntu: sudo apt update && sudo apt install stockfish")
            print("Or download a Linux binary and pass it with --engine-path /path/to/stockfish")
            return

    rng = random.Random(args.seed)

    model_cpls = []
    human_cpls = []

    iterator = list(iterate_games_or_positions(pq_path, min_elo=args.min_elo, max_elo=args.max_elo))
    if not iterator:
        print("No games/positions matched the elo filter")
        return

    # Sample up to args.num_games
    samples = rng.sample(iterator, min(args.num_games, len(iterator)))

    for idx, item in enumerate(samples, 1):
        # Either grouped game or single row
        rows = item.get("rows")
        if rows is not None:
            # pick at most one position per game for speed: choose middle ply
            # rows is a pandas DataFrame
            mid = rows.shape[0] // 2
            row = rows.iloc[mid]
        else:
            row = item["row"]

        # Try to extract canonical board history
        if "board_history" not in row:
            print("Row missing board_history, skipping")
            continue

        board_history_raw = row["board_history"]
        # Convert numpy object array to list of lists of ints
        board_history = [list(map(int, bh)) for bh in board_history_raw]
        # board_history is nested list newest-first; use index 0 as current
        encoded_current = board_history[0]
        repetition_flags = list(row.get("repetition_flags", [0] * 8))
        board = decode_board_from_encoded(encoded_current)

        # Engine evaluate current position and after ground-truth move and model move
        # Get ground-truth policy move if present
        gt_move = None
        if "policy_move" in row and row["policy_move"] is not None:
            pm = row["policy_move"]
            from_sq, to_sq, promo = int(pm[0]), int(pm[1]), int(pm[2])
            if promo == 0:
                gt_move = chess.Move(from_sq, to_sq)
            else:
                promo_map = {1: chess.QUEEN, 2: chess.ROOK, 3: chess.BISHOP, 4: chess.KNIGHT}
                gt_move = chess.Move(from_sq, to_sq, promotion=promo_map.get(promo, chess.QUEEN))

        # model move (argmax)
        ctx = ContextOptions(
            active_elo=int(row.get("active_elo", 1900)),
            opponent_elo=int(row.get("opp_elo", row.get("opp_elo", 1900))),
            active_clock_s=float(row.get("active_clock", 300.0)),
            opponent_clock_s=float(row.get("opp_clock", 300.0)),
            active_inc_s=float(row.get("active_inc", 0.0)),
            opponent_inc_s=float(row.get("opp_inc", 0.0)),
            halfmove_clock=int(row.get("halfmove_clock", 0)),
        )

        model_move = get_model_move(
            loaded.model, loaded.device, board, board_history, repetition_flags, ctx,
            temperature=args.temperature, top_p=args.top_p, rng=rng
        )

        # Evaluate current position and resulting positions
        limit = chess.engine.Limit(time=args.engine_time) if args.engine_depth is None else chess.engine.Limit(depth=args.engine_depth)
        try:
            # Get the best move evaluation from engine (this is the "optimal" play)
            base_info = engine.analyse(board, limit)
            base_score = base_info.get("score")
            if not isinstance(base_score, chess.engine.PovScore):
                continue
            # base_cp is the eval of the best move from the side-to-move's perspective
            base_cp = score_to_cp(base_score)

            # Evaluate human (ground-truth) move
            human_cpl = None
            if gt_move is not None and gt_move in board.legal_moves:
                board_human = board.copy(stack=False)
                board_human.push(gt_move)
                human_info = engine.analyse(board_human, limit)
                human_score = human_info.get("score")
                if isinstance(human_score, chess.engine.PovScore):
                    human_cp_after = -score_to_cp(human_score)
                    human_cpl = max(0, base_cp - human_cp_after)
                    human_cpl = min(human_cpl, 1500)

            # Evaluate model move
            model_cpl = None
            if model_move is not None and model_move in board.legal_moves:
                board_model = board.copy(stack=False)
                board_model.push(model_move)
                model_info = engine.analyse(board_model, limit)
                model_score = model_info.get("score")
                if isinstance(model_score, chess.engine.PovScore):
                    model_cp_after = -score_to_cp(model_score)
                    model_cpl = max(0, base_cp - model_cp_after)
                    model_cpl = min(model_cpl, 1500)

            # Only count positions where we have both evaluations
            if human_cpl is not None and model_cpl is not None:
                human_cpls.append(human_cpl)
                model_cpls.append(model_cpl)

        except Exception as e:
            print("Engine evaluation failed for position", e)
            continue

        if idx % 100 == 0:
            print(f"Processed {idx}/{len(samples)}")

    engine.quit()

    if not model_cpls:
        print("No CPLs computed")
        return

    def print_stats(name: str, cpls: list):
        mean_acpl = statistics.mean(cpls)
        stdev = statistics.pstdev(cpls)
        median_acpl = statistics.median(cpls)
        
        perfect_moves = sum(1 for x in cpls if x == 0)
        good_moves = sum(1 for x in cpls if 0 < x <= 50)
        ok_moves = sum(1 for x in cpls if 50 < x <= 100)
        inaccuracies = sum(1 for x in cpls if 100 < x <= 300)
        mistakes = sum(1 for x in cpls if 300 < x <= 500)
        blunders = sum(1 for x in cpls if x > 500)
        
        print(f"\n=== {name} ACPL Results ===")
        print(f"Samples: {len(cpls)}")
        print(f"Mean ACPL: {mean_acpl:.2f} cp")
        print(f"Median ACPL: {median_acpl:.2f} cp")
        print(f"Stddev: {stdev:.2f} cp")
        print(f"\n  Move Quality Distribution:")
        print(f"  Perfect (0 cp):        {perfect_moves:4d} ({100*perfect_moves/len(cpls):.1f}%)")
        print(f"  Good (1-50 cp):        {good_moves:4d} ({100*good_moves/len(cpls):.1f}%)")
        print(f"  OK (51-100 cp):        {ok_moves:4d} ({100*ok_moves/len(cpls):.1f}%)")
        print(f"  Inaccuracies (101-300):{inaccuracies:4d} ({100*inaccuracies/len(cpls):.1f}%)")
        print(f"  Mistakes (301-500):    {mistakes:4d} ({100*mistakes/len(cpls):.1f}%)")
        print(f"  Blunders (>500):       {blunders:4d} ({100*blunders/len(cpls):.1f}%)")

    print_stats("Human", human_cpls)
    print_stats("Model", model_cpls)
    
    # Summary comparison
    human_mean = statistics.mean(human_cpls)
    model_mean = statistics.mean(model_cpls)
    diff = model_mean - human_mean
    print(f"\n=== Comparison ===")
    print(f"Human Mean ACPL:  {human_mean:.2f} cp")
    print(f"Model Mean ACPL:  {model_mean:.2f} cp")
    print(f"Difference:       {diff:+.2f} cp ({'model worse' if diff > 0 else 'model better'})")


if __name__ == "__main__":
    main()
