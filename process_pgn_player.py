#!/usr/bin/env python3
"""
Single-player PGN processor for fine-tuning.

Extracts training data from only ONE player's perspective across all their games.
Key differences from process_pgn.py:
  - Only saves positions from the target player's side (not the opponent's)
  - No bucket quotas (we want ALL data from this player)
  - Lower ultra-bullet filter (60s instead of 180s) to include 1+0 bullet
  - Chronological validation split (last N games) instead of random hash
  - Same output format (parquet) — fully compatible with existing dataset.py / train.py
"""
from __future__ import annotations

import argparse
import chess
import chess.pgn
import gc
import os
import sys
from collections import Counter, deque
from dataclasses import dataclass
from pathlib import Path
from typing import Deque, Dict, List, Optional, Tuple

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from tqdm import tqdm

# ───────────────────────────────────────────────────────────────────────────────
# CONFIGURATION (reuse constants from process_pgn.py for format compatibility)
# ───────────────────────────────────────────────────────────────────────────────
HISTORY_LEN = 8
CHUNK_SIZE = 50_000  # Smaller chunks since dataset is smaller

# Special policy indices for resign/flag
RESIGN_MOVE_INDEX = 4096
FLAG_MOVE_INDEX = 4097
RESIGN_SENTINEL_MOVE = [RESIGN_MOVE_INDEX, RESIGN_MOVE_INDEX, 0]
FLAG_SENTINEL_MOVE = [FLAG_MOVE_INDEX, FLAG_MOVE_INDEX, 0]

# Minimum game duration filter (seconds): base + 40*inc
# 60s allows 1+0 bullet (duration=60), filters 0+1 ultrabullet (duration=40)
MIN_GAME_DURATION = 60

# Minimum ELO for the target player
MIN_PLAYER_ELO = 0  # No ELO filter for the target player (they're the focus)
MIN_OPPONENT_ELO = 1000  # Minimum opponent ELO (filter out complete beginners)

# Termination detection
FLAG_TERMS = ("time forfeit", "forfeit on time", "flagged", "flag", "time ran out", "time expired")
RESIGN_TERMINATIONS = {"", "Normal"}
FLAG_TERMINATIONS = {"Time forfeit"}

PIECE_MAP = {
    chess.PAWN: 1, chess.KNIGHT: 2, chess.BISHOP: 3,
    chess.ROOK: 4, chess.QUEEN: 5, chess.KING: 6,
}
PROMOTION_MAP = {None: 0, chess.QUEEN: 1, chess.ROOK: 2, chess.BISHOP: 3, chess.KNIGHT: 4}

SCHEMA_COLUMNS = [
    "game_id", "ply", "split", "board_history", "legal_moves", "time_history",
    "active_elo", "opp_elo", "move_ply", "halfmove_clock", "repetition_flags",
    "castling_rights", "ep_square", "tc_cat", "active_clock", "opp_clock",
    "active_inc", "opp_inc", "y_val", "policy_move", "policy_is_resign",
    "policy_is_flag", "time_target",
]


# ───────────────────────────────────────────────────────────────────────────────
# UTILITY FUNCTIONS (identical to process_pgn.py for compatibility)
# ───────────────────────────────────────────────────────────────────────────────
def parse_time_control(tc_string: str) -> Optional[Tuple[int, int]]:
    if not tc_string or tc_string == "-":
        return None
    try:
        if "+" in tc_string:
            base, inc = tc_string.split("+")
            return int(base), int(inc)
        return int(tc_string), 0
    except ValueError:
        return None


def get_tc_category(base: int, inc: int) -> int:
    duration = base + 40 * inc
    if duration < 600: return 0    # Blitz
    if duration < 1800: return 1   # Rapid
    return 2                        # Classical


def encode_board(board: chess.Board) -> List[int]:
    tokens = [0] * 64
    for square in range(64):
        piece = board.piece_at(square)
        if not piece:
            continue
        val = PIECE_MAP[piece.piece_type]
        if piece.color == chess.WHITE:
            tokens[square] = val
        else:
            tokens[square] = val + 6
    return tokens


def canonicalize(board: chess.Board, turn: chess.Color) -> chess.Board:
    return board if turn == chess.WHITE else board.mirror()


def encode_legal_moves(board: chess.Board) -> List[List[int]]:
    moves = []
    for move in board.legal_moves:
        promo = PROMOTION_MAP.get(move.promotion, 0)
        moves.append([move.from_square, move.to_square, promo])
    return moves


def repetition_flags(history: List[chess.Board]) -> List[int]:
    flags: List[int] = []
    seen: Counter = Counter()
    for b in history:
        key = b.board_fen()
        seen[key] += 1
        flags.append(1 if seen[key] >= 2 else 0)
    while len(flags) < HISTORY_LEN:
        flags.insert(0, 0)
    return flags[:HISTORY_LEN]


def extract_time_seconds(node: chess.pgn.ChildNode) -> Optional[float]:
    clk = node.clock()
    if clk is None:
        return None
    try:
        return float(clk)
    except ValueError:
        return None


# ───────────────────────────────────────────────────────────────────────────────
# GAME PROCESSING (single-player variant)
# ───────────────────────────────────────────────────────────────────────────────
@dataclass
class PositionSample:
    row: dict


def process_game_for_player(
    game: chess.pgn.Game,
    target_username: str,
) -> List[PositionSample]:
    """
    Process a single game and return samples ONLY from the target player's side.
    
    Unlike process_pgn.py which saves both sides, this only extracts positions
    where it's the target player's turn to move.
    """
    headers = game.headers

    # --- Identify which color the target player is ---
    white_player = headers.get("White", "").lower()
    black_player = headers.get("Black", "").lower()
    target_lower = target_username.lower()

    if white_player == target_lower:
        player_color = chess.WHITE
    elif black_player == target_lower:
        player_color = chess.BLACK
    else:
        return []  # Target player not in this game

    # --- Time control filter ---
    tc = parse_time_control(headers.get("TimeControl", ""))
    if not tc:
        return []
    base, inc = tc
    duration = base + 40 * inc
    if duration < MIN_GAME_DURATION:
        return []

    # --- ELO filters ---
    white_elo = int(headers.get("WhiteElo", 0))
    black_elo = int(headers.get("BlackElo", 0))

    player_elo = white_elo if player_color == chess.WHITE else black_elo
    opp_elo = black_elo if player_color == chess.WHITE else white_elo

    if player_elo < MIN_PLAYER_ELO:
        return []
    if opp_elo < MIN_OPPONENT_ELO:
        return []

    # --- Game metadata ---
    result = headers.get("Result", "*")
    termination = headers.get("Termination", "")
    game_id = headers.get("Site", "unknown").split("/")[-1]
    tc_cat = get_tc_category(base, inc)
    termination_lc = termination.lower()
    is_flag_loss = any(term in termination_lc for term in FLAG_TERMS)

    # --- Walk through moves ---
    board = game.board()
    node = game
    ply = 0
    board_history: Deque[List[int]] = deque(maxlen=HISTORY_LEN)
    raw_boards: Deque[chess.Board] = deque(maxlen=HISTORY_LEN)
    time_history: Deque[float] = deque(maxlen=HISTORY_LEN)
    samples: List[PositionSample] = []

    white_clock = float(base)
    black_clock = float(base)
    white_last = 0.0
    black_last = 0.0

    while True:
        next_node = node.next()
        if next_node is None:
            break
        move = next_node.move
        turn = board.turn

        canonical = canonicalize(board, turn)
        board_tokens = encode_board(canonical)
        board_history.appendleft(board_tokens)
        raw_boards.appendleft(canonical.copy())

        turn_clock = white_clock if turn == chess.WHITE else black_clock
        opp_clock_val = black_clock if turn == chess.WHITE else white_clock
        turn_inc = inc
        opp_inc = inc

        # --- Only save positions from the target player's turns ---
        if turn == player_color:
            if turn == chess.WHITE:
                active_elo, active_opp_elo = white_elo, black_elo
                policy_from = move.from_square
                policy_to = move.to_square
                policy_promo = PROMOTION_MAP.get(move.promotion, 0)
                y_val = 1.0 if result == "1-0" else 0.0 if result == "0-1" else 0.5
            else:
                active_elo, active_opp_elo = black_elo, white_elo
                policy_from = move.from_square ^ 56  # Mirror for canonical view
                policy_to = move.to_square ^ 56
                policy_promo = PROMOTION_MAP.get(move.promotion, 0)
                y_val = 0.0 if result == "1-0" else 1.0 if result == "0-1" else 0.5

            history_tensor = list(board_history)
            while len(history_tensor) < HISTORY_LEN:
                history_tensor.append([0] * 64)
            history_tensor = history_tensor[:HISTORY_LEN]

            rep = repetition_flags(list(raw_boards))
            legal_moves = encode_legal_moves(canonical)

            time_hist = list(time_history)
            while len(time_hist) < HISTORY_LEN:
                time_hist.insert(0, 0.0)
            time_hist = time_hist[-HISTORY_LEN:]

            row = {
                "game_id": f"{game_id}_{ply}",
                "ply": ply,
                "split": "",  # Filled after all games processed
                "board_history": history_tensor,
                "legal_moves": legal_moves,
                "time_history": time_hist,
                "active_elo": active_elo,
                "opp_elo": active_opp_elo,
                "move_ply": ply,
                "halfmove_clock": board.halfmove_clock,
                "repetition_flags": rep,
                "castling_rights": [
                    int(board.has_kingside_castling_rights(chess.WHITE)),
                    int(board.has_queenside_castling_rights(chess.WHITE)),
                    int(board.has_kingside_castling_rights(chess.BLACK)),
                    int(board.has_queenside_castling_rights(chess.BLACK)),
                ],
                "ep_square": board.ep_square if board.ep_square is not None else -1,
                "tc_cat": tc_cat,
                "active_clock": turn_clock,
                "opp_clock": opp_clock_val,
                "active_inc": turn_inc,
                "opp_inc": opp_inc,
                "y_val": y_val,
                "policy_move": [policy_from, policy_to, policy_promo],
                "policy_is_resign": False,
                "policy_is_flag": False,
                "time_target": 0.0,  # Updated below when clock info is available
            }
            samples.append(PositionSample(row=row))

        # --- Update clocks (always, even on opponent turns — needed for history) ---
        clk_seconds = extract_time_seconds(next_node)
        if clk_seconds is not None:
            elapsed = max(0.0, turn_clock + turn_inc - clk_seconds)
            time_history.append(elapsed)
            if turn == chess.WHITE:
                white_clock = clk_seconds
                white_last = elapsed
            else:
                black_clock = clk_seconds
                black_last = elapsed
            # Update time_target for the last sample we added (if it was ours)
            if turn == player_color and samples:
                samples[-1].row["time_target"] = elapsed
        else:
            time_history.append(0.0)

        board.push(move)
        ply += 1
        node = next_node

    # --- Handle resign position (only if the TARGET PLAYER resigned) ---
    if result in {"1-0", "0-1"} and termination in RESIGN_TERMINATIONS:
        if not board.is_checkmate() and not board.is_stalemate():
            loser = chess.BLACK if result == "1-0" else chess.WHITE
            if loser == player_color and board.turn == loser:
                canonical = canonicalize(board, loser)
                board_history.appendleft(encode_board(canonical))
                raw_boards.appendleft(canonical.copy())

                history_tensor = list(board_history)
                while len(history_tensor) < HISTORY_LEN:
                    history_tensor.append([0] * 64)

                rep = repetition_flags(list(raw_boards))
                legal_moves = encode_legal_moves(canonical)
                active_elo_val = white_elo if loser == chess.WHITE else black_elo
                opp_elo_val = black_elo if loser == chess.WHITE else white_elo
                active_clock_val = white_clock if loser == chess.WHITE else black_clock

                row = {
                    "game_id": f"{game_id}_{ply}_resign",
                    "ply": ply,
                    "split": "",
                    "board_history": history_tensor[:HISTORY_LEN],
                    "legal_moves": legal_moves,
                    "time_history": list(time_history)[-HISTORY_LEN:],
                    "active_elo": active_elo_val,
                    "opp_elo": opp_elo_val,
                    "move_ply": ply,
                    "halfmove_clock": board.halfmove_clock,
                    "repetition_flags": rep,
                    "castling_rights": [
                        int(board.has_kingside_castling_rights(chess.WHITE)),
                        int(board.has_queenside_castling_rights(chess.WHITE)),
                        int(board.has_kingside_castling_rights(chess.BLACK)),
                        int(board.has_queenside_castling_rights(chess.BLACK)),
                    ],
                    "ep_square": board.ep_square if board.ep_square is not None else -1,
                    "tc_cat": tc_cat,
                    "active_clock": active_clock_val,
                    "opp_clock": black_clock if loser == chess.WHITE else white_clock,
                    "active_inc": inc,
                    "opp_inc": inc,
                    "y_val": 0.0,
                    "policy_move": RESIGN_SENTINEL_MOVE,
                    "policy_is_resign": True,
                    "policy_is_flag": False,
                    "time_target": 0.0,
                }
                samples.append(PositionSample(row=row))

    # --- Handle flag loss position (only if the TARGET PLAYER flagged) ---
    if (is_flag_loss or termination in FLAG_TERMINATIONS) and result in {"1-0", "0-1"}:
        loser = chess.BLACK if result == "1-0" else chess.WHITE
        if loser == player_color and board.turn == loser:
            canonical = canonicalize(board, loser)
            board_history.appendleft(encode_board(canonical))
            raw_boards.appendleft(canonical.copy())

            history_tensor = list(board_history)
            while len(history_tensor) < HISTORY_LEN:
                history_tensor.append([0] * 64)

            rep = repetition_flags(list(raw_boards))
            legal_moves = encode_legal_moves(canonical)
            active_elo_val = white_elo if loser == chess.WHITE else black_elo
            opp_elo_val = black_elo if loser == chess.WHITE else white_elo
            active_clock_val = white_clock if loser == chess.WHITE else black_clock

            flag_time_target = white_last if loser == chess.WHITE else black_last
            row = {
                "game_id": f"{game_id}_{ply}_flag",
                "ply": ply,
                "split": "",
                "board_history": history_tensor[:HISTORY_LEN],
                "legal_moves": legal_moves,
                "time_history": list(time_history)[-HISTORY_LEN:],
                "active_elo": active_elo_val,
                "opp_elo": opp_elo_val,
                "move_ply": ply,
                "halfmove_clock": board.halfmove_clock,
                "repetition_flags": rep,
                "castling_rights": [
                    int(board.has_kingside_castling_rights(chess.WHITE)),
                    int(board.has_queenside_castling_rights(chess.WHITE)),
                    int(board.has_kingside_castling_rights(chess.BLACK)),
                    int(board.has_queenside_castling_rights(chess.BLACK)),
                ],
                "ep_square": board.ep_square if board.ep_square is not None else -1,
                "tc_cat": tc_cat,
                "active_clock": active_clock_val,
                "opp_clock": black_clock if loser == chess.WHITE else white_clock,
                "active_inc": inc,
                "opp_inc": inc,
                "y_val": 0.0,
                "policy_move": FLAG_SENTINEL_MOVE,
                "policy_is_resign": False,
                "policy_is_flag": True,
                "time_target": flag_time_target,
            }
            samples.append(PositionSample(row=row))

    return samples


# ───────────────────────────────────────────────────────────────────────────────
# MAIN
# ───────────────────────────────────────────────────────────────────────────────
def main():
    # Declare module-level globals we may override from CLI early so they
    # can be referenced as globals throughout this function without
    # triggering Python's local-variable rules.
    global MIN_GAME_DURATION, MIN_OPPONENT_ELO
    parser = argparse.ArgumentParser(
        description="Process PGN files for single-player fine-tuning data"
    )
    parser.add_argument(
        "--username", type=str, required=True,
        help="Lichess username of the target player (case-insensitive)"
    )
    parser.add_argument(
        "--input-dir", type=str, default="data/user-data-raw",
        help="Directory containing PGN file(s)"
    )
    parser.add_argument(
        "--output-dir", type=str, default="data/user",
        help="Output directory for parquet files"
    )
    parser.add_argument(
        "--val-games", type=int, default=0,
        help="Number of most recent games to use for validation (chronological split). "
             "If 0, uses --val-fraction instead."
    )
    parser.add_argument(
        "--val-fraction", type=float, default=0.10,
        help="Fraction of games for validation (used when --val-games=0). Default: 0.10"
    )
    parser.add_argument(
        "--min-duration", type=int, default=MIN_GAME_DURATION,
        help=f"Minimum game duration in seconds (base + 40*inc). Default: {MIN_GAME_DURATION}"
    )
    parser.add_argument(
        "--min-opponent-elo", type=int, default=MIN_OPPONENT_ELO,
        help=f"Minimum opponent ELO. Default: {MIN_OPPONENT_ELO}"
    )
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)

    if not input_dir.exists():
        print(f"ERROR: Input directory {input_dir} does not exist.")
        sys.exit(1)

    pgn_files = sorted(input_dir.glob("*.pgn"))
    if not pgn_files:
        print(f"No PGN files found in {input_dir}")
        sys.exit(1)

    # Create output directories
    (output_dir / "train").mkdir(parents=True, exist_ok=True)

    print(f"=== Single-Player PGN Processor ===")
    print(f"Target player: {args.username}")
    print(f"Input: {input_dir.resolve()}")
    print(f"Output: {output_dir.resolve()}")
    print(f"Min game duration: {args.min_duration}s")
    print(f"Min opponent ELO: {args.min_opponent_elo}")
    print(f"PGN files: {len(pgn_files)}")
    if args.val_games > 0:
        print(f"Val split: last {args.val_games} games (chronological)")
    else:
        print(f"Val split: {args.val_fraction*100:.0f}% of games (chronological)")
    print()

    # Override module-level defaults with CLI args
    MIN_GAME_DURATION = args.min_duration
    MIN_OPPONENT_ELO = args.min_opponent_elo

    # ─── Pass 1: Process all games, collect samples with game order ─────
    # We process sequentially since Lichess exports are chronological
    # and we need to preserve game order for chronological splitting.
    all_game_samples: List[Tuple[str, List[dict]]] = []  # (game_id, [rows])
    total_games = 0
    total_samples = 0
    skipped_not_player = 0
    skipped_tc = 0
    skipped_elo = 0
    tc_distribution: Dict[int, int] = {0: 0, 1: 0, 2: 0}

    for pgn_path in pgn_files:
        file_size = os.path.getsize(pgn_path)
        pbar = tqdm(total=file_size, unit='B', unit_scale=True, desc=pgn_path.name)

        with open(pgn_path) as pgn_file:
            last_pos = 0
            while True:
                try:
                    game = chess.pgn.read_game(pgn_file)
                except Exception:
                    continue

                if game is None:
                    break

                current_pos = pgn_file.tell()
                pbar.update(current_pos - last_pos)
                last_pos = current_pos

                samples = process_game_for_player(game, args.username)

                if not samples:
                    # Track skip reason
                    headers = game.headers
                    w = headers.get("White", "").lower()
                    b = headers.get("Black", "").lower()
                    if args.username.lower() not in (w, b):
                        skipped_not_player += 1
                    else:
                        tc = parse_time_control(headers.get("TimeControl", ""))
                        if tc:
                            base, inc = tc
                            dur = base + 40 * inc
                            if dur < MIN_GAME_DURATION:
                                skipped_tc += 1
                            else:
                                skipped_elo += 1
                    del game
                    continue

                total_games += 1
                game_id = samples[0].row["game_id"].rsplit("_", 1)[0]
                rows = [s.row for s in samples]
                total_samples += len(rows)

                # Track TC distribution
                tc_cat = rows[0]["tc_cat"]
                tc_distribution[tc_cat] = tc_distribution.get(tc_cat, 0) + 1

                all_game_samples.append((game_id, rows))

                del samples, game
                if total_games % 2000 == 0:
                    gc.collect()

        pbar.close()

    print(f"\n=== Processing Summary ===")
    print(f"Games processed: {total_games:,}")
    print(f"Positions extracted: {total_samples:,}")
    print(f"Skipped (not target player): {skipped_not_player:,}")
    print(f"Skipped (time control < {MIN_GAME_DURATION}s): {skipped_tc:,}")
    print(f"Skipped (ELO filter): {skipped_elo:,}")
    tc_names = {0: "Blitz", 1: "Rapid", 2: "Classical"}
    for tc_cat, count in sorted(tc_distribution.items()):
        print(f"  {tc_names.get(tc_cat, f'TC{tc_cat}')}: {count:,} games")

    if total_games == 0:
        print("No games found for this player!")
        sys.exit(1)

    # ─── Pass 2: Chronological train/val split ─────────────────────────
    # Lichess exports are oldest-first, so later entries = more recent games.
    if args.val_games > 0:
        val_count = min(args.val_games, total_games // 2)  # Never more than half
    else:
        val_count = max(1, int(total_games * args.val_fraction))

    train_count = total_games - val_count
    print(f"\nTrain games: {train_count:,} | Val games: {val_count:,}")

    train_rows: List[dict] = []
    val_rows: List[dict] = []

    for i, (game_id, rows) in enumerate(all_game_samples):
        split = "train" if i < train_count else "val"
        for row in rows:
            row["split"] = split
        if split == "train":
            train_rows.extend(rows)
        else:
            val_rows.extend(rows)

    print(f"Train positions: {len(train_rows):,} | Val positions: {len(val_rows):,}")

    # ─── Pass 3: Write parquet files ───────────────────────────────────
    # Write train and val into SEPARATE directories so dataset.py's hash-based
    # file splitter doesn't mix them. Train goes to output_dir/train/,
    # val goes to output_dir-val/train/ (the nested "train" is required by
    # dataset.py which always looks for a train/ subdirectory).
    val_output_dir = output_dir.parent / f"{output_dir.name}-val"

    def write_split(rows: List[dict], split: str):
        if not rows:
            return
        if split == "val":
            split_dir = val_output_dir / "train"
        else:
            split_dir = output_dir / "train"
        split_dir.mkdir(parents=True, exist_ok=True)
        
        # Write in chunks
        for chunk_start in range(0, len(rows), CHUNK_SIZE):
            chunk_rows = rows[chunk_start:chunk_start + CHUNK_SIZE]
            df = pd.DataFrame(chunk_rows, columns=SCHEMA_COLUMNS)
            table = pa.Table.from_pandas(df, preserve_index=False)
            chunk_idx = chunk_start // CHUNK_SIZE
            filename = f"player_{split}_{chunk_idx:05d}.parquet"
            out_path = split_dir / filename
            pq.write_table(table, out_path)
            print(f"  Wrote {out_path} ({len(chunk_rows):,} rows)")
            del df, table
            gc.collect()

    print(f"\nWriting train data...")
    write_split(train_rows, "train")
    print(f"Writing val data...")
    write_split(val_rows, "val")

    # ─── Summary stats ─────────────────────────────────────────────────
    print(f"\n=== Done ===")
    print(f"Train data: {output_dir.resolve()}")
    print(f"Val data:   {val_output_dir.resolve()}")
    train_files = list((output_dir / 'train').glob('*.parquet'))
    val_files = list((val_output_dir / 'train').glob('*.parquet')) if val_output_dir.exists() else []
    print(f"Parquet files: {len(train_files)} train, {len(val_files)} val")
    print(f"\nFor fine-tuning, use:")
    print(f"  --data-dir {output_dir} --val-data-dir {val_output_dir}")

    # ELO distribution summary
    all_rows = train_rows + val_rows
    elos = [r["active_elo"] for r in all_rows]
    if elos:
        import statistics
        print(f"\nPlayer ELO stats:")
        print(f"  Min: {min(elos)}, Max: {max(elos)}")
        print(f"  Mean: {statistics.mean(elos):.0f}, Median: {statistics.median(elos):.0f}")
        print(f"  Stdev: {statistics.stdev(elos):.0f}")


if __name__ == "__main__":
    main()
