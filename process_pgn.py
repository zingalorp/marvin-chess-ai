#!/usr/bin/env python3
"""
High-performance PGN processor with:
- Bucket quotas for data balancing (elo, ply, clock)
- Stratified train/val/test splits
- Flag/resign move as policy targets
- Multi-worker parallel processing with per-file progress bars
- No shared-state lock contention (workers are independent)
"""
from __future__ import annotations

import argparse
import chess
import chess.pgn
import gc
import hashlib
import json
import multiprocessing as mp
import os
import random
import sys
from collections import Counter, defaultdict, deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Deque, Dict, List, Optional, Tuple

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from tqdm import tqdm

# ───────────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ───────────────────────────────────────────────────────────────────────────────
INPUT_DIR = Path("data/raw")
OUTPUT_ROOT = Path("data")
HISTORY_LEN = 8
CHUNK_SIZE = 250_000

# Special policy indices for resign/flag
RESIGN_MOVE_INDEX = 4096
FLAG_MOVE_INDEX = 4097
RESIGN_SENTINEL_MOVE = [RESIGN_MOVE_INDEX, RESIGN_MOVE_INDEX, 0]
FLAG_SENTINEL_MOVE = [FLAG_MOVE_INDEX, FLAG_MOVE_INDEX, 0]

# Bucket config
ELO_MIN, ELO_MAX, ELO_STEP = 1200, 2900, 100
PLY_BANDS = [20, 40, 60, 80]
CLOCK_BANDS = [15, 60, 300, 600]
BUCKET_QUOTA = 150_000  # Per (elo, ply, clock) bucket, ignoring increment

# Split ratios
SPLIT_RATIOS = {"train": 0.85, "val": 0.10, "test": 0.05}

# Worker config
DEFAULT_NUM_WORKERS = 8
GLOBAL_SEED = 42

# Termination detection
FLAG_TERMS = ("time forfeit", "forfeit on time", "flagged", "flag", "time ran out", "time expired")
# For resignations, only count Normal terminations (not time forfeit, abandoned)
RESIGN_TERMINATIONS = {"", "Normal"}
# For flag losses, look for time forfeit terminations
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
# GLOBAL WORKER STATE
# ───────────────────────────────────────────────────────────────────────────────
_worker_quota_counts: Dict[Tuple[int, int, int], int] = {}
_worker_quota_max: int = 0
_worker_split_totals: Dict[Tuple[int, int, int, int], int] = {}
_worker_split_counts: Dict[str, Dict[Tuple[int, int, int, int], int]] = {}
_worker_rng: random.Random = random.Random()
_worker_id: int = 0
_worker_seed: int = 0


def init_worker(worker_id: int, bucket_quota: int, seed: int):
    """Initialize worker-local state."""
    global _worker_quota_counts, _worker_quota_max, _worker_split_totals
    global _worker_split_counts, _worker_rng, _worker_id, _worker_seed
    
    _worker_quota_counts = defaultdict(int)
    _worker_quota_max = bucket_quota
    _worker_split_totals = defaultdict(int)
    _worker_split_counts = {split: defaultdict(int) for split in SPLIT_RATIOS}
    _worker_rng = random.Random(seed + worker_id)
    _worker_id = worker_id
    _worker_seed = seed


# ───────────────────────────────────────────────────────────────────────────────
# UTILITY FUNCTIONS
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
    # Pad to HISTORY_LEN
    while len(flags) < HISTORY_LEN:
        flags.insert(0, 0)
    return flags[:HISTORY_LEN]


def bucket_key(elo: int, ply: int, clock_seconds: float, inc_seconds: float) -> Tuple[int, int, int, int]:
    """Returns (b_elo, b_ply, b_clock, b_inc). Quotas ignore b_inc."""
    if elo < ELO_MIN:
        b_elo = 0
    elif elo >= ELO_MAX:
        b_elo = (ELO_MAX - ELO_MIN) // ELO_STEP + 1
    else:
        b_elo = (elo - ELO_MIN) // ELO_STEP + 1

    b_ply = len(PLY_BANDS)
    for i, limit in enumerate(PLY_BANDS):
        if ply < limit:
            b_ply = i
            break

    b_clk = len(CLOCK_BANDS)
    for i, limit in enumerate(CLOCK_BANDS):
        if clock_seconds < limit:
            b_clk = i
            break

    b_inc = 1 if inc_seconds > 0 else 0
    return b_elo, b_ply, b_clk, b_inc


def quota_accept(bucket: Tuple[int, int, int, int]) -> bool:
    """Check if a sample should be kept based on local bucket quota."""
    if _worker_quota_max <= 0:
        return True
    key = bucket[:3]  # Ignore increment
    count = _worker_quota_counts[key]
    if count >= _worker_quota_max:
        return False
    _worker_quota_counts[key] = count + 1
    return True


def assign_split(game_id: str, bucket_counts: Dict[Tuple[int, int, int, int], int]) -> str:
    """Assign game to split based on stratified need."""
    if not bucket_counts:
        return "train"
    
    best_split = None
    best_score = -1e9
    
    for split in SPLIT_RATIOS:
        score = 0.0
        split_dict = _worker_split_counts[split]
        for bucket, delta in bucket_counts.items():
            total = _worker_split_totals.get(bucket, 0)
            split_count = split_dict.get(bucket, 0)
            frac = split_count / total if total > 0 else 0.0
            need = SPLIT_RATIOS[split] - frac
            score += need * delta
        # Deterministic tie-breaker
        payload = f"{game_id}|{split}|{_worker_seed}".encode()
        digest = hashlib.blake2s(payload, digest_size=4).digest()
        score -= int.from_bytes(digest, "little") / 2**32 * 1e-3
        if score > best_score:
            best_score = score
            best_split = split
    
    chosen = best_split or "train"
    
    # Update local tracking
    for bucket, delta in bucket_counts.items():
        _worker_split_totals[bucket] = _worker_split_totals.get(bucket, 0) + delta
        _worker_split_counts[chosen][bucket] = _worker_split_counts[chosen].get(bucket, 0) + delta
    
    return chosen


# ───────────────────────────────────────────────────────────────────────────────
# SPLIT WRITER (Per-Worker)
# ───────────────────────────────────────────────────────────────────────────────
class SplitWriter:
    def __init__(self, root: Path, worker_id: int, chunk_size: int = CHUNK_SIZE):
        self.root = root
        self.worker_id = worker_id
        self.chunk_size = chunk_size
        self.buffers: Dict[str, List[dict]] = defaultdict(list)
        self.counters: Dict[str, int] = defaultdict(int)
        for split in SPLIT_RATIOS:
            (self.root / split).mkdir(parents=True, exist_ok=True)

    def add_rows(self, split: str, rows: List[dict]):
        buf = self.buffers[split]
        buf.extend(rows)
        # Flush when ANY split buffer exceeds threshold
        if len(buf) >= self.chunk_size:
            self._flush(split)
        # Also check total memory pressure across all buffers
        total_rows = sum(len(b) for b in self.buffers.values())
        if total_rows >= self.chunk_size * 2:
            self._flush_largest()

    def finalize(self):
        for split in list(self.buffers.keys()):
            if self.buffers[split]:
                self._flush(split)

    def _flush_largest(self):
        """Flush the largest buffer to relieve memory pressure."""
        if not self.buffers:
            return
        largest_split = max(self.buffers.keys(), key=lambda s: len(self.buffers[s]))
        if self.buffers[largest_split]:
            self._flush(largest_split)

    def _flush(self, split: str):
        rows = self.buffers[split]
        if not rows:
            return
        df = pd.DataFrame(rows, columns=SCHEMA_COLUMNS)
        table = pa.Table.from_pandas(df, preserve_index=False)
        shard = self.counters[split]
        out_path = self.root / split / f"chunk_w{self.worker_id:02d}_{split}_{shard:05d}.parquet"
        pq.write_table(table, out_path)
        self.buffers[split] = []
        self.counters[split] += 1
        del df, table
        gc.collect()


# ───────────────────────────────────────────────────────────────────────────────
# GAME PROCESSING
# ───────────────────────────────────────────────────────────────────────────────
@dataclass
class PositionSample:
    row: dict
    bucket: Tuple[int, int, int, int]


def extract_time_seconds(node: chess.pgn.ChildNode) -> Optional[float]:
    clk = node.clock()
    if clk is None:
        return None
    try:
        return float(clk)
    except ValueError:
        return None


def process_game(game: chess.pgn.Game) -> List[PositionSample]:
    """
    Process a single game and return samples.
    Uses worker-local quota manager and split assignment.
    """
    headers = game.headers
    tc = parse_time_control(headers.get("TimeControl", ""))
    if not tc:
        return []
    base, inc = tc
    duration = base + 40 * inc
    if duration < 180:  # Skip ultra-bullet
        return []

    white_elo = int(headers.get("WhiteElo", 0))
    black_elo = int(headers.get("BlackElo", 0))
    if white_elo < 1200 or black_elo < 1200:
        return []

    result = headers.get("Result", "*")
    termination = headers.get("Termination", "")
    game_id = headers.get("Site", "unknown").split("/")[-1]
    tc_cat = get_tc_category(base, inc)
    termination_lc = termination.lower()
    is_flag_loss = any(term in termination_lc for term in FLAG_TERMS)

    board = game.board()
    node = game
    ply = 0
    board_history: Deque[List[int]] = deque(maxlen=HISTORY_LEN)
    raw_boards: Deque[chess.Board] = deque(maxlen=HISTORY_LEN)
    time_history: Deque[float] = deque(maxlen=HISTORY_LEN)
    samples: List[PositionSample] = []
    bucket_counts: Dict[Tuple[int, int, int, int], int] = defaultdict(int)
    
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
        opp_clock = black_clock if turn == chess.WHITE else white_clock
        turn_inc = inc
        opp_inc = inc

        if turn == chess.WHITE:
            active_elo, opp_elo = white_elo, black_elo
            policy_from = move.from_square
            policy_to = move.to_square
            policy_promo = PROMOTION_MAP.get(move.promotion, 0)
            y_val = 1.0 if result == "1-0" else 0.0 if result == "0-1" else 0.5
        else:
            active_elo, opp_elo = black_elo, white_elo
            policy_from = move.from_square ^ 56  # Mirror for canonical view
            policy_to = move.to_square ^ 56
            policy_promo = PROMOTION_MAP.get(move.promotion, 0)
            y_val = 0.0 if result == "1-0" else 1.0 if result == "0-1" else 0.5

        bucket = bucket_key(active_elo, ply, turn_clock, turn_inc)
        keep_sample = quota_accept(bucket)
        row_ref: Optional[dict] = None

        if keep_sample:
            bucket_counts[bucket] += 1
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
                "split": "",  # Filled after game ends
                "board_history": history_tensor,
                "legal_moves": legal_moves,
                "time_history": time_hist,
                "active_elo": active_elo,
                "opp_elo": opp_elo,
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
                "opp_clock": opp_clock,
                "active_inc": turn_inc,
                "opp_inc": opp_inc,
                "y_val": y_val,
                "policy_move": [policy_from, policy_to, policy_promo],
                "policy_is_resign": False,
                "policy_is_flag": False,
                "time_target": 0.0,
            }
            samples.append(PositionSample(row=row, bucket=bucket))
            row_ref = row

        # Update clocks
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
            if row_ref is not None:
                row_ref["time_target"] = elapsed
        else:
            time_history.append(0.0)

        board.push(move)
        ply += 1
        node = next_node

    # Handle resign positions (only for Normal terminations, not time forfeit)
    if result in {"1-0", "0-1"} and termination in RESIGN_TERMINATIONS:
        if not board.is_checkmate() and not board.is_stalemate():
            loser = chess.BLACK if result == "1-0" else chess.WHITE
            if board.turn == loser:
                canonical = canonicalize(board, loser)
                board_history.appendleft(encode_board(canonical))
                raw_boards.appendleft(canonical.copy())
                
                history_tensor = list(board_history)
                while len(history_tensor) < HISTORY_LEN:
                    history_tensor.append([0] * 64)
                
                rep = repetition_flags(list(raw_boards))
                legal_moves = encode_legal_moves(canonical)
                active_elo = white_elo if loser == chess.WHITE else black_elo
                opp_elo = black_elo if loser == chess.WHITE else white_elo
                active_clock = white_clock if loser == chess.WHITE else black_clock
                
                bucket = bucket_key(active_elo, ply, active_clock, inc)
                if quota_accept(bucket):
                    bucket_counts[bucket] += 1
                    row = {
                        "game_id": f"{game_id}_{ply}_resign",
                        "ply": ply,
                        "split": "",
                        "board_history": history_tensor[:HISTORY_LEN],
                        "legal_moves": legal_moves,
                        "time_history": list(time_history)[-HISTORY_LEN:],
                        "active_elo": active_elo,
                        "opp_elo": opp_elo,
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
                        "active_clock": active_clock,
                        "opp_clock": black_clock if loser == chess.WHITE else white_clock,
                        "active_inc": inc,
                        "opp_inc": inc,
                        "y_val": 0.0,
                        "policy_move": RESIGN_SENTINEL_MOVE,
                        "policy_is_resign": True,
                        "policy_is_flag": False,
                        "time_target": 0.0,
                    }
                    samples.append(PositionSample(row=row, bucket=bucket))

    # Handle flag loss positions (time forfeit terminations)
    if (is_flag_loss or termination in FLAG_TERMINATIONS) and result in {"1-0", "0-1"}:
        loser = chess.BLACK if result == "1-0" else chess.WHITE
        if board.turn == loser:
            canonical = canonicalize(board, loser)
            board_history.appendleft(encode_board(canonical))
            raw_boards.appendleft(canonical.copy())
            
            history_tensor = list(board_history)
            while len(history_tensor) < HISTORY_LEN:
                history_tensor.append([0] * 64)
            
            rep = repetition_flags(list(raw_boards))
            legal_moves = encode_legal_moves(canonical)
            active_elo = white_elo if loser == chess.WHITE else black_elo
            opp_elo = black_elo if loser == chess.WHITE else white_elo
            active_clock = white_clock if loser == chess.WHITE else black_clock
            
            bucket = bucket_key(active_elo, ply, active_clock, inc)
            if quota_accept(bucket):
                bucket_counts[bucket] += 1
                flag_time_target = white_last if loser == chess.WHITE else black_last
                row = {
                    "game_id": f"{game_id}_{ply}_flag",
                    "ply": ply,
                    "split": "",
                    "board_history": history_tensor[:HISTORY_LEN],
                    "legal_moves": legal_moves,
                    "time_history": list(time_history)[-HISTORY_LEN:],
                    "active_elo": active_elo,
                    "opp_elo": opp_elo,
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
                    "active_clock": active_clock,
                    "opp_clock": black_clock if loser == chess.WHITE else white_clock,
                    "active_inc": inc,
                    "opp_inc": inc,
                    "y_val": 0.0,
                    "policy_move": FLAG_SENTINEL_MOVE,
                    "policy_is_resign": False,
                    "policy_is_flag": True,
                    "time_target": flag_time_target,
                }
                samples.append(PositionSample(row=row, bucket=bucket))

    # Assign split and update rows
    split = assign_split(game_id, bucket_counts)
    for sample in samples:
        sample.row["split"] = split
    
    return samples


# ───────────────────────────────────────────────────────────────────────────────
# WORKER ENTRY POINT
# ───────────────────────────────────────────────────────────────────────────────
def process_single_pgn(file_path: str, worker_id: int, bucket_quota: int, seed: int) -> Dict:
    """
    Process a single PGN file. Called by each worker.
    Each worker has its own progress bar for the file.
    """
    # Initialize worker-local state
    init_worker(worker_id, bucket_quota, seed)
    
    writer = SplitWriter(OUTPUT_ROOT, worker_id, CHUNK_SIZE)
    path = Path(file_path)
    
    games_processed = 0
    samples_written = 0
    
    try:
        file_size = os.path.getsize(file_path)
        pbar = tqdm(total=file_size, unit='B', unit_scale=True, desc=f"W{worker_id}")
        
        with open(file_path) as pgn_file:
            last_pos = 0
            while True:
                try:
                    game = chess.pgn.read_game(pgn_file)
                except Exception:
                    continue
                
                if game is None:
                    break  # EOF
                
                # Update progress bar
                current_pos = pgn_file.tell()
                pbar.update(current_pos - last_pos)
                last_pos = current_pos
                
                samples = process_game(game)
                if not samples:
                    del game  # Free game memory
                    continue
                
                games_processed += 1
                
                # Group by split and write
                split_rows: Dict[str, List[dict]] = defaultdict(list)
                for sample in samples:
                    split_rows[sample.row["split"]].append(sample.row)
                    samples_written += 1
                
                for split, rows in split_rows.items():
                    writer.add_rows(split, rows)
                
                # Explicitly free memory
                del samples, split_rows, game
                
                # Periodic garbage collection every 1000 games
                if games_processed % 1000 == 0:
                    gc.collect()
        
        pbar.close()
    except Exception as e:
        print(f"[Worker {worker_id}] Error processing {file_path}: {e}", file=sys.stderr, flush=True)
        return {"worker_id": worker_id, "games": 0, "samples": 0, "error": str(e)}
    
    writer.finalize()
    
    return {
        "worker_id": worker_id,
        "games": games_processed,
        "samples": samples_written,
        "quota_counts": dict(_worker_quota_counts),
    }


def process_file_wrapper(args):
    """Wrapper for multiprocessing Pool."""
    return process_single_pgn(*args)


# ───────────────────────────────────────────────────────────────────────────────
# MAIN
# ───────────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Process PGN files into stratified parquet shards")
    parser.add_argument("--seed", type=int, default=GLOBAL_SEED, help="Random seed for reproducibility")
    parser.add_argument("--bucket-quota", type=int, default=BUCKET_QUOTA, 
                        help="Max rows per (elo, ply, clock) bucket. Use --global-quota to split across workers.")
    parser.add_argument("--global-quota", action="store_true",
                        help="Divide bucket-quota by num-workers for approximate global limit")
    parser.add_argument("--num-workers", type=int, default=DEFAULT_NUM_WORKERS, help="Number of parallel workers")
    args = parser.parse_args()
    
    # Calculate effective per-worker quota
    if args.global_quota:
        effective_quota = max(1, args.bucket_quota // args.num_workers)
        quota_mode = f"{args.bucket_quota:,} global (≈{effective_quota:,} per worker)"
    else:
        effective_quota = args.bucket_quota
        quota_mode = f"{args.bucket_quota:,} per worker (max {args.bucket_quota * args.num_workers:,} global)"
    
    print(f"=== PGN Processor ===")
    print(f"Input: {INPUT_DIR.resolve()}")
    print(f"Output: {OUTPUT_ROOT.resolve()}")
    print(f"Workers: {args.num_workers}")
    print(f"Bucket quota: {quota_mode}")
    print(f"Seed: {args.seed}")
    print()
    
    if not INPUT_DIR.exists():
        print(f"ERROR: Input directory {INPUT_DIR} does not exist.")
        sys.exit(1)
    
    # Create output directories
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    for split in SPLIT_RATIOS:
        (OUTPUT_ROOT / split).mkdir(exist_ok=True)
    
    pgn_files = sorted(INPUT_DIR.glob("*.pgn"))
    print(f"Found {len(pgn_files)} PGN files.")
    
    if not pgn_files:
        print("No PGN files to process.")
        return
    
    # --- Resume Logic ---
    # Check which files have already been processed by looking at existing chunks
    # Each file gets worker_id = file_index, so we can check for existing output
    existing_chunks = list((OUTPUT_ROOT / "train").glob("chunk_w*_train_*.parquet"))
    processed_ids = set()
    for chunk_path in existing_chunks:
        try:
            # Extract worker ID from filename like "chunk_w00_train_00000.parquet"
            filename = chunk_path.name
            parts = filename.split('_')
            if len(parts) >= 2 and parts[0] == "chunk":
                worker_id = int(parts[1][1:])  # Remove 'w' prefix
                processed_ids.add(worker_id)
        except (ValueError, IndexError):
            continue
    
    start_index = 0
    if processed_ids:
        # Find the last fully processed file
        # We'll re-process the max ID in case it was partial
        max_processed_id = max(processed_ids)
        start_index = max_processed_id  # Restart from last file (may be partial)
        
        # Remove chunks from the file we're restarting
        for split in SPLIT_RATIOS:
            chunks_to_remove = list((OUTPUT_ROOT / split).glob(f"chunk_w{start_index:02d}_*.parquet"))
            # Handle both 2-digit and 3-digit worker IDs
            chunks_to_remove.extend((OUTPUT_ROOT / split).glob(f"chunk_w{start_index:03d}_*.parquet"))
            for c in chunks_to_remove:
                try:
                    c.unlink()
                    print(f"Removed partial chunk: {c.name}")
                except OSError:
                    pass
        
        print(f"Resuming from file index {start_index} ({len(processed_ids)} files already processed)")
    else:
        print("Starting fresh (no existing data found)")
    
    # Build tasks: each file gets one worker, skip already processed
    tasks = []
    for i, pgn_file in enumerate(pgn_files):
        if i >= start_index:
            tasks.append((str(pgn_file), i, effective_quota, args.seed))
    
    if not tasks:
        print("All files already processed!")
        return
    
    print(f"Processing {len(tasks)} remaining files with {args.num_workers} workers...\n")
    
    # Use Pool.imap_unordered for efficient parallelism
    total_games = 0
    total_samples = 0
    
    with mp.Pool(processes=args.num_workers) as pool:
        results = list(pool.imap_unordered(process_file_wrapper, tasks))
    
    # Summarize results
    print(f"\n=== Processing Complete ===")
    for result in results:
        if "error" in result:
            print(f"Worker {result['worker_id']}: ERROR - {result['error']}")
        else:
            total_games += result["games"]
            total_samples += result["samples"]
    
    print(f"Total games processed: {total_games:,}")
    print(f"Total samples written: {total_samples:,}")


if __name__ == "__main__":
    main()
