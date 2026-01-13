"""Dataset for Marvin Chess with board history and legal moves."""

from __future__ import annotations

import hashlib
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch
from torch.utils.data import DataLoader, IterableDataset, get_worker_info

__all__ = [
    "StreamingDataset",
    "BalancedValDataset",
    "build_dataloaders",
    "create_dataloader",
    "create_balanced_val_loader",
    "count_samples",
]

# Special policy indices
RESIGN_MOVE_INDEX = 4096
FLAG_MOVE_INDEX = 4097
NUM_POLICY_OUTPUTS = 4098  # 64*64 + resign + flag


@dataclass
class Sample:
    """A single training sample."""
    # Board history: (8, 64) - 8 board states, each with 64 piece codes
    board_history: np.ndarray
    # Legal moves: list of (from, to, promo) tuples
    legal_moves: List[Tuple[int, int, int]]
    # Time history: (8,) - time spent on last 8 moves
    time_history: np.ndarray
    # Repetition flags: (8,) - binary flags for each history position
    repetition_flags: np.ndarray
    # Castling rights: (4,) - K, Q, k, q
    castling_rights: np.ndarray
    # EP square: -1 if none, else 0-63
    ep_square: int
    # Scalars
    active_elo: int
    opp_elo: int
    move_ply: int
    halfmove_clock: int
    tc_cat: int
    active_clock: float
    opp_clock: float
    active_inc: int
    opp_inc: int
    # Targets
    policy_move: Tuple[int, int, int]  # (from, to, promo)
    policy_is_resign: bool
    policy_is_flag: bool
    y_val: float
    time_target: float


def _sample_to_tensors(sample: Sample) -> Dict[str, torch.Tensor]:
    """Convert a Sample to model input tensors."""
    
    # Board history: (8, 64) int64
    board_history = torch.from_numpy(sample.board_history).long()
    
    # Time history: (8,) float32 - model expects most-recent first.
    # Parquet rows from `process_pgn.py` store time_history oldest->newest.
    time_history = torch.from_numpy(sample.time_history[::-1].copy()).float() / 60.0
    
    # Repetition flags: (8,) float32
    rep_flags = torch.from_numpy(sample.repetition_flags).float()
    
    # Castling rights: (4,) float32
    castling = torch.from_numpy(sample.castling_rights).float()
    
    # EP square: one-hot-ish encoding - (64,) with 1 at ep_square if valid
    ep_mask = torch.zeros(64, dtype=torch.float32)
    if 0 <= sample.ep_square < 64:
        ep_mask[sample.ep_square] = 1.0
    
    # Scalars
    # Normalize ELO to roughly [-1, 1] range: (elo - 1900) / 700
    active_elo_norm = (sample.active_elo - 1900) / 700.0
    opp_elo_norm = (sample.opp_elo - 1900) / 700.0
    # Normalize ply: ply / 100
    ply_norm = sample.move_ply / 100.0
    # Normalize clocks: log(1 + seconds) / 10
    active_clock_norm = np.log1p(sample.active_clock) / 10.0
    opp_clock_norm = np.log1p(sample.opp_clock) / 10.0
    # Normalize increment: inc / 30
    active_inc_norm = sample.active_inc / 30.0
    opp_inc_norm = sample.opp_inc / 30.0
    # Halfmove clock: hmc / 100
    hmc_norm = sample.halfmove_clock / 100.0
    
    scalars = torch.tensor([
        active_elo_norm, opp_elo_norm, ply_norm,
        active_clock_norm, opp_clock_norm,
        active_inc_norm, opp_inc_norm, hmc_norm,
    ], dtype=torch.float32)
    
    # TC category: int
    tc_cat = torch.tensor(sample.tc_cat, dtype=torch.long)
    
    # Legal move mask: (4098,) bool - True for legal moves
    legal_mask = torch.zeros(NUM_POLICY_OUTPUTS, dtype=torch.bool)
    for from_sq, to_sq, promo in sample.legal_moves:
        if 0 <= from_sq < 64 and 0 <= to_sq < 64:
            move_idx = from_sq * 64 + to_sq
            legal_mask[move_idx] = True
    
    # Policy target
    promo_target = 0
    promo_file = 0
    
    if sample.policy_is_resign:
        policy_target = RESIGN_MOVE_INDEX
    elif sample.policy_is_flag:
        policy_target = FLAG_MOVE_INDEX
    else:
        from_sq, to_sq, promo = sample.policy_move
        policy_target = from_sq * 64 + to_sq
        promo_target = promo
        if 56 <= to_sq <= 63:
            promo_file = to_sq - 56

    # Always unmask Resign/Flag for training so the model learns to suppress them
    legal_mask[RESIGN_MOVE_INDEX] = True
    legal_mask[FLAG_MOVE_INDEX] = True

    policy_target = torch.tensor(policy_target, dtype=torch.long)
    
    # Value target (regression: 0=loss, 0.5=draw, 1=win)
    y_val = torch.tensor(sample.y_val, dtype=torch.float32)
    
    # Value classification target (WDL: 0=loss, 1=draw, 2=win)
    # y_val is 0.0, 0.5, or 1.0 -> class 0, 1, 2
    y_val_cls = torch.tensor(int(sample.y_val * 2), dtype=torch.long)
    
    # Time target: normalize as fraction of remaining clock (like v1)
    # This makes the target invariant to time control
    # time_spent / clock_before, with 4th root scaling to compress range
    clock_before = max(1.0, sample.active_clock)  # Avoid division by zero
    
    # Dequantization: Add uniform noise to time_target (which is integer seconds)
    noise = torch.rand(1).item() - 0.5
    noisy_time = max(0.0, sample.time_target + noise)
    
    time_ratio = noisy_time / clock_before
    # Apply Square Root scaling (was 4th root) to compress range
    # Square root is less aggressive at the low end than 4th root, better for noisy data
    time_target_scaled = time_ratio ** 0.5
    time_target = torch.tensor(time_target_scaled, dtype=torch.float32)
    
    # Time classification target: 256 bins over [0, 1] range of scaled time
    # Clamp to [0, 1] to handle edge cases, then quantize to 256 bins
    time_bin = int(min(max(time_target_scaled, 0.0), 0.9999) * 256)
    time_target_cls = torch.tensor(time_bin, dtype=torch.long)
    
    return {
        "board_history": torch.tensor(sample.board_history, dtype=torch.long),
        "time_history": time_history,
        "rep_flags": torch.tensor(sample.repetition_flags, dtype=torch.float32),
        "castling": torch.tensor(sample.castling_rights, dtype=torch.float32),
        "ep_mask": ep_mask,
        "scalars": scalars,
        "tc_cat": tc_cat,
        "legal_mask": legal_mask,
        "policy_target": policy_target,
        "y_val": y_val,
        "y_val_cls": y_val_cls,
        "time_target_cls": time_target_cls,
        "promo_target": torch.tensor(promo_target, dtype=torch.long),
        "promo_file": torch.tensor(promo_file, dtype=torch.long),
    }


def _sample_to_tensors_fast(
    board_history: np.ndarray,
    legal_moves: List[Tuple[int, int, int]],
    time_history: np.ndarray,
    rep_flags: np.ndarray,
    castling: np.ndarray,
    ep_square: int,
    active_elo: int,
    opp_elo: int,
    move_ply: int,
    halfmove_clock: int,
    tc_cat: int,
    active_clock: float,
    opp_clock: float,
    active_inc: int,
    opp_inc: int,
    policy_move: Tuple[int, int, int],
    policy_is_resign: bool,
    policy_is_flag: bool,
    y_val: float,
    time_target: float,
    noise: float,
) -> Dict[str, torch.Tensor]:
    """Optimized tensor conversion - takes direct arguments to avoid Sample object creation."""
    
    # Time history: reverse and normalize (most-recent first)
    time_hist_reversed = time_history[::-1].copy() / 60.0
    
    # EP mask
    ep_mask = np.zeros(64, dtype=np.float32)
    if 0 <= ep_square < 64:
        ep_mask[ep_square] = 1.0
    
    # Scalars - pre-compute normalization
    scalars = np.array([
        (active_elo - 1900) / 700.0,
        (opp_elo - 1900) / 700.0,
        move_ply / 100.0,
        np.log1p(active_clock) / 10.0,
        np.log1p(opp_clock) / 10.0,
        active_inc / 30.0,
        opp_inc / 30.0,
        halfmove_clock / 100.0,
    ], dtype=np.float32)
    
    # Legal move mask - use numpy for speed
    legal_mask = np.zeros(NUM_POLICY_OUTPUTS, dtype=np.bool_)
    for from_sq, to_sq, promo in legal_moves:
        if 0 <= from_sq < 64 and 0 <= to_sq < 64:
            legal_mask[from_sq * 64 + to_sq] = True
    legal_mask[RESIGN_MOVE_INDEX] = True
    legal_mask[FLAG_MOVE_INDEX] = True
    
    # Policy target
    promo_target = 0
    promo_file = 0
    
    if policy_is_resign:
        policy_target_idx = RESIGN_MOVE_INDEX
    elif policy_is_flag:
        policy_target_idx = FLAG_MOVE_INDEX
    else:
        from_sq, to_sq, promo = policy_move
        policy_target_idx = from_sq * 64 + to_sq
        promo_target = promo
        if 56 <= to_sq <= 63:
            promo_file = to_sq - 56
    
    # Value classification target
    y_val_cls = int(y_val * 2)
    
    # Time target with dequantization noise
    clock_before = max(1.0, active_clock)
    noisy_time = max(0.0, time_target + noise)
    time_ratio = noisy_time / clock_before
    time_target_scaled = time_ratio ** 0.5
    time_bin = int(min(max(time_target_scaled, 0.0), 0.9999) * 256)
    
    # Convert to tensors
    return {
        "board_history": torch.from_numpy(board_history),
        "time_history": torch.from_numpy(time_hist_reversed),
        "rep_flags": torch.from_numpy(rep_flags.astype(np.float32)),
        "castling": torch.from_numpy(castling.astype(np.float32)),
        "ep_mask": torch.from_numpy(ep_mask),
        "scalars": torch.from_numpy(scalars),
        "tc_cat": torch.tensor(tc_cat, dtype=torch.long),
        "legal_mask": torch.from_numpy(legal_mask),
        "policy_target": torch.tensor(policy_target_idx, dtype=torch.long),
        "y_val": torch.tensor(y_val, dtype=torch.float32),
        "y_val_cls": torch.tensor(y_val_cls, dtype=torch.long),
        "time_target_cls": torch.tensor(time_bin, dtype=torch.long),
        "promo_target": torch.tensor(promo_target, dtype=torch.long),
        "promo_file": torch.tensor(promo_file, dtype=torch.long),
    }


def _collate(samples: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    """Collate a batch of samples - optimized with pre-allocated stacking."""
    if not samples:
        return {}
    keys = samples[0].keys()
    return {
        key: torch.stack([s[key] for s in samples])
        for key in keys
    }


class StreamingDataset(IterableDataset):
    """Streaming dataset for parquet data."""
    
    def __init__(
        self,
        files: Sequence[str],
        *,
        shuffle_files: bool = True,
        shuffle_rows: bool = True,
        seed: int = 42,
        read_batch_rows: int = 4096,
    ) -> None:
        self.files = list(files)
        self.shuffle_files = shuffle_files
        self.shuffle_rows = shuffle_rows
        self.seed = seed
        self.read_batch_rows = read_batch_rows
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        # Used to vary shuffling across epochs while staying deterministic.
        self.epoch = int(epoch)

    def _epoch_seed(self, base_seed: int) -> int:
        # Large odd multiplier to avoid small-cycle patterns.
        return int(base_seed + self.epoch * 1_000_003)

    def _iter_files_for_worker(self) -> Iterable[str]:
        worker = get_worker_info()
        files = self.files
        if worker is not None:
            files = files[worker.id :: worker.num_workers]
            seed = self.seed + worker.id * 17
        else:
            seed = self.seed

        order = list(files)
        if self.shuffle_files:
            rng = random.Random(self._epoch_seed(seed))
            rng.shuffle(order)
        return order

    def __iter__(self):
        for file_path in self._iter_files_for_worker():
            yield from self._stream_file(file_path)

    def _stream_file(self, file_path: str):
        try:
            parquet_file = pq.ParquetFile(file_path)
        except Exception as exc:
            print(f"[dataset] Failed to open {file_path}: {exc}")
            return

        # Deterministic RNG for this file
        file_hash = int(hashlib.md5(file_path.encode("utf-8")).hexdigest(), 16)
        rng = np.random.default_rng(self._epoch_seed(self.seed) + (file_hash % 1000000))

        for batch in parquet_file.iter_batches(batch_size=self.read_batch_rows):
            batch_len = batch.num_rows
            if batch_len == 0:
                continue

            # Shuffle rows within batch
            if self.shuffle_rows:
                shuffle_indices = rng.permutation(batch_len)
                batch = batch.take(shuffle_indices)

            # Vectorized batch processing - much faster than row-by-row
            try:
                yield from self._process_batch_vectorized(batch, rng)
            except Exception as e:
                print(f"[dataset] Error processing batch in {file_path}: {e}")
                continue

    def _process_batch_vectorized(self, batch, rng):
        """Process entire batch using vectorized operations instead of row-by-row."""
        batch_len = batch.num_rows
        
        # Extract all columns at once using to_pylist() - faster than individual .as_py()
        board_history_list = batch["board_history"].to_pylist()
        legal_moves_list = batch["legal_moves"].to_pylist()
        time_history_list = batch["time_history"].to_pylist()
        rep_flags_list = batch["repetition_flags"].to_pylist()
        castling_list = batch["castling_rights"].to_pylist()
        
        # Scalar columns - convert to numpy arrays for fast access
        ep_square_arr = batch["ep_square"].to_numpy()
        active_elo_arr = batch["active_elo"].to_numpy()
        opp_elo_arr = batch["opp_elo"].to_numpy()
        move_ply_arr = batch["move_ply"].to_numpy()
        halfmove_clock_arr = batch["halfmove_clock"].to_numpy()
        tc_cat_arr = batch["tc_cat"].to_numpy()
        active_clock_arr = batch["active_clock"].to_numpy()
        opp_clock_arr = batch["opp_clock"].to_numpy()
        active_inc_arr = batch["active_inc"].to_numpy()
        opp_inc_arr = batch["opp_inc"].to_numpy()
        
        # Target columns
        policy_move_list = batch["policy_move"].to_pylist()
        # Boolean columns require zero_copy_only=False
        policy_is_resign_arr = batch["policy_is_resign"].to_numpy(zero_copy_only=False)
        policy_is_flag_arr = batch["policy_is_flag"].to_numpy(zero_copy_only=False)
        y_val_arr = batch["y_val"].to_numpy()
        time_target_arr = batch["time_target"].to_numpy()
        
        # Pre-generate noise for dequantization
        noise_arr = rng.random(batch_len) - 0.5
        
        for i in range(batch_len):
            try:
                # Board history
                bh = board_history_list[i]
                board_history = np.array([np.array(b) for b in bh], dtype=np.int64)
                if board_history.shape != (8, 64):
                    continue
                
                # Legal moves
                legal_moves = [(int(m[0]), int(m[1]), int(m[2])) for m in legal_moves_list[i]]
                
                # Time history
                th = time_history_list[i]
                time_history = np.array(th, dtype=np.float32) if len(th) == 8 else np.zeros(8, dtype=np.float32)
                
                # Rep flags
                rf = rep_flags_list[i]
                rep_flags = np.array(rf, dtype=np.int64) if len(rf) == 8 else np.zeros(8, dtype=np.int64)
                
                # Castling
                cr = castling_list[i]
                castling = np.array(cr, dtype=np.int64) if len(cr) == 4 else np.zeros(4, dtype=np.int64)
                
                # Policy move
                pm = policy_move_list[i]
                policy_move = (int(pm[0]), int(pm[1]), int(pm[2]))
                
                yield _sample_to_tensors_fast(
                    board_history=board_history,
                    legal_moves=legal_moves,
                    time_history=time_history,
                    rep_flags=rep_flags,
                    castling=castling,
                    ep_square=int(ep_square_arr[i]),
                    active_elo=int(active_elo_arr[i]),
                    opp_elo=int(opp_elo_arr[i]),
                    move_ply=int(move_ply_arr[i]),
                    halfmove_clock=int(halfmove_clock_arr[i]),
                    tc_cat=int(tc_cat_arr[i]),
                    active_clock=float(active_clock_arr[i]),
                    opp_clock=float(opp_clock_arr[i]),
                    active_inc=int(active_inc_arr[i]),
                    opp_inc=int(opp_inc_arr[i]),
                    policy_move=policy_move,
                    policy_is_resign=bool(policy_is_resign_arr[i]),
                    policy_is_flag=bool(policy_is_flag_arr[i]),
                    y_val=float(y_val_arr[i]),
                    time_target=float(time_target_arr[i]),
                    noise=noise_arr[i],
                )
            except Exception:
                continue

    def _extract_sample(self, batch, idx: int) -> Optional[Sample]:
        """Extract a single sample from a batch at the given index."""
        try:
            # Board history: nested array (8, 64)
            board_history_raw = batch["board_history"][idx].as_py()
            board_history = np.array([np.array(b) for b in board_history_raw], dtype=np.int64)
            if board_history.shape != (8, 64):
                return None

            # Legal moves: list of [from, to, promo]
            legal_moves_raw = batch["legal_moves"][idx].as_py()
            legal_moves = [(int(m[0]), int(m[1]), int(m[2])) for m in legal_moves_raw]

            # Time history: (8,)
            time_history = np.array(batch["time_history"][idx].as_py(), dtype=np.float32)
            if len(time_history) != 8:
                time_history = np.zeros(8, dtype=np.float32)

            # Repetition flags: (8,)
            rep_flags = np.array(batch["repetition_flags"][idx].as_py(), dtype=np.int64)
            if len(rep_flags) != 8:
                rep_flags = np.zeros(8, dtype=np.int64)

            # Castling rights: (4,)
            castling = np.array(batch["castling_rights"][idx].as_py(), dtype=np.int64)
            if len(castling) != 4:
                castling = np.zeros(4, dtype=np.int64)

            # Scalars
            ep_square = int(batch["ep_square"][idx].as_py())
            active_elo = int(batch["active_elo"][idx].as_py())
            opp_elo = int(batch["opp_elo"][idx].as_py())
            move_ply = int(batch["move_ply"][idx].as_py())
            halfmove_clock = int(batch["halfmove_clock"][idx].as_py())
            tc_cat = int(batch["tc_cat"][idx].as_py())
            active_clock = float(batch["active_clock"][idx].as_py())
            opp_clock = float(batch["opp_clock"][idx].as_py())
            active_inc = int(batch["active_inc"][idx].as_py())
            opp_inc = int(batch["opp_inc"][idx].as_py())

            # Targets
            policy_move_raw = batch["policy_move"][idx].as_py()
            policy_move = (int(policy_move_raw[0]), int(policy_move_raw[1]), int(policy_move_raw[2]))
            policy_is_resign = bool(batch["policy_is_resign"][idx].as_py())
            policy_is_flag = bool(batch["policy_is_flag"][idx].as_py())
            y_val = float(batch["y_val"][idx].as_py())
            time_target = float(batch["time_target"][idx].as_py())

            return Sample(
                board_history=board_history,
                legal_moves=legal_moves,
                time_history=time_history,
                repetition_flags=rep_flags,
                castling_rights=castling,
                ep_square=ep_square,
                active_elo=active_elo,
                opp_elo=opp_elo,
                move_ply=move_ply,
                halfmove_clock=halfmove_clock,
                tc_cat=tc_cat,
                active_clock=active_clock,
                opp_clock=opp_clock,
                active_inc=active_inc,
                opp_inc=opp_inc,
                policy_move=policy_move,
                policy_is_resign=policy_is_resign,
                policy_is_flag=policy_is_flag,
                y_val=y_val,
                time_target=time_target,
            )
        except Exception as e:
            return None


def build_dataloaders(
    train_dir: str,
    val_dir: str,
    batch_size: int = 256,
    num_workers: int = 4,
    seed: int = 42,
) -> Tuple[DataLoader, DataLoader]:
    """Build train and validation dataloaders."""
    
    train_files = sorted(Path(train_dir).glob("*.parquet"))
    val_files = sorted(Path(val_dir).glob("*.parquet"))
    
    print(f"[dataset] Found {len(train_files)} train files, {len(val_files)} val files")
    
    train_ds = StreamingDataset(
        [str(f) for f in train_files],
        shuffle_files=True,
        shuffle_rows=True,
        seed=seed,
    )
    
    val_ds = StreamingDataset(
        [str(f) for f in val_files],
        shuffle_files=False,
        shuffle_rows=False,
        seed=seed + 1,
    )
    
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        num_workers=num_workers,
        collate_fn=_collate,
        pin_memory=True,
        prefetch_factor=2 if num_workers > 0 else None,
    )
    
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        num_workers=num_workers,
        collate_fn=_collate,
        pin_memory=True,
        prefetch_factor=2 if num_workers > 0 else None,
    )
    
    return train_loader, val_loader


def _split_files_deterministic(files: List[Path], val_fraction: float = 0.05) -> Tuple[List[Path], List[Path]]:
    """Split files into train/val using deterministic hash-based assignment.
    
    Each file is assigned to train or val based on the MD5 hash of its name.
    This ensures:
    1. Repeatability - same files always go to same split
    2. No data leakage - files are never in both splits
    3. Approximate balance - hash distribution is uniform
    
    Args:
        files: List of parquet files to split
        val_fraction: Fraction of files to use for validation (default 5%)
    
    Returns:
        (train_files, val_files) tuple
    """
    train_files = []
    val_files = []
    
    for f in files:
        # Hash the filename (not full path) for determinism
        file_hash = int(hashlib.md5(f.name.encode('utf-8')).hexdigest(), 16)
        # Use modulo to get a value in [0, 1)
        hash_val = (file_hash % 100000) / 100000.0
        
        if hash_val < val_fraction:
            val_files.append(f)
        else:
            train_files.append(f)
    
    return train_files, val_files


def create_dataloader(
    data_dir: str,
    split: str = "train",
    batch_size: int = 256,
    num_workers: int = 4,
    seed: int = 42,
    shuffle: bool = None,
    pin_memory: bool = True,
    prefetch_factor: int = None,
    val_fraction: float = 0.05,
    persistent_workers: bool = True,
    drop_last: bool = True,
) -> DataLoader:
    """Create a single dataloader for a split.
    
    Now splits the training data deterministically instead of using separate directories.
    
    Args:
        data_dir: Root directory containing train/ subdirectory
        split: 'train' or 'val' - determines which split to load
        val_fraction: Fraction of files to use for validation (default 5%)
        persistent_workers: Keep workers alive between epochs (faster)
        drop_last: Drop last incomplete batch (better for training)
        Other args: standard DataLoader configuration
    """
    
    # Always load from train directory and split deterministically
    train_dir = Path(data_dir) / "train"
    all_files = sorted(train_dir.glob("*.parquet"))
    
    if len(all_files) == 0:
        raise FileNotFoundError(f"No parquet files found in {train_dir}")
    
    # Split files deterministically
    train_files, val_files = _split_files_deterministic(all_files, val_fraction)
    
    if split == "train":
        files = train_files
    elif split == "val":
        files = val_files
    else:
        raise ValueError(f"Unknown split: {split}. Use 'train' or 'val'.")
    
    print(f"[dataset] Split: {split} | Files: {len(files)}/{len(all_files)} ({len(files)/len(all_files)*100:.1f}%)")
    
    is_train = split == "train"
    should_shuffle = shuffle if shuffle is not None else is_train
    
    ds = StreamingDataset(
        [str(f) for f in files],
        shuffle_files=should_shuffle,
        shuffle_rows=should_shuffle,
        seed=seed,
    )
    
    # Use persistent_workers to avoid worker restart overhead between epochs
    use_persistent = persistent_workers and num_workers > 0
    
    return DataLoader(
        ds,
        batch_size=batch_size,
        num_workers=num_workers,
        collate_fn=_collate,
        pin_memory=pin_memory,
        prefetch_factor=prefetch_factor if num_workers > 0 else None,
        persistent_workers=use_persistent,
        drop_last=drop_last if is_train else False,
    )


def count_samples(data_dir: str, split: str = "train", val_fraction: float = 0.05) -> int:
    """Estimate number of samples in a split by reading parquet metadata.
    
    Now uses deterministic file splitting instead of separate directories.
    """
    train_dir = Path(data_dir) / "train"
    all_files = sorted(train_dir.glob("*.parquet"))
    
    # Split files the same way as create_dataloader
    train_files, val_files = _split_files_deterministic(all_files, val_fraction)
    
    if split == "train":
        files = train_files
    elif split == "val":
        files = val_files
    else:
        raise ValueError(f"Unknown split: {split}")
    
    total = 0
    for f in files:
        try:
            pf = pq.ParquetFile(f)
            total += pf.metadata.num_rows
        except Exception:
            pass
    
    return total


class BalancedValDataset(torch.utils.data.Dataset):
    """
    Map-style dataset for balanced validation set loaded entirely into memory.
    
    This is more efficient for validation than streaming since:
    1. The balanced val set is small (~70k samples)
    2. We iterate over it multiple times
    3. Random access enables proper shuffling
    """
    
    def __init__(self, parquet_path: str):
        self.parquet_path = parquet_path
        
        # Load entire table into memory
        print(f"[dataset] Loading balanced validation set from {parquet_path}...")
        table = pq.read_table(parquet_path)
        self.num_samples = len(table)
        print(f"[dataset] Loaded {self.num_samples:,} validation samples")
        
        # Convert to pandas for easier indexing
        self.df = table.to_pandas()
        
        # Pre-parse legal moves (they're stored as nested arrays)
        self._preprocess()
    
    def _preprocess(self):
        """Pre-process columns that need special handling."""
        # Legal moves are stored as list of lists - convert once
        self.legal_moves_list = [
            [(int(m[0]), int(m[1]), int(m[2])) for m in moves]
            for moves in self.df["legal_moves"].values
        ]
        
        # Board history: nested arrays
        self.board_history_list = [
            np.array([np.array(b) for b in bh], dtype=np.int64)
            for bh in self.df["board_history"].values
        ]
        
        # Time history
        self.time_history_list = [
            np.array(th, dtype=np.float32) if len(th) == 8 else np.zeros(8, dtype=np.float32)
            for th in self.df["time_history"].values
        ]
        
        # Repetition flags
        self.rep_flags_list = [
            np.array(rf, dtype=np.int64) if len(rf) == 8 else np.zeros(8, dtype=np.int64)
            for rf in self.df["repetition_flags"].values
        ]
        
        # Castling rights
        self.castling_list = [
            np.array(cr, dtype=np.int64)
            for cr in self.df["castling_rights"].values
        ]
    
    def __len__(self) -> int:
        return self.num_samples
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        row = self.df.iloc[idx]
        
        # policy_move is stored as numpy array [from, to, promo]
        policy_move = row["policy_move"]
        
        sample = Sample(
            board_history=self.board_history_list[idx],
            legal_moves=self.legal_moves_list[idx],
            time_history=self.time_history_list[idx],
            repetition_flags=self.rep_flags_list[idx],
            castling_rights=self.castling_list[idx],
            ep_square=int(row["ep_square"]),
            active_elo=int(row["active_elo"]),
            opp_elo=int(row["opp_elo"]),
            move_ply=int(row["move_ply"]),
            halfmove_clock=int(row["halfmove_clock"]),
            tc_cat=int(row["tc_cat"]),
            active_clock=float(row["active_clock"]),
            opp_clock=float(row["opp_clock"]),
            active_inc=int(row["active_inc"]),
            opp_inc=int(row["opp_inc"]),
            policy_move=(int(policy_move[0]), int(policy_move[1]), int(policy_move[2])),
            policy_is_resign=bool(row.get("policy_is_resign", False)),
            policy_is_flag=bool(row.get("policy_is_flag", False)),
            y_val=float(row["y_val"]),
            time_target=float(row["time_target"]),
        )
        
        return _sample_to_tensors(sample)


def create_balanced_val_loader(
    parquet_path: str,
    batch_size: int = 256,
    num_workers: int = 4,
    pin_memory: bool = True,
) -> DataLoader:
    """
    Create a dataloader for the balanced validation set.
    
    Args:
        parquet_path: Path to the balanced validation parquet file
        batch_size: Batch size
        num_workers: Number of dataloader workers
        pin_memory: Whether to pin memory for GPU transfer
    
    Returns:
        DataLoader for the balanced validation set
    """
    dataset = BalancedValDataset(parquet_path)
    
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,  # No need to shuffle validation
        num_workers=num_workers,
        collate_fn=_collate,
        pin_memory=pin_memory,
    )
