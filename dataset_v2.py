"""Dataset for Marvin Chess v2 data format with board history and legal moves."""

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
    "StreamingDatasetV2",
    "BalancedValDataset",
    "build_dataloaders_v2",
    "create_dataloader_v2",
    "create_balanced_val_loader",
    "count_samples",
]

# Special policy indices
RESIGN_MOVE_INDEX = 4096
FLAG_MOVE_INDEX = 4097
NUM_POLICY_OUTPUTS = 4098  # 64*64 + resign + flag


@dataclass
class SampleV2:
    """A single training sample from v2 data."""
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


def _sample_to_tensors(sample: SampleV2) -> Dict[str, torch.Tensor]:
    """Convert a SampleV2 to model input tensors."""
    
    # Board history: (8, 64) int64
    board_history = torch.from_numpy(sample.board_history).long()
    
    # Time history: (8,) float32 - normalize by dividing by 60 (minutes)
    time_history = torch.from_numpy(sample.time_history).float() / 60.0
    
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
    if sample.policy_is_resign:
        policy_target = RESIGN_MOVE_INDEX
        legal_mask[RESIGN_MOVE_INDEX] = True  # Add to legal mask for cross-entropy
    elif sample.policy_is_flag:
        policy_target = FLAG_MOVE_INDEX
        legal_mask[FLAG_MOVE_INDEX] = True  # Add to legal mask for cross-entropy
    else:
        from_sq, to_sq, promo = sample.policy_move
        policy_target = from_sq * 64 + to_sq
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
    time_ratio = sample.time_target / clock_before
    # Apply 4th root scaling (like v1's encode_time_bin) to compress range
    # This gives more resolution to quick moves
    time_target_scaled = time_ratio ** 0.25
    time_target = torch.tensor(time_target_scaled, dtype=torch.float32)
    
    # Time classification target: 256 bins over [0, 1] range of scaled time
    # Clamp to [0, 1] to handle edge cases, then quantize to 256 bins
    time_bin = int(min(max(time_target_scaled, 0.0), 0.9999) * 256)
    time_target_cls = torch.tensor(time_bin, dtype=torch.long)
    
    return {
        "board_history": board_history,      # (8, 64) long
        "time_history": time_history,        # (8,) float
        "rep_flags": rep_flags,              # (8,) float
        "castling": castling,                # (4,) float
        "ep_mask": ep_mask,                  # (64,) float
        "scalars": scalars,                  # (8,) float
        "tc_cat": tc_cat,                    # () long
        "legal_mask": legal_mask,            # (4098,) bool
        "policy_target": policy_target,      # () long
        "y_val": y_val,                      # () float (regression)
        "y_val_cls": y_val_cls,              # () long (WDL class 0/1/2)
        "time_target": time_target,          # () float - (time_spent/clock)^0.25
        "time_target_cls": time_target_cls,  # () long (bin 0-255)
    }


def _collate_v2(samples: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    """Collate a batch of samples."""
    return {
        key: torch.stack([s[key] for s in samples])
        for key in samples[0].keys()
    }


class StreamingDatasetV2(IterableDataset):
    """Streaming dataset for v2 parquet data."""
    
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
            rng = random.Random(seed)
            rng.shuffle(order)
        return order

    def __iter__(self):
        for file_path in self._iter_files_for_worker():
            yield from self._stream_file(file_path)

    def _stream_file(self, file_path: str):
        try:
            parquet_file = pq.ParquetFile(file_path)
        except Exception as exc:
            print(f"[dataset_v2] Failed to open {file_path}: {exc}")
            return

        # Deterministic RNG for this file
        file_hash = int(hashlib.md5(file_path.encode("utf-8")).hexdigest(), 16)
        rng = np.random.default_rng(self.seed + (file_hash % 1000000))

        for batch in parquet_file.iter_batches(batch_size=self.read_batch_rows):
            batch_len = batch.num_rows
            if batch_len == 0:
                continue

            # Shuffle rows within batch
            if self.shuffle_rows:
                shuffle_indices = rng.permutation(batch_len)
                batch = batch.take(shuffle_indices)

            # Extract columns
            try:
                for i in range(batch_len):
                    sample = self._extract_sample(batch, i)
                    if sample is not None:
                        yield _sample_to_tensors(sample)
            except Exception as e:
                print(f"[dataset_v2] Error processing batch in {file_path}: {e}")
                continue

    def _extract_sample(self, batch, idx: int) -> Optional[SampleV2]:
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

            return SampleV2(
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


def build_dataloaders_v2(
    train_dir: str,
    val_dir: str,
    batch_size: int = 256,
    num_workers: int = 4,
    seed: int = 42,
) -> Tuple[DataLoader, DataLoader]:
    """Build train and validation dataloaders for v2 data."""
    
    train_files = sorted(Path(train_dir).glob("*.parquet"))
    val_files = sorted(Path(val_dir).glob("*.parquet"))
    
    print(f"[dataset_v2] Found {len(train_files)} train files, {len(val_files)} val files")
    
    train_ds = StreamingDatasetV2(
        [str(f) for f in train_files],
        shuffle_files=True,
        shuffle_rows=True,
        seed=seed,
    )
    
    val_ds = StreamingDatasetV2(
        [str(f) for f in val_files],
        shuffle_files=False,
        shuffle_rows=False,
        seed=seed + 1,
    )
    
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        num_workers=num_workers,
        collate_fn=_collate_v2,
        pin_memory=True,
        prefetch_factor=2 if num_workers > 0 else None,
    )
    
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        num_workers=num_workers,
        collate_fn=_collate_v2,
        pin_memory=True,
        prefetch_factor=2 if num_workers > 0 else None,
    )
    
    return train_loader, val_loader


def create_dataloader_v2(
    data_dir: str,
    split: str = "train",
    batch_size: int = 256,
    num_workers: int = 4,
    seed: int = 42,
    shuffle: bool = None,
    pin_memory: bool = True,
    prefetch_factor: int = None,
) -> DataLoader:
    """Create a single dataloader for a split."""
    
    split_dir = Path(data_dir) / split
    files = sorted(split_dir.glob("*.parquet"))
    
    print(f"[dataset_v2] Found {len(files)} {split} files")
    
    is_train = split == "train"
    should_shuffle = shuffle if shuffle is not None else is_train
    
    ds = StreamingDatasetV2(
        [str(f) for f in files],
        shuffle_files=should_shuffle,
        shuffle_rows=should_shuffle,
        seed=seed,
    )
    
    return DataLoader(
        ds,
        batch_size=batch_size,
        num_workers=num_workers,
        collate_fn=_collate_v2,
        pin_memory=pin_memory,
        prefetch_factor=prefetch_factor if num_workers > 0 else None,
    )


def count_samples(data_dir: str, split: str = "train") -> int:
    """Estimate number of samples in a split by reading parquet metadata."""
    split_dir = Path(data_dir) / split
    files = sorted(split_dir.glob("*.parquet"))
    
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
        print(f"[dataset_v2] Loading balanced validation set from {parquet_path}...")
        table = pq.read_table(parquet_path)
        self.num_samples = len(table)
        print(f"[dataset_v2] Loaded {self.num_samples:,} validation samples")
        
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
        
        sample = SampleV2(
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
        collate_fn=_collate_v2,
        pin_memory=pin_memory,
    )
