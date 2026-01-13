"""compare_ACPL.py

Compute Average Centipawn Loss (ACPL) and CPL variance for a model
versus a dataset of human games (e.g. `data/val_balanced.parquet`).

The script is intentionally defensive: it probes the parquet to find
useful columns, can operate per-game or per-position, and uses Stockfish
via `python-chess` for centipawn evaluations. If no engine path is
provided the script will attempt to use `stockfish` on PATH.

Usage examples:
  python compare_ACPL.py --parquet data/val_balanced.parquet --num-games 2000 \
      --engine-path /usr/bin/stockfish --model-py model.py --checkpoint inference/chessformer_smolgen_best.pt

Notes:
- The script picks the model's top policy move (argmax over legal moves)
  rather than sampling.
- Engine parameters (time or depth) are configurable; longer evals are
  slower but more accurate.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import statistics
from collections import defaultdict
from dataclasses import dataclass, field
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


# ============================================================
# ELO and Move Quality Configuration
# ============================================================

# ELO bins for stratified analysis
ELO_BINS = [(1200, 1400), (1400, 1600), (1600, 1800), (1800, 2000), (2000, 2200), (2200, 2400), (2400, 2600)]
ELO_BIN_NAMES = [f"{lo}-{hi}" for lo, hi in ELO_BINS]

# Move quality bins (in centipawns of loss)
# Perfect: 0 cp, Good: 1-50 cp, OK: 51-100, Inaccuracy: 101-300, Mistake: 301-500, Blunder: >500
QUALITY_BINS = [
    (0, 0, "Perfect"),
    (1, 50, "Good"),
    (51, 100, "OK"),
    (101, 300, "Inaccuracy"),
    (301, 500, "Mistake"),
    (501, 10000, "Blunder"),
]
QUALITY_BIN_NAMES = [q[2] for q in QUALITY_BINS]


def get_elo_bin(elo: int) -> Optional[str]:
    """Return the ELO bin name for a given ELO, or None if out of range."""
    for (lo, hi), name in zip(ELO_BINS, ELO_BIN_NAMES):
        if lo <= elo < hi:
            return name
    return None


def get_quality_bin(cpl: int) -> str:
    """Return the quality bin name for a given centipawn loss."""
    for lo, hi, name in QUALITY_BINS:
        if lo <= cpl <= hi:
            return name
    return "Blunder"  # fallback for very large values


@dataclass
class PositionResult:
    """Result of evaluating a single position."""
    active_elo: int
    human_cpl: int
    model_cpl: int
    human_move: chess.Move
    model_move: chess.Move
    move_matches: bool
    model_better: bool  # model_cpl < human_cpl
    
    @property
    def elo_bin(self) -> Optional[str]:
        return get_elo_bin(self.active_elo)
    
    @property
    def human_quality_bin(self) -> str:
        return get_quality_bin(self.human_cpl)


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


# ============================================================
# ELO-Stratified Analysis Functions
# ============================================================

@dataclass
class EloQualityStats:
    """Statistics for a single ELO x Quality cell."""
    total: int = 0
    matches: int = 0
    model_better: int = 0
    
    @property
    def match_rate(self) -> float:
        return self.matches / self.total if self.total > 0 else 0.0
    
    @property
    def model_better_rate(self) -> float:
        return self.model_better / self.total if self.total > 0 else 0.0


@dataclass
class EloQualityAnalysis:
    """Complete analysis results for ELO x Quality stratification."""
    # 2D matrix: elo_bin -> quality_bin -> stats
    matrix: Dict[str, Dict[str, EloQualityStats]] = field(default_factory=dict)
    # Aggregate stats per ELO bin
    elo_totals: Dict[str, EloQualityStats] = field(default_factory=dict)
    # Aggregate stats per quality bin
    quality_totals: Dict[str, EloQualityStats] = field(default_factory=dict)
    # Overall stats
    overall: EloQualityStats = field(default_factory=EloQualityStats)


def analyze_elo_quality_matching(
    results: List[PositionResult],
    min_samples: int = 30,
) -> EloQualityAnalysis:
    """Analyze move matching rates stratified by ELO and move quality.
    
    Args:
        results: List of PositionResult from evaluation
        min_samples: Minimum samples per cell to be considered valid
        
    Returns:
        EloQualityAnalysis with all statistics
    """
    analysis = EloQualityAnalysis()
    
    # Initialize matrix structure
    for elo_name in ELO_BIN_NAMES:
        analysis.matrix[elo_name] = {}
        for quality_name in QUALITY_BIN_NAMES:
            analysis.matrix[elo_name][quality_name] = EloQualityStats()
        analysis.elo_totals[elo_name] = EloQualityStats()
    
    for quality_name in QUALITY_BIN_NAMES:
        analysis.quality_totals[quality_name] = EloQualityStats()
    
    # Populate the matrix
    for r in results:
        elo_bin = r.elo_bin
        if elo_bin is None:
            continue  # ELO out of range
        
        quality_bin = r.human_quality_bin
        
        # Update cell
        cell = analysis.matrix[elo_bin][quality_bin]
        cell.total += 1
        if r.move_matches:
            cell.matches += 1
        if r.model_better:
            cell.model_better += 1
        
        # Update ELO totals
        elo_agg = analysis.elo_totals[elo_bin]
        elo_agg.total += 1
        if r.move_matches:
            elo_agg.matches += 1
        if r.model_better:
            elo_agg.model_better += 1
        
        # Update quality totals
        qual_agg = analysis.quality_totals[quality_bin]
        qual_agg.total += 1
        if r.move_matches:
            qual_agg.matches += 1
        if r.model_better:
            qual_agg.model_better += 1
        
        # Update overall
        analysis.overall.total += 1
        if r.move_matches:
            analysis.overall.matches += 1
        if r.model_better:
            analysis.overall.model_better += 1
    
    return analysis


def print_elo_quality_analysis(analysis: EloQualityAnalysis, min_samples: int = 30) -> None:
    """Print formatted analysis results to console."""
    print("\n" + "=" * 80)
    print("ELO-STRATIFIED MOVE MATCHING ANALYSIS")
    print("=" * 80)
    
    print(f"\nOverall: {analysis.overall.total} positions, "
          f"{analysis.overall.match_rate:.1%} move match, "
          f"{analysis.overall.model_better_rate:.1%} model better")
    
    # Print ELO totals
    print("\n--- Match Rate by ELO ---")
    print(f"{'ELO Range':<12} {'Positions':>10} {'Match Rate':>12} {'Model Better':>14}")
    print("-" * 50)
    for elo_name in ELO_BIN_NAMES:
        stats = analysis.elo_totals[elo_name]
        flag = "*" if stats.total < min_samples else ""
        print(f"{elo_name:<12} {stats.total:>10} {stats.match_rate:>11.1%}{flag} {stats.model_better_rate:>13.1%}{flag}")
    
    # Print Quality totals
    print("\n--- Match Rate by Human Move Quality ---")
    print(f"{'Quality':<12} {'Positions':>10} {'Match Rate':>12} {'Model Better':>14}")
    print("-" * 50)
    for quality_name in QUALITY_BIN_NAMES:
        stats = analysis.quality_totals[quality_name]
        flag = "*" if stats.total < min_samples else ""
        print(f"{quality_name:<12} {stats.total:>10} {stats.match_rate:>11.1%}{flag} {stats.model_better_rate:>13.1%}{flag}")
    
    # Print heatmap-style table
    print("\n--- Match Rate Matrix (ELO x Quality) ---")
    print(f"{'':>12}", end="")
    for quality_name in QUALITY_BIN_NAMES:
        print(f"{quality_name:>10}", end="")
    print()
    print("-" * (12 + 10 * len(QUALITY_BIN_NAMES)))
    
    for elo_name in ELO_BIN_NAMES:
        print(f"{elo_name:<12}", end="")
        for quality_name in QUALITY_BIN_NAMES:
            stats = analysis.matrix[elo_name][quality_name]
            if stats.total < min_samples:
                print(f"{'--':>10}", end="")
            else:
                print(f"{stats.match_rate:>9.1%}", end=" ")
        print()
    
    print("\n* Low sample count (less than threshold)")


def save_analysis_results(
    analysis: EloQualityAnalysis,
    results: List[PositionResult],
    output_dir: Path,
) -> None:
    """Save analysis results to JSON file."""
    output = {
        "summary": {
            "total_positions": analysis.overall.total,
            "match_rate": analysis.overall.match_rate,
            "model_better_rate": analysis.overall.model_better_rate,
        },
        "by_elo": {},
        "by_quality": {},
        "matrix": {},
    }
    
    for elo_name in ELO_BIN_NAMES:
        stats = analysis.elo_totals[elo_name]
        output["by_elo"][elo_name] = {
            "total": stats.total,
            "matches": stats.matches,
            "model_better": stats.model_better,
            "match_rate": stats.match_rate,
            "model_better_rate": stats.model_better_rate,
        }
    
    for quality_name in QUALITY_BIN_NAMES:
        stats = analysis.quality_totals[quality_name]
        output["by_quality"][quality_name] = {
            "total": stats.total,
            "matches": stats.matches,
            "model_better": stats.model_better,
            "match_rate": stats.match_rate,
            "model_better_rate": stats.model_better_rate,
        }
    
    for elo_name in ELO_BIN_NAMES:
        output["matrix"][elo_name] = {}
        for quality_name in QUALITY_BIN_NAMES:
            stats = analysis.matrix[elo_name][quality_name]
            output["matrix"][elo_name][quality_name] = {
                "total": stats.total,
                "matches": stats.matches,
                "model_better": stats.model_better,
                "match_rate": stats.match_rate,
                "model_better_rate": stats.model_better_rate,
            }
    
    output_path = output_dir / "elo_quality_analysis.json"
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved analysis to {output_path}")


def generate_elo_quality_plots(
    analysis: EloQualityAnalysis,
    output_dir: Path,
    min_samples: int = 30,
) -> None:
    """Generate visualizations for the ELO x Quality analysis."""
    try:
        import matplotlib.pyplot as plt
        import matplotlib.colors as mcolors
    except ImportError:
        print("Warning: matplotlib not available, skipping plots")
        return
    
    # 1. Heatmap: Match Rate by ELO (x) and Quality (y)
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Build data matrix
    data = np.zeros((len(QUALITY_BIN_NAMES), len(ELO_BIN_NAMES)))
    mask = np.zeros_like(data, dtype=bool)  # mask for low-sample cells
    annotations = [['' for _ in ELO_BIN_NAMES] for _ in QUALITY_BIN_NAMES]
    
    for i, quality_name in enumerate(QUALITY_BIN_NAMES):
        for j, elo_name in enumerate(ELO_BIN_NAMES):
            stats = analysis.matrix[elo_name][quality_name]
            if stats.total < min_samples:
                mask[i, j] = True
                annotations[i][j] = f"n={stats.total}"
            else:
                data[i, j] = stats.match_rate * 100
                annotations[i][j] = f"{stats.match_rate:.0%}\nn={stats.total}"
    
    # Create heatmap
    cmap = plt.cm.RdYlGn  # Red (low) to Green (high)
    im = ax.imshow(data, cmap=cmap, aspect='auto', vmin=0, vmax=100)
    
    # Add colorbar
    cbar = ax.figure.colorbar(im, ax=ax)
    cbar.ax.set_ylabel("Match Rate (%)", rotation=-90, va="bottom")
    
    # Set ticks and labels
    ax.set_xticks(np.arange(len(ELO_BIN_NAMES)))
    ax.set_yticks(np.arange(len(QUALITY_BIN_NAMES)))
    ax.set_xticklabels(ELO_BIN_NAMES)
    ax.set_yticklabels(QUALITY_BIN_NAMES)
    ax.set_xlabel("Player ELO Range")
    ax.set_ylabel("Human Move Quality")
    ax.set_title("Model Move Matching Rate by ELO and Human Move Quality")
    
    # Rotate x-axis labels
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    
    # Add text annotations
    for i in range(len(QUALITY_BIN_NAMES)):
        for j in range(len(ELO_BIN_NAMES)):
            color = "gray" if mask[i, j] else ("black" if data[i, j] > 50 else "white")
            ax.text(j, i, annotations[i][j], ha="center", va="center", 
                   color=color, fontsize=8)
    
    plt.tight_layout()
    heatmap_path = output_dir / "elo_quality_heatmap.png"
    plt.savefig(heatmap_path, dpi=150)
    plt.close()
    print(f"Saved heatmap to {heatmap_path}")
    
    # 2. Line plot: Match Rate vs ELO, grouped by Quality
    fig, ax = plt.subplots(figsize=(10, 6))
    
    colors = plt.cm.viridis(np.linspace(0, 1, len(QUALITY_BIN_NAMES)))
    
    for idx, quality_name in enumerate(QUALITY_BIN_NAMES):
        x_vals = []
        y_vals = []
        for j, elo_name in enumerate(ELO_BIN_NAMES):
            stats = analysis.matrix[elo_name][quality_name]
            if stats.total >= min_samples:
                # Use midpoint of ELO range
                lo, hi = ELO_BINS[j]
                x_vals.append((lo + hi) / 2)
                y_vals.append(stats.match_rate * 100)
        
        if x_vals:
            ax.plot(x_vals, y_vals, marker='o', label=quality_name, color=colors[idx], linewidth=2)
    
    ax.set_xlabel("Player ELO")
    ax.set_ylabel("Match Rate (%)")
    ax.set_title("Model Move Matching Rate vs Player ELO by Human Move Quality")
    ax.legend(title="Human Move Quality", loc="best")
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 100)
    
    plt.tight_layout()
    lineplot_path = output_dir / "elo_quality_lineplot.png"
    plt.savefig(lineplot_path, dpi=150)
    plt.close()
    print(f"Saved line plot to {lineplot_path}")
    
    # 3. Bar chart: Model Better Rate by ELO
    fig, ax = plt.subplots(figsize=(10, 5))
    
    elo_positions = np.arange(len(ELO_BIN_NAMES))
    better_rates = [analysis.elo_totals[n].model_better_rate * 100 for n in ELO_BIN_NAMES]
    sample_counts = [analysis.elo_totals[n].total for n in ELO_BIN_NAMES]
    
    bars = ax.bar(elo_positions, better_rates, color='steelblue', edgecolor='black')
    
    # Add sample count labels on bars
    for i, (bar, count) in enumerate(zip(bars, sample_counts)):
        height = bar.get_height()
        color = "gray" if count < min_samples else "black"
        ax.annotate(f'n={count}',
                   xy=(bar.get_x() + bar.get_width() / 2, height),
                   xytext=(0, 3), textcoords="offset points",
                   ha='center', va='bottom', fontsize=8, color=color)
    
    ax.set_xticks(elo_positions)
    ax.set_xticklabels(ELO_BIN_NAMES, rotation=45, ha='right')
    ax.set_xlabel("Player ELO Range")
    ax.set_ylabel("Model Better Rate (%)")
    ax.set_title("Rate at Which Model Plays Better Move Than Human (by ELO)")
    ax.axhline(y=50, color='red', linestyle='--', alpha=0.5, label='50% baseline')
    ax.legend()
    ax.set_ylim(0, max(better_rates) * 1.2 if better_rates else 100)
    
    plt.tight_layout()
    better_path = output_dir / "model_better_by_elo.png"
    plt.savefig(better_path, dpi=150)
    plt.close()
    print(f"Saved model-better chart to {better_path}")


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
    p.add_argument("--parquet", required=True, help="Path to val_parquet (e.g. data/val_balanced.parquet)")
    p.add_argument("--num-games", type=int, default=48000, help="Number of games (or positions) to sample")
    p.add_argument("--min-elo", type=int, default=1150)
    p.add_argument("--max-elo", type=int, default=2550)
    p.add_argument("--engine-path", type=str, default="stockfish", help="Path to stockfish binary (or 'stockfish' on PATH)")
    p.add_argument("--engine-time", type=float, default=0.01, help="Engine time per position in seconds (or use --engine-depth)")
    p.add_argument("--engine-depth", type=int, default=None, help="Engine depth (overrides engine-time if set)")
    p.add_argument("--model-py", default="model.py")
    p.add_argument("--checkpoint", default="inference/chessformer_smolgen_best.pt")
    p.add_argument("--config-name", default="smolgen")
    p.add_argument("--device", default="cpu")
    p.add_argument("--seed", type=int, default=11)
    p.add_argument("--temperature", type=float, default=0.0, help="Sampling temperature (0=argmax, >0=sample)")
    p.add_argument("--top-p", type=float, default=0.95, help="Nucleus sampling threshold (1.0=no filtering)")
    p.add_argument("--output-dir", type=str, default="acpl_results", help="Directory for output files")
    p.add_argument("--no-plots", action="store_true", help="Skip generating plots")
    p.add_argument("--min-samples-per-cell", type=int, default=10, help="Minimum samples for a heatmap cell to be shown")
    args = p.parse_args()
    
    print(f"Temperature: {args.temperature}, Top-p: {args.top_p}")

    pq_path = Path(args.parquet)
    if not pq_path.exists():
        raise SystemExit(f"Parquet not found: {pq_path}")

    print(f"Probing parquet columns...")
    cols = probe_parquet_columns(pq_path)
    print("Columns:", cols)

    print("Loading model... This may take a few seconds.")
    loaded = model_loader.load_chessformer(
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
    position_results: List[PositionResult] = []  # Collect detailed per-position data

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
                
                # Collect detailed position result
                active_elo = int(row.get("active_elo", 1900))
                position_results.append(PositionResult(
                    active_elo=active_elo,
                    human_cpl=human_cpl,
                    model_cpl=model_cpl,
                    human_move=gt_move,
                    model_move=model_move,
                    move_matches=(gt_move == model_move),
                    model_better=(model_cpl < human_cpl),
                ))

        except Exception as e:
            print("Engine evaluation failed for position", e)
            continue

        if idx % 100 == 0:
            print(f"Processed {idx}/{len(samples)}")

    engine.quit()

    if not model_cpls:
        print("No CPLs computed")
        return

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ============================================================
    # ELO-Stratified Move Matching Analysis
    # ============================================================
    
    elo_quality_analysis = analyze_elo_quality_matching(position_results, args.min_samples_per_cell)
    print_elo_quality_analysis(elo_quality_analysis, args.min_samples_per_cell)
    
    # Save analysis results to JSON
    save_analysis_results(elo_quality_analysis, position_results, output_dir)
    
    # Generate plots
    if not args.no_plots:
        print("\nGenerating plots...")
        generate_elo_quality_plots(elo_quality_analysis, output_dir, args.min_samples_per_cell)

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
