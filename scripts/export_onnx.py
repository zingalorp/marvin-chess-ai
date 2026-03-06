"""
Export Marvin Chessformer PyTorch checkpoints (.pt) to ONNX format.

The exported model uses dynamic batch axes so it works with any batch size,
enabling batched MCTS evaluation for ~4-5x GPU throughput improvement.

Usage:
    python scripts/export_onnx.py --input inference/marvin_small.pt
    python scripts/export_onnx.py --input inference/marvin_large.pt --validate --benchmark
    python scripts/export_onnx.py --input inference/marvin_tiny.pt -o inference/marvin_tiny.onnx
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
import torch.nn as nn
import numpy as np

# Add repo root to path
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from model import Chessformer, CONFIG_SMALL, CONFIG_LARGE, CONFIG_TINY

# Map config name keywords to config dicts.
CONFIGS = {
    "small": CONFIG_SMALL,
    "large": CONFIG_LARGE,
    "tiny": CONFIG_TINY,
}

INPUT_NAMES = [
    "board_history", "time_history", "rep_flags", "castling",
    "ep_mask", "scalars", "tc_cat", "legal_mask",
]
OUTPUT_NAMES = [
    "move_logits", "value_out", "value_cls_out",
    "value_error_out", "time_cls_out", "start_square_logits", "promo_logits",
]


def _guess_config_name(path: Path) -> str:
    """Guess config name from checkpoint filename."""
    stem = path.stem.lower()
    if "large" in stem:
        return "large"
    if "tiny" in stem:
        return "tiny"
    # Default to small (most common)
    return "small"


class ChessformerONNXWrapper(nn.Module):
    """Wraps Chessformer to accept flat tensor arguments instead of a dict.

    ONNX doesn't support dict inputs, so we take individual named tensors
    and repack them into the batch dict that ``Chessformer.forward()`` expects.

    Returns all 7 outputs including promotion logits.
    """

    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

    def forward(
        self,
        board_history: torch.Tensor,   # (B, 8, 64) int64
        time_history: torch.Tensor,    # (B, 8) float32
        rep_flags: torch.Tensor,       # (B, 8) float32
        castling: torch.Tensor,        # (B, 4) float32
        ep_mask: torch.Tensor,         # (B, 64) float32
        scalars: torch.Tensor,         # (B, 8) float32
        tc_cat: torch.Tensor,          # (B,) int64
        legal_mask: torch.Tensor,      # (B, 4098) bool
    ):
        batch = {
            "board_history": board_history,
            "time_history": time_history,
            "rep_flags": rep_flags,
            "castling": castling,
            "ep_mask": ep_mask,
            "scalars": scalars,
            "tc_cat": tc_cat,
            "legal_mask": legal_mask,
        }
        return self.model(batch, return_promo=True)


def _create_dummy_inputs(batch_size: int, device: torch.device) -> tuple:
    """Create dummy inputs for ONNX export tracing."""
    return (
        torch.randint(0, 13, (batch_size, 8, 64), dtype=torch.int64, device=device),
        torch.randn(batch_size, 8, dtype=torch.float32, device=device),
        torch.randint(0, 2, (batch_size, 8), dtype=torch.float32, device=device),
        torch.randint(0, 2, (batch_size, 4), dtype=torch.float32, device=device),
        torch.zeros(batch_size, 64, dtype=torch.float32, device=device),
        torch.randn(batch_size, 8, dtype=torch.float32, device=device),
        torch.randint(0, 3, (batch_size,), dtype=torch.int64, device=device),
        torch.ones(batch_size, 4098, dtype=torch.bool, device=device),
    )


def _load_model(weights_path: Path, config_name: str, device: torch.device) -> nn.Module:
    """Load a Chessformer from a .pt checkpoint."""
    config = CONFIGS[config_name]
    model = Chessformer(config)

    state_dict = torch.load(str(weights_path), map_location="cpu", weights_only=True)
    # Unwrap DDP / compiled wrappers if present.
    if any(k.startswith("module.") for k in state_dict):
        state_dict = {k.removeprefix("module."): v for k, v in state_dict.items()}
    if any(k.startswith("_orig_mod.") for k in state_dict):
        state_dict = {k.removeprefix("_orig_mod."): v for k, v in state_dict.items()}

    info = model.load_state_dict(state_dict, strict=False)
    if info.missing_keys:
        print(f"  WARNING: {len(info.missing_keys)} missing keys: {info.missing_keys[:5]}...")
    if info.unexpected_keys:
        print(f"  WARNING: {len(info.unexpected_keys)} unexpected keys: {info.unexpected_keys[:5]}...")

    model.to(device)
    model.eval()
    return model


def export_to_onnx(
    weights_path: Path,
    output_path: Path,
    config_name: str,
    device: str = "cuda",
    opset_version: int = 17,
) -> Path:
    """Export a Chessformer .pt checkpoint to ONNX with dynamic batch axes."""
    dev = torch.device(device if torch.cuda.is_available() else "cpu")

    print(f"Loading Chessformer ({config_name}) from {weights_path.name} ...")
    model = _load_model(weights_path, config_name, dev)
    wrapper = ChessformerONNXWrapper(model)
    wrapper.eval()

    # Use batch_size=2 for tracing so the exporter sees a non-degenerate batch dim.
    dummy = _create_dummy_inputs(batch_size=2, device=dev)

    # Dynamic batch axes for every input and output.
    dynamic_axes = {name: {0: "batch"} for name in INPUT_NAMES + OUTPUT_NAMES}

    print(f"Exporting to ONNX (opset {opset_version}, dynamic batch) ...")
    with torch.no_grad():
        torch.onnx.export(
            wrapper,
            dummy,
            str(output_path),
            input_names=INPUT_NAMES,
            output_names=OUTPUT_NAMES,
            opset_version=opset_version,
            do_constant_folding=True,
            export_params=True,
            dynamic_axes=dynamic_axes,
            dynamo=False,  # Legacy TorchScript exporter for compatibility
        )

    # Validate the graph.
    import onnx
    onnx_model = onnx.load(str(output_path))
    onnx.checker.check_model(onnx_model)

    size_mb = output_path.stat().st_size / (1024 * 1024)
    print(f"  OK: {output_path}  ({size_mb:.1f} MB)")
    print(f"  Inputs:  {[i.name for i in onnx_model.graph.input]}")
    print(f"  Outputs: {[o.name for o in onnx_model.graph.output]}")
    return output_path


def validate_onnx_vs_pytorch(
    weights_path: Path,
    onnx_path: Path,
    config_name: str,
    device: str = "cuda",
) -> bool:
    """Compare ONNX outputs against PyTorch outputs for numerical validation."""
    import onnxruntime as ort

    print("\n" + "=" * 60)
    print("Validating ONNX vs PyTorch ...")
    print("=" * 60)

    dev = torch.device(device if torch.cuda.is_available() else "cpu")
    model = _load_model(weights_path, config_name, dev)
    model.eval()

    providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    session = ort.InferenceSession(str(onnx_path), providers=providers)
    print(f"  ONNX Runtime provider: {session.get_providers()[0]}")

    all_ok = True
    for bs in [1, 4, 16]:
        dummy = _create_dummy_inputs(bs, dev)
        with torch.no_grad():
            batch = {n: t for n, t in zip(INPUT_NAMES, dummy)}
            torch_outs = model(batch, return_promo=True)

        ort_inputs = {n: t.cpu().numpy() for n, t in zip(INPUT_NAMES, dummy)}
        onnx_outs = session.run(None, ort_inputs)

        max_diffs = []
        for name, t_out, o_out in zip(OUTPUT_NAMES, torch_outs, onnx_outs):
            diff = float(np.abs(t_out.cpu().numpy() - o_out).max())
            max_diffs.append(diff)
            if diff > 1e-2:
                all_ok = False

        worst = max(max_diffs)
        status = "OK" if worst < 1e-2 else "FAIL"
        print(f"  [{status}] batch={bs:>2d}:  worst max_diff = {worst:.2e}")

    if all_ok:
        print("  All outputs match within tolerance.")
    else:
        print("  WARNING: Some outputs differ -- check model config / weights.")
    return all_ok


def benchmark_inference(onnx_path: Path, device: str = "cuda", num_iters: int = 200) -> None:
    """Benchmark ONNX Runtime inference speed at various batch sizes."""
    import onnxruntime as ort
    import time

    print("\n" + "=" * 60)
    print("Benchmarking ONNX Runtime Inference ...")
    print("=" * 60)

    dev = torch.device(device if torch.cuda.is_available() else "cpu")
    providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    sess_options = ort.SessionOptions()
    sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    session = ort.InferenceSession(str(onnx_path), sess_options, providers=providers)
    print(f"  Provider: {session.get_providers()[0]}")

    for bs in [1, 4, 8, 16, 32, 64]:
        dummy = _create_dummy_inputs(bs, dev)
        ort_inputs = {n: t.cpu().numpy() for n, t in zip(INPUT_NAMES, dummy)}

        # Warmup
        for _ in range(5):
            session.run(None, ort_inputs)

        n = max(20, num_iters // bs)
        t0 = time.perf_counter()
        for _ in range(n):
            session.run(None, ort_inputs)
        elapsed = time.perf_counter() - t0

        evals = n * bs
        ms_call = (elapsed / n) * 1000
        ms_eval = (elapsed / evals) * 1000
        evals_s = evals / elapsed
        print(f"  batch={bs:>2d}: {ms_call:6.2f} ms/call, {ms_eval:5.2f} ms/eval, {evals_s:7.0f} evals/s")


def main():
    parser = argparse.ArgumentParser(
        description="Export Chessformer .pt -> ONNX (dynamic batch axes)",
    )
    parser.add_argument(
        "--input", "-i", type=Path, required=True,
        help="Path to a .pt checkpoint (e.g. inference/marvin_small.pt)",
    )
    parser.add_argument(
        "--output", "-o", type=Path, default=None,
        help="Output ONNX path. Default: replaces .pt with .onnx",
    )
    parser.add_argument(
        "--config", "-c", type=str, default=None,
        choices=list(CONFIGS.keys()),
        help="Model config name. Auto-detected from filename if omitted.",
    )
    parser.add_argument(
        "--device", "-d", type=str, default="cuda",
        help="Device for export tracing (cuda or cpu)",
    )
    parser.add_argument(
        "--opset", type=int, default=17,
        help="ONNX opset version (default: 17)",
    )
    parser.add_argument("--validate", action="store_true", help="Validate ONNX vs PyTorch outputs")
    parser.add_argument("--benchmark", action="store_true", help="Run inference benchmark")

    args = parser.parse_args()

    weights = args.input.resolve()
    if not weights.exists():
        print(f"Error: {weights} not found")
        sys.exit(1)
    if weights.suffix.lower() != ".pt":
        print(f"Error: expected a .pt checkpoint, got {weights.suffix}")
        sys.exit(1)

    config_name = args.config or _guess_config_name(weights)
    output = args.output or weights.with_suffix(".onnx")

    export_to_onnx(weights, output, config_name, device=args.device, opset_version=args.opset)

    if args.validate:
        validate_onnx_vs_pytorch(weights, output, config_name, device=args.device)

    if args.benchmark:
        benchmark_inference(output, device=args.device)

    print("\nDone!")


if __name__ == "__main__":
    main()
