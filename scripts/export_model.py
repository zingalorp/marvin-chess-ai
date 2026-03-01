#!/usr/bin/env python3
"""
Export a training checkpoint to a lightweight BF16 inference checkpoint.

This strips out optimizer state, scheduler state, and other training metadata,
keeping only the model weights converted to bfloat16.

Usage:
    python export_model.py <input_checkpoint> [output_path]
    
Examples:
    python export_model.py checkpoints/chessformer_v2_smolgen_best.pt
    python export_model.py checkpoints/chessformer_v2_smolgen_best.pt inference/my_model_bf16.pt
"""

import argparse
import sys
from pathlib import Path

import torch


def export_checkpoint(input_path: Path, output_path: Path) -> None:
    """Export a training checkpoint to BF16 inference weights."""
    
    print(f"Loading checkpoint: {input_path}")
    ckpt = torch.load(input_path, map_location="cpu", weights_only=False)
    
    # Extract model state dict
    if isinstance(ckpt, dict) and "model" in ckpt:
        state = ckpt["model"]
        print(f"  Found 'model' key in checkpoint")
        if "step" in ckpt:
            print(f"  Training step: {ckpt['step']}")
        if "best_val_loss" in ckpt:
            print(f"  Best val loss: {ckpt['best_val_loss']:.4f}")
    else:
        # Assume it's already just a state dict
        state = ckpt
        print(f"  Checkpoint is a raw state dict")
    
    # Strip _orig_mod. prefix from torch.compile'd models
    orig_keys = list(state.keys())
    if any(k.startswith("_orig_mod.") for k in orig_keys):
        state = {k.replace("_orig_mod.", ""): v for k, v in state.items()}
        print(f"  Stripped '_orig_mod.' prefix from {len(orig_keys)} keys")
    
    # Convert to bfloat16
    bf16_state = {}
    converted = 0
    for k, v in state.items():
        if torch.is_tensor(v) and v.is_floating_point():
            bf16_state[k] = v.to(torch.bfloat16)
            converted += 1
        else:
            bf16_state[k] = v
    
    print(f"  Converted {converted} tensors to bfloat16")
    print(f"  Total parameters: {len(bf16_state)}")
    
    # Calculate sizes
    input_size = input_path.stat().st_size / (1024 * 1024)
    
    # Save
    torch.save(bf16_state, output_path)
    output_size = output_path.stat().st_size / (1024 * 1024)
    
    print(f"\nExported to: {output_path}")
    print(f"  Input size:  {input_size:.1f} MB")
    print(f"  Output size: {output_size:.1f} MB")
    print(f"  Reduction:   {(1 - output_size/input_size) * 100:.1f}%")


def main():
    parser = argparse.ArgumentParser(
        description="Export a training checkpoint to BF16 inference weights"
    )
    parser.add_argument(
        "input", 
        type=Path, 
        help="Path to input checkpoint (.pt file)"
    )
    parser.add_argument(
        "output", 
        type=Path, 
        nargs="?",
        default=None,
        help="Output path (default: inference/<input_name>_bf16.pt)"
    )
    
    args = parser.parse_args()
    
    if not args.input.exists():
        print(f"Error: Input file not found: {args.input}")
        sys.exit(1)
    
    # Generate default output path if not specified
    if args.output is None:
        stem = args.input.stem.replace("_best", "").replace("_bf16", "")
        args.output = Path("inference") / f"{stem}_bf16.pt"
    
    # Ensure output directory exists
    args.output.parent.mkdir(parents=True, exist_ok=True)
    
    export_checkpoint(args.input, args.output)
    print("\nDone!")


if __name__ == "__main__":
    main()
