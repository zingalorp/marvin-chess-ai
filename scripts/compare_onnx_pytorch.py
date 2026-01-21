"""
Compare ONNX Runtime vs PyTorch outputs for numerical accuracy.

Verifies that the ONNX export produces the same results as the original PyTorch model.
"""

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import torch
import chess

from inference.onnx_runtime import ONNXChessformer
from inference.runtime import load_default_chessformer
from inference.encoding import (
    make_model_batch,
    build_history_from_position,
    ContextOptions,
    TC_BLITZ,
)


def compare_outputs(position_name: str, board: chess.Board, moves: list[str]):
    """Compare PyTorch and ONNX outputs for a given position."""
    print(f"\n{'='*60}")
    print(f"Position: {position_name}")
    print(f"{'='*60}")
    
    # Build position
    final_board, board_history, rep_flags = build_history_from_position(board, moves)
    print(f"FEN: {final_board.fen()}")
    
    # Create batch
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ctx = ContextOptions(active_elo=1900, tc_base_s=180.0)
    batch = make_model_batch(
        board=final_board,
        board_history=board_history,
        repetition_flags=rep_flags,
        ctx=ctx,
        tc_cat=TC_BLITZ,
        device=device,
    )
    
    # Load PyTorch model
    _, pytorch_model, _ = load_default_chessformer(
        repo_root=REPO_ROOT,
        device=device,
        compile_model=False,
    )
    pytorch_model.eval()
    
    # Load ONNX model  
    onnx_model = ONNXChessformer()
    
    # PyTorch inference
    with torch.no_grad():
        pytorch_outputs = pytorch_model(batch)
    
    # ONNX inference
    np_batch = {k: v.cpu().numpy() for k, v in batch.items()}
    onnx_result = onnx_model.predict_from_batch(np_batch)
    
    # Compare outputs
    output_names = [
        'move_logits', 'value_out', 'value_cls_out',
        'value_error_out', 'time_cls_out', 'start_square_logits'
    ]
    onnx_outputs = [
        onnx_result.move_logits,
        onnx_result.value_out,
        onnx_result.value_cls_out,
        onnx_result.value_error_out,
        onnx_result.time_cls_out,
        onnx_result.start_square_logits,
    ]
    
    all_close = True
    print(f"\nOutput Comparison:")
    print(f"{'Output':<25} {'Max Diff':>12} {'Mean Diff':>12} {'Status':>10}")
    print("-" * 60)
    
    for name, pt_out, onnx_out in zip(output_names, pytorch_outputs, onnx_outputs):
        pt_arr = pt_out.cpu().numpy()
        max_diff = np.abs(pt_arr - onnx_out).max()
        mean_diff = np.abs(pt_arr - onnx_out).mean()
        
        # Tolerance for bf16 models
        is_close = np.allclose(pt_arr, onnx_out, rtol=1e-2, atol=1e-3)
        status = "✓" if is_close else "✗"
        
        print(f"{name:<25} {max_diff:>12.2e} {mean_diff:>12.2e} {status:>10}")
        
        if not is_close:
            all_close = False
    
    # Compare top moves
    print(f"\nTop Move Comparison:")
    legal_mask = np_batch['legal_mask'][0]
    
    # PyTorch top move
    pt_logits = pytorch_outputs[0][0].cpu().numpy()
    pt_masked = np.where(legal_mask[:4096], pt_logits[:4096], -np.inf)
    pt_top_idx = np.argmax(pt_masked)
    pt_from, pt_to = pt_top_idx // 64, pt_top_idx % 64
    pt_move = chess.Move(pt_from, pt_to)
    
    # ONNX top move
    onnx_logits = onnx_result.move_logits[0]
    onnx_masked = np.where(legal_mask[:4096], onnx_logits[:4096], -np.inf)
    onnx_top_idx = np.argmax(onnx_masked)
    onnx_from, onnx_to = onnx_top_idx // 64, onnx_top_idx % 64
    onnx_move = chess.Move(onnx_from, onnx_to)
    
    pt_san = final_board.san(pt_move) if pt_move in final_board.legal_moves else str(pt_move)
    onnx_san = final_board.san(onnx_move) if onnx_move in final_board.legal_moves else str(onnx_move)
    
    moves_match = pt_top_idx == onnx_top_idx
    print(f"  PyTorch top move: {pt_san}")
    print(f"  ONNX top move:    {onnx_san}")
    print(f"  Match: {'✓' if moves_match else '✗'}")
    
    # Compare values
    print(f"\nValue Comparison:")
    pt_value = pytorch_outputs[1][0, 0].item()
    onnx_value = onnx_result.value_out[0, 0]
    print(f"  PyTorch value: {pt_value:.4f}")
    print(f"  ONNX value:    {onnx_value:.4f}")
    print(f"  Diff:          {abs(pt_value - onnx_value):.4e}")
    
    return all_close and moves_match


def main():
    print("\n" + "=" * 60)
    print("PyTorch vs ONNX Comparison Test")
    print("=" * 60)
    
    test_positions = [
        ("Starting Position", chess.Board(), []),
        ("Scandinavian Defense", chess.Board(), ['e2e4', 'd7d5', 'e4d5', 'd8d5']),
        ("Italian Game", chess.Board(), ['e2e4', 'e7e5', 'g1f3', 'b8c6', 'f1c4']),
        ("Sicilian Defense", chess.Board(), ['e2e4', 'c7c5', 'g1f3', 'd7d6', 'd2d4', 'c5d4']),
        ("Queen's Gambit", chess.Board(), ['d2d4', 'd7d5', 'c2c4']),
    ]
    
    results = []
    for name, board, moves in test_positions:
        try:
            passed = compare_outputs(name, board, moves)
            results.append((name, passed))
        except Exception as e:
            print(f"\nError testing {name}: {e}")
            results.append((name, False))
    
    # Summary
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    
    all_passed = True
    for name, passed in results:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"  {name}: {status}")
        if not passed:
            all_passed = False
    
    print()
    if all_passed:
        print("All positions match! ✓")
    else:
        print("Some positions had differences (may be due to bf16 precision)")
    
    return all_passed


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
