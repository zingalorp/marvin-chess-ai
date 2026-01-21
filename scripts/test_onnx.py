"""
Full functionality test for ONNX Runtime inference.

Tests the complete pipeline:
1. Load ONNX model with CUDA
2. Create a real chess position  
3. Encode position using the encoding module
4. Run inference
5. Decode and display results
"""

import sys
from pathlib import Path

# Add repo root to path
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import chess
import torch

from inference.onnx_runtime import ONNXChessformer
from inference.encoding import (
    make_model_batch,
    build_history_from_position,
    ContextOptions,
    TC_BLITZ,
    TC_RAPID,
)


def test_basic_inference():
    """Test basic ONNX inference on a starting position."""
    print("=== Test 1: Basic Inference ===\n")
    
    # Load model
    print("Loading ONNX model...")
    model = ONNXChessformer()
    print(f"  Provider: {model.provider}")
    print(f"  Is CUDA: {model.is_cuda}")
    
    # Create random batch
    batch_size = 1
    np_batch = {
        'board_history': np.random.randint(0, 13, (batch_size, 8, 64), dtype=np.int64),
        'time_history': np.random.randn(batch_size, 8).astype(np.float32),
        'rep_flags': np.random.randint(0, 2, (batch_size, 8)).astype(np.float32),
        'castling': np.random.randint(0, 2, (batch_size, 4)).astype(np.float32),
        'ep_mask': np.zeros((batch_size, 64), dtype=np.float32),
        'scalars': np.random.randn(batch_size, 8).astype(np.float32),
        'tc_cat': np.random.randint(0, 3, (batch_size,), dtype=np.int64),
        'legal_mask': np.ones((batch_size, 4098), dtype=np.bool_),
    }
    
    # Run inference
    print("\nRunning inference...")
    result = model.predict_from_batch(np_batch)
    
    print(f"  move_logits: {result.move_logits.shape}")
    print(f"  value_out: {result.value_out.shape}")
    print(f"  value_cls_out: {result.value_cls_out.shape}")
    print(f"  time_cls_out: {result.time_cls_out.shape}")
    
    print("\n✓ Basic inference passed!\n")
    return model


def test_real_position(model: ONNXChessformer):
    """Test inference on a real chess position."""
    print("=== Test 2: Real Chess Position ===\n")
    
    # Create position: Scandinavian Defense
    board = chess.Board()
    moves = ['e2e4', 'd7d5', 'e4d5', 'd8d5']
    final_board, board_history, rep_flags = build_history_from_position(board, moves)
    
    print(f"Position: {final_board.fen()}")
    print(f"Legal moves: {len(list(final_board.legal_moves))}")
    
    # Create batch
    device = torch.device('cpu')
    ctx = ContextOptions(active_elo=1900, tc_base_s=180.0)
    batch = make_model_batch(
        board=final_board,
        board_history=board_history,
        repetition_flags=rep_flags,
        ctx=ctx,
        tc_cat=TC_BLITZ,
        device=device,
    )
    
    # Convert to numpy
    np_batch = {k: v.numpy() for k, v in batch.items()}
    
    # Run inference
    print("\nRunning inference...")
    result = model.predict_from_batch(np_batch)
    
    # Value prediction
    value = result.value_out[0, 0]
    wdl = np.exp(result.value_cls_out[0])
    wdl = wdl / wdl.sum()
    print(f"  Value: {value:.4f}")
    print(f"  WDL: W={wdl[0]:.1%} D={wdl[1]:.1%} L={wdl[2]:.1%}")
    
    # Top moves
    legal_mask = np_batch['legal_mask'][0]
    masked_logits = np.where(legal_mask[:4096], result.move_logits[0, :4096], -np.inf)
    
    # Get top 5 moves
    top_indices = np.argsort(masked_logits)[-5:][::-1]
    probs = np.exp(masked_logits - masked_logits.max())
    probs = probs / probs.sum()
    
    print("\nTop 5 moves:")
    for idx in top_indices:
        from_sq = idx // 64
        to_sq = idx % 64
        move = chess.Move(from_sq, to_sq)
        if move in final_board.legal_moves:
            san = final_board.san(move)
            prob = probs[idx]
            print(f"  {san}: {prob:.1%}")
    
    print("\n✓ Real position test passed!\n")


def test_batched_inference(model: ONNXChessformer):
    """Test batched inference."""
    print("=== Test 3: Batched Inference ===\n")
    
    from inference.onnx_runtime import BatchedONNXInference
    
    batcher = BatchedONNXInference(model, batch_size=8)
    
    # Add positions
    for i in range(20):
        inputs = {
            'board_history': np.random.randint(0, 13, (1, 8, 64), dtype=np.int64)[0],
            'time_history': np.random.randn(1, 8).astype(np.float32)[0],
            'rep_flags': np.random.randint(0, 2, (1, 8)).astype(np.float32)[0],
            'castling': np.random.randint(0, 2, (1, 4)).astype(np.float32)[0],
            'ep_mask': np.zeros((1, 64), dtype=np.float32)[0],
            'scalars': np.random.randn(1, 8).astype(np.float32)[0],
            'tc_cat': np.random.randint(0, 3, (1,), dtype=np.int64)[0],
            'legal_mask': np.ones((1, 4098), dtype=np.bool_)[0],
        }
        batcher.add(inputs)
    
    results = batcher.get_results()
    print(f"Processed {len(results)} positions in batches")
    print(f"  First result move_logits shape: {results[0].move_logits.shape}")
    
    print("\n✓ Batched inference passed!\n")


def test_speed():
    """Benchmark inference speed."""
    print("=== Test 4: Speed Benchmark ===\n")
    
    import time
    
    model = ONNXChessformer()
    
    # Create batch
    batch_size = 1
    np_batch = {
        'board_history': np.random.randint(0, 13, (batch_size, 8, 64), dtype=np.int64),
        'time_history': np.random.randn(batch_size, 8).astype(np.float32),
        'rep_flags': np.random.randint(0, 2, (batch_size, 8)).astype(np.float32),
        'castling': np.random.randint(0, 2, (batch_size, 4)).astype(np.float32),
        'ep_mask': np.zeros((batch_size, 64), dtype=np.float32),
        'scalars': np.random.randn(batch_size, 8).astype(np.float32),
        'tc_cat': np.random.randint(0, 3, (batch_size,), dtype=np.int64),
        'legal_mask': np.ones((batch_size, 4098), dtype=np.bool_),
    }
    
    # Warmup
    for _ in range(10):
        model.predict_from_batch(np_batch)
    
    # Benchmark
    num_iters = 100
    start = time.perf_counter()
    for _ in range(num_iters):
        model.predict_from_batch(np_batch)
    elapsed = time.perf_counter() - start
    
    avg_ms = (elapsed / num_iters) * 1000
    pos_per_sec = num_iters / elapsed
    
    print(f"Provider: {model.provider}")
    print(f"Average latency: {avg_ms:.2f} ms")
    print(f"Throughput: {pos_per_sec:.0f} positions/sec")
    
    print("\n✓ Speed benchmark passed!\n")


def main():
    print("\n" + "=" * 60)
    print("ONNX Runtime Full Functionality Test")
    print("=" * 60 + "\n")
    
    # Run tests
    model = test_basic_inference()
    test_real_position(model)
    test_batched_inference(model)
    test_speed()
    
    print("=" * 60)
    print("All tests passed! ✓")
    print("=" * 60)


if __name__ == "__main__":
    main()
