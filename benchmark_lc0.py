#!/usr/bin/env python3
"""Benchmark Marvin model via lc0 MCTS on both CPU and GPU backends."""

import subprocess
import time
import sys
import os

LC0_PATH = "/home/victor/projects/lco-marvin/build/release/lc0"
WEIGHTS = "/home/victor/projects/marvin-chess/marvin_for_leela.pb.gz"

# Environment with CUDA 12 libs
ENV = os.environ.copy()
ENV["LD_LIBRARY_PATH"] = (
    "/home/victor/onnxruntime/lib:"
    "/usr/local/cuda-12.8/lib64:"
    + ENV.get("LD_LIBRARY_PATH", "")
)

POSITIONS = [
    ("startpos", "position startpos"),
    ("sicilian", "position startpos moves e2e4 c7c5 g1f3 d7d6 d2d4 c5d4 f3d4 g8f6 b1c3"),
    ("ruy_lopez", "position startpos moves e2e4 e7e5 g1f3 b8c6 f1b5 a7a6 b5a4 g8f6"),
    ("queens_gambit", "position startpos moves d2d4 d7d5 c2c4 e7e6 b1c3 g8f6 c1g5"),
]


def run_lc0_benchmark(backend: str, nodes: int, position_cmd: str, threads: int = 1):
    """Run lc0 with given backend and return (nps, best_move, time_s, depth)."""
    cmd = [
        LC0_PATH,
        f"--weights={WEIGHTS}",
        f"--backend={backend}",
        f"--threads={threads}",
    ]
    
    proc = subprocess.Popen(
        cmd,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        env=ENV,
        cwd="/home/victor/projects/lco-marvin/build/release",
    )
    
    # Send UCI init and wait for readyok before searching
    proc.stdin.write(b"uci\n")
    proc.stdin.flush()
    
    # Read until uciok
    while True:
        raw = proc.stdout.readline()
        if not raw:
            return None, None, 0, 0, "lc0 crashed during init"
        line = raw.decode().strip()
        if line == "uciok":
            break
    
    proc.stdin.write(b"isready\n")
    proc.stdin.flush()
    
    # Read until readyok
    while True:
        raw = proc.stdout.readline()
        if not raw:
            return None, None, 0, 0, "lc0 crashed during isready"
        line = raw.decode().strip()
        if line == "readyok":
            break
    
    # Send position and go
    proc.stdin.write(f"{position_cmd}\n".encode())
    proc.stdin.write(f"go nodes {nodes}\n".encode())
    proc.stdin.flush()
    
    # Read until bestmove
    best_move = None
    last_nps = 0
    last_depth = 0
    last_time_ms = 0
    last_nodes = 0
    
    import select
    deadline = time.time() + 600  # 10 min timeout
    
    while time.time() < deadline:
        raw = proc.stdout.readline()
        if not raw:  # Process died / EOF
            break
        line = raw.decode().strip()
        if not line:
            continue
        if line.startswith("bestmove"):
            parts = line.split()
            best_move = parts[1] if len(parts) > 1 else "???"
            break
        elif line.startswith("info"):
            parts = line.split()
            for i, p in enumerate(parts):
                if p == "nps" and i + 1 < len(parts):
                    last_nps = int(parts[i + 1])
                if p == "depth" and i + 1 < len(parts):
                    last_depth = int(parts[i + 1])
                if p == "time" and i + 1 < len(parts):
                    last_time_ms = int(parts[i + 1])
                if p == "nodes" and i + 1 < len(parts):
                    last_nodes = int(parts[i + 1])
    
    # Quit
    try:
        proc.stdin.write(b"quit\n")
        proc.stdin.flush()
        proc.wait(timeout=5)
    except:
        proc.kill()
        proc.wait()
    
    search_time = last_time_ms / 1000.0 if last_time_ms else 0
    return last_nps, best_move, search_time, last_depth, None


def main():
    nodes_list = [10, 50, 100]
    backends = ["onnx-cpu", "onnx-cuda"]
    
    print("=" * 80)
    print("MARVIN via lc0 - BENCHMARK")
    print(f"Model: {WEIGHTS}")
    print(f"Engine: {LC0_PATH}")
    print("=" * 80)
    
    for backend in backends:
        print(f"\n{'=' * 60}")
        print(f"BACKEND: {backend}")
        print(f"{'=' * 60}")
        
        # Warm-up run
        print(f"  Warming up {backend}...")
        nps, move, t, depth, err = run_lc0_benchmark(backend, 5, "position startpos")
        if err:
            print(f"  ERROR: {err}")
            print(f"  Skipping {backend}")
            continue
        print(f"  Warm-up done: {move or '???'}, {nps} NPS, {t:.1f}s")
        
        for nodes in nodes_list:
            print(f"\n  --- {nodes} nodes ---")
            for pos_name, pos_cmd in POSITIONS:
                nps, move, t, depth, err = run_lc0_benchmark(backend, nodes, pos_cmd)
                if err:
                    print(f"  {pos_name:15s}: ERROR - {err}")
                else:
                    move_str = move or "???"
                    print(f"  {pos_name:15s}: {move_str:6s}  depth={depth:2d}  "
                          f"NPS={nps:5d}  time={t:6.1f}s")
        
        # Test with multiple threads (for parallelized MCTS)
        if backend == "onnx-cuda":
            print(f"\n  --- GPU + Multi-thread test (100 nodes) ---")
            for threads in [1, 2, 4]:
                nps, move, t, depth, err = run_lc0_benchmark(
                    backend, 100, "position startpos", threads=threads
                )
                if err:
                    print(f"  threads={threads}: ERROR - {err}")
                else:
                    move_str = move or "???"
                    print(f"  threads={threads}: {move_str:6s}  depth={depth:2d}  "
                          f"NPS={nps:5d}  time={t:6.1f}s")
    
    # Also benchmark raw ONNX inference (no MCTS overhead)
    print(f"\n{'=' * 60}")
    print("RAW ONNX INFERENCE (no MCTS)")
    print(f"{'=' * 60}")
    
    try:
        import onnxruntime as ort
        import numpy as np
        
        onnx_path = "/home/victor/projects/marvin-chess/marvin_for_leela.onnx"
        
        # Prepare input
        inp = np.zeros((1, 112, 8, 8), dtype=np.float32)
        inp[0, 111, :, :] = 1.0
        inp[0, 5, 0, 4] = 1.0
        inp[0, 11, 7, 4] = 1.0
        
        for provider_name, providers in [
            ("CPU", ["CPUExecutionProvider"]),
            ("CUDA", ["CUDAExecutionProvider", "CPUExecutionProvider"]),
        ]:
            try:
                sess = ort.InferenceSession(onnx_path, providers=providers)
                active = sess.get_providers()
                if provider_name == "CUDA" and "CUDAExecutionProvider" not in active:
                    print(f"  {provider_name}: CUDA provider not available, skipping")
                    continue
                    
                # Warm up
                for _ in range(3):
                    sess.run(None, {"/input/planes": inp})
                
                # Benchmark batch=1
                for batch_size in [1, 8, 32]:
                    inp_batch = np.tile(inp, (batch_size, 1, 1, 1))
                    n_iters = max(10, 100 // batch_size)
                    
                    t0 = time.time()
                    for _ in range(n_iters):
                        sess.run(None, {"/input/planes": inp_batch})
                    t_total = time.time() - t0
                    
                    evals_per_sec = (n_iters * batch_size) / t_total
                    ms_per_eval = (t_total / (n_iters * batch_size)) * 1000
                    print(f"  {provider_name} batch={batch_size:2d}: "
                          f"{evals_per_sec:7.1f} evals/s  "
                          f"({ms_per_eval:.1f} ms/eval)")
            except Exception as e:
                print(f"  {provider_name}: {e}")
    except ImportError:
        print("  onnxruntime not available")
    
    # Also benchmark via PyTorch directly (what your MCTS uses)
    print(f"\n{'=' * 60}")
    print("RAW PYTORCH INFERENCE (what your MCTS uses)")
    print(f"{'=' * 60}")
    
    try:
        import torch
        sys.path.insert(0, "/home/victor/projects/marvin-chess")
        from inference.model_loader import load_model
        
        model, config = load_model("/home/victor/projects/marvin-chess/inference/marvin_large.pt")
        
        for device_name in ["cpu", "cuda"]:
            try:
                device = torch.device(device_name)
                model_dev = model.to(device)
                model_dev.eval()
                
                # Create a minimal batch
                batch = {
                    'board_history': torch.zeros(1, 8, 64, dtype=torch.long, device=device),
                    'time_history': torch.zeros(1, 8, device=device),
                    'rep_flags': torch.zeros(1, 8, device=device),
                    'castling': torch.zeros(1, 4, device=device),
                    'ep_mask': torch.zeros(1, 64, device=device),
                    'scalars': torch.zeros(1, 8, device=device),
                    'tc_cat': torch.zeros(1, dtype=torch.long, device=device),
                    'legal_mask': None,
                }
                # Put some pieces
                batch['board_history'][0, 0, 4] = 6   # White king e1
                batch['board_history'][0, 0, 60] = 12  # Black king e8
                
                # Warm up
                with torch.no_grad():
                    for _ in range(3):
                        model_dev(batch)
                    if device_name == "cuda":
                        torch.cuda.synchronize()
                
                for batch_size in [1, 8, 32]:
                    batch_big = {
                        k: v.expand(batch_size, *v.shape[1:]).contiguous() 
                        if isinstance(v, torch.Tensor) else v
                        for k, v in batch.items()
                    }
                    n_iters = max(10, 100 // batch_size)
                    
                    if device_name == "cuda":
                        torch.cuda.synchronize()
                    t0 = time.time()
                    with torch.no_grad():
                        for _ in range(n_iters):
                            model_dev(batch_big)
                    if device_name == "cuda":
                        torch.cuda.synchronize()
                    t_total = time.time() - t0
                    
                    evals_per_sec = (n_iters * batch_size) / t_total
                    ms_per_eval = (t_total / (n_iters * batch_size)) * 1000
                    print(f"  PyTorch {device_name:4s} batch={batch_size:2d}: "
                          f"{evals_per_sec:7.1f} evals/s  "
                          f"({ms_per_eval:.1f} ms/eval)")
                    
            except Exception as e:
                print(f"  PyTorch {device_name}: {e}")
    except Exception as e:
        print(f"  PyTorch benchmark failed: {e}")
    
    print(f"\n{'=' * 80}")
    print("BENCHMARK COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()
