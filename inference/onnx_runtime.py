"""
ONNX Runtime Inference for Marvin Chess AI

Provides CUDA-accelerated inference using ONNX Runtime.
Falls back to CPU if CUDA is unavailable.
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple, Dict, Any

import numpy as np

# Configure DLL search path for cuDNN on Windows
# ONNX Runtime CUDA requires cuDNN 9.x which is bundled with PyTorch
if sys.platform == 'win32':
    try:
        import torch
        torch_lib_path = Path(torch.__file__).parent / 'lib'
        if torch_lib_path.exists():
            # Add to DLL search path (Python 3.8+)
            os.add_dll_directory(str(torch_lib_path))
    except (ImportError, OSError):
        pass  # PyTorch not installed or DLL directory couldn't be added

try:
    import onnxruntime as ort
except ImportError:
    raise ImportError(
        "onnxruntime-gpu is required. Install with: pip install onnxruntime-gpu"
    )



@dataclass
class ONNXInferenceResult:
    """Result from ONNX inference."""
    move_logits: np.ndarray      # (B, 4098) policy logits
    value_out: np.ndarray        # (B, 1) value prediction
    value_cls_out: np.ndarray    # (B, 3) WDL classification
    value_error_out: np.ndarray  # (B, 1) value error/volatility
    time_cls_out: np.ndarray     # (B, 256) time bin logits
    start_square_logits: np.ndarray  # (B, 64) start square prediction


class ONNXChessformer:
    """
    ONNX Runtime wrapper for Chessformer inference.
    
    Uses CUDA if available, falls back to CPU.
    """
    
    def __init__(
        self,
        model_path: Optional[Path] = None,
        providers: Optional[List[str]] = None,
        use_tensorrt: bool = False,
    ):
        """
        Initialize ONNX inference session.
        
        Args:
            model_path: Path to ONNX model. Defaults to inference/marvin_small.onnx
            providers: List of execution providers. Defaults to CUDA -> CPU fallback.
            use_tensorrt: If True, use TensorRT provider (faster but slower startup)
        """
        # Default model path
        if model_path is None:
            model_path = Path(__file__).parent / "marvin_small.onnx"
        
        self.model_path = Path(model_path)
        if not self.model_path.exists():
            raise FileNotFoundError(
                f"ONNX model not found at {self.model_path}. "
                f"Run scripts/export_onnx.py first."
            )
        
        # Default providers: prefer CUDA, fallback to CPU
        if providers is None:
            available = ort.get_available_providers()
            providers = []
            
            if use_tensorrt and 'TensorrtExecutionProvider' in available:
                providers.append('TensorrtExecutionProvider')
            
            if 'CUDAExecutionProvider' in available:
                providers.append('CUDAExecutionProvider')
            
            providers.append('CPUExecutionProvider')  # Always include CPU fallback
        
        self.requested_providers = providers
        
        # Session options for optimization
        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        
        # Enable memory pattern optimization
        sess_options.enable_mem_pattern = True
        sess_options.enable_mem_reuse = True
        
        # Create session
        self.session = ort.InferenceSession(
            str(self.model_path),
            sess_options,
            providers=providers,
        )
        
        # Get actual active provider
        self.active_providers = self.session.get_providers()
        self.provider = self.active_providers[0] if self.active_providers else "Unknown"
        
        # Cache input/output names
        self.input_names = [inp.name for inp in self.session.get_inputs()]
        self.output_names = [out.name for out in self.session.get_outputs()]
        
        print(f"ONNXChessformer initialized:")
        print(f"  Model: {self.model_path.name}")
        print(f"  Provider: {self.provider}")
    
    @property
    def is_cuda(self) -> bool:
        """Check if running on CUDA."""
        return 'CUDA' in self.provider
    
    def predict(
        self,
        board_history: np.ndarray,   # (B, 8, 64) int64
        time_history: np.ndarray,    # (B, 8) float32
        rep_flags: np.ndarray,       # (B, 8) float32
        castling: np.ndarray,        # (B, 4) float32
        ep_mask: np.ndarray,         # (B, 64) float32
        scalars: np.ndarray,         # (B, 8) float32
        tc_cat: np.ndarray,          # (B,) int64
        legal_mask: np.ndarray,      # (B, 4098) bool
    ) -> ONNXInferenceResult:
        """
        Run inference on a batch of positions.
        
        Returns:
            ONNXInferenceResult with all model outputs.
        """
        # Ensure correct dtypes
        inputs = {
            'board_history': board_history.astype(np.int64),
            'time_history': time_history.astype(np.float32),
            'rep_flags': rep_flags.astype(np.float32),
            'castling': castling.astype(np.float32),
            'ep_mask': ep_mask.astype(np.float32),
            'scalars': scalars.astype(np.float32),
            'tc_cat': tc_cat.astype(np.int64),
            'legal_mask': legal_mask.astype(np.bool_),
        }
        
        # Run inference
        outputs = self.session.run(self.output_names, inputs)
        
        return ONNXInferenceResult(
            move_logits=outputs[0],
            value_out=outputs[1],
            value_cls_out=outputs[2],
            value_error_out=outputs[3],
            time_cls_out=outputs[4],
            start_square_logits=outputs[5],
        )
    
    def predict_from_batch(self, batch: Dict[str, np.ndarray]) -> ONNXInferenceResult:
        """
        Run inference from a dict batch (convenience method).
        
        Args:
            batch: Dict with keys matching input names
            
        Returns:
            ONNXInferenceResult
        """
        return self.predict(
            board_history=batch['board_history'],
            time_history=batch['time_history'],
            rep_flags=batch['rep_flags'],
            castling=batch['castling'],
            ep_mask=batch['ep_mask'],
            scalars=batch['scalars'],
            tc_cat=batch['tc_cat'],
            legal_mask=batch['legal_mask'],
        )


class BatchedONNXInference:
    """
    Batched inference for higher throughput.
    
    Collects positions and runs inference in batches.
    """
    
    def __init__(
        self,
        model: ONNXChessformer,
        batch_size: int = 32,
    ):
        self.model = model
        self.batch_size = batch_size
        self.pending: List[Dict[str, np.ndarray]] = []
        self.results: List[ONNXInferenceResult] = []
    
    def add(self, inputs: Dict[str, np.ndarray]) -> None:
        """Add a single position to the batch queue."""
        self.pending.append(inputs)
        
        if len(self.pending) >= self.batch_size:
            self._flush()
    
    def _flush(self) -> None:
        """Run inference on pending positions."""
        if not self.pending:
            return
        
        # Stack inputs into batch
        batch = {
            key: np.stack([p[key] for p in self.pending])
            for key in self.pending[0].keys()
        }
        
        result = self.model.predict_from_batch(batch)
        
        # Split results back into individual positions
        for i in range(len(self.pending)):
            self.results.append(ONNXInferenceResult(
                move_logits=result.move_logits[i:i+1],
                value_out=result.value_out[i:i+1],
                value_cls_out=result.value_cls_out[i:i+1],
                value_error_out=result.value_error_out[i:i+1],
                time_cls_out=result.time_cls_out[i:i+1],
                start_square_logits=result.start_square_logits[i:i+1],
            ))
        
        self.pending = []
    
    def get_results(self) -> List[ONNXInferenceResult]:
        """Flush any pending and return all results."""
        self._flush()
        results = self.results
        self.results = []
        return results


def compare_pytorch_vs_onnx(
    num_positions: int = 100,
    batch_size: int = 1,
) -> Tuple[float, float]:
    """
    Compare PyTorch vs ONNX Runtime inference speed.
    
    Returns:
        Tuple of (pytorch_ms, onnx_ms) average latencies
    """
    import sys
    import time
    import torch
    
    repo_root = Path(__file__).parent.parent
    sys.path.insert(0, str(repo_root))
    
    from inference.runtime import load_default_chessformer
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load PyTorch model
    print("Loading PyTorch model...")
    _, pytorch_model, _ = load_default_chessformer(
        repo_root=repo_root,
        device=device,
        compile_model=False,
    )
    pytorch_model.eval()
    
    # Load ONNX model
    print("Loading ONNX model...")
    onnx_model = ONNXChessformer()
    
    # Create test inputs
    def make_inputs():
        return {
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
    print("Warming up...")
    for _ in range(10):
        inputs = make_inputs()
        onnx_model.predict_from_batch(inputs)
        
        torch_batch = {k: torch.from_numpy(v).to(device) for k, v in inputs.items()}
        with torch.no_grad():
            pytorch_model(torch_batch)
    
    if device.type == 'cuda':
        torch.cuda.synchronize()
    
    # Benchmark PyTorch
    print(f"Benchmarking PyTorch ({num_positions} positions)...")
    start = time.perf_counter()
    for _ in range(num_positions):
        inputs = make_inputs()
        torch_batch = {k: torch.from_numpy(v).to(device) for k, v in inputs.items()}
        with torch.no_grad():
            pytorch_model(torch_batch)
    if device.type == 'cuda':
        torch.cuda.synchronize()
    pytorch_elapsed = time.perf_counter() - start
    pytorch_ms = (pytorch_elapsed / num_positions) * 1000
    
    # Benchmark ONNX
    print(f"Benchmarking ONNX Runtime ({num_positions} positions)...")
    start = time.perf_counter()
    for _ in range(num_positions):
        inputs = make_inputs()
        onnx_model.predict_from_batch(inputs)
    onnx_elapsed = time.perf_counter() - start
    onnx_ms = (onnx_elapsed / num_positions) * 1000
    
    print(f"\nResults (batch_size={batch_size}):")
    print(f"  PyTorch:     {pytorch_ms:.2f} ms/position")
    print(f"  ONNX Runtime: {onnx_ms:.2f} ms/position")
    print(f"  Speedup: {pytorch_ms / onnx_ms:.2f}x")
    
    return pytorch_ms, onnx_ms


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="ONNX Runtime Inference")
    parser.add_argument(
        "--compare",
        action="store_true",
        help="Compare PyTorch vs ONNX speed"
    )
    parser.add_argument(
        "--positions", "-n",
        type=int,
        default=100,
        help="Number of positions for benchmark"
    )
    parser.add_argument(
        "--batch-size", "-b",
        type=int,
        default=1,
        help="Batch size for benchmark"
    )
    
    args = parser.parse_args()
    
    if args.compare:
        compare_pytorch_vs_onnx(
            num_positions=args.positions,
            batch_size=args.batch_size,
        )
    else:
        # Quick test
        model = ONNXChessformer()
        print(f"\nProvider: {model.provider}")
        print(f"Is CUDA: {model.is_cuda}")
        print(f"Inputs: {model.input_names}")
        print(f"Outputs: {model.output_names}")
