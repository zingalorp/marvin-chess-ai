from __future__ import annotations

import platform
import sys
from pathlib import Path
from typing import TextIO

import torch

from inference.model_loader import LoadedModel, load_chessformer
from inference.config import get_model_path, get_config_name


def resolve_repo_root(*, cwd: Path | None = None) -> Path:
    cwd = (cwd or Path.cwd()).resolve()
    if (cwd / "model.py").exists():
        return cwd
    if (cwd.parent / "model.py").exists():
        return cwd.parent
    return cwd


def ensure_repo_on_syspath(repo_root: Path) -> None:
    repo_root = repo_root.resolve()
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))


def default_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def print_device_info(device: torch.device, *, file: TextIO | None = None, prefix: str = "") -> None:
    """Print comprehensive device information for debugging and user feedback.
    
    Args:
        device: The torch device being used.
        file: Output stream (defaults to sys.stdout if None).
        prefix: Optional prefix for each line (e.g., "# " for UCI output).
    """
    import sys as _sys
    out = file or _sys.stdout
    p = prefix
    
    def _print(msg: str) -> None:
        out.write(f"{p}{msg}\n")
        out.flush()
    
    _print("="*60)
    _print("DEVICE INFORMATION")
    _print("="*60)
    
    # Platform info
    _print(f"Platform: {platform.system()} {platform.release()} ({platform.machine()})")
    _print(f"Python: {platform.python_version()}")
    _print(f"PyTorch: {torch.__version__}")
    
    # CUDA availability
    cuda_available = torch.cuda.is_available()
    _print(f"")
    _print(f"CUDA Available: {cuda_available}")
    
    if cuda_available:
        _print(f"CUDA Version: {torch.version.cuda}")
        cudnn_version = torch.backends.cudnn.version()
        _print(f"cuDNN Version: {cudnn_version}")
        _print(f"cuDNN Enabled: {torch.backends.cudnn.enabled}")
        
        # GPU count and details
        gpu_count = torch.cuda.device_count()
        _print(f"GPU Count: {gpu_count}")
        
        for i in range(gpu_count):
            gpu_name = torch.cuda.get_device_name(i)
            props = torch.cuda.get_device_properties(i)
            total_mem_gb = props.total_memory / (1024**3)
            _print(f"")
            _print(f"GPU {i}: {gpu_name}")
            _print(f"  - Compute Capability: {props.major}.{props.minor}")
            _print(f"  - Total Memory: {total_mem_gb:.2f} GB")
            _print(f"  - Multi-Processor Count: {props.multi_processor_count}")
            
            # Current memory usage if this is the active device
            if i == (device.index if device.index is not None else 0) and device.type == "cuda":
                allocated = torch.cuda.memory_allocated(i) / (1024**3)
                reserved = torch.cuda.memory_reserved(i) / (1024**3)
                _print(f"  - Memory Allocated: {allocated:.3f} GB")
                _print(f"  - Memory Reserved: {reserved:.3f} GB")
    else:
        _print("")
        _print("CUDA is not available. Possible reasons:")
        _print("  - No NVIDIA GPU installed")
        _print("  - NVIDIA drivers not installed or outdated")
        _print("  - PyTorch installed without CUDA support")
        _print("  - Incompatible CUDA/PyTorch versions")
        _print("")
        _print("To enable GPU support on Windows:")
        _print("  1. Install NVIDIA GPU drivers from nvidia.com")
        _print("  2. Install PyTorch with CUDA: pip install torch --index-url https://download.pytorch.org/whl/cu121")
    
    # Selected device
    _print("")
    _print("-"*60)
    if device.type == "cuda":
        gpu_idx = device.index if device.index is not None else 0
        gpu_name = torch.cuda.get_device_name(gpu_idx)
        _print(f">>> USING DEVICE: {device} ({gpu_name}) <<<")
        _print(f">>> GPU ACCELERATION: ENABLED <<<")
    else:
        _print(f">>> USING DEVICE: {device} <<<")
        _print(f">>> GPU ACCELERATION: DISABLED (running on CPU) <<<")
    _print("-"*60)
    _print("")


def load_default_chessformer(*, repo_root: Path | None = None, device: torch.device | None = None, compile_model: bool = True, checkpoint_path: Path | None = None, config_name: str | None = None) -> tuple[LoadedModel, torch.nn.Module, Path]:
    repo_root = (repo_root or resolve_repo_root()).resolve()
    ensure_repo_on_syspath(repo_root)

    device = device or default_device()

    model_py_path = repo_root / "model.py"
    # Use provided checkpoint or get from centralized config
    if checkpoint_path is None:
        checkpoint_path = get_model_path(repo_root)
    # Use provided config or get from centralized config
    if config_name is None:
        config_name = get_config_name()
    if not checkpoint_path.exists():
        print(f"Error: model weights are missing. Expected checkpoint at {checkpoint_path}.")
        raise FileNotFoundError(checkpoint_path)

    loaded = load_chessformer(
        model_py_path=model_py_path,
        config_name=config_name,
        checkpoint_path=checkpoint_path,
        device=device,
    )
    model = loaded.model
    model.eval()

    if compile_model:
        # torch.compile with Triton backend doesn't work on Windows
        if platform.system() == "Windows":
            print("Note: Skipping torch.compile on Windows (Triton not supported).")
        else:
            try:
                model = torch.compile(model)
            except Exception as e:
                print(f"Warning: torch.compile failed: {e}. Falling back to eager mode.")

    return loaded, model, checkpoint_path


def count_attn_layers(model: torch.nn.Module) -> int:
    n = 0
    # Handle torch.compile wrapper
    orig_model = getattr(model, "_orig_mod", model)
    for mod in orig_model.modules():
        if hasattr(mod, "last_attn_probs"):
            n += 1
    return n