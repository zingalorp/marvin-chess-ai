from __future__ import annotations

import sys
from pathlib import Path
from typing import Union

import torch

from inference.model_loader import LoadedModel, load_chessformer
from inference.config import get_model_path, get_config_name
from inference.backend import PyTorchBackend, OnnxBackend


# Type alias for either backend.
InferenceBackend = Union[PyTorchBackend, OnnxBackend]


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
        try:
            model = torch.compile(model, mode="default")
        except Exception as e:
            print(f"Warning: torch.compile failed: {e}. Falling back to eager mode.")

    return loaded, model, checkpoint_path


def load_default_backend(
    *,
    repo_root: Path | None = None,
    device: torch.device | None = None,
    compile_model: bool = True,
    checkpoint_path: Path | None = None,
    config_name: str | None = None,
) -> tuple[InferenceBackend, Path]:
    """Load a PyTorch or ONNX backend based on the weights file extension.

    Returns ``(backend, weights_path)``.
    """
    repo_root = (repo_root or resolve_repo_root()).resolve()
    ensure_repo_on_syspath(repo_root)

    device = device or default_device()

    if checkpoint_path is None:
        checkpoint_path = get_model_path(repo_root)
    checkpoint_path = Path(checkpoint_path)

    if not checkpoint_path.exists():
        print(f"Error: model weights are missing. Expected weights at {checkpoint_path}.")
        raise FileNotFoundError(checkpoint_path)

    suffix = checkpoint_path.suffix.lower()

    if suffix == ".onnx":
        # For ONNX we still need to determine config_name for metadata.
        # Infer from filename (marvin_large.onnx -> large, etc.).
        if config_name is None or config_name == "auto":
            name_lower = checkpoint_path.stem.lower()
            if "large" in name_lower:
                config_name = "large"
            elif "tiny" in name_lower:
                config_name = "tiny"
            elif "small" in name_lower:
                config_name = "small"
            else:
                config_name = "unknown"

        backend: InferenceBackend = OnnxBackend(checkpoint_path, device=device, config_name=config_name)
        print(f"[runtime] Loaded ONNX backend: {checkpoint_path.name} (config: {config_name}, device: {device})")
        return backend, checkpoint_path

    # Default: PyTorch .pt weights
    loaded, model, checkpoint_path = load_default_chessformer(
        repo_root=repo_root,
        device=device,
        compile_model=compile_model,
        checkpoint_path=checkpoint_path,
        config_name=config_name,
    )
    backend = PyTorchBackend(model, device=loaded.device, config_name=loaded.config_name)
    print(f"[runtime] Loaded PyTorch backend: {checkpoint_path.name} (config: {loaded.config_name}, device: {loaded.device})")
    return backend, checkpoint_path