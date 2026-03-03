from __future__ import annotations

import os
import sys
from pathlib import Path

from inference.config import get_model_path, get_config_name
from inference.backend import OnnxBackend


# Type alias kept for downstream type annotations.
InferenceBackend = OnnxBackend


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


def default_device() -> str:
    """Return the best available device as a plain string ("cpu" or "cuda")."""
    override = os.environ.get("MARVIN_DEVICE")
    if override:
        return str(override).split(":")[0]
    try:
        import onnxruntime as ort
        if "CUDAExecutionProvider" in ort.get_available_providers():
            return "cuda"
    except ImportError:
        pass
    return "cpu"


def load_default_backend(
    *,
    repo_root: Path | None = None,
    device: str | None = None,
    checkpoint_path: Path | None = None,
    config_name: str | None = None,
) -> tuple[OnnxBackend, Path]:
    """Load an ONNX backend.

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
    if suffix != ".onnx":
        raise ValueError(
            f"Only .onnx weights are supported for inference, got '{suffix}'. "
            f"Convert .pt weights to ONNX first."
        )

    # Infer config_name from filename (marvin_large.onnx -> large, etc.).
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

    backend = OnnxBackend(checkpoint_path, device=device, config_name=config_name)
    print(f"[runtime] Loaded ONNX backend: {checkpoint_path.name} (config: {config_name}, device: {backend.device})")
    return backend, checkpoint_path