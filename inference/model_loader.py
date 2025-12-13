from __future__ import annotations

import importlib.util
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Literal, Tuple

import torch


ConfigName = Literal["leela", "deep", "smolgen", "100m"]


@dataclass(frozen=True)
class LoadedModel:
    model: torch.nn.Module
    config: Dict[str, Any]
    device: torch.device


def _load_model_module(model_py_path: Path):
    spec = importlib.util.spec_from_file_location("marvin_model_v2_1", model_py_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load module spec from {model_py_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_chessformer_v2(
    *,
    model_py_path: str | Path = "model_v2-1.py",
    config_name: ConfigName = "smolgen",
    checkpoint_path: str | Path = "checkpoints/chessformer_v2_smolgen_best.pt",
    device: str | torch.device = "cuda",
) -> LoadedModel:
    model_py_path = Path(model_py_path)
    checkpoint_path = Path(checkpoint_path)
    if not model_py_path.exists():
        raise FileNotFoundError(model_py_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(checkpoint_path)

    module = _load_model_module(model_py_path)

    if config_name == "deep":
        config = dict(module.CONFIG_V2_DEEP)
    elif config_name == "leela":
        config = dict(module.CONFIG_V2_LEELA)
    elif config_name == "100m":
        config = dict(module.CONFIG_100M_BALANCED)
    else:
        config = dict(module.CONFIG_V2_SMOLGEN)

    device = torch.device(device)
    model = module.ChessformerV2(config).to(device)

    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state = ckpt.get("model", ckpt)
    try:
        model.load_state_dict(state)
    except RuntimeError:
        # torch.compile prefix
        state = {k.replace("_orig_mod.", ""): v for k, v in state.items()}
        model.load_state_dict(state)

    model.eval()
    return LoadedModel(model=model, config=config, device=device)
