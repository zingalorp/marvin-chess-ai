"""
Unified inference backend abstraction for PyTorch and ONNX models.

Both backends expose the same interface:
    backend(batch_dict, return_promo=True) -> tuple of 7 tensors

The tuple contains:
    (move_logits, value_out, value_cls_out, value_error_out,
     time_cls_out, start_square_logits, promo_logits)

All outputs are torch tensors on the backend's device, regardless of
whether the underlying engine is PyTorch or ONNX Runtime.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch


class PyTorchBackend:
    """Wraps a torch.nn.Module for inference with autocast + inference_mode."""

    def __init__(self, model: torch.nn.Module, device: torch.device, config_name: str) -> None:
        self._model = model
        self._device = device
        self._config_name = config_name

    # ------------------------------------------------------------------
    @property
    def device(self) -> torch.device:
        return self._device

    @property
    def config_name(self) -> str:
        return self._config_name

    @property
    def kind(self) -> str:
        return "pytorch"

    # ------------------------------------------------------------------
    def __call__(self, batch: dict[str, torch.Tensor], return_promo: bool = True) -> tuple:
        with torch.inference_mode():
            with torch.autocast(device_type=self._device.type, enabled=(self._device.type == "cuda")):
                return self._model(batch, return_promo=return_promo)


class OnnxBackend:
    """Wraps an ONNX Runtime InferenceSession.

    Accepts the same batch dict of torch tensors and returns torch tensors,
    so call-sites don't need to know the backend type.
    """

    # Ordered output names matching the 7-tuple convention.
    _OUTPUT_NAMES = (
        "move_logits",
        "value_out",
        "value_cls_out",
        "value_error_out",
        "time_cls_out",
        "start_square_logits",
        "promo_logits",
    )

    # Ordered input names matching the batch dict keys used in encoding.py.
    _INPUT_NAMES = (
        "board_history",
        "time_history",
        "rep_flags",
        "castling",
        "ep_mask",
        "scalars",
        "tc_cat",
        "legal_mask",
    )

    def __init__(self, session_path: str | Path, device: torch.device, config_name: str) -> None:
        import onnxruntime as ort

        providers: list[str] = []
        if device.type == "cuda":
            providers.append("CUDAExecutionProvider")
        providers.append("CPUExecutionProvider")

        self._session = ort.InferenceSession(str(session_path), providers=providers)
        self._device = device
        self._config_name = config_name

        # Cache the actual provider in use.
        active = self._session.get_providers()
        if "CUDAExecutionProvider" in active:
            print(f"[OnnxBackend] Using CUDAExecutionProvider")
        else:
            print(f"[OnnxBackend] Using CPUExecutionProvider")

        # Build input-name → dtype map from the session metadata.
        self._input_meta = {inp.name: inp for inp in self._session.get_inputs()}

    # ------------------------------------------------------------------
    @property
    def device(self) -> torch.device:
        return self._device

    @property
    def config_name(self) -> str:
        return self._config_name

    @property
    def kind(self) -> str:
        return "onnx"

    # ------------------------------------------------------------------
    @staticmethod
    def _to_numpy(t: torch.Tensor, target_dtype: str | None = None) -> np.ndarray:
        """Convert a torch tensor to numpy, coercing dtype if needed."""
        arr = t.detach().cpu().numpy()
        if target_dtype is not None:
            onnx_to_np = {
                "tensor(float)": np.float32,
                "tensor(int64)": np.int64,
                "tensor(bool)": np.bool_,
                "tensor(float16)": np.float16,
                "tensor(double)": np.float64,
                "tensor(int32)": np.int32,
            }
            np_dtype = onnx_to_np.get(target_dtype)
            if np_dtype is not None and arr.dtype != np_dtype:
                arr = arr.astype(np_dtype)
        return arr

    def __call__(self, batch: dict[str, torch.Tensor], return_promo: bool = True) -> tuple:
        # Build the ONNX feed dict from the batch, matching expected input names.
        feed: dict[str, np.ndarray] = {}
        for name in self._INPUT_NAMES:
            if name not in batch:
                continue
            meta = self._input_meta.get(name)
            target_dtype = meta.type if meta is not None else None
            feed[name] = self._to_numpy(batch[name], target_dtype)

        output_names = list(self._OUTPUT_NAMES)
        raw_outputs = self._session.run(output_names, feed)

        # Wrap outputs as torch tensors on the target device.
        results = []
        for arr in raw_outputs:
            t = torch.from_numpy(arr)
            if self._device.type != "cpu":
                t = t.to(self._device)
            results.append(t)

        return tuple(results)
