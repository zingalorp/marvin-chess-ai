"""
ONNX inference backend for Chessformer models.

The backend exposes a simple callable interface:
    backend(batch_dict, return_promo=True) -> tuple of 7 numpy arrays

The tuple contains:
    (move_logits, value_out, value_cls_out, value_error_out,
     time_cls_out, start_square_logits, promo_logits)

All outputs are numpy arrays (no PyTorch dependency required).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np


class OnnxBackend:
    """Wraps an ONNX Runtime InferenceSession.

    Accepts batch dicts of numpy arrays and returns numpy arrays.
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

    def __init__(self, session_path: str | Path, device: str = "cpu", config_name: str = "") -> None:
        import onnxruntime as ort

        device_str = str(device).split(":")[0] if device else "cpu"

        self._session_path = str(session_path)
        self._device_str = device_str
        self._config_name = config_name

        self._session, self._input_meta = self._create_session()

    def _create_session(self):
        import onnxruntime as ort

        providers: list[str] = []
        if self._device_str == "cuda":
            providers.append("CUDAExecutionProvider")
        providers.append("CPUExecutionProvider")

        session = ort.InferenceSession(self._session_path, providers=providers)

        # Cache the actual provider in use.
        active = session.get_providers()
        if "CUDAExecutionProvider" in active:
            print(f"[OnnxBackend] Using CUDAExecutionProvider")
        else:
            print(f"[OnnxBackend] Using CPUExecutionProvider")

        input_meta = {inp.name: inp for inp in session.get_inputs()}
        return session, input_meta

    # ------------------------------------------------------------------
    @property
    def device(self) -> str:
        """Return the device as a plain string (``"cpu"`` or ``"cuda"``)."""
        return self._device_str

    @property
    def config_name(self) -> str:
        return self._config_name

    @property
    def kind(self) -> str:
        return "onnx"

    # ------------------------------------------------------------------
    @staticmethod
    def _to_numpy(t: np.ndarray, target_dtype: str | None = None) -> np.ndarray:
        """Ensure array has the dtype expected by the ONNX session."""
        arr = np.asarray(t)
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

    def __call__(self, batch: dict[str, np.ndarray], return_promo: bool = True) -> tuple:
        # Build the ONNX feed dict from the batch, matching expected input names.
        feed: dict[str, np.ndarray] = {}
        for name in self._INPUT_NAMES:
            if name not in batch:
                continue
            meta = self._input_meta.get(name)
            target_dtype = meta.type if meta is not None else None
            feed[name] = self._to_numpy(batch[name], target_dtype)

        output_names = list(self._OUTPUT_NAMES)
        try:
            raw_outputs = self._session.run(output_names, feed)
        except Exception as e:
            # CUDA context loss (e.g. CUBLAS_STATUS_NOT_INITIALIZED) can happen after
            # a GPU driver event.  Recreate the ONNX session and retry once.
            err_str = str(e)
            if any(kw in err_str for kw in ("CUBLAS", "CUDA", "CudaCall", "cudnn", "ONNXRuntimeError")):
                print(f"[OnnxBackend] CUDA error detected, recreating session: {err_str[:120]}")
                try:
                    self._session, self._input_meta = self._create_session()
                    # Rebuild feed dict with fresh input metadata.
                    feed = {}
                    for name in self._INPUT_NAMES:
                        if name not in batch:
                            continue
                        meta = self._input_meta.get(name)
                        target_dtype = meta.type if meta is not None else None
                        feed[name] = self._to_numpy(batch[name], target_dtype)
                    raw_outputs = self._session.run(output_names, feed)
                    print("[OnnxBackend] Session recreated and retry succeeded.")
                except Exception as retry_exc:
                    raise RuntimeError(f"[OnnxBackend] Session recreation failed: {retry_exc}") from retry_exc
            else:
                raise

        # Return plain numpy arrays.
        return tuple(raw_outputs)
