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

import threading
import time as _time
from pathlib import Path

import numpy as np


class OnnxBackend:
    """Wraps an ONNX Runtime InferenceSession.

    Accepts batch dicts of numpy arrays and returns numpy arrays.

    Thread-safety: all calls to ``session.run()`` are serialised through a
    lock so that concurrent Flask / engine threads cannot trigger CUDA
    "illegal memory access" errors (ONNX Runtime CUDA sessions are NOT
    thread-safe for concurrent ``.run()`` calls).
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

    # After a CUDA failure forces a CPU fallback, wait this many seconds
    # before trying to re-create a CUDA session.
    _CUDA_RETRY_COOLDOWN_S = 30.0

    def __init__(self, session_path: str | Path, device: str = "cpu", config_name: str = "") -> None:
        import onnxruntime as ort

        device_str = str(device).split(":")[0] if device else "cpu"

        self._session_path = str(session_path)
        self._device_str = device_str          # requested device
        self._active_device_str = device_str   # actual device in use (may fall back to cpu)
        self._config_name = config_name
        self._lock = threading.Lock()
        self._last_cuda_failure: float | None = None   # monotonic timestamp

        self._session, self._input_meta = self._create_session()

    def _create_session(self, *, force_cpu: bool = False):
        import onnxruntime as ort

        providers: list[str] = []
        if self._device_str == "cuda" and not force_cpu:
            providers.append("CUDAExecutionProvider")
        providers.append("CPUExecutionProvider")

        session = ort.InferenceSession(self._session_path, providers=providers)

        # Cache the actual provider in use.
        active = session.get_providers()
        if "CUDAExecutionProvider" in active:
            self._active_device_str = "cuda"
            print("[OnnxBackend] Using CUDAExecutionProvider")
        else:
            self._active_device_str = "cpu"
            print("[OnnxBackend] Using CPUExecutionProvider")

        input_meta = {inp.name: inp for inp in session.get_inputs()}
        return session, input_meta

    # ------------------------------------------------------------------
    @property
    def device(self) -> str:
        """Return the device actually in use (``"cpu"`` or ``"cuda"``)."""
        return self._active_device_str

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

    def _build_feed(self, batch: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        """Build the ONNX feed dict, casting dtypes to match the session."""
        feed: dict[str, np.ndarray] = {}
        for name in self._INPUT_NAMES:
            if name not in batch:
                continue
            meta = self._input_meta.get(name)
            target_dtype = meta.type if meta is not None else None
            feed[name] = self._to_numpy(batch[name], target_dtype)
        return feed

    def _maybe_promote_to_cuda(self) -> None:
        """If we fell back to CPU after a CUDA error, try re-creating a CUDA
        session once the cooldown has elapsed.  Called inside the lock."""
        if self._device_str != "cuda" or self._active_device_str == "cuda":
            return  # nothing to do
        if self._last_cuda_failure is None:
            return
        elapsed = _time.monotonic() - self._last_cuda_failure
        if elapsed < self._CUDA_RETRY_COOLDOWN_S:
            return
        print("[OnnxBackend] Cooldown elapsed — attempting to restore CUDA session …")
        try:
            self._session, self._input_meta = self._create_session()
            if self._active_device_str == "cuda":
                print("[OnnxBackend] CUDA session restored successfully.")
                self._last_cuda_failure = None
        except Exception as exc:
            print(f"[OnnxBackend] CUDA restore failed, staying on CPU: {exc!s:.120}")
            self._last_cuda_failure = _time.monotonic()

    def __call__(self, batch: dict[str, np.ndarray], return_promo: bool = True) -> tuple:
        output_names = list(self._OUTPUT_NAMES)

        with self._lock:
            # Opportunistically try to get back onto CUDA after a fallback.
            self._maybe_promote_to_cuda()

            feed = self._build_feed(batch)

            try:
                raw_outputs = self._session.run(output_names, feed)
            except Exception as e:
                err_str = str(e)
                if any(kw in err_str for kw in ("CUBLAS", "CUDA", "CudaCall", "cudnn", "ONNXRuntimeError")):
                    print(f"[OnnxBackend] CUDA error detected, recreating session: {err_str[:120]}")
                    self._last_cuda_failure = _time.monotonic()
                    try:
                        self._session, self._input_meta = self._create_session()
                        feed = self._build_feed(batch)
                        raw_outputs = self._session.run(output_names, feed)
                        print("[OnnxBackend] Session recreated and retry succeeded.")
                    except Exception as retry_exc:
                        # Last resort: force CPU so subsequent calls don't keep crashing.
                        print(f"[OnnxBackend] Retry failed, falling back to CPU: {retry_exc!s:.120}")
                        self._session, self._input_meta = self._create_session(force_cpu=True)
                        feed = self._build_feed(batch)
                        raw_outputs = self._session.run(output_names, feed)
                        print("[OnnxBackend] CPU fallback succeeded.")
                else:
                    raise

        # Return plain numpy arrays.
        return tuple(raw_outputs)
