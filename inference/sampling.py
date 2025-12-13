from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import torch


PROMO_INDEX_TO_CHAR = {0: "q", 1: "r", 2: "b", 3: "n"}


@dataclass(frozen=True)
class SampleResult:
    move_index: int
    prob: float


def _top_p_filter(probs: np.ndarray, top_p: float) -> np.ndarray:
    if top_p >= 1.0:
        return probs
    if top_p <= 0.0:
        out = np.zeros_like(probs)
        out[int(np.argmax(probs))] = 1.0
        return out

    order = np.argsort(-probs)
    sorted_probs = probs[order]
    cdf = np.cumsum(sorted_probs)

    keep = cdf <= top_p
    # Always keep at least one
    if not np.any(keep):
        keep[0] = True
    else:
        # also keep the first prob that crosses top_p
        first_over = int(np.argmax(cdf > top_p))
        keep[first_over] = True

    mask = np.zeros_like(probs, dtype=bool)
    mask[order[keep]] = True
    out = np.where(mask, probs, 0.0)
    s = float(out.sum())
    return out / s if s > 0 else probs


def sample_from_logits(
    logits: torch.Tensor,
    *,
    temperature: float = 1.0,
    top_p: float = 1.0,
    rng: Optional[np.random.Generator] = None,
) -> SampleResult:
    """Samples a single index from logits using temperature + nucleus (top-p)."""
    if rng is None:
        rng = np.random.default_rng()

    x = logits.detach().float().cpu().numpy()
    if temperature <= 0.0:
        idx = int(np.argmax(x))
        return SampleResult(move_index=idx, prob=1.0)

    x = x / float(temperature)
    x = x - x.max()
    probs = np.exp(x)
    probs = probs / (probs.sum() + 1e-12)

    probs = _top_p_filter(probs, float(top_p))
    idx = int(rng.choice(len(probs), p=probs))
    return SampleResult(move_index=idx, prob=float(probs[idx]))


def select_promo(
    promo_logits_row: torch.Tensor,
    *,
    temperature: float,
    top_p: float,
    rng: Optional[np.random.Generator] = None,
) -> str:
    """Selects promotion piece char from a 4-logit vector (q,r,b,n)."""
    res = sample_from_logits(promo_logits_row, temperature=temperature, top_p=top_p, rng=rng)
    return PROMO_INDEX_TO_CHAR.get(res.move_index, "q")
