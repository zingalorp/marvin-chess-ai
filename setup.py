"""
setup.py — Downloads the default model weights (marvin_small.onnx) from Hugging Face.

Run once after cloning:
    python setup.py

The script requires only the standard library + huggingface_hub (installed via
requirements.txt).  If huggingface_hub is not yet available it falls back to a
plain urllib download so you can run this before `pip install -r requirements.txt`.
"""

from __future__ import annotations

import os
import sys
import urllib.request
from pathlib import Path

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
REPO_ID = "holymolyyy/marvin"
FILENAME = "marvin_small.onnx"
DEST = Path(__file__).parent / "inference" / FILENAME

HF_URL = (
    f"https://huggingface.co/{REPO_ID}/resolve/main/{FILENAME}"
)

EXPECTED_SIZE_MB = 92  # rough expected size – used for a progress sanity check


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _fmt_mb(n_bytes: int) -> str:
    return f"{n_bytes / 1_048_576:.1f} MB"


def _download_urllib(url: str, dest: Path) -> None:
    """Fallback downloader using stdlib urllib (no progress bar)."""
    print(f"Downloading {FILENAME} from Hugging Face …")
    dest.parent.mkdir(parents=True, exist_ok=True)

    def _reporthook(count: int, block_size: int, total_size: int) -> None:
        if total_size <= 0:
            return
        done = min(count * block_size, total_size)
        pct = done * 100 // total_size
        bar = "#" * (pct // 2) + "-" * (50 - pct // 2)
        print(f"\r  [{bar}] {pct}%  {_fmt_mb(done)} / {_fmt_mb(total_size)}", end="", flush=True)

    urllib.request.urlretrieve(url, dest, reporthook=_reporthook)
    print()  # newline after progress bar


def _download_hf(dest: Path) -> None:
    """Download via huggingface_hub (handles auth, caching, resume)."""
    from huggingface_hub import hf_hub_download  # type: ignore

    print(f"Downloading {FILENAME} from Hugging Face (via huggingface_hub) …")
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = hf_hub_download(
        repo_id=REPO_ID,
        filename=FILENAME,
        local_dir=str(dest.parent),
    )
    # hf_hub_download already writes to local_dir/filename, so no copy needed
    _ = tmp  # path returned; file is already at dest


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    if DEST.exists():
        size_mb = DEST.stat().st_size / 1_048_576
        print(f"{DEST} already exists ({size_mb:.1f} MB). Nothing to do.")
        return

    # Prefer huggingface_hub when available
    try:
        import huggingface_hub  # noqa: F401
        _download_hf(DEST)
    except ImportError:
        print(
            "huggingface_hub not installed - falling back to direct URL download.\n"
            "Tip: run `pip install huggingface_hub` for resumable downloads."
        )
        _download_urllib(HF_URL, DEST)

    if DEST.exists():
        size_mb = DEST.stat().st_size / 1_048_576
        print(f"Saved to {DEST} ({size_mb:.1f} MB)")
    else:
        print("Download failed - file not found at expected path.", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
