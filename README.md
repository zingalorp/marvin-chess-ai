# Marvin Chess

[![License: AGPL v3](https://img.shields.io/badge/License-AGPL_v3-blue.svg)](https://www.gnu.org/licenses/agpl-3.0)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Models-yellow)](https://huggingface.co/holymolyyy/marvin)

Marvin is a **human-like** chess transformer trained on millions of [Lichess](https://lichess.org) games. Instead of playing the best move, it mimics human play across skill levels (1200-2400 Elo) and adapts to clock time and time control.

<p align="center">
  <img src="docs/accuracy_comparison.png" width="700" alt="Move Matching Accuracy">
  <br>
  <em>Human move-matching accuracy vs <a href="https://arxiv.org/abs/2409.20553">Maia2</a> and <a href="https://arxiv.org/abs/2410.03893">Allie</a>.</em>
</p>

## Architecture

Transformer with 6 conditioning tokens (Elo, clock, time control) prepended to 64 square tokens. Outputs: policy, value (WDL), and predicted thinking time.

Two sizes:
- **Large** (~49M params) - full quality
- **Tiny** (~5M params) - fast and lightweight

## Installation

```bash
git clone https://github.com/zingalorp/marvin-chess.git
cd marvin-chess
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
```

## Usage

| Mode | Command | Weights | Notes |
|---|---|---|---|
| Web interface | `python -m inference.app` | `.pt` | Full parameter control via UI |
| UCI engine | `python -m inference.uci_engine` | `.pt` | All conditioning as UCI options |
| UCI engine (no PyTorch) | `python -m inference.uci_onnx` | `.onnx` | Export first with `scripts/export_onnx.py` |
| lc0 / chess GUI | see below | `.pb.gz` | Elo fixed per file, no Python needed |

## Model Weights

All weights are on [Hugging Face](https://huggingface.co/holymolyyy/marvin).

- **`.pt`** - Native PyTorch checkpoints (`marvin_large.pt`, `marvin_tiny.pt`). Used by the app and UCI engine. All conditioning adjustable at runtime.
- **`.onnx`** - Exported from `.pt` for use without PyTorch. Generated via `python scripts/export_onnx.py`.
- **`.pb.gz`** - For use with [Leela Chess Zero](https://lczero.org). One file per Elo level (1600-2500 in steps of 100), with Elo baked in at export time.

## Using Marvin with lc0

The `.pb.gz` files drop into lc0 like any other weight file, as long as you use an **ONNX build** of lc0. The standard CUDA/DNNL builds use lc0's native backend and will not work.

**Windows:** Download `lc0-vX.X.X-windows-onnx-dml.zip` from the [lc0 releases](https://github.com/LeelaChessZero/lc0/releases). Drop the `.pb.gz` into the lc0 folder, add lc0 as an engine in your GUI, and point it at the weights file. Works on any GPU and falls back to CPU automatically.

**Linux:** No pre-built ONNX binary is available. Build lc0 from source with ONNX Runtime:
```bash
git clone --recurse-submodules https://github.com/LeelaChessZero/lc0.git && cd lc0
./build.sh -Donnx_include=<ORT_PATH>/include -Donnx_libdir=<ORT_PATH>/lib
```

**Recommended lc0 settings:**

| Option | Value | Notes |
|---|---|---|
| nodes | `1` | Policy-only play; can be increased for stronger play|
| `--backend` | `onnx-cuda` or `onnx-cpu` | Required |
| `--threads` | `1` | More threads reduces GPU throughput |
| `--PolicyTemperature` | `0.0-1.0` | Lower = less random |

## Play on Lichess

[marvin-1200](https://lichess.org/@/marvin-1200) / [marvin-1600](https://lichess.org/@/marvin-1600) / [marvin-2000](https://lichess.org/@/marvin-2000) / [marvin-2400](https://lichess.org/@/marvin-2400)

## Acknowledgments

- [Lichess](https://lichess.org) for training data and bot infrastructure
- [Leela Chess Zero](https://lczero.org) for architectural inspiration and the MCTS inference framework
- [lichess-bot](https://github.com/lichess-bot-devs/lichess-bot) for the bot framework
- [Chessground](https://github.com/lichess-org/chessground) for the web board UI

## License

AGPL-3.0