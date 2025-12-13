This projects develops a Human-Like Chess Transformer designed to mimic human behavior across the 1200-2600 ELO range.

The goal is to reach SOTA human move-matching accuracy. 
Basically [Allie](https://arxiv.org/pdf/2410.03893) on a diet, using techniques described by the [Leela team](https://lczero.org/blog/2024/02/transformer-progress/).

This model predicts the move played, time spent, and game termination (including resignation).
This repo includes everything needed to recreate the trained model.

## Play in a notebook

- See `notebooks/play_vs_chessformer_v2.ipynb`

## UCI engine (no search)

There is a minimal UCI engine wrapper for the v2 model in `uci/engine.py` (no MCTS/search yet).

- Run it with: `python -m uci.engine --checkpoint checkpoints/chessformer_v2_smolgen_best.pt --config smolgen`
## UCI engine (no search)

There is a minimal UCI engine wrapper for the v2 model in `uci/engine.py` (no MCTS/search yet).
