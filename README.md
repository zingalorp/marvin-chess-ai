This projects develops a Human-Like Chess Transformer designed to mimic human behavior across the 1200-2600 ELO range.

The goal is to reach SOTA human move-matching accuracy. 
Basically [Allie](https://arxiv.org/pdf/2410.03893) on a diet, using techniques described by the [Leela team](https://lczero.org/blog/2024/02/transformer-progress/).

This model predicts the move played, time spent, and game termination (including resignation).
Trained on stratified dataset of over 2 billion positions from online lichess.org games, the model can distinguish between deep calculation in endgames and panic-induced reactions in blitz time scrambles.

This repo includes everything needed to recreate the trained model.
