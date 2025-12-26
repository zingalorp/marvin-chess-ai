# Marvin Chess

A Human-Like Chess Transformer designed to mimic human behavior across the 1200-2600 ELO range.

The goal is to reach SOTA human move-matching accuracy. 
Basically [Allie](https://arxiv.org/pdf/2410.03893) on a diet, using techniques described by the [Leela team](https://lczero.org/blog/2024/02/transformer-progress/).

This model predicts:
- **Move played** - policy head with resign/flag support
- **Time spent** - 256-bin classification head  
- **Game value** - WDL (Win/Draw/Loss) prediction

## Project Structure

```
marvin-chess/
├── model.py          # ChessformerV2 model architecture
├── train.py          # Training script
├── dataset.py        # DataLoader for parquet training data
├── process_pgn.py    # PGN preprocessing to parquet format
├── inference/        # Inference runtime and UCI engine
│   ├── uci_engine.py # UCI protocol implementation
│   ├── app.py        # Flask web app for play/analysis
│   └── ...
├── lichess-bot/      # lichess-bot with custom modifications
└── marvin_wrapper.sh # Wrapper script for lichess-bot
```

## Quick Start

### Play in a notebook
See `notebooks/play_vs_chessformer_v2.ipynb`

### UCI Engine
```bash
python -m inference.uci_engine
```

### Web Interface
```bash
python -m inference.app
```

### lichess-bot Integration
The `lichess-bot/` directory contains a modified version of [lichess-bot](https://github.com/lichess-bot-devs/lichess-bot) with:
- Custom time control filtering (`extra_game_handlers.py`)
- Resign/flag action support (`lib/engine_wrapper.py`)

Setup:
1. Copy `lichess-bot/config.example.yml` to `lichess-bot/config.yml`
2. Add your Lichess OAuth2 token
3. Update paths to point to your installation
4. Run: `cd lichess-bot && python lichess-bot.py`

## Training

```bash
# Preprocess PGN files
python process_pgn.py --input data/raw/*.pgn --output data_v2/

# Train model
python train.py --config smolgen --data-dir data_v2/
```
