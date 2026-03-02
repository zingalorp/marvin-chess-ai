from __future__ import annotations

import chess


# Settings (Defaults) copied from `inference/app.py`.
DEFAULT_GAME_SETTINGS: dict = {
    "temperature": 0.9,
    "time_temperature": 0.9,
    "top_p": 0.95,
    "time_top_p": 0.95,
    "opening_temperature": 1.2,
    "opening_length": 10,
    "human_elo": 1800,
    "engine_elo": 1800,
    "human_color": chess.WHITE,
    "compile_model": True,
    "auto_play": False,
    "simulate_thinking_time": False,
    "internal_clock": False,
    "use_real_time": False,
    "use_mode_time": False,
    "use_expected_time": True,
    "start_clock_s": 300.0,
    "inc_s": 3.0,
    # Device selection for model inference: 'auto' (default), 'cuda', or 'cpu'
    "device": "cuda",

    # MCTS (disabled by default)
    "use_mcts": False,
    "mcts_simulations": 256,
    "mcts_c_puct": 2.0,
    "mcts_max_children": 16,
    "mcts_root_dirichlet_alpha": 0.0,
    "mcts_root_exploration_frac": 0.0,

    "mcts_final_temperature": 0.0,
    "mcts_final_top_p": 0.90,  # Top-p (nucleus) sampling for final move selection (1.0 = disabled)
    "mcts_max_depth": 24,
    "mcts_adaptive": True,
    "mcts_adaptive_scale": 30.0,
    "mcts_contempt": 0.15,  # Penalize draws to avoid drawish positions when ahead
    "mcts_simulate_time": False,  # Simulate remaining thinking time after MCTS completes
    "mcts_start_ply": 0,  # Ply at which MCTS kicks in (0 = from start, higher = skip opening for variety)
    "mcts_tree_reuse": True,  # Reuse search tree from previous position for faster follow-up moves
    "show_mcts_stats": True,
    
    # Pondering (thinking during opponent's time)
    "ponder": False,

    # Overlays

    # Arrow for top move
    "show_arrows": True,
}


# Fixed Context (copied from `inference/app.py`).
START_CLOCK_S = 300.0
INC_S = 0


# App-level RNG seed (copied from `inference/app.py`).
DEFAULT_RNG_SEED = 67
