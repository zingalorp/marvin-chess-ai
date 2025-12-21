from __future__ import annotations

import chess


# Settings (Defaults) copied from `inference/app.py`.
DEFAULT_GAME_SETTINGS: dict = {
    "temperature": 0.9,
    "time_temperature": 0.6,
    "top_p": 0.95,
    "time_top_p": 0.75,
    "human_elo": 1900,
    "engine_elo": 1900,
    "human_color": chess.WHITE,
    "compile_model": True,
    "simulate_thinking_time": False,
    "internal_clock": False,
    "use_real_time": False,
    "use_mode_time": False,
    "use_expected_time": False,
    "start_clock_s": 180.0,
    "inc_s": 0.0,

    # MCTS (disabled by default)
    "use_mcts": False,
    "mcts_simulations": 256,
    "mcts_c_puct": 2.0,
    "mcts_max_children": 48,
    "mcts_root_dirichlet_alpha": 0.0,
    "mcts_root_exploration_frac": 0.0,
    "mcts_final_temperature": 0.0,
    "mcts_max_depth": 96,
    "mcts_adaptive": False,
    "mcts_adaptive_scale": 500.0,
    "show_mcts_stats": False,

    # Overlays

    # Attention viz
    "show_attention": False,

    # Attention viz settings
    # - attn_layer: -1 => aggregate across all layers, else 0..N-1
    # - attn_head_agg: 'avg' | 'max' | 'smolgen'
    # - attn_focus: 'outbound' | 'inbound' (frontend rendering)
    "attn_layer": -1,
    "attn_head_agg": "avg",
    "attn_focus": "outbound",
}


# Fixed Context (copied from `inference/app.py`).
START_CLOCK_S = 180.0
INC_S = 0


# App-level RNG seed (copied from `inference/app.py`).
DEFAULT_RNG_SEED = 67
