import sys
import json
import logging
import queue
import threading
import argparse
from pathlib import Path
import numpy as np
import chess
import chess.pgn
import chess.svg
import io
from flask import Flask, render_template, render_template_string, request, jsonify, Response
import time

try:
    import torch
except ImportError:
    torch = None  # type: ignore[assignment]

from inference.app_settings import DEFAULT_GAME_SETTINGS, DEFAULT_RNG_SEED, INC_S, START_CLOCK_S

# =============================================================================
# CLI ARGUMENT PARSING
# =============================================================================
def parse_args():
    parser = argparse.ArgumentParser(description="Marvin Chess Inference Server")
    parser.add_argument("--weights", type=str, default=None,
                        help="Path to model weights (.pt or .onnx file). "
                             "Default: inference/marvin_small.onnx")
    parser.add_argument("--port", type=int, default=5000,
                        help="Port to run the server on (default: 5000)")
    return parser.parse_args()

# Parse args early so we can resolve checkpoint path before imports
_cli_args = parse_args()

from inference.engine_logic import analyze_position as analyze_position_core
from inference.engine_logic import choose_engine_move as choose_engine_move_core
from inference.runtime import (
    ensure_repo_on_syspath,
    load_default_backend,
    resolve_repo_root,
)

# ==========================================
# 1. SETUP & MODEL LOADING
# ==========================================

# Settings (Defaults)
game_settings = dict(DEFAULT_GAME_SETTINGS)

# Paths
repo_root = resolve_repo_root()
ensure_repo_on_syspath(repo_root)

from inference.encoding import ContextOptions, build_history_from_position, canonicalize, make_model_batch
from inference.chessformer_policy import choose_move
from inference.mcts import MCTSSettings, mcts_choose_move
from inference.sampling import sample_from_logits
from inference.config import print_config

# Resolve checkpoint path from CLI arg or default
_checkpoint_path_arg = Path(_cli_args.weights) if _cli_args.weights else None

print("Loading model...")
if _checkpoint_path_arg:
    print(f"Weights: {_checkpoint_path_arg}")
else:
    print_config(repo_root)

# Load backend (auto-detects .pt vs .onnx and picks best device)
backend, _checkpoint_path = load_default_backend(
    repo_root=repo_root,
    checkpoint_path=_checkpoint_path_arg,
)
device = backend.device
print(f"Model loaded: {_checkpoint_path.name} (config: {backend.config_name}, backend: {backend.kind})")
print(f"[inference] Device: {device}")

# ==========================================
# 2. ANALYSIS LOGIC
# ==========================================
PROMO_INDEX = {chess.QUEEN: 0, chess.ROOK: 1, chess.BISHOP: 2, chess.KNIGHT: 3}
rng = np.random.default_rng(DEFAULT_RNG_SEED)

def _mirror_square(sq: int) -> int: return sq ^ 56


def _softmax_1d(x) -> np.ndarray:
    a = np.asarray(x, dtype=np.float64).ravel()
    a = a - np.max(a)
    e = np.exp(a)
    return (e / np.sum(e)).astype(np.float32)
def _time_bin_to_seconds(bin_idx: int, active_clock_s: float) -> float:
    scaled_mid = (bin_idx + 0.5) / 256.0
    # Inverse of sqrt scaling: target = sqrt(time_ratio), so time_ratio = target^2
    return float((scaled_mid ** 2) * max(1e-6, active_clock_s))


def _apply_ply_clock(clock_s: float, *, spent_s: float, inc_s: float) -> float:
    # Simple Fischer clock: subtract, clamp at 0, then add increment.
    return max(0.0, float(clock_s) - max(0.0, float(spent_s))) + float(inc_s)


def _clocks_at_cursor(cursor: int) -> dict[chess.Color, float]:
    """Return remaining clocks (seconds) for white/black at the given cursor."""
    start_s = float(game_settings.get("start_clock_s", START_CLOCK_S))
    inc_s = float(game_settings.get("inc_s", INC_S))
    w = start_s
    b = start_s
    hist = game_state.get("history", [])[:cursor]
    for ply_idx, item in enumerate(hist):
        uci = item.get("uci")
        spent = float(item.get("pred_time_s", 0.0))
        mover = chess.WHITE if (ply_idx % 2 == 0) else chess.BLACK
        if uci in ("resign", "flag"):
            # Treat as a ply that consumes time but doesn't change board.
            pass
        if mover == chess.WHITE:
            w = _apply_ply_clock(w, spent_s=spent, inc_s=inc_s)
        else:
            b = _apply_ply_clock(b, spent_s=spent, inc_s=inc_s)
    return {chess.WHITE: w, chess.BLACK: b}

def _canonical_to_real_move(move: chess.Move, real_turn: chess.Color) -> chess.Move:
    if real_turn == chess.WHITE: return move
    return chess.Move(_mirror_square(move.from_square), _mirror_square(move.to_square), promotion=move.promotion)

def _pred_time_history_s_at_cursor(cursor: int) -> list[float]:
    """Return last 8 predicted move times (seconds), newest-first.

    We store one predicted time per played ply in `game_state['history'][i]['pred_time_s']`.
    """
    hist = game_state.get("history", [])[:cursor]
    times = [float(it.get("pred_time_s", 0.0)) for it in hist]
    # Newest-first, pad with zeros to HISTORY_LEN=8.
    out = list(reversed(times[-8:]))
    while len(out) < 8:
        out.append(0.0)
    return out[:8]


def analyze_position(
    board: chess.Board,
    moves_uci: list[str],
    *,
    active_clock_s: float,
    opponent_clock_s: float,
    time_history_s: list[float] | None = None,
) -> dict:
    return analyze_position_core(
        backend=backend,
        settings=game_settings,
        rng=rng,
        board=board,
        moves_uci=moves_uci,
        active_clock_s=active_clock_s,
        opponent_clock_s=opponent_clock_s,
        active_inc_s=float(game_settings.get("inc_s", INC_S)),
        opponent_inc_s=float(game_settings.get("inc_s", INC_S)),
        time_history_s=time_history_s,
    )

def format_stats_html(data: dict) -> str:
    if not data: return "<div style='color:#666;padding:8px'>Evaluating...</div>"
    def mk_bar(p, c='#629924'): return f"<div style='background:{c}; width:{max(0,min(100,p*100))}%; height:100%;'></div>"
    
    def _value_color(v):
        """Color by model value: green (good, v≥0.6) → yellow (neutral, ~0.5) → red (bad, v≤0.4)."""
        if v is None: return '#629924'  # fallback green
        # Clamp to [0,1] and map: 0.0=red, 0.5=yellow, 1.0=green
        v = max(0.0, min(1.0, v))
        if v >= 0.5:
            t = (v - 0.5) * 2  # 0..1 within yellow→green
            r = int(0xb5 + (0x62 - 0xb5) * t)
            g = int(0xa0 + (0x99 - 0xa0) * t)
            b = int(0x30 + (0x24 - 0x30) * t)
        else:
            t = v * 2  # 0..1 within red→yellow
            r = int(0xcc + (0xb5 - 0xcc) * t)
            g = int(0x44 + (0xa0 - 0x44) * t)
            b = int(0x44 + (0x30 - 0x44) * t)
        return f'#{r:02x}{g:02x}{b:02x}'
    
    rows = ""
    for m in data['top_moves']:
        mv_val = m.get('value')  # per-move value (win% for side to move)
        color = _value_color(mv_val)
        eval_txt = f"<span class='val' style='color:{color}; width:32px;'>{mv_val:.0%}</span>" if mv_val is not None else "<span class='val' style='width:32px;'></span>"
        rows += f"<div class='row'><span class='lbl'>{m['label']}</span><div class='bar-bg'>{mk_bar(m['prob'], color)}</div><span class='val'>{m['prob']:.1%}</span>{eval_txt}</div>"
    
    extras = f"<div class='sub'>Resign: {data['resign']:.1%} | Flag: {data['flag']:.1%}</div>"
    
    policy_html = f"<div class='panel'><h3>Policy (Effective)</h3><div class='policy-scroll'>{rows}</div>{extras}</div>"

    w, d, l = data['wdl']['w'], data['wdl']['d'], data['wdl']['l']
    wdl_html = f"<div class='panel'><h3>WDL</h3><div style='display:flex;height:6px;overflow:hidden;margin-bottom:4px;'><div style='width:{w*100:.1f}%;background:#c8c8c8'></div><div style='width:{d*100:.1f}%;background:#555'></div><div style='width:{l*100:.1f}%;background:#1e1e1e;border:1px solid #333'></div></div><div class='sub'>W {w:.1%} &nbsp; D {d:.1%} &nbsp; L {l:.1%}</div></div>"
    
    val_html = f"<div class='panel'><h3>Value (Win%)</h3><div class='big-val'>{data['value']:.1%} <span class='err'>±{data['value_error']:.1%}</span></div></div>"
    
    sampled = data.get('time_sample_s', None)
    sampled_prob = data.get('time_sample_prob', None)
    expected_t = data.get('expected_time_s', None)
    mode_t = data.get('mode_time_s', None)
    
    time_dist = data.get('time_dist', [])
    if time_dist:
        max_p = max(d['prob'] for d in time_dist) if time_dist else 1.0
        bars = []
        
        # Find closest bin to mode_t to highlight it
        mode_idx = -1
        if mode_t is not None:
            min_diff = float('inf')
            for i, item in enumerate(time_dist):
                diff = abs(item['sec'] - mode_t)
                if diff < min_diff:
                    min_diff = diff
                    mode_idx = i

        for i, item in enumerate(time_dist):
            h = (item['prob'] / max_p) * 100
            color = '#555'
            if i == mode_idx:
                color = '#888' # Lighter for Mode
            bars.append(f"<div title='{item['sec']:.1f}s ({item['prob']:.1%})' style='flex:1; background:{color}; height:{h}%; min-width:2px;'></div>")
        
        start_t = time_dist[0]['sec']
        end_t = time_dist[-1]['sec']
        
        marker_html = ""
        # Show Expected marker (Orange, dashed)
        if expected_t is not None:
            closest_i = -1
            min_diff = float('inf')
            for i, item in enumerate(time_dist):
                diff = abs(item['sec'] - expected_t)
                if diff < min_diff:
                    min_diff = diff
                    closest_i = i
            
            if closest_i >= 0:
                N = len(time_dist)
                pct = ((closest_i + 0.5) / N) * 100
                marker_html += f"<div style='position:absolute; left:{pct}%; bottom:0; height:100%; width:1px; border-left:1px dashed #cc8833; opacity:0.9; pointer-events:none; z-index:9;' title='Expected: {expected_t:.1f}s'></div>"

        graph_html = f"""
        <div style="position:relative; height:60px; margin-top:8px; background:#111; padding:4px; border:1px solid #252525;">
            <div style="display:flex; align-items:flex-end; height:100%; gap:1px;">
                {"".join(bars)}
            </div>
            {marker_html}
        </div>
        <div style="display:flex; justify-content:space-between; font-size:9px; color:#666; margin-top:2px; padding:0 2px;">
            <span>{start_t:.1f}s</span>
            <span>{end_t:.1f}s</span>
        </div>
        """
        
        sampled_txt = ""
        if sampled is not None:
            sampled_txt = f"<div class='sub'>Sampled: <b style='color:#eee'>{sampled:.1f}s</b> <span style='color:#777'>({sampled_prob:.1%})</span>"
            if mode_t is not None:
                sampled_txt += f" | Mode: <b style='color:#aaa'>{mode_t:.1f}s</b>"
            if expected_t is not None:
                sampled_txt += f" | Exp: <span style='color:#cc8833'>{expected_t:.1f}s</span>"
            sampled_txt += "</div>"
            
        time_top_p = data.get('time_top_p', 0.95)
        time_html = f"<div class='panel'><h3>Time Distribution ({time_top_p:.0%})</h3>{sampled_txt}{graph_html}</div>"
    else:
        time_html = ""

    # Engine Inputs Panel
    # Removed from here, now sent separately as 'engine_inputs'
    return f"""
    <div class="stats-grid">
        <div style="grid-column: span 2">{policy_html}</div>
        {wdl_html}{val_html}
        <div style="grid-column: span 2">{time_html}</div>
    </div>
    """

def format_mcts_stats_html(stats: dict) -> str:
    root_val = stats.get("root_value", 0.0)
    children = stats.get("children", [])
    
    # Root Value (Static)
    val_html = f"<div class='panel'><h3>MCTS Static Value</h3><div class='big-val'>{root_val:+.2f}</div></div>"
    
    # Children List
    rows = ""
    max_visits = children[0]['visits'] if children else 1
    
    # Header
    rows += """
    <div class='row' style='font-size:9px; color:#777; border-bottom:1px solid #222; padding-bottom:2px; margin-bottom:4px;'>
        <span style='width:40px;'>Move</span>
        <span style='flex:1;'>Visits</span>
        <span style='width:30px; text-align:right;'>N</span>
        <span style='width:40px; text-align:right;'>Q</span>
        <span style='width:30px; text-align:right;'>P</span>
    </div>
    """

    def _q_color(q):
        """Color by Q-value: green (+1) → yellow (0) → red (−1)."""
        v = (q + 1.0) / 2.0  # map [-1,+1] → [0,1]
        v = max(0.0, min(1.0, v))
        if v >= 0.5:
            t = (v - 0.5) * 2
            r = int(0xb5 + (0x62 - 0xb5) * t)
            g = int(0xa0 + (0x99 - 0xa0) * t)
            b = int(0x30 + (0x24 - 0x30) * t)
        else:
            t = v * 2
            r = int(0xcc + (0xb5 - 0xcc) * t)
            g = int(0x44 + (0xa0 - 0x44) * t)
            b = int(0x44 + (0x30 - 0x44) * t)
        return f'#{r:02x}{g:02x}{b:02x}'

    for child in children:
        move = child.get('san', child['move'])
        visits = child['visits']
        q = child['q']
        prior = child['prior']
        
        # Bar colored by Q-value
        width = (visits / max_visits) * 100
        bar_color = _q_color(q)
        bar = f"<div style='background:{bar_color}; width:{width}%; height:100%;'></div>"
        q_color = _q_color(q)
        
        rows += f"""
        <div class='row' style='font-size:10px; gap:4px;'>
            <span class='lbl' style='width:40px;'>{move}</span>
            <div class='bar-bg' style='flex:1;'>{bar}</div>
            <span class='val' style='width:30px; text-align:right;'>{visits}</span>
            <span class='val' style='width:40px; text-align:right; color:{q_color};'>{q:+.2f}</span>
            <span class='val' style='width:30px; text-align:right; color:#777;'>{prior:.0%}</span>
        </div>
        """
        
    list_html = f"<div class='panel'><h3>MCTS Search ({len(children)})</h3><div class='policy-scroll' style='max-height:250px;'>{rows}</div></div>"
    
    return f"""
    <div class="stats-grid">
        <div style="grid-column: span 2">{list_html}</div>
        {val_html}
    </div>
    """

# ==========================================
# 3. WEB SERVER & STATE
# ==========================================

app = Flask(__name__)

game_state = {
    "history": [], 
    "cursor": 0,
    # Cache per-position sampled time for the *current* position.
    # Key: cursor index, Value: {fen, time_temperature, time_top_p, time_sample_s, time_sample_prob}
    "pos_cache": {},
    # MCTS tree reuse state
    "last_mcts_result": None,  # MCTSResult from last search
    "last_mcts_ply": -1,  # ply count when last MCTS search was done
}

# MCTS progress streaming
mcts_progress_queue = queue.Queue()
mcts_progress_lock = threading.Lock()
mcts_final_stats = None  # Store final stats for display after MCTS completes

# Background MCTS analysis state
_analysis_thread = None
_analysis_cancel = threading.Event()
_analysis_done = threading.Event()
_analysis_done.set()  # Start in 'done' state

# Background engine move state
_engine_thread = None
_engine_cancel = threading.Event()
_engine_done = threading.Event()
_engine_done.set()  # Start in 'done' state


def _mcts_progress_callback(stats: dict) -> None:
    """Called by MCTS during search to report progress."""
    global mcts_final_stats
    try:
        # Non-blocking put - if queue is full, skip this update
        mcts_progress_queue.put_nowait(stats)
        # Also store as final stats (will be overwritten each iteration)
        with mcts_progress_lock:
            mcts_final_stats = stats
    except queue.Full:
        pass


def _clear_pos_cache() -> None:
    game_state["pos_cache"] = {}


def _clear_mcts_tree() -> None:
    """Clear the MCTS tree reuse state."""
    game_state["last_mcts_result"] = None
    game_state["last_mcts_ply"] = -1


def get_board_at_cursor():
    board = chess.Board()
    for i in range(game_state["cursor"]):
        move_uci = game_state["history"][i]['uci']
        board.push(chess.Move.from_uci(move_uci))
    return board

def get_uci_list_at_cursor():
    return [item['uci'] for item in game_state["history"][:game_state["cursor"]]]

def prepare_response(board):
    fen = board.fen()
    moves_uci = get_uci_list_at_cursor()
    cursor = game_state["cursor"]
    time_hist = _pred_time_history_s_at_cursor(cursor)
    clocks = _clocks_at_cursor(cursor)
    active_clock = float(clocks[board.turn])
    current_stats = analyze_position(
        board,
        moves_uci,
        active_clock_s=active_clock,
        opponent_clock_s=float(clocks[not board.turn]),
        time_history_s=time_hist,
    )

    # Persist the sampled time for this position so navigating away/back doesn't resample.
    # - If we're in history, show the sampled time that was *actually used* for the next move.
    # - If we're at the latest position, cache the sampled time for the current position.
    if cursor < len(game_state["history"]):
        hist_item = game_state["history"][cursor]
        if "pred_time_s" in hist_item:
            current_stats["time_sample_s"] = float(hist_item["pred_time_s"])
        if "pred_time_prob" in hist_item:
            current_stats["time_sample_prob"] = float(hist_item["pred_time_prob"])
    else:
        cache = game_state.get("pos_cache", {})
        entry = cache.get(cursor)
        meta = {
            "fen": fen,
            "time_temperature": float(game_settings.get("time_temperature", 1.0)),
            "time_top_p": float(game_settings.get("time_top_p", 0.95)),
        }
        if (
            entry
            and entry.get("fen") == meta["fen"]
            and float(entry.get("time_temperature", -1.0)) == meta["time_temperature"]
            and float(entry.get("time_top_p", -1.0)) == meta["time_top_p"]
        ):
            current_stats["time_sample_s"] = float(entry.get("time_sample_s", current_stats.get("time_sample_s", 0.0)))
            current_stats["time_sample_prob"] = float(entry.get("time_sample_prob", current_stats.get("time_sample_prob", 0.0)))
        else:
            cache[cursor] = {
                **meta,
                "time_sample_s": float(current_stats.get("time_sample_s", 0.0)),
                "time_sample_prob": float(current_stats.get("time_sample_prob", 0.0)),
            }
            game_state["pos_cache"] = cache

    # If we are reviewing history, use the originally sampled time for this move
    if cursor < len(game_state["history"]):
        hist_item = game_state["history"][cursor]
        if 'pred_time_s' in hist_item:
            current_stats['time_sample_s'] = float(hist_item['pred_time_s'])
        if 'pred_time_prob' in hist_item:
            current_stats['time_sample_prob'] = float(hist_item['pred_time_prob'])

    # Attach display clocks in human/engine terms.
    human_color = game_settings["human_color"]
    current_stats['clocks'] = {
        'human': float(clocks[human_color]),
        'engine': float(clocks[not human_color]),
    }
    current_html = format_stats_html(current_stats)
    
    # Extract top move UCI for arrow drawing
    top_move_uci = None
    if current_stats.get('top_moves') and len(current_stats['top_moves']) > 0:
        top_move_uci = current_stats['top_moves'][0].get('uci')

    prev_html = "<div style='color:#666;padding:8px'>Start of game.</div>"
    if game_state["cursor"] > 0:
        prev_item = game_state["history"][game_state["cursor"] - 1]
        prev_html = prev_item.get('prev_stats_html', '')
        if not prev_html:
            prev_html = "<div style='padding:8px'>Human moved.</div>"

    # GENERATE PGN for the full line
    pgn_board = chess.Board()
    full_moves = []
    for item in game_state["history"]:
        uci = item['uci']
        if uci in ('resign', 'flag'):
            continue
        full_moves.append(chess.Move.from_uci(uci))
    
    pgn_str = pgn_board.variation_san(full_moves)

    # Get legal moves as UCI strings for Chessground
    legal_moves = [move.uci() for move in board.legal_moves]
    
    # Get last move for highlighting
    last_move = None
    if game_state["cursor"] > 0:
        last_item = game_state["history"][game_state["cursor"] - 1]
        last_uci = last_item.get('uci', '')
        if last_uci and last_uci not in ('resign', 'flag'):
            last_move = [last_uci[:2], last_uci[2:4]]
    
    # Check if in check
    in_check = board.is_check()

    # Build clock state for frontend ticking clock display
    clocks_white = float(clocks[chess.WHITE])
    clocks_black = float(clocks[chess.BLACK])
    active_color = "white" if board.turn == chess.WHITE else "black"
    
    # WDL from White's perspective (model returns side-to-move perspective)
    stm_wdl = current_stats.get("wdl", {"w": 0.33, "d": 0.34, "l": 0.33})
    if board.turn == chess.WHITE:
        wdl_white = stm_wdl
    else:
        wdl_white = {"w": stm_wdl["l"], "d": stm_wdl["d"], "l": stm_wdl["w"]}

    return jsonify({
        "fen": fen,
        "orientation": "white" if game_settings["human_color"] == chess.WHITE else "black",
        "current_stats": current_html,
        "prev_engine_stats": prev_html,
        "pgn": pgn_str,
        "cursor": game_state["cursor"],
        "max_cursor": len(game_state["history"]),
        "is_game_over": board.is_game_over(),
        "result": board.result() if board.is_game_over() else "",
        "settings": game_settings,
        "legal_moves": legal_moves,
        "last_move": last_move,
        "in_check": in_check,
        "top_move_uci": top_move_uci,
        "clocks": {"white": clocks_white, "black": clocks_black, "active": active_color},
        "value_white": float(current_stats.get("value", 0.5)) if board.turn == chess.WHITE else 1.0 - float(current_stats.get("value", 0.5)),
        "wdl": wdl_white,
    })

@app.route('/')
def index():
    # Cancel any running analysis or engine move
    _analysis_cancel.set()
    if _analysis_thread is not None:
        _analysis_thread.join(timeout=2.0)
    _analysis_done.set()
    _engine_cancel.set()
    if _engine_thread is not None:
        _engine_thread.join(timeout=2.0)
    _engine_done.set()

    # Reset game state on every page load so refresh = fresh start
    game_state["history"] = []
    game_state["cursor"] = 0
    _clear_pos_cache()
    _clear_mcts_tree()
    return render_template('index.html')

@app.route('/state', methods=['GET'])
def get_state():
    board = get_board_at_cursor()
    return prepare_response(board)


@app.route('/load_pgn', methods=['POST'])
def load_pgn():
    """Load a PGN into the game history so users can analyze any game."""
    data = request.json
    pgn_text = data.get('pgn', '')
    if not pgn_text.strip():
        return jsonify({"error": "Empty PGN"}), 400

    try:
        pgn_io = io.StringIO(pgn_text)
        game = chess.pgn.read_game(pgn_io)
        if game is None:
            return jsonify({"error": "Could not parse PGN"}), 400

        # Replay moves into history
        board = chess.Board()
        new_history = []
        for move in game.mainline_moves():
            if move not in board.legal_moves:
                break
            new_history.append({
                'uci': move.uci(),
                'pred_time_s': 0.0,
                'pred_time_prob': 0.0,
                'prev_stats_html': '<div style="padding:10px; color:#aaa">Imported move</div>',
            })
            board.push(move)

        game_state["history"] = new_history
        game_state["cursor"] = 0  # Start at beginning so user can navigate
        _clear_pos_cache()
        _clear_mcts_tree()

        board_at_cursor = get_board_at_cursor()
        return prepare_response(board_at_cursor)

    except Exception as e:
        return jsonify({"error": str(e)}), 400


def _run_analysis_thread(board, moves_uci, cursor_now, time_hist, clocks, analyze_settings, mcts_reuse_root, mcts_reuse_moves):
    """Run MCTS analysis in a background thread."""
    global mcts_final_stats
    try:
        def stop_check():
            return _analysis_cancel.is_set()

        out, engine_stats, mcts_stats, mcts_result = choose_engine_move_core(
            backend=backend,
            settings=analyze_settings,
            rng=rng,
            board=board,
            moves_uci=moves_uci,
            active_clock_s=float(clocks[board.turn]),
            opponent_clock_s=float(clocks[not board.turn]),
            active_inc_s=float(game_settings.get("inc_s", INC_S)),
            opponent_inc_s=float(game_settings.get("inc_s", INC_S)),
            time_history_s=time_hist,
            stop_check=stop_check,
            allow_ponder_sleep=False,
            mcts_progress_callback=_mcts_progress_callback,
            mcts_reuse_root=mcts_reuse_root,
            mcts_reuse_moves=mcts_reuse_moves,
        )

        # Store MCTS result for tree reuse
        if mcts_result is not None:
            game_state["last_mcts_result"] = mcts_result
            game_state["last_mcts_ply"] = cursor_now
    except Exception as e:
        print(f"MCTS analysis error: {e}")
    finally:
        _analysis_done.set()


@app.route('/analyze_mcts', methods=['POST'])
def analyze_mcts_endpoint():
    """Start MCTS analysis on the current position in a background thread."""
    global mcts_final_stats, _analysis_thread

    # If already running, don't start another
    if not _analysis_done.is_set():
        return jsonify({"status": "already_running"}), 200

    # Clear any previous MCTS stats
    with mcts_progress_lock:
        mcts_final_stats = None
    while not mcts_progress_queue.empty():
        try:
            mcts_progress_queue.get_nowait()
        except queue.Empty:
            break

    board = get_board_at_cursor()
    if board.is_game_over():
        return jsonify({"status": "game_over"}), 200

    moves_uci = get_uci_list_at_cursor()
    cursor_now = game_state["cursor"]
    time_hist = _pred_time_history_s_at_cursor(cursor_now)
    clocks = _clocks_at_cursor(cursor_now)

    # Prepare tree reuse data
    mcts_reuse_root = None
    mcts_reuse_moves = []
    if (
        game_settings.get("mcts_tree_reuse", False)
        and game_state.get("last_mcts_result") is not None
        and game_state.get("last_mcts_ply", -1) >= 0
    ):
        last_result = game_state["last_mcts_result"]
        if last_result.chosen_move is not None:
            moves_since = cursor_now - game_state["last_mcts_ply"]
            if 0 < moves_since <= 2:
                mcts_reuse_root = last_result.root
                for i in range(game_state["last_mcts_ply"], cursor_now):
                    if i < len(moves_uci):
                        mcts_reuse_moves.append(chess.Move.from_uci(moves_uci[i]))

    # Force MCTS on for this analysis
    analyze_settings = dict(game_settings)
    analyze_settings['use_mcts'] = True
    analyze_settings['show_mcts_stats'] = True

    # Start background thread
    _analysis_cancel.clear()
    _analysis_done.clear()
    _analysis_thread = threading.Thread(
        target=_run_analysis_thread,
        args=(board, moves_uci, cursor_now, time_hist, clocks, analyze_settings, mcts_reuse_root, mcts_reuse_moves),
        daemon=True
    )
    _analysis_thread.start()
    return jsonify({"status": "started"}), 200


@app.route('/analysis_status', methods=['GET'])
def analysis_status():
    """Check if background MCTS analysis is complete."""
    done = _analysis_done.is_set()
    if done:
        return get_state()
    return jsonify({"status": "running"}), 200


@app.route('/cancel_analysis', methods=['POST'])
def cancel_analysis():
    """Request cancellation of the running background analysis."""
    _analysis_cancel.set()
    # Wait briefly for thread to finish
    if _analysis_thread is not None:
        _analysis_thread.join(timeout=2.0)
    _analysis_done.set()
    return jsonify({"status": "cancelled"}), 200

@app.route('/settings', methods=['POST'])
def update_settings():
    data = request.json
    
    if 'auto_play' in data:
        game_settings['auto_play'] = bool(data['auto_play'])
    
    game_settings['temperature'] = float(data.get('temperature', game_settings['temperature']))
    game_settings['time_temperature'] = float(data.get('time_temperature', game_settings.get('time_temperature', 1.0)))
    game_settings['top_p'] = float(data.get('top_p', game_settings['top_p']))
    game_settings['time_top_p'] = float(data.get('time_top_p', game_settings.get('time_top_p', 0.95)))
    game_settings['human_elo'] = int(data.get('human_elo', game_settings['human_elo']))
    game_settings['engine_elo'] = int(data.get('engine_elo', game_settings['engine_elo']))
    
    if 'human_color' in data:
        game_settings['human_color'] = chess.WHITE if data['human_color'] == 'white' else chess.BLACK
    
    if 'simulate_thinking_time' in data:
        game_settings['simulate_thinking_time'] = bool(data['simulate_thinking_time'])
        
    if 'use_real_time' in data:
        game_settings['use_real_time'] = bool(data['use_real_time'])

    if 'use_mode_time' in data:
        game_settings['use_mode_time'] = bool(data['use_mode_time'])

    if 'use_expected_time' in data:
        game_settings['use_expected_time'] = bool(data['use_expected_time'])

    if 'start_clock_s' in data:
        game_settings['start_clock_s'] = float(data['start_clock_s'])
    
    if 'inc_s' in data:
        game_settings['inc_s'] = float(data['inc_s'])

    # MCTS
    if 'use_mcts' in data:
        game_settings['use_mcts'] = bool(data['use_mcts'])
    if 'mcts_simulations' in data:
        game_settings['mcts_simulations'] = int(float(data['mcts_simulations']))
    if 'mcts_c_puct' in data:
        game_settings['mcts_c_puct'] = float(data['mcts_c_puct'])
    if 'mcts_max_children' in data:
        game_settings['mcts_max_children'] = int(float(data['mcts_max_children']))
    if 'mcts_final_temperature' in data:
        game_settings['mcts_final_temperature'] = float(data['mcts_final_temperature'])
    if 'mcts_max_depth' in data:
        game_settings['mcts_max_depth'] = int(float(data['mcts_max_depth']))
    if 'mcts_adaptive' in data:
        game_settings['mcts_adaptive'] = bool(data['mcts_adaptive'])
    if 'mcts_adaptive_scale' in data:
        game_settings['mcts_adaptive_scale'] = float(data['mcts_adaptive_scale'])
    if 'mcts_tree_reuse' in data:
        game_settings['mcts_tree_reuse'] = bool(data['mcts_tree_reuse'])
        if not game_settings['mcts_tree_reuse']:
            _clear_mcts_tree()  # Clear tree when disabling
    if 'show_mcts_stats' in data:
        game_settings['show_mcts_stats'] = bool(data['show_mcts_stats'])
    if 'show_arrows' in data:
        game_settings['show_arrows'] = bool(data['show_arrows'])
    if 'opening_temperature' in data:
        game_settings['opening_temperature'] = float(data['opening_temperature'])
    if 'opening_length' in data:
        game_settings['opening_length'] = int(float(data['opening_length']))

    # Sampling-related settings changes should reset the per-position sampled-time cache.
    if any(k in data for k in ('time_temperature', 'time_top_p', 'use_real_time', 'use_mode_time', 'use_expected_time')):
        _clear_pos_cache()
    
    print(f"DEBUG: Settings Updated: SimulateThinkingTime={game_settings.get('simulate_thinking_time')}, RealTime={game_settings.get('use_real_time')}")

    board = get_board_at_cursor()
    
    # If it's the start of the game and human is Black, engine (White) must move.
    # Frontend will trigger engine move if needed.
        
    return prepare_response(board)

@app.route('/navigate', methods=['POST'])
def navigate():
    data = request.json
    action = data.get('action')
    if action == 'start': game_state["cursor"] = 0
    elif action == 'prev': game_state["cursor"] = max(0, game_state["cursor"] - 1)
    elif action == 'next': game_state["cursor"] = min(len(game_state["history"]), game_state["cursor"] + 1)
    elif action == 'end': game_state["cursor"] = len(game_state["history"])
    return get_state()

@app.route('/navigate_to', methods=['POST'])
def navigate_to():
    """Jump directly to a specific ply number."""
    data = request.json
    target = int(data.get('ply', 0))
    target = max(0, min(len(game_state["history"]), target))
    game_state["cursor"] = target
    return get_state()

def _play_engine_move(board, stop_check=None):
    if board.is_game_over():
        return

    moves_uci = get_uci_list_at_cursor()
    cursor_engine = game_state["cursor"]
    time_hist_engine = _pred_time_history_s_at_cursor(cursor_engine)
    clocks_engine = _clocks_at_cursor(cursor_engine)
    
    # Use progress callback if MCTS is enabled and show_mcts_stats is on
    progress_cb = None
    if game_settings.get("use_mcts", False) and game_settings.get("show_mcts_stats", False):
        progress_cb = _mcts_progress_callback
    
    # Prepare tree reuse data
    mcts_reuse_root = None
    mcts_reuse_moves = []
    
    if (
        game_settings.get("mcts_tree_reuse", False)
        and game_state.get("last_mcts_result") is not None
        and game_state.get("last_mcts_ply", -1) >= 0
    ):
        last_result = game_state["last_mcts_result"]
        if last_result.chosen_move is not None:
            # Calculate moves played since last search
            moves_since = cursor_engine - game_state["last_mcts_ply"]
            if 0 < moves_since <= 2:
                mcts_reuse_root = last_result.root
                for i in range(game_state["last_mcts_ply"], cursor_engine):
                    if i < len(moves_uci):
                        mcts_reuse_moves.append(chess.Move.from_uci(moves_uci[i]))
    
    out, engine_stats, mcts_stats, mcts_result = choose_engine_move_core(
        backend=backend,
        settings=game_settings,
        rng=rng,
        board=board,
        moves_uci=moves_uci,
        active_clock_s=float(clocks_engine[board.turn]),
        opponent_clock_s=float(clocks_engine[not board.turn]),
        active_inc_s=float(game_settings.get("inc_s", INC_S)),
        opponent_inc_s=float(game_settings.get("inc_s", INC_S)),
        time_history_s=time_hist_engine,
        stop_check=stop_check,
        allow_ponder_sleep=True,
        mcts_progress_callback=progress_cb,
        mcts_reuse_root=mcts_reuse_root,
        mcts_reuse_moves=mcts_reuse_moves,
    )
    
    # Store MCTS result for tree reuse
    if mcts_result is not None:
        game_state["last_mcts_result"] = mcts_result
        game_state["last_mcts_ply"] = cursor_engine

    stats_html = format_stats_html(engine_stats)
    engine_pred_time_s = float(engine_stats.get("time_sample_s", 0.0))
    engine_pred_time_prob = float(engine_stats.get("time_sample_prob", 0.0))
    
    # Cap first-move think time to avoid berserk-related artifacts in training data.
    FIRST_MOVE_TIME_CAP_S = 2.0
    if cursor_engine == 0 and engine_pred_time_s > FIRST_MOVE_TIME_CAP_S:
        engine_pred_time_s = FIRST_MOVE_TIME_CAP_S

    if mcts_stats and game_settings.get("show_mcts_stats", False):
        stats_html = format_mcts_stats_html(mcts_stats)
    
    if out.is_resign:
        chosen_txt = "Resigns"
        chosen_html = f"<div class='chosen-move' style='color:#c44'>Engine Resigns ({out.policy_prob:.1%})</div>"
        full_prev_html = f"<div style='padding:4px'>{stats_html}</div>{chosen_html}"
        game_state["history"].append({
            'uci': 'resign',
            'pred_time_s': engine_pred_time_s,
            'pred_time_prob': engine_pred_time_prob,
            'prev_stats_html': full_prev_html
        })
    elif out.is_flag:
        chosen_txt = "Flags"
        chosen_html = f"<div class='chosen-move' style='color:#c44'>Engine Flags ({out.policy_prob:.1%})</div>"
        full_prev_html = f"<div style='padding:4px'>{stats_html}</div>{chosen_html}"
        game_state["history"].append({
            'uci': 'flag',
            'pred_time_s': engine_pred_time_s,
            'pred_time_prob': engine_pred_time_prob,
            'prev_stats_html': full_prev_html
        })
    else:
        chosen_txt = board.san(out.move)
        chosen_html = f"<div class='chosen-move'>Chosen: {chosen_txt} ({out.policy_prob:.1%})</div>"
        full_prev_html = f"<div style='padding:4px'>{stats_html}</div>{chosen_html}"
        
        board.push(out.move)
        
        # Ensure history is truncated to cursor before appending (in case of race conditions or logic bugs)
        game_state["history"] = game_state["history"][:game_state["cursor"]]
        
        game_state["history"].append({
            'uci': out.move.uci(),
            'pred_time_s': engine_pred_time_s,
            'pred_time_prob': engine_pred_time_prob,
            'prev_stats_html': full_prev_html
        })
        game_state["cursor"] += 1

@app.route('/move', methods=['POST'])
def make_move():
    data = request.json
    source = data.get('from')
    target = data.get('to')
    promotion = data.get('promotion', None)
    elapsed_s = data.get('elapsed_s', None)
    
    board = get_board_at_cursor()
    
    try:
        move_uci = f"{source}{target}"
        if promotion: move_uci += promotion
        move = chess.Move.from_uci(move_uci)
        
        if move not in board.legal_moves and not promotion:
             if chess.square_rank(move.to_square) in (0, 7) and board.piece_at(move.from_square).piece_type == chess.PAWN:
                 move = chess.Move.from_uci(move_uci + 'q')

        if move in board.legal_moves:
            game_state["history"] = game_state["history"][:game_state["cursor"]]
            _clear_pos_cache()

            # Predict human move time from current position (top-1 bin).
            moves_uci_before = get_uci_list_at_cursor()
            cursor_before = game_state["cursor"]
            time_hist_before = _pred_time_history_s_at_cursor(cursor_before)
            clocks_before = _clocks_at_cursor(cursor_before)
            human_stats = analyze_position(
                board,
                moves_uci_before,
                active_clock_s=float(clocks_before[board.turn]),
                opponent_clock_s=float(clocks_before[not board.turn]),
                time_history_s=time_hist_before,
            )
            
            if game_settings.get("use_real_time", False) and elapsed_s is not None:
                human_pred_time_s = float(elapsed_s)
                human_pred_time_prob = 1.0
            else:
                human_pred_time_s = float(human_stats.get("time_sample_s", 0.0))
                human_pred_time_prob = float(human_stats.get("time_sample_prob", 0.0))
            
            board.push(move)
            game_state["history"].append({
                'uci': move.uci(),
                'pred_time_s': human_pred_time_s,
                'pred_time_prob': human_pred_time_prob,
                'prev_stats_html': f"<div style='padding:10px; color:#aaa'>Human played <b>{board.pop().uci()}</b> ({human_pred_time_s:.1f}s)</div>"
            })
            board.push(move)
            game_state["cursor"] += 1
        else:
            return jsonify({"error": "Illegal move"}), 400
    except Exception as e:
        return jsonify({"error": str(e)}), 400

    if not board.is_game_over():
        # Frontend will trigger engine move
        pass

    return get_state()

def _run_engine_move_thread(board):
    """Run engine move in a background thread."""
    try:
        def stop_check():
            return _engine_cancel.is_set()
        _play_engine_move(board, stop_check=stop_check)
    except Exception as e:
        print(f"Engine move error: {e}")
    finally:
        _engine_done.set()


@app.route('/engine_move', methods=['POST'])
def engine_move_endpoint():
    global mcts_final_stats, _engine_thread

    # If already running, don't start another
    if not _engine_done.is_set():
        return jsonify({"status": "already_running"}), 200

    # Clear any previous MCTS stats
    with mcts_progress_lock:
        mcts_final_stats = None
    while not mcts_progress_queue.empty():
        try:
            mcts_progress_queue.get_nowait()
        except queue.Empty:
            break
    
    board = get_board_at_cursor()
    if board.is_game_over() or board.turn == game_settings['human_color']:
        return get_state()

    # Start engine move in background thread
    _engine_cancel.clear()
    _engine_done.clear()
    _engine_thread = threading.Thread(
        target=_run_engine_move_thread,
        args=(board,),
        daemon=True
    )
    _engine_thread.start()
    return jsonify({"status": "started"}), 200


@app.route('/engine_move_status', methods=['GET'])
def engine_move_status():
    """Check if background engine move is complete."""
    done = _engine_done.is_set()
    if done:
        return get_state()
    return jsonify({"status": "running"}), 200


@app.route('/cancel_engine_move', methods=['POST'])
def cancel_engine_move():
    """Request cancellation of the running engine move."""
    _engine_cancel.set()
    if _engine_thread is not None:
        _engine_thread.join(timeout=2.0)
    _engine_done.set()
    return jsonify({"status": "cancelled"}), 200


@app.route('/mcts_progress')
def mcts_progress_stream():
    """SSE endpoint for streaming MCTS progress updates."""
    def generate():
        while True:
            try:
                # Wait for new progress with timeout
                stats = mcts_progress_queue.get(timeout=0.1)
                yield f"data: {json.dumps(stats)}\n\n"
            except queue.Empty:
                # Send keepalive
                yield f": keepalive\n\n"
    
    return Response(generate(), mimetype='text/event-stream', headers={
        'Cache-Control': 'no-cache',
        'Connection': 'keep-alive',
        'X-Accel-Buffering': 'no'
    })


@app.route('/mcts_stats')
def get_mcts_stats():
    """Get the final MCTS stats after search completes."""
    with mcts_progress_lock:
        if mcts_final_stats:
            return jsonify(mcts_final_stats)
        return jsonify(None)


@app.route('/reset', methods=['POST'])
def reset():
    # Cancel any running analysis or engine move
    _analysis_cancel.set()
    if _analysis_thread is not None:
        _analysis_thread.join(timeout=2.0)
    _analysis_done.set()
    _engine_cancel.set()
    if _engine_thread is not None:
        _engine_thread.join(timeout=2.0)
    _engine_done.set()

    game_state["history"] = []
    game_state["cursor"] = 0
    _clear_pos_cache()
    _clear_mcts_tree()
    
    # Frontend will trigger engine move if needed
        
    return get_state()

if __name__ == '__main__':
    port = _cli_args.port
    print(f"Starting Flask Server on http://localhost:{port}")
    app.run(host='0.0.0.0', port=port, debug=False, threaded=True)