import sys
import json
import logging
from pathlib import Path
import numpy as np
import torch
import chess
import chess.svg
from flask import Flask, render_template_string, request, jsonify
import time

# ==========================================
# 1. SETUP & MODEL LOADING
# ==========================================

# Settings (Defaults)
game_settings = {
    "temperature": 0.9,
    "time_temperature": 0.6,
    "top_p": 0.95,
    "time_top_p": 0.75,
    "human_elo": 1900,
    "engine_elo": 1900,
    "human_color": chess.WHITE,
    "ponder": False,
    "use_real_time": False,
    "use_mode_time": False,
    "use_expected_time": False,

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

# Fixed Context
START_CLOCK_S = 180.0
INC_S = 0

# Paths
cwd = Path.cwd().resolve()
if (cwd / 'model_v2-1.py').exists():
    repo_root = cwd
elif (cwd.parent / 'model_v2-1.py').exists():
    repo_root = cwd.parent
else:
    repo_root = cwd 

if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from inference.model_loader import load_chessformer_v2
from inference.encoding import ContextOptions, build_history_from_position, canonicalize, make_model_batch
from inference.chessformer_policy import choose_move
from inference.mcts import MCTSSettings, mcts_choose_move
from inference.sampling import sample_from_logits

print("Loading model...")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model_path = repo_root / 'checkpoints/chessformer_v2_smolgen_best.pt'
if not model_path.exists():
    print(f"Warning: Model not found at {model_path}.")

loaded = load_chessformer_v2(
    model_py_path=repo_root / 'model_v2-1.py',
    config_name='smolgen',
    checkpoint_path=model_path,
    device=device,
)
model = loaded.model
model.eval()
print("Model loaded.")

def _count_attn_layers(model: torch.nn.Module) -> int:
    n = 0
    for mod in model.modules():
        if hasattr(mod, "last_attn_probs"):
            n += 1
    return n

ATTN_NUM_LAYERS = _count_attn_layers(model)
ATTN_HAS_SMOLGEN = bool(getattr(model, "smolgen", None) is not None)

# Expose read-only attention metadata to the frontend via the settings blob.
game_settings["attn_num_layers"] = int(ATTN_NUM_LAYERS)
game_settings["attn_has_smolgen"] = bool(ATTN_HAS_SMOLGEN)

# ==========================================
# 2. ANALYSIS LOGIC
# ==========================================
PROMO_INDEX = {chess.QUEEN: 0, chess.ROOK: 1, chess.BISHOP: 2, chess.KNIGHT: 3}
rng = np.random.default_rng(67)

def _mirror_square(sq: int) -> int: return sq ^ 56


def _extract_attention_64(
    model: torch.nn.Module,
    *,
    real_turn: chess.Color,
    layer: int,
    head_agg: str,
) -> list[float] | None:
    """Extract a 64x64 attention matrix according to visualization settings.

    Returns flattened list length 4096 in *real* board square indexing (python-chess: a1=0..h8=63).
    Requires the model's attention modules to have `last_attn_probs` populated from a just-completed forward.

    head_agg:
      - 'avg': average over heads (and over layers if layer == -1)
      - 'max': elementwise max over heads (and over layers if layer == -1)
      - 'smolgen': use smolgen bias only (softmax over bias), averaged over heads
    """

    head_agg = str(head_agg or "avg").lower().strip()
    if head_agg not in ("avg", "max", "smolgen"):
        head_agg = "avg"

    # Smolgen-only view (pure dynamic bias).
    if head_agg == "smolgen":
        bias_h: torch.Tensor | None = None
        for mod in model.modules():
            b = getattr(mod, "last_smolgen_bias", None)
            if b is None:
                continue
            if not torch.is_tensor(b) or b.ndim != 4 or b.shape[-2:] != (64, 64):
                continue
            bias_h = b[0].float()  # (H, 64, 64)
            break

        if bias_h is None:
            return None

        # Convert smolgen logits to attention probabilities.
        probs_h = torch.softmax(bias_h, dim=-1)  # (H, 64, 64)
        mat = probs_h.mean(dim=0)  # (64, 64)
    else:
        layers_h: list[torch.Tensor] = []
        for mod in model.modules():
            attn = getattr(mod, "last_attn_probs", None)
            if attn is None:
                continue
            if not torch.is_tensor(attn) or attn.ndim != 4 or attn.shape[-2:] != (64, 64):
                continue
            layers_h.append(attn[0].float())  # (H, 64, 64)

        if not layers_h:
            return None

        if int(layer) >= 0:
            idx = int(layer)
            if idx >= len(layers_h):
                idx = len(layers_h) - 1
            layers_h = [layers_h[idx]]

        if head_agg == "avg":
            mats = [a.mean(dim=0) for a in layers_h]  # (64,64) per layer
            mat = torch.stack(mats, dim=0).mean(dim=0)
        else:  # 'max'
            mats = [a.max(dim=0).values for a in layers_h]  # (64,64) per layer
            mat = mats[0]
            for m in mats[1:]:
                mat = torch.maximum(mat, m)

    # Model uses canonical boards (mirrors when black-to-move). Convert back to real-square indexing.
    if real_turn == chess.BLACK:
        perm = torch.tensor([_mirror_square(i) for i in range(64)], device=mat.device, dtype=torch.long)
        mat = mat.index_select(0, perm).index_select(1, perm)

    return mat.detach().cpu().reshape(-1).tolist()
def _softmax_1d(x: torch.Tensor) -> torch.Tensor:
    x = x.float() - torch.max(x)
    return torch.softmax(x, dim=0)
def _time_bin_to_seconds(bin_idx: int, active_clock_s: float) -> float:
    scaled_mid = (bin_idx + 0.5) / 256.0
    # Inverse of sqrt scaling: target = sqrt(time_ratio), so time_ratio = target^2
    return float((scaled_mid ** 2) * max(1e-6, active_clock_s))


def _format_clock_s(sec: float) -> str:
    sec = max(0.0, float(sec))
    m = int(sec) // 60
    s = int(sec) % 60
    return f"{m}:{s:02d}"


def _apply_ply_clock(clock_s: float, *, spent_s: float, inc_s: float) -> float:
    # Simple Fischer clock: subtract, clamp at 0, then add increment.
    return max(0.0, float(clock_s) - max(0.0, float(spent_s))) + float(inc_s)


def _clocks_at_cursor(cursor: int) -> dict[chess.Color, float]:
    """Return remaining clocks (seconds) for white/black at the given cursor."""
    w = float(START_CLOCK_S)
    b = float(START_CLOCK_S)
    hist = game_state.get("history", [])[:cursor]
    for ply_idx, item in enumerate(hist):
        uci = item.get("uci")
        spent = float(item.get("pred_time_s", 0.0))
        mover = chess.WHITE if (ply_idx % 2 == 0) else chess.BLACK
        if uci in ("resign", "flag"):
            # Treat as a ply that consumes time but doesn't change board.
            pass
        if mover == chess.WHITE:
            w = _apply_ply_clock(w, spent_s=spent, inc_s=INC_S)
        else:
            b = _apply_ply_clock(b, spent_s=spent, inc_s=INC_S)
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
    _final_board, board_history, repetition_flags = build_history_from_position(chess.Board(), moves_uci)
    
    if board.turn == game_settings["human_color"]:
        active_elo = game_settings["human_elo"]
        opp_elo = game_settings["engine_elo"]
    else:
        active_elo = game_settings["engine_elo"]
        opp_elo = game_settings["human_elo"]

    ctx = ContextOptions(
        active_elo=active_elo, opponent_elo=opp_elo,
        active_clock_s=active_clock_s, opponent_clock_s=opponent_clock_s,
        active_inc_s=INC_S, opponent_inc_s=INC_S,
        halfmove_clock=int(board.halfmove_clock),
    )

    batch = make_model_batch(
        board=board,
        board_history=board_history,
        repetition_flags=repetition_flags,
        time_history_s=time_history_s,
        ctx=ctx,
        device=loaded.device,
    )

    # For analysis only: unmask resign/flag so we can inspect their probabilities.
    # We still keep them disabled for actual move selection elsewhere.
    batch_for_rf = dict(batch)
    if "legal_mask" in batch_for_rf:
        lm = batch_for_rf["legal_mask"].clone()
        if lm.shape[-1] > 4097:
            lm[..., 4096] = True
            lm[..., 4097] = True
        batch_for_rf["legal_mask"] = lm

    with torch.inference_mode():
        with torch.autocast(device_type=loaded.device.type, enabled=(loaded.device.type == 'cuda')):
            (m_logits, v_raw, v_cls, v_err, t_logits, ss_logits, p_logits) = model(batch, return_promo=True)
            (m_logits_rf, *_rest) = model(batch_for_rf, return_promo=False)

    attention64 = None
    if bool(game_settings.get("show_attention", False)):
        attention64 = _extract_attention_64(
            model,
            real_turn=board.turn,
            layer=int(game_settings.get("attn_layer", -1)),
            head_agg=str(game_settings.get("attn_head_agg", "avg")),
        )

    m_logits = m_logits[0]
    promo_p = torch.softmax(p_logits[0].float(), dim=-1)
    
    # Effective Policy
    canonical_board = canonicalize(board)
    legal_moves_data = []

    for mv in canonical_board.legal_moves:
        base_idx = mv.from_square * 64 + mv.to_square
        logit = float(m_logits[base_idx].item())
        if mv.promotion is not None and 56 <= mv.to_square <= 63:
            file_idx = mv.to_square - 56
            p_idx = PROMO_INDEX.get(mv.promotion, 0)
            promo_prob = float(promo_p[file_idx, p_idx].item())
            logit += np.log(max(1e-8, promo_prob))
        legal_moves_data.append({'move': mv, 'logit': logit})

    if not legal_moves_data:
        policy_display = []
    else:
        T = max(1e-4, game_settings['temperature'])
        logits_vec = torch.tensor([x['logit'] for x in legal_moves_data], device=device)
        logits_vec = logits_vec / T
        probs_vec = torch.softmax(logits_vec, dim=0)
        
        sorted_probs, sorted_indices = torch.sort(probs_vec, descending=True)
        cumulative_probs = torch.cumsum(sorted_probs, dim=0)
        
        target_p = game_settings['top_p']
        cutoff_index = torch.searchsorted(cumulative_probs, target_p).item()
        cutoff_index = min(len(sorted_probs) - 1, cutoff_index + 1)
        
        kept_indices = sorted_indices[:cutoff_index]
        kept_probs = sorted_probs[:cutoff_index]
        kept_probs = kept_probs / kept_probs.sum()
        
        policy_display = []
        real_turn = board.turn
        for i, prob in zip(kept_indices.tolist(), kept_probs.tolist()):
            mv = legal_moves_data[i]['move']
            real_mv = _canonical_to_real_move(mv, real_turn)
            try: label = board.san(real_mv)
            except: label = real_mv.uci()
            policy_display.append({'label': label, 'prob': prob, 'uci': real_mv.uci()})

    wdl = _softmax_1d(v_cls[0])
    
    # Apply Time Temperature
    T_time = max(1e-4, float(game_settings.get('time_temperature', 1.0)))
    t_logits_scaled = t_logits[0].float() / T_time
    time_p = torch.softmax(t_logits_scaled - torch.max(t_logits_scaled), dim=0)

    # Time Distribution (Top X%)
    target_time_p = float(game_settings.get('time_top_p', 0.95))
    t_probs = time_p.cpu().numpy()

    # Calculate raw stats (Temp=1.0) for "True" Model Opinion
    t_probs_raw = torch.softmax(t_logits[0].float(), dim=0).cpu().numpy()
    
    # Mode (Argmax)
    mode_bin = np.argmax(t_probs_raw)
    mode_time_s = _time_bin_to_seconds(mode_bin, active_clock_s)

    t_sorted_idx = np.argsort(t_probs)[::-1]
    t_cumsum = np.cumsum(t_probs[t_sorted_idx])
    t_cutoff = np.searchsorted(t_cumsum, target_time_p) + 1
    t_active_idx = t_sorted_idx[:t_cutoff]
    t_active_idx.sort()
    
    time_dist = []
    expected_time_s = 0.0
    total_prob_mass = 0.0
    
    for idx in t_active_idx:
        sec = _time_bin_to_seconds(idx, active_clock_s)
        prob = float(t_probs[idx])
        time_dist.append({'sec': sec, 'prob': prob})
        
        # Accumulate for expected time calculation (weighted by prob)
        expected_time_s += prob * sec
        total_prob_mass += prob
        
    # Normalize expected time by the total probability mass of the top-p subset
    if total_prob_mass > 0:
        expected_time_s /= total_prob_mass

    # Sample a time bin (like we sample moves), but with fixed temperature=1.
    # We re-use the app's top_p setting to avoid sampling extreme tails.
    if game_settings.get("use_mode_time", False):
        # Use Mode (Argmax) for deterministic behavior
        time_sample_s = mode_time_s
        time_sample_prob = float(t_probs_raw[mode_bin])
    elif game_settings.get("use_expected_time", False):
        # Use Expected Value (Mean of Top-P)
        time_sample_s = expected_time_s
        time_sample_prob = 1.0 # It's an aggregate, not a single bin prob
    else:
        time_sample = sample_from_logits(
            t_logits[0],
            temperature=T_time,
            top_p=target_time_p,
            rng=rng,
        )
        time_sample_bin = int(time_sample.move_index)
        time_sample_s = _time_bin_to_seconds(time_sample_bin, active_clock_s)
        time_sample_prob = float(time_sample.prob)

    # Note: `m_logits` is masked by legal_mask, which keeps resign/flag at -inf.
    # Use the analysis forward-pass logits where resign/flag are unmasked.
    raw_policy_all = _softmax_1d(m_logits_rf[0])

    # Value processing
    # v_raw is logit, sigmoid gives 0-1 probability (win%)
    win_prob = float(torch.sigmoid(v_raw[0].squeeze(-1)).item())
    
    # v_err is predicted squared error. Take sqrt to get std dev / error bar.
    # Clamp to 0 just in case.
    pred_sq_error = v_err[0].squeeze(-1).item()
    error_bar = np.sqrt(max(0.0, pred_sq_error))

    return {
        'top_moves': policy_display,
        'resign': float(raw_policy_all[4096].item()),
        'flag': float(raw_policy_all[4097].item()),
        'wdl': {'w': wdl[2].item(), 'd': wdl[1].item(), 'l': wdl[0].item()},
        'value': win_prob,
        'value_error': error_bar,
        'time_dist': time_dist,
        'time_top_p': target_time_p,
        'time_sample_s': float(time_sample_s),
        'time_sample_prob': time_sample_prob,
        'expected_time_s': expected_time_s,
        'mode_time_s': mode_time_s,
        'attention64': attention64,
    }

def format_stats_html(data: dict) -> str:
    if not data: return "<div style='color:#666;padding:10px'><i>Evaluating...</i></div>"
    def mk_bar(p, c='#4caf50'): return f"<div style='background:{c}; width:{max(0,min(100,p*100))}%; height:100%;'></div>"
    
    rows = "".join([f"<div class='row'><span class='lbl'>{m['label']}</span><div class='bar-bg'>{mk_bar(m['prob'], '#2196F3')}</div><span class='val'>{m['prob']:.1%}</span></div>" for m in data['top_moves']])
    
    extras = f"<div class='sub'>Resign: {data['resign']:.1%} | Flag: {data['flag']:.1%}</div>"
    
    policy_html = f"<div class='panel'><h3>Policy (Effective)</h3><div class='policy-scroll'>{rows}</div>{extras}</div>"

    w, d, l = data['wdl']['w'], data['wdl']['d'], data['wdl']['l']
    wdl_html = f"<div class='panel'><h3>WDL</h3><div style='display:flex;height:8px;border-radius:2px;overflow:hidden;'><div style='width:{w*100}%;background:#4caf50'></div><div style='width:{d*100}%;background:#9e9e9e'></div><div style='width:{l*100}%;background:#f44336'></div></div><div class='sub'>W {w:.1%} D {d:.1%} L {l:.1%}</div></div>"
    
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
            color = '#2196F3'
            if i == mode_idx:
                color = '#ffeb3b' # Yellow for Mode
            bars.append(f"<div title='{item['sec']:.1f}s ({item['prob']:.1%})' style='flex:1; background:{color}; height:{h}%; border-radius:1px 1px 0 0; min-width:2px;'></div>")
        
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
                marker_html += f"<div style='position:absolute; left:{pct}%; bottom:0; height:100%; width:1px; border-left:1px dashed #ff9800; opacity:0.9; pointer-events:none; z-index:9;' title='Expected: {expected_t:.1f}s'></div>"

        graph_html = f"""
        <div style="position:relative; height:60px; margin-top:8px; background:#111; padding:4px; border-radius:4px;">
            <div style="display:flex; align-items:flex-end; height:100%; gap:1px;">
                {"".join(bars)}
            </div>
            {marker_html}
        </div>
        <div style="display:flex; justify-content:space-between; font-size:10px; color:#666; margin-top:2px; padding:0 2px;">
            <span>{start_t:.1f}s</span>
            <span>{end_t:.1f}s</span>
        </div>
        """
        
        sampled_txt = ""
        if sampled is not None:
            sampled_txt = f"<div class='sub'>Sampled: <b style='color:#eee'>{sampled:.1f}s</b> <span style='color:#666'>({sampled_prob:.1%})</span>"
            if mode_t is not None:
                sampled_txt += f" | Mode: <b style='color:#ffeb3b'>{mode_t:.1f}s</b>"
            if expected_t is not None:
                sampled_txt += f" | Exp: <span style='color:#ff9800'>{expected_t:.1f}s</span>"
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
        {wdl_html}{val_html}<div style="grid-column: span 2">{time_html}</div>
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
    <div class='row' style='font-size:10px; color:#888; border-bottom:1px solid #333; padding-bottom:2px; margin-bottom:4px;'>
        <span style='width:40px;'>Move</span>
        <span style='flex:1;'>Visits</span>
        <span style='width:30px; text-align:right;'>N</span>
        <span style='width:40px; text-align:right;'>Q</span>
        <span style='width:30px; text-align:right;'>P</span>
    </div>
    """

    for child in children:
        move = child['move']
        visits = child['visits']
        q = child['q']
        prior = child['prior']
        
        # Bar for visits
        width = (visits / max_visits) * 100
        bar = f"<div style='background:#4caf50; width:{width}%; height:100%;'></div>"
        
        rows += f"""
        <div class='row' style='font-size:11px; gap:4px;'>
            <span class='lbl' style='width:40px; font-family:monospace;'>{move}</span>
            <div class='bar-bg' style='flex:1;'>{bar}</div>
            <span class='val' style='width:30px; text-align:right;'>{visits}</span>
            <span class='val' style='width:40px; text-align:right; color:#aaa;'>{q:+.2f}</span>
            <span class='val' style='width:30px; text-align:right; color:#666;'>{prior:.0%}</span>
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
}


def _clear_pos_cache() -> None:
    game_state["pos_cache"] = {}

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
    
    # Construct Engine Inputs HTML
    if board.turn == game_settings["human_color"]:
        act_color = "White" if board.turn == chess.WHITE else "Black"
        act_elo = game_settings["human_elo"]
        opp_elo = game_settings["engine_elo"]
        act_clk = clocks[board.turn]
        opp_clk = clocks[not board.turn]
    else:
        act_color = "White" if board.turn == chess.WHITE else "Black"
        act_elo = game_settings["engine_elo"]
        opp_elo = game_settings["human_elo"]
        act_clk = clocks[board.turn]
        opp_clk = clocks[not board.turn]

    castling_str = board.fen().split(' ')[2]
    ep_str = board.fen().split(' ')[3]
    
    inputs_html = f"""
    <div style="font-size:14px; font-weight:bold; margin-bottom:10px; border-bottom:1px solid #333; padding-bottom:5px; margin-top:20px;">ENGINE INPUTS</div>
    <div style="font-size:12px; color:#aaa; line-height:1.8;">
        <div style="display:flex; justify-content:space-between"><span>Side to Move</span> <b style="color:#eee">{act_color}</b></div>
        <div style="display:flex; justify-content:space-between"><span>Active Clock</span> <b style="color:#eee">{_format_clock_s(act_clk)}</b></div>
        <div style="display:flex; justify-content:space-between"><span>Opponent Clock</span> <b style="color:#eee">{_format_clock_s(opp_clk)}</b></div>
        <div style="display:flex; justify-content:space-between"><span>Active ELO</span> <b style="color:#eee">{act_elo}</b></div>
        <div style="display:flex; justify-content:space-between"><span>Opponent ELO</span> <b style="color:#eee">{opp_elo}</b></div>
        <div style="display:flex; justify-content:space-between"><span>Halfmove Clock</span> <b style="color:#eee">{board.halfmove_clock}</b></div>
        <div style="display:flex; justify-content:space-between"><span>Castling</span> <b style="color:#eee">{castling_str}</b></div>
        <div style="display:flex; justify-content:space-between"><span>En Passant</span> <b style="color:#eee">{ep_str}</b></div>
    </div>
    """
    
    prev_html = "<div style='color:#666;font-style:italic;padding:10px'>Start of game.</div>"
    if game_state["cursor"] > 0:
        prev_item = game_state["history"][game_state["cursor"] - 1]
        prev_html = prev_item.get('prev_stats_html', '')
        if not prev_html:
            prev_html = "<div style='padding:10px'>Human moved.</div>"

    # GENERATE PGN for the full line
    pgn_board = chess.Board()
    full_moves = []
    for item in game_state["history"]:
        uci = item['uci']
        if uci in ('resign', 'flag'):
            continue
        full_moves.append(chess.Move.from_uci(uci))
    
    pgn_str = pgn_board.variation_san(full_moves)

    return jsonify({
        "fen": fen,
        "orientation": "white" if game_settings["human_color"] == chess.WHITE else "black",
        "current_stats": current_html,
        "engine_inputs": inputs_html,
        "prev_engine_stats": prev_html,
        "pgn": pgn_str,
        "cursor": game_state["cursor"],
        "max_cursor": len(game_state["history"]),
        "is_game_over": board.is_game_over(),
        "result": board.result() if board.is_game_over() else "",
        "attention64": current_stats.get("attention64"),
        "settings": game_settings
    })

@app.route('/')
def index():
    return render_template_string(HTML_TEMPLATE)

@app.route('/state', methods=['GET'])
def get_state():
    board = get_board_at_cursor()
    return prepare_response(board)

@app.route('/settings', methods=['POST'])
def update_settings():
    data = request.json
    game_settings['temperature'] = float(data.get('temperature', game_settings['temperature']))
    game_settings['time_temperature'] = float(data.get('time_temperature', game_settings.get('time_temperature', 1.0)))
    game_settings['top_p'] = float(data.get('top_p', game_settings['top_p']))
    game_settings['time_top_p'] = float(data.get('time_top_p', game_settings.get('time_top_p', 0.95)))
    game_settings['human_elo'] = int(data.get('human_elo', game_settings['human_elo']))
    game_settings['engine_elo'] = int(data.get('engine_elo', game_settings['engine_elo']))
    
    if 'human_color' in data:
        game_settings['human_color'] = chess.WHITE if data['human_color'] == 'white' else chess.BLACK
    
    if 'ponder' in data:
        game_settings['ponder'] = bool(data['ponder'])
        
    if 'use_real_time' in data:
        game_settings['use_real_time'] = bool(data['use_real_time'])

    if 'use_mode_time' in data:
        game_settings['use_mode_time'] = bool(data['use_mode_time'])

    if 'use_expected_time' in data:
        game_settings['use_expected_time'] = bool(data['use_expected_time'])

    # MCTS
    if 'use_mcts' in data:
        game_settings['use_mcts'] = bool(data['use_mcts'])
    if 'mcts_simulations' in data:
        game_settings['mcts_simulations'] = int(float(data['mcts_simulations']))
    if 'mcts_c_puct' in data:
        game_settings['mcts_c_puct'] = float(data['mcts_c_puct'])
    if 'mcts_max_children' in data:
        game_settings['mcts_max_children'] = int(float(data['mcts_max_children']))
    if 'mcts_root_dirichlet_alpha' in data:
        game_settings['mcts_root_dirichlet_alpha'] = float(data['mcts_root_dirichlet_alpha'])
    if 'mcts_root_exploration_frac' in data:
        game_settings['mcts_root_exploration_frac'] = float(data['mcts_root_exploration_frac'])
    if 'mcts_final_temperature' in data:
        game_settings['mcts_final_temperature'] = float(data['mcts_final_temperature'])
    if 'mcts_max_depth' in data:
        game_settings['mcts_max_depth'] = int(float(data['mcts_max_depth']))
    if 'mcts_adaptive' in data:
        game_settings['mcts_adaptive'] = bool(data['mcts_adaptive'])
    if 'mcts_adaptive_scale' in data:
        game_settings['mcts_adaptive_scale'] = float(data['mcts_adaptive_scale'])
    if 'show_mcts_stats' in data:
        game_settings['show_mcts_stats'] = bool(data['show_mcts_stats'])
    if 'show_attention' in data:
        game_settings['show_attention'] = bool(data['show_attention'])

    if 'attn_layer' in data:
        try:
            game_settings['attn_layer'] = int(float(data['attn_layer']))
        except Exception:
            game_settings['attn_layer'] = int(game_settings.get('attn_layer', -1))

    if 'attn_head_agg' in data:
        v = str(data['attn_head_agg']).lower().strip()
        if v not in ('avg', 'max', 'smolgen'):
            v = 'avg'
        # If smolgen is unavailable, fall back to avg.
        if v == 'smolgen' and not bool(game_settings.get('attn_has_smolgen', False)):
            v = 'avg'
        game_settings['attn_head_agg'] = v

    if 'attn_focus' in data:
        v = str(data['attn_focus']).lower().strip()
        if v not in ('outbound', 'inbound'):
            v = 'outbound'
        game_settings['attn_focus'] = v

    # Sampling-related settings changes should reset the per-position sampled-time cache.
    if any(k in data for k in ('time_temperature', 'time_top_p', 'use_real_time', 'use_mode_time', 'use_expected_time')):
        _clear_pos_cache()
    
    print(f"DEBUG: Settings Updated: Ponder={game_settings.get('ponder')}, RealTime={game_settings.get('use_real_time')}")

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

def _play_engine_move(board):
    if board.is_game_over():
        return

    moves_uci = get_uci_list_at_cursor()
    cursor_engine = game_state["cursor"]
    time_hist_engine = _pred_time_history_s_at_cursor(cursor_engine)
    clocks_engine = _clocks_at_cursor(cursor_engine)
    
    # 1. Analyze (for display stats)
    engine_stats = analyze_position(
        board,
        moves_uci,
        active_clock_s=float(clocks_engine[board.turn]),
        opponent_clock_s=float(clocks_engine[not board.turn]),
        time_history_s=time_hist_engine,
    )
    stats_html = format_stats_html(engine_stats)
    engine_pred_time_s = float(engine_stats.get("time_sample_s", 0.0))
    engine_pred_time_prob = float(engine_stats.get("time_sample_prob", 0.0))
    
    # 2. Ponder (Wait)
    if game_settings.get("ponder", False):
        time.sleep(engine_pred_time_s)

    # 3. Choose Move
    ctx = ContextOptions(
        active_elo=game_settings["engine_elo"], opponent_elo=game_settings["human_elo"],
        active_clock_s=float(clocks_engine[board.turn]), opponent_clock_s=float(clocks_engine[not board.turn]),
        active_inc_s=INC_S, opponent_inc_s=INC_S,
        halfmove_clock=int(board.halfmove_clock),
    )
    if bool(game_settings.get("use_mcts", False)):
        sims = int(game_settings.get("mcts_simulations", 128))
        cpuct = float(game_settings.get("mcts_c_puct", 1.5))
        
        if bool(game_settings.get("mcts_adaptive", False)):
            scale = float(game_settings.get("mcts_adaptive_scale", 100.0))
            # Adaptive Logic:
            # Simulations = scale * predicted_time
            # C_PUCT = base + (predicted_time / 20.0)
            sims = max(16, int(scale * engine_pred_time_s))
            cpuct = min(6.0, cpuct + (engine_pred_time_s / 20.0))

        mcts_settings = MCTSSettings(
            simulations=sims,
            c_puct=cpuct,
            max_children=int(game_settings.get("mcts_max_children", 48)),
            max_depth=int(game_settings.get("mcts_max_depth", 96)),
            root_dirichlet_alpha=float(game_settings.get("mcts_root_dirichlet_alpha", 0.0)),
            root_exploration_frac=float(game_settings.get("mcts_root_exploration_frac", 0.0)),
            final_temperature=float(game_settings.get("mcts_final_temperature", 0.0)),
        )
        out = mcts_choose_move(
            model=model,
            board=board,
            ctx=ctx,
            time_history_s=time_hist_engine,
            device=loaded.device,
            settings=mcts_settings,
            rng=rng,
        )
        if out.stats and game_settings.get("show_mcts_stats", False):
            stats_html = format_mcts_stats_html(out.stats)
    else:
        _fb, bh, rf = build_history_from_position(chess.Board(), moves_uci)
        out = choose_move(
            model=model, board=board, board_history=bh, repetition_flags=rf,
            ctx=ctx, temperature=game_settings['temperature'], top_p=game_settings['top_p'],
            time_history_s=time_hist_engine,
            device=loaded.device, rng=rng
        )
    
    if out.is_resign:
        chosen_txt = "Resigns"
        chosen_html = f"<div class='chosen-move' style='color:#f44336'>Engine Resigns ({out.policy_prob:.1%})</div>"
        full_prev_html = f"<div style='padding:4px'>{stats_html}</div>{chosen_html}"
        game_state["history"].append({
            'uci': 'resign',
            'pred_time_s': engine_pred_time_s,
            'pred_time_prob': engine_pred_time_prob,
            'prev_stats_html': full_prev_html
        })
    elif out.is_flag:
        chosen_txt = "Flags"
        chosen_html = f"<div class='chosen-move' style='color:#f44336'>Engine Flags ({out.policy_prob:.1%})</div>"
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

@app.route('/engine_move', methods=['POST'])
def engine_move_endpoint():
    board = get_board_at_cursor()
    # Only play if it is NOT the human's turn (i.e. it is engine's turn)
    if not board.is_game_over() and board.turn != game_settings['human_color']:
        _play_engine_move(board)
    return get_state()

@app.route('/reset', methods=['POST'])
def reset():
    game_state["history"] = []
    game_state["cursor"] = 0
    _clear_pos_cache()
    
    # Frontend will trigger engine move if needed
        
    return get_state()

# ==========================================
# 4. FRONTEND TEMPLATE
# ==========================================

HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>ChessFormer Web</title>
    <link rel="stylesheet" href="https://unpkg.com/@chrisoakman/chessboardjs@1.0.0/dist/chessboard-1.0.0.min.css">
    <script src="https://code.jquery.com/jquery-3.5.1.min.js"></script>
    <script src="https://unpkg.com/@chrisoakman/chessboardjs@1.0.0/dist/chessboard-1.0.0.min.js"></script>
    <style>
        body { background-color: #0e0e0e; color: #e0e0e0; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; margin: 0; padding: 20px; }
        
        .container { display: flex; gap: 20px; max-width: 1300px; margin: 0 auto; align-items: flex-start; justify-content: center; }
        
        /* 1. Settings Panel */
        .settings-panel { width: 250px; background: #181818; padding: 12px; border-radius: 8px; border: 1px solid #333; max-height: 85vh; overflow-y: auto; }
        .settings-panel::-webkit-scrollbar { width: 4px; }
        .settings-panel::-webkit-scrollbar-thumb { background: #444; border-radius: 2px; }
        .settings-panel::-webkit-scrollbar-track { background: #222; }

        .setting-group { margin-bottom: 8px; }
        .setting-label { font-size: 11px; font-weight: 600; color: #aaa; margin-bottom: 3px; display: flex; justify-content: space-between; align-items: center; }
        .setting-val { color: #4caf50; font-family: monospace; font-size: 11px; }
        input[type=range] { width: 100%; -webkit-appearance: none; background: #333; height: 4px; border-radius: 2px; outline: none; margin: 5px 0; display: block; }
        input[type=range]::-webkit-slider-thumb { -webkit-appearance: none; width: 12px; height: 12px; background: #eee; border-radius: 50%; cursor: pointer; }
        
        .section-header { font-size: 10px; font-weight: 700; color: #555; text-transform: uppercase; margin: 15px 0 8px 0; border-bottom: 1px solid #222; padding-bottom: 4px; letter-spacing: 0.5px; }
        .section-header:first-of-type { margin-top: 0; }

        .lock-btn { cursor:pointer; font-size:11px; color:#666; display:flex; align-items:center; gap:6px; margin: -6px 0 12px 0; justify-content:flex-end; user-select:none; }
        .lock-btn:hover { color:#999; }
        .lock-btn.active { color: #4caf50; }
        .lock-btn.active svg { fill: #4caf50; }
        .lock-btn svg { width:12px; height:12px; fill:#666; transition:fill 0.2s; }

        /* Simple color blobs for Play-As control */
        .color-blob { width:18px; height:18px; border-radius:50%; display:inline-block; box-sizing:border-box; }
        .color-blob.white { background: #fff; border: 2px solid #ccc; }
        /* Make black visible on the dark background by giving a light border and subtle outline */
        .color-blob.black { background: #000; border: 2px solid rgba(255,255,255,0.9); box-shadow: 0 0 0 2px rgba(255,255,255,0.06); }

        /* 2. Board Area */
        #board-area { width: 500px; }
        #board { width: 100%; }
        
        /* 3. Stats Panel */
        .sidebar { width: 320px; }
        
        /* Stats Styling */
        .stats-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 8px; }
        .panel { background: #1f1f1f; border: 1px solid #333; padding: 10px; border-radius: 6px; margin-bottom: 8px;}
        .panel h3 { margin: 0 0 8px 0; font-size: 11px; text-transform: uppercase; color: #888; border-bottom: 1px solid #333; padding-bottom: 4px; }
        
        .policy-scroll { max-height: 300px; overflow-y: auto; padding-right:4px; }
        .policy-scroll::-webkit-scrollbar { width: 4px; }
        .policy-scroll::-webkit-scrollbar-thumb { background: #444; border-radius: 2px; }
        .policy-scroll::-webkit-scrollbar-track { background: #222; }

        .row { display: flex; align-items: center; gap: 8px; margin-bottom: 4px; font-size: 13px; }
        .lbl { width: 50px; font-family: monospace; font-weight: bold; flex-shrink:0; }
        .bar-bg { flex-grow: 1; height: 6px; background: #333; border-radius: 2px; overflow: hidden; }
        .val { width: 35px; text-align: right; font-size: 11px; color: #aaa; }
        .sub { font-size: 11px; color: #888; margin-top: 4px; }
        .big-val { font-size: 20px; font-weight: bold; color: #eee; }
        .err { font-size: 12px; color: #f44336; font-weight: normal; }
        
        .section-title { font-size: 12px; font-weight: 600; color: #aaa; margin: 20px 0 6px 0; letter-spacing: 0.5px; }
        .chosen-move { padding: 8px; background: #2c2c2c; border-top: 1px solid #444; font-weight: bold; color: #4caf50; margin-top: 8px; }
        
        /* Controls */
        /* Slight right offset so the arrow buttons sit visually centered under the board */
        .controls { display: flex; gap: 5px; justify-content: center; margin-top: 10px; margin-left: 84px; }
        .btn { background: #333; color: white; border: none; padding: 8px 14px; cursor: pointer; border-radius: 4px; font-size: 16px; min-width: 40px;}
        .btn:hover { background: #444; }
        .btn-reset { font-size: 14px; background: #442222; }
        #status { text-align:center; min-height: 20px; color: #888; font-size: 13px; margin-top: 8px; }

        /* PGN Container */
        .pgn-container { margin-top: 15px; background: #181818; padding: 10px; border-radius: 4px; border: 1px solid #333; position: relative; min-height:40px; }
        #pgn-text { font-family: monospace; font-size: 12px; color: #aaa; line-height: 1.4; word-wrap: break-word; padding-right: 24px; white-space: pre-wrap; }
        .copy-btn { position: absolute; top: 5px; right: 5px; background: transparent; border: none; cursor: pointer; opacity: 0.4; font-size: 16px; transition: opacity 0.2s; }
        .copy-btn:hover { opacity: 1; }

        /* Attention overlay (8x8 grid over board) */
        #board-wrap { position: relative; width: 100%; }
        #attn-overlay { position: absolute; inset: 0; display: grid; grid-template-columns: repeat(8, 1fr); grid-template-rows: repeat(8, 1fr); pointer-events: none; }
        .attn-cell { background: #4caf50; opacity: 0; }

        
    </style>
</head>
<body>

<div class="container">
    <!-- LEFT: Settings -->
    <div class="settings-panel">
        <div class="section-header">Game Setup</div>
        
        <div class="setting-group" style="display:flex; align-items:center; justify-content:space-between; gap:8px;">
            <div class="setting-label" style="margin:0; flex:1">Play As</div>
            <div style="display:flex; gap:8px; align-items:center;">
                <div class="color-choice" id="i-color-white" data-color="white" title="Play as White" style="cursor:pointer; padding:4px; border-radius:4px; background:#222; border:1px solid #333;">
                    <div class="color-blob white"></div>
                </div>
                <div class="color-choice" id="i-color-black" data-color="black" title="Play as Black" style="cursor:pointer; padding:4px; border-radius:4px; background:#111; border:1px solid #333;">
                    <div class="color-blob black"></div>
                </div>
            </div>
        </div>

        <div class="setting-group" style="margin-bottom:4px">
            <div class="setting-label">Engine Elo <span id="v-eelo" class="setting-val">1900</span></div>
            <input type="range" id="i-eelo" min="1000" max="2800" step="50" value="1900">
        </div>

        <div class="lock-btn" id="lock-toggle" onclick="toggleEloLock()">
            <svg viewBox="0 0 24 24"><path d="M12 17a2 2 0 100-4 2 2 0 000 4zm6-9h-1V6a5 5 0 00-10 0v2H6a2 2 0 00-2 2v10a2 2 0 002 2h12a2 2 0 002-2V10a2 2 0 00-2-2zM9 6a3 3 0 116 0v2H9V6z"/></svg>
            <span>Link Elos</span>
        </div>

        <div class="setting-group">
            <div class="setting-label">Human Elo <span id="v-helo" class="setting-val">1900</span></div>
            <input type="range" id="i-helo" min="1000" max="2800" step="50" value="1900">
        </div>
        
        <div class="setting-group" style="display:flex; justify-content:space-between; align-items:center;">
            <div class="setting-label" style="margin:0">
                Ponder
                <span title="Engine waits for predicted time" style="cursor:help; color:#666; margin-left:4px;">ⓘ</span>
            </div>
            <input type="checkbox" id="i-ponder">
        </div>

        <div class="setting-group" style="display:flex; justify-content:space-between; align-items:center;">
            <div class="setting-label" style="margin:0">
                Real Time
                <span title="Use actual wall clock time" style="cursor:help; color:#666; margin-left:4px;">ⓘ</span>
            </div>
            <input type="checkbox" id="i-realtime">
        </div>

        <div class="setting-group" style="display:flex; justify-content:space-between; align-items:center;">
            <div class="setting-label" style="margin:0">
                Use Mode Time
                <span title="Use most likely time (Mode) instead of sampling" style="cursor:help; color:#666; margin-left:4px;">ⓘ</span>
            </div>
            <input type="checkbox" id="i-modetime">
        </div>

        <div class="setting-group" style="display:flex; justify-content:space-between; align-items:center;">
            <div class="setting-label" style="margin:0">
                Use Expected Time
                <span title="Use mean of top-p time distribution" style="cursor:help; color:#666; margin-left:4px;">ⓘ</span>
            </div>
            <input type="checkbox" id="i-exptime">
        </div>

        <div class="setting-group" style="display:flex; justify-content:space-between; align-items:center;">
            <div class="setting-label" style="margin:0">Show Attention</div>
            <input type="checkbox" id="i-attn" onchange="toggleAttnPanel()">
        </div>

        <div id="attn-params" style="display:none; margin-top:6px;">
            <div class="setting-group">
                <div class="setting-label">Layer</div>
                <select id="i-attn-layer" style="width:100%"></select>
            </div>

            <div class="setting-group">
                <div class="setting-label">Head Aggregation</div>
                <select id="i-attn-agg" style="width:100%">
                    <option value="avg">Average All Heads</option>
                    <option value="max">Max Head</option>
                    <option value="smolgen">Smolgen Only</option>
                </select>
            </div>

            <div class="setting-group">
                <div class="setting-label">Focus Mode</div>
                <select id="i-attn-focus" style="width:100%">
                    <option value="outbound">Outbound (from square)</option>
                    <option value="inbound">Inbound (to square)</option>
                </select>
            </div>
        </div>

        

        <div class="section-header">Sampling</div>

        <div class="setting-group">
            <div class="setting-label">Temperature <span id="v-temp" class="setting-val">0.9</span></div>
            <input type="range" id="i-temp" min="0" max="2.0" step="0.1" value="0.9">
        </div>
        
        <div class="setting-group">
            <div class="setting-label">Top P <span id="v-topp" class="setting-val">0.95</span></div>
            <input type="range" id="i-topp" min="0.5" max="1.0" step="0.01" value="0.95">
        </div>

        <div class="setting-group">
            <div class="setting-label">Time Temp <span id="v-ttemp" class="setting-val">1.0</span></div>
            <input type="range" id="i-ttemp" min="0.1" max="5.0" step="0.1" value="1.0">
        </div>

        <div class="setting-group">
            <div class="setting-label">Time Top P <span id="v-ttopp" class="setting-val">0.95</span></div>
            <input type="range" id="i-ttopp" min="0.5" max="1.0" step="0.01" value="0.95">
        </div>

        <div class="section-header">MCTS</div>

        <div class="setting-group" style="display:flex; justify-content:space-between; align-items:center;">
            <div class="setting-label" style="margin:0">Enable MCTS</div>
            <input type="checkbox" id="i-use-mcts" onchange="toggleMctsPanel()">
        </div>

        <div id="mcts-params">
            <div class="setting-group" style="display:flex; justify-content:space-between; align-items:center;">
                <div class="setting-label" style="margin:0">Show Stats</div>
                <input type="checkbox" id="i-mcts-stats">
            </div>

            <div class="setting-group">
                <div class="setting-label">Simulations <span id="v-mcts-sims" class="setting-val">128</span></div>
                <input type="range" id="i-mcts-sims" min="16" max="5000" step="16" value="128">
            </div>

            <div class="setting-group">
                <div class="setting-label">C_PUCT <span id="v-mcts-cpuct" class="setting-val">1.5</span></div>
                <input type="range" id="i-mcts-cpuct" min="0.2" max="6.0" step="0.1" value="1.5">
            </div>

            <div class="setting-group">
                <div class="setting-label">Max Children <span id="v-mcts-k" class="setting-val">48</span></div>
                <input type="range" id="i-mcts-k" min="8" max="128" step="8" value="48">
            </div>

            <div class="setting-group">
                <div class="setting-label">Max Depth <span id="v-mcts-depth" class="setting-val">96</span></div>
                <input type="range" id="i-mcts-depth" min="1" max="200" step="1" value="96">
            </div>

            <div class="setting-group" style="display:flex; justify-content:space-between; align-items:center; margin-top:10px; border-top:1px solid #333; padding-top:5px;">
                <div class="setting-label" style="margin:0">Adaptive MCTS</div>
                <input type="checkbox" id="i-mcts-adaptive" onchange="toggleAdaptivePanel()">
            </div>
            
            <div id="adaptive-params" style="display:none; margin-bottom:10px; border-bottom:1px solid #333; padding-bottom:5px;">
                <div class="setting-group">
                    <div class="setting-label">Scaling Factor <span id="v-mcts-ascale" class="setting-val">100</span></div>
                    <input type="range" id="i-mcts-ascale" min="10" max="1000" step="10" value="100">
                </div>
            </div>

            <div class="setting-group">
                <div class="setting-label">Root Noise (frac) <span id="v-mcts-frac" class="setting-val">0.0</span></div>
                <input type="range" id="i-mcts-frac" min="0.0" max="0.5" step="0.05" value="0.0">
            </div>

            <div class="setting-group">
                <div class="setting-label">Root Noise (alpha) <span id="v-mcts-alpha" class="setting-val">0.0</span></div>
                <input type="range" id="i-mcts-alpha" min="0.0" max="1.0" step="0.05" value="0.0">
            </div>

            <div class="setting-group">
                <div class="setting-label">Final Temp <span id="v-mcts-ft" class="setting-val">0.0</span></div>
                <input type="range" id="i-mcts-ft" min="0.0" max="2.0" step="0.1" value="0.0">
            </div>
        </div>

        <div id="engine-inputs"></div>
    </div>

    <!-- CENTER: Board -->
    <div id="board-area">
            <div id="board-wrap">
            <div id="board"></div>
            <div id="attn-overlay"></div>
        </div>
        <div class="controls">
            <button class="btn" onclick="nav('start')" title="Start"><<</button>
            <button class="btn" onclick="nav('prev')" title="Previous"><</button>
            <button class="btn" onclick="nav('next')" title="Next">></button>
            <button class="btn" onclick="nav('end')" title="End">>></button>
            <div style="width:20px"></div>
            <button class="btn btn-reset" onclick="resetGame()">New</button>
        </div>
        <div id="status"></div>
        
        <div class="pgn-container">
            <div id="pgn-text"></div>
            <button class="copy-btn" onclick="copyPgn()" title="Copy PGN">📋</button>
        </div>
    </div>
    
    <!-- RIGHT: Stats -->
    <div class="sidebar">
        <div class="section-title" style="margin-top:0">CURRENT EVALUATION (Live)</div>
        <div id="current-stats">Waiting...</div>
        
        <div class="section-title">PREVIOUS MOVE</div>
        <div id="prev-engine-stats">None.</div>
    </div>
</div>

<script>
var board = null;
var game_fen = 'start';
var cursor = 0, max_cursor = 0;
var eloLocked = false;
var moveStartTime = 0;
var humanColor = 'white'; // 'white' or 'black'
var attention64 = null;
var overlayOrientation = null;
var attnFocusMode = 'outbound';
 

function toggleMctsPanel() {
    var checked = $('#i-use-mcts').is(':checked');
    if (checked) {
        $('#mcts-params').slideDown(200);
        $('#i-temp, #i-topp, #i-ttemp, #i-ttopp, #i-ponder').prop('disabled', true).closest('.setting-group').css('opacity', 0.5);
        toggleAdaptivePanel();
    } else {
        $('#mcts-params').slideUp(200);
        $('#i-temp, #i-topp, #i-ttemp, #i-ttopp, #i-ponder').prop('disabled', false).closest('.setting-group').css('opacity', 1.0);
    }
}

function toggleAttnPanel() {
    var checked = $('#i-attn').is(':checked');
    if (checked) {
        $('#attn-params').slideDown(200);
        updateAttnControlEnables();
    } else {
        $('#attn-params').slideUp(200);
        clearAttentionOverlay();
    }
}

function ensureAttnLayerOptions(numLayers) {
    numLayers = parseInt(numLayers || 0, 10);
    var $sel = $('#i-attn-layer');
    if ($sel.data('builtFor') === numLayers) return;
    $sel.empty();
    $sel.append("<option value='-1'>All Layers</option>");
    for (var i = 0; i < numLayers; i++) {
        $sel.append("<option value='" + i + "'>Layer " + i + "</option>");
    }
    $sel.data('builtFor', numLayers);
}

function updateAttnControlEnables() {
    var agg = $('#i-attn-agg').val();
    var smolgenSelected = (agg === 'smolgen');
    $('#i-attn-layer').prop('disabled', smolgenSelected).closest('.setting-group').css('opacity', smolgenSelected ? 0.5 : 1.0);
}

function toggleAdaptivePanel() {
    var checked = $('#i-mcts-adaptive').is(':checked');
    if (checked) {
        $('#adaptive-params').slideDown(200);
        $('#i-mcts-sims, #i-mcts-cpuct').prop('disabled', true).closest('.setting-group').css('opacity', 0.5);
    } else {
        $('#adaptive-params').slideUp(200);
        $('#i-mcts-sims, #i-mcts-cpuct').prop('disabled', false).closest('.setting-group').css('opacity', 1.0);
    }
}

function squareToIndex(sq) {
    // a1=0 .. h8=63
    var file = sq.charCodeAt(0) - 'a'.charCodeAt(0);
    var rank = parseInt(sq[1], 10) - 1;
    return rank * 8 + file;
}

function buildAttentionOverlay(orientation) {
    if (overlayOrientation === orientation) return;
    overlayOrientation = orientation;
    var $o = $('#attn-overlay');
    $o.empty();

    var filesW = ['a','b','c','d','e','f','g','h'];
    var filesB = ['h','g','f','e','d','c','b','a'];

    if (orientation === 'white') {
        for (var r = 8; r >= 1; r--) {
            for (var f = 0; f < 8; f++) {
                var sq = filesW[f] + r;
                $o.append("<div class='attn-cell' data-square='" + sq + "'></div>");
            }
        }
    } else {
        for (var r2 = 1; r2 <= 8; r2++) {
            for (var f2 = 0; f2 < 8; f2++) {
                var sq2 = filesB[f2] + r2;
                $o.append("<div class='attn-cell' data-square='" + sq2 + "'></div>");
            }
        }
    }
}

function clearAttentionOverlay() {
    $('#attn-overlay .attn-cell').css('opacity', 0);
}

 



function renderAttentionFromSquare(fromSq) {
    if (!attention64 || attention64.length !== 4096) return;

    var fromIdx = squareToIndex(fromSq);
    var vals = new Array(64);

    if (attnFocusMode === 'inbound') {
        // Column: which query squares attend to this square.
        for (var i = 0; i < 64; i++) vals[i] = attention64[i * 64 + fromIdx];
    } else {
        // Row: where this square attends.
        var rowStart = fromIdx * 64;
        for (var j = 0; j < 64; j++) vals[j] = attention64[rowStart + j];
    }

    var maxV = 0;
    for (var k = 0; k < 64; k++) if (vals[k] > maxV) maxV = vals[k];
    if (maxV <= 0) { clearAttentionOverlay(); return; }

    $('#attn-overlay .attn-cell').each(function() {
        var sq = $(this).attr('data-square');
        if (!sq) return;
        var idx = squareToIndex(sq);
        var a = vals[idx] / maxV;
        $(this).css('opacity', Math.max(0, Math.min(1, a)));
    });
}

// Settings Logic
function sendSettings() {
    var payload = {
        temperature: $('#i-temp').val(),
        time_temperature: $('#i-ttemp').val(),
        top_p: $('#i-topp').val(),
        time_top_p: $('#i-ttopp').val(),
        use_mcts: $('#i-use-mcts').is(':checked'),
        show_mcts_stats: $('#i-mcts-stats').is(':checked'),
        show_attention: $('#i-attn').is(':checked'),

        attn_layer: $('#i-attn-layer').val(),
        attn_head_agg: $('#i-attn-agg').val(),
        attn_focus: $('#i-attn-focus').val(),
        
        mcts_simulations: $('#i-mcts-sims').val(),
        mcts_c_puct: $('#i-mcts-cpuct').val(),
        mcts_max_children: $('#i-mcts-k').val(),
        mcts_max_depth: $('#i-mcts-depth').val(),
        mcts_adaptive: $('#i-mcts-adaptive').is(':checked'),
        mcts_adaptive_scale: $('#i-mcts-ascale').val(),
        mcts_root_exploration_frac: $('#i-mcts-frac').val(),
        mcts_root_dirichlet_alpha: $('#i-mcts-alpha').val(),
        mcts_final_temperature: $('#i-mcts-ft').val(),
        engine_elo: $('#i-eelo').val(),
        human_elo: $('#i-helo').val(),
        human_color: humanColor,
        ponder: $('#i-ponder').is(':checked'),
        use_real_time: $('#i-realtime').is(':checked'),
        use_mode_time: $('#i-modetime').is(':checked'),
        use_expected_time: $('#i-exptime').is(':checked')
    };
    $('#v-temp').text(payload.temperature);
    $('#v-ttemp').text(payload.time_temperature);
    $('#v-topp').text(payload.top_p);
    $('#v-ttopp').text(payload.time_top_p);
    $('#v-eelo').text(payload.engine_elo);
    $('#v-helo').text(payload.human_elo);

    $('#v-mcts-sims').text(payload.mcts_simulations);
    $('#v-mcts-cpuct').text(payload.mcts_c_puct);
    $('#v-mcts-k').text(payload.mcts_max_children);
    $('#v-mcts-depth').text(payload.mcts_max_depth);
    $('#v-mcts-ascale').text(payload.mcts_adaptive_scale);
    $('#v-mcts-frac').text(payload.mcts_root_exploration_frac);
    $('#v-mcts-alpha').text(payload.mcts_root_dirichlet_alpha);
    $('#v-mcts-ft').text(payload.mcts_final_temperature);
    
    humanColor = payload.human_color;

    $.ajax({
        url: '/settings', type: 'POST', contentType: 'application/json',
        data: JSON.stringify(payload),
        success: function(data) { 
            updateUI(data); 
            moveStartTime = Date.now(); // Reset clock on settings change
            checkEngineMove(data);
        }
    });
}

function checkEngineMove(data) {
    // If game not over, and it's engine's turn, trigger engine move
    // Only trigger if we are at the latest position (not reviewing history)
    if (!data.is_game_over && data.cursor === data.max_cursor) {
        var turn = data.fen.split(' ')[1]; // 'w' or 'b'
        var engineTurn = (humanColor === 'white' && turn === 'b') || (humanColor === 'black' && turn === 'w');
        
        if (engineTurn) {
            $('#status').text("Engine thinking...");
            $.post('/engine_move', function(data2) {
                updateUI(data2);
                moveStartTime = Date.now(); // Reset clock after engine moves
            });
        }
    }
}

function toggleEloLock() {
    eloLocked = !eloLocked;
    $('#lock-toggle').toggleClass('active', eloLocked);
    if (eloLocked) {
        var val = $('#i-eelo').val();
        $('#i-helo').val(val);
        $('#v-helo').text(val);
        sendSettings();
    }
}

$('#i-temp').on('input', function() { $('#v-temp').text($(this).val()); });
$('#i-ttemp').on('input', function() { $('#v-ttemp').text($(this).val()); });
$('#i-topp').on('input', function() { $('#v-topp').text($(this).val()); });
$('#i-ttopp').on('input', function() { $('#v-ttopp').text($(this).val()); });
$('#i-mcts-sims').on('input', function() { $('#v-mcts-sims').text($(this).val()); });
$('#i-mcts-cpuct').on('input', function() { $('#v-mcts-cpuct').text($(this).val()); });
$('#i-mcts-k').on('input', function() { $('#v-mcts-k').text($(this).val()); });
$('#i-mcts-depth').on('input', function() { $('#v-mcts-depth').text($(this).val()); });
$('#i-mcts-ascale').on('input', function() { $('#v-mcts-ascale').text($(this).val()); });
$('#i-mcts-frac').on('input', function() { $('#v-mcts-frac').text($(this).val()); });
$('#i-mcts-alpha').on('input', function() { $('#v-mcts-alpha').text($(this).val()); });
$('#i-mcts-ft').on('input', function() { $('#v-mcts-ft').text($(this).val()); });

$('#i-eelo').on('input', function() {
    var val = $(this).val();
    $('#v-eelo').text(val);
    if(eloLocked) {
        $('#i-helo').val(val);
        $('#v-helo').text(val);
    }
});

$('#i-helo').on('input', function() {
    var val = $(this).val();
    $('#v-helo').text(val);
    if(eloLocked) {
        $('#i-eelo').val(val);
        $('#v-eelo').text(val);
    }
});

$('#i-temp, #i-ttemp, #i-topp, #i-ttopp, #i-use-mcts, #i-mcts-stats, #i-mcts-sims, #i-mcts-cpuct, #i-mcts-k, #i-mcts-depth, #i-mcts-adaptive, #i-mcts-ascale, #i-mcts-frac, #i-mcts-alpha, #i-mcts-ft, #i-eelo, #i-helo, #i-ponder, #i-realtime, #i-modetime, #i-exptime, #i-attn, #i-attn-layer, #i-attn-agg, #i-attn-focus').on('change', function() {
    updateAttnControlEnables();
    sendSettings();
});

// Attention overlay hover (delegated)
$('#board').on('mouseenter', '.square-55d63', function() {
    if (!$('#i-attn').is(':checked')) return;
    if (!attention64) return;
    var sq = $(this).attr('data-square');
    if (!sq) return;
    renderAttentionFromSquare(sq);
});

$('#board').on('mouseleave', function() {
    clearAttentionOverlay();
});

// Color choice click handlers
$('.color-choice').on('click', function() {
    var c = $(this).data('color');
    humanColor = c;
    $('.color-choice').removeClass('selected').css('box-shadow','none');
    $(this).addClass('selected').css('box-shadow','0 0 0 2px rgba(76,175,80,0.15)');
    sendSettings();
});

// Board Logic
function onDragStart (source, piece) { 
    var orientation = board.orientation();
    if ((orientation === 'white' && piece.search(/^b/) !== -1) ||
        (orientation === 'black' && piece.search(/^w/) !== -1)) {
        return false;
    }
}
function onDrop (source, target) {
  if (source === target) return;
  var elapsed = (Date.now() - moveStartTime) / 1000.0;
  $.ajax({
      url: '/move', type: 'POST', contentType: 'application/json',
      data: JSON.stringify({ from: source, to: target, elapsed_s: elapsed }),
      success: function(data) { 
          updateUI(data); 
          checkEngineMove(data);
      },
      error: function() { board.position(game_fen); }
  });
}
function nav(action) {
    $.ajax({
        url: '/navigate', type: 'POST', contentType: 'application/json',
        data: JSON.stringify({ action: action }),
        success: function(data) { updateUI(data); }
    });
}
function updateUI(data) {
    game_fen = data.fen;
    cursor = data.cursor; max_cursor = data.max_cursor;
    board.orientation(data.orientation);
    board.position(data.fen);

    attention64 = data.attention64;
    buildAttentionOverlay(data.orientation);

    
    $('#current-stats').html(data.current_stats);
    $('#prev-engine-stats').html(data.prev_engine_stats);
    $('#pgn-text').text(data.pgn);
    if(data.engine_inputs) $('#engine-inputs').html(data.engine_inputs);
    
    var status = "Move " + Math.floor(cursor/2 + 1);
    if (data.is_game_over) status += " | " + data.result;
    else if (cursor < max_cursor) status += " (Analysis Mode)";
    $('#status').text(status);
    
    // NOTE: We do NOT reset moveStartTime here anymore, to avoid resetting it during navigation.
    // It is reset in init, resetGame, sendSettings, and after engine move.

    if (data.settings) {
        var s = data.settings;
        $('#i-temp').val(s.temperature); $('#v-temp').text(s.temperature);
        $('#i-ttemp').val(s.time_temperature); $('#v-ttemp').text(s.time_temperature);
        $('#i-topp').val(s.top_p); $('#v-topp').text(s.top_p);
        $('#i-ttopp').val(s.time_top_p); $('#v-ttopp').text(s.time_top_p);

        $('#i-attn').prop('checked', !!s.show_attention);
        ensureAttnLayerOptions(s.attn_num_layers || 0);
        $('#i-attn-layer').val((s.attn_layer != null) ? String(s.attn_layer) : '-1');
        $('#i-attn-agg').val(s.attn_head_agg || 'avg');
        $('#i-attn-focus').val(s.attn_focus || 'outbound');
        attnFocusMode = $('#i-attn-focus').val();
        toggleAttnPanel();
        if (!s.show_attention) clearAttentionOverlay();

        $('#i-use-mcts').prop('checked', !!s.use_mcts);
        toggleMctsPanel();
        $('#i-mcts-stats').prop('checked', !!s.show_mcts_stats);
        $('#i-mcts-sims').val(s.mcts_simulations); $('#v-mcts-sims').text(s.mcts_simulations);
        $('#i-mcts-cpuct').val(s.mcts_c_puct); $('#v-mcts-cpuct').text(s.mcts_c_puct);
        $('#i-mcts-k').val(s.mcts_max_children); $('#v-mcts-k').text(s.mcts_max_children);
        $('#i-mcts-depth').val(s.mcts_max_depth || 96); $('#v-mcts-depth').text(s.mcts_max_depth || 96);
        
        $('#i-mcts-adaptive').prop('checked', !!s.mcts_adaptive);
        toggleAdaptivePanel();
        $('#i-mcts-ascale').val(s.mcts_adaptive_scale || 100); $('#v-mcts-ascale').text(s.mcts_adaptive_scale || 100);

        $('#i-mcts-frac').val(s.mcts_root_exploration_frac); $('#v-mcts-frac').text(s.mcts_root_exploration_frac);
        $('#i-mcts-alpha').val(s.mcts_root_dirichlet_alpha); $('#v-mcts-alpha').text(s.mcts_root_dirichlet_alpha);
        $('#i-mcts-ft').val(s.mcts_final_temperature); $('#v-mcts-ft').text(s.mcts_final_temperature);
        $('#i-eelo').val(s.engine_elo); $('#v-eelo').text(s.engine_elo);
        $('#i-helo').val(s.human_elo); $('#v-helo').text(s.human_elo);
        
        $('#i-ponder').prop('checked', s.ponder);
        $('#i-realtime').prop('checked', s.use_real_time);
        $('#i-modetime').prop('checked', s.use_mode_time);
        $('#i-exptime').prop('checked', s.use_expected_time);
        
        var color = s.human_color ? 'white' : 'black';
        humanColor = color;
        $('.color-choice').removeClass('selected').css('box-shadow','none');
        if (color === 'white') {
            $('#i-color-white').addClass('selected').css('box-shadow','0 0 0 2px rgba(76,175,80,0.15)');
        } else {
            $('#i-color-black').addClass('selected').css('box-shadow','0 0 0 2px rgba(76,175,80,0.15)');
        }
    }
}

 
function resetGame() { 
    $.post('/reset', function(data) { 
        updateUI(data); 
        moveStartTime = Date.now();
        checkEngineMove(data);
    }); 
}

function copyPgn() {
    var text = $('#pgn-text').text();
    navigator.clipboard.writeText(text).then(function() {
        var btn = $('.copy-btn');
        var original = btn.text();
        btn.text('✅');
        setTimeout(function() { btn.text(original); }, 1500);
    });
}

function init() {
    board = Chessboard('board', {
        draggable: true, position: 'start',
        onDragStart: onDragStart, onDrop: onDrop,
        pieceTheme: 'https://chessboardjs.com/img/chesspieces/wikipedia/{piece}.png'
    });
    $.get('/state', function(data) { 
        updateUI(data); 
        moveStartTime = Date.now();
        humanColor = data.orientation;
        checkEngineMove(data);
    });
    $(document).keydown(function(e) {
        if (e.keyCode == 37) nav('prev');
        else if (e.keyCode == 39) nav('next');
    });
}
$(document).ready(init);
</script>
</body>
</html>
"""

if __name__ == '__main__':
    print("Starting Flask Server on http://localhost:5000")
    app.run(host='0.0.0.0', port=5000, debug=False)