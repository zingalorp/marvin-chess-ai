from __future__ import annotations

import math
import time
from typing import Callable

import numpy as np
import torch
import chess

from inference.chessformer_policy import PolicyOutput, choose_move
from inference.encoding import ContextOptions, build_history_from_position, canonicalize, make_model_batch
from inference.mcts import MCTSSettings, MCTSResult, mcts_choose_move, find_subtree_by_move_sequence, _Node
from inference.sampling import sample_from_logits


PROMO_INDEX = {chess.QUEEN: 0, chess.ROOK: 1, chess.BISHOP: 2, chess.KNIGHT: 3}


def _mirror_square(sq: int) -> int:
    return sq ^ 56


def _canonical_to_real_move(move: chess.Move, real_turn: chess.Color) -> chess.Move:
    if real_turn == chess.WHITE:
        return move
    return chess.Move(_mirror_square(move.from_square), _mirror_square(move.to_square), promotion=move.promotion)


def _softmax_1d(x: torch.Tensor) -> torch.Tensor:
    x = x.float() - torch.max(x)
    return torch.softmax(x, dim=0)


def _time_bin_to_seconds(bin_idx: int, active_clock_s: float) -> float:
    scaled_mid = (bin_idx + 0.5) / 256.0
    return float((scaled_mid**2) * max(1e-6, active_clock_s))


def _extract_attention_64(
    model: torch.nn.Module,
    *,
    real_turn: chess.Color,
    layer: int,
    head_agg: str,
) -> list[float] | None:
    """Copied from `inference/app.py` (flattened 64x64 in real square indexing)."""

    head_agg = str(head_agg or "avg").lower().strip()
    if head_agg not in ("avg", "max", "smolgen"):
        head_agg = "avg"

    orig_model = getattr(model, "_orig_mod", model)

    if head_agg == "smolgen":
        bias_h: torch.Tensor | None = None
        for mod in orig_model.modules():
            b = getattr(mod, "last_smolgen_bias", None)
            if b is None:
                continue
            if not torch.is_tensor(b) or b.ndim != 4 or b.shape[-2:] != (64, 64):
                continue
            bias_h = b[0].float()  # (H,64,64)
            break

        if bias_h is None:
            return None

        probs_h = torch.softmax(bias_h, dim=-1)
        mat = probs_h.mean(dim=0)
    else:
        layers_h: list[torch.Tensor] = []
        for mod in orig_model.modules():
            attn = getattr(mod, "last_attn_probs", None)
            if attn is None:
                continue
            if not torch.is_tensor(attn) or attn.ndim != 4 or attn.shape[-2:] != (64, 64):
                continue
            layers_h.append(attn[0].float())

        if not layers_h:
            return None

        if int(layer) >= 0:
            idx = int(layer)
            if idx >= len(layers_h):
                idx = len(layers_h) - 1
            layers_h = [layers_h[idx]]

        if head_agg == "avg":
            mats = [a.mean(dim=0) for a in layers_h]
            mat = torch.stack(mats, dim=0).mean(dim=0)
        else:
            mats = [a.max(dim=0).values for a in layers_h]
            mat = mats[0]
            for m in mats[1:]:
                mat = torch.maximum(mat, m)

    if real_turn == chess.BLACK:
        perm = torch.tensor([_mirror_square(i) for i in range(64)], device=mat.device, dtype=torch.long)
        mat = mat.index_select(0, perm).index_select(1, perm)

    return mat.detach().cpu().reshape(-1).tolist()


def analyze_position(
    *,
    model: torch.nn.Module,
    device: torch.device,
    settings: dict,
    rng: np.random.Generator,
    board: chess.Board,
    moves_uci: list[str],
    active_clock_s: float,
    opponent_clock_s: float,
    active_inc_s: float = 0.0,
    opponent_inc_s: float = 0.0,
    time_history_s: list[float] | None = None,
    tc_base_s: float | None = None,
) -> dict:
    """Same analysis as `inference/app.py.analyze_position`, but dependency-injected."""

    _final_board, board_history, repetition_flags = build_history_from_position(chess.Board(), moves_uci)

    if board.turn == settings["human_color"]:
        active_elo = settings["human_elo"]
        opp_elo = settings["engine_elo"]
    else:
        active_elo = settings["engine_elo"]
        opp_elo = settings["human_elo"]

    ctx = ContextOptions(
        active_elo=active_elo,
        opponent_elo=opp_elo,
        active_clock_s=active_clock_s,
        opponent_clock_s=opponent_clock_s,
        active_inc_s=float(active_inc_s),
        opponent_inc_s=float(opponent_inc_s),
        tc_base_s=(
            float(tc_base_s)
            if tc_base_s is not None
            else (float(settings.get("start_clock_s")) if settings.get("start_clock_s") is not None else None)
        ),
        halfmove_clock=int(board.halfmove_clock),
    )

    batch = make_model_batch(
        board=board,
        board_history=board_history,
        repetition_flags=repetition_flags,
        time_history_s=time_history_s,
        ctx=ctx,
        device=device,
    )

    # For analysis only: unmask resign/flag so we can inspect their probabilities.
    batch_for_rf = dict(batch)
    if "legal_mask" in batch_for_rf:
        lm = batch_for_rf["legal_mask"].clone()
        if lm.shape[-1] > 4097:
            lm[..., 4096] = True
            lm[..., 4097] = True
        batch_for_rf["legal_mask"] = lm

    with torch.inference_mode():
        with torch.autocast(device_type=device.type, enabled=(device.type == "cuda")):
            (m_logits, v_raw, v_cls, v_err, t_logits, _ss_logits, p_logits) = model(batch, return_promo=True)
            (m_logits_rf, *_rest) = model(batch_for_rf, return_promo=False)

    attention64 = None
    if bool(settings.get("show_attention", False)):
        attention64 = _extract_attention_64(
            model,
            real_turn=board.turn,
            layer=int(settings.get("attn_layer", -1)),
            head_agg=str(settings.get("attn_head_agg", "avg")),
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
            logit += float(np.log(max(1e-8, promo_prob)))
        legal_moves_data.append({"move": mv, "logit": logit})

    if not legal_moves_data:
        policy_display = []
    else:
        T = max(1e-4, float(settings["temperature"]))
        logits_vec = torch.tensor([x["logit"] for x in legal_moves_data], device=device)
        logits_vec = logits_vec / T
        probs_vec = torch.softmax(logits_vec, dim=0)

        sorted_probs, sorted_indices = torch.sort(probs_vec, descending=True)
        cumulative_probs = torch.cumsum(sorted_probs, dim=0)

        target_p = float(settings["top_p"])
        cutoff_index = torch.searchsorted(cumulative_probs, target_p).item()
        # `cutoff_index` is an index (0-based). We want a *count* of items to keep.
        # Ensure we keep at least 1 move, including the single-legal-move case.
        cutoff_count = min(len(sorted_probs), int(cutoff_index) + 1)
        cutoff_count = max(1, int(cutoff_count))

        kept_indices = sorted_indices[:cutoff_count]
        kept_probs = sorted_probs[:cutoff_count]
        kept_probs = kept_probs / kept_probs.sum()

        policy_display = []
        real_turn = board.turn
        for i, prob in zip(kept_indices.tolist(), kept_probs.tolist()):
            mv = legal_moves_data[i]["move"]
            real_mv = _canonical_to_real_move(mv, real_turn)
            try:
                label = board.san(real_mv)
            except Exception:
                label = real_mv.uci()
            policy_display.append({"label": label, "prob": prob, "uci": real_mv.uci()})

    wdl = _softmax_1d(v_cls[0])

    # Apply Time Temperature
    T_time = max(1e-4, float(settings.get("time_temperature", 1.0)))
    t_logits_scaled = t_logits[0].float() / T_time
    time_p = torch.softmax(t_logits_scaled - torch.max(t_logits_scaled), dim=0)

    # Time Distribution (Top X%)
    target_time_p = float(settings.get("time_top_p", 0.95))
    t_probs = time_p.cpu().numpy()

    # Calculate raw stats (Temp=1.0) for "True" Model Opinion
    t_probs_raw = torch.softmax(t_logits[0].float(), dim=0).cpu().numpy()

    # Mode (Argmax)
    mode_bin = int(np.argmax(t_probs_raw))
    mode_time_s = _time_bin_to_seconds(mode_bin, active_clock_s)

    t_sorted_idx = np.argsort(t_probs)[::-1]
    t_cumsum = np.cumsum(t_probs[t_sorted_idx])
    t_cutoff = int(np.searchsorted(t_cumsum, target_time_p) + 1)
    t_active_idx = t_sorted_idx[:t_cutoff]
    t_active_idx.sort()

    time_dist = []
    expected_time_s = 0.0
    total_prob_mass = 0.0

    for idx in t_active_idx:
        sec = _time_bin_to_seconds(int(idx), active_clock_s)
        prob = float(t_probs[idx])
        time_dist.append({"sec": sec, "prob": prob})

        expected_time_s += prob * sec
        total_prob_mass += prob

    if total_prob_mass > 0:
        expected_time_s /= total_prob_mass

    # Sample a time bin (like we sample moves), but with fixed temperature=1.
    if settings.get("use_mode_time", False):
        time_sample_s = mode_time_s
        time_sample_prob = float(t_probs_raw[mode_bin])
    elif settings.get("use_expected_time", False):
        time_sample_s = expected_time_s
        time_sample_prob = 1.0
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

    # Use the analysis forward-pass logits where resign/flag are unmasked.
    raw_policy_all = _softmax_1d(m_logits_rf[0])

    win_prob = float(torch.sigmoid(v_raw[0].squeeze(-1)).item())

    pred_sq_error = float(v_err[0].squeeze(-1).item())
    error_bar = float(np.sqrt(max(0.0, pred_sq_error)))

    return {
        "top_moves": policy_display,
        "resign": float(raw_policy_all[4096].item()),
        "flag": float(raw_policy_all[4097].item()),
        "wdl": {"w": wdl[2].item(), "d": wdl[1].item(), "l": wdl[0].item()},
        "value": win_prob,
        "value_error": error_bar,
        "time_dist": time_dist,
        "time_top_p": target_time_p,
        "time_sample_s": float(time_sample_s),
        "time_sample_prob": float(time_sample_prob),
        "expected_time_s": float(expected_time_s),
        "mode_time_s": float(mode_time_s),
        "attention64": attention64,
    }


def choose_engine_move(
    *,
    model: torch.nn.Module,
    device: torch.device,
    settings: dict,
    rng: np.random.Generator,
    board: chess.Board,
    moves_uci: list[str],
    active_clock_s: float,
    opponent_clock_s: float,
    active_inc_s: float = 0.0,
    opponent_inc_s: float = 0.0,
    time_history_s: list[float] | None,
    stop_check: Callable[[], bool] | None = None,
    allow_ponder_sleep: bool = True,
    mcts_progress_callback: Callable[[dict], None] | None = None,
    mcts_reuse_root: _Node | None = None,
    mcts_reuse_moves: list[chess.Move] | None = None,
) -> tuple[PolicyOutput, dict, dict | None, MCTSResult | None]:
    """Implements the engine selection path from `_play_engine_move` in `inference/app.py`.

    Returns: (policy_out, analysis_dict, mcts_stats_or_none, mcts_result_or_none)
    
    The mcts_result contains the search tree root for potential reuse in subsequent searches.
    Pass mcts_reuse_root and mcts_reuse_moves to enable tree reuse from a previous search.
    """

    if board.is_game_over():
        return PolicyOutput(move=None, policy_prob=1.0), {}, None, None

    engine_stats = analyze_position(
        model=model,
        device=device,
        settings=settings,
        rng=rng,
        board=board,
        moves_uci=moves_uci,
        active_clock_s=float(active_clock_s),
        opponent_clock_s=float(opponent_clock_s),
        active_inc_s=float(active_inc_s),
        opponent_inc_s=float(opponent_inc_s),
        time_history_s=time_history_s,
        tc_base_s=(float(settings.get("start_clock_s")) if settings.get("start_clock_s") is not None else None),
    )

    engine_pred_time_s = float(engine_stats.get("time_sample_s", 0.0))

    # For non-MCTS mode: simulate thinking time before returning
    # For MCTS mode: we handle timing after MCTS completes (see below)
    use_mcts = bool(settings.get("use_mcts", False))
    current_ply = len(moves_uci)
    mcts_start_ply = int(settings.get("mcts_start_ply", 0))
    
    # Skip MCTS for early plies to allow varied openings
    use_mcts_this_move = use_mcts and (current_ply >= mcts_start_ply)

    if allow_ponder_sleep and bool(settings.get("simulate_thinking_time", False)) and not use_mcts_this_move:
        # In the web app, this is an intentional delay. In UCI, we want to be
        # responsive to `stop`/new `position`, so make it interruptible.
        if stop_check is None:
            time.sleep(engine_pred_time_s)
        else:
            deadline = time.time() + max(0.0, engine_pred_time_s)
            while time.time() < deadline:
                if stop_check():
                    break
                time.sleep(0.05)

    ctx = ContextOptions(
        active_elo=int(settings["engine_elo"]),
        opponent_elo=int(settings["human_elo"]),
        active_clock_s=float(active_clock_s),
        opponent_clock_s=float(opponent_clock_s),
        active_inc_s=float(active_inc_s),
        opponent_inc_s=float(opponent_inc_s),
        tc_base_s=(float(settings.get("start_clock_s")) if settings.get("start_clock_s") is not None else None),
        halfmove_clock=int(board.halfmove_clock),
    )

    mcts_stats = None
    use_mcts = bool(settings.get("use_mcts", False))
    mcts_start_ply = int(settings.get("mcts_start_ply", 0))
    current_ply = len(moves_uci)
    
    # Check if we should use MCTS (enabled and past start ply)
    if use_mcts and current_ply >= mcts_start_ply:
        sims = int(settings.get("mcts_simulations", 128))
        cpuct = float(settings.get("mcts_c_puct", 1.5))

        if bool(settings.get("mcts_adaptive", False)):
            scale = float(settings.get("mcts_adaptive_scale", 100.0))
            base_sims = int(settings.get("mcts_simulations", 128))
            
            # Adaptive sims: scale * predicted_time
            sims = max(16, int(scale * engine_pred_time_s))
            
            # Adaptive cpuct: scale by sqrt(sims / base_sims) to maintain constant regularization strength
            # relative to the policy as the search budget changes.
            # We clamp the multiplier to avoid extreme values.
            sim_ratio = sims / max(1, base_sims)
            cpuct_scale = math.sqrt(sim_ratio)
            cpuct = cpuct * cpuct_scale
            
            # Safety clamp
            cpuct = max(0.1, min(8.0, cpuct))

        mcts_settings = MCTSSettings(
            simulations=sims,
            c_puct=cpuct,
            max_children=int(settings.get("mcts_max_children", 48)),
            max_depth=int(settings.get("mcts_max_depth", 96)),
            root_dirichlet_alpha=float(settings.get("mcts_root_dirichlet_alpha", 0.0)),
            root_exploration_frac=float(settings.get("mcts_root_exploration_frac", 0.0)),
            final_temperature=float(settings.get("mcts_final_temperature", 0.0)),
            final_top_p=float(settings.get("mcts_final_top_p", 1.0)),
            fpu_reduction=float(settings.get("mcts_fpu_reduction", 0.0)),
            contempt=float(settings.get("mcts_contempt", 0.15)),
            tree_reuse=bool(settings.get("mcts_tree_reuse", False)),
            leaf_batch_size=int(settings.get("mcts_leaf_batch_size", 8)),
        )

        # Track MCTS execution time for timing simulation
        mcts_start_time = time.time()
        
        # Try to find a reusable subtree from the previous search
        reuse_root: _Node | None = None
        if mcts_settings.tree_reuse and mcts_reuse_root is not None and mcts_reuse_moves:
            reuse_root = find_subtree_by_move_sequence(mcts_reuse_root, mcts_reuse_moves)
        
        mcts_result = mcts_choose_move(
            model=model,
            board=board,
            ctx=ctx,
            time_history_s=time_history_s,
            device=device,
            settings=mcts_settings,
            rng=rng,
            stop_check=stop_check,
            claim_draw=False,
            progress_callback=mcts_progress_callback,
            reuse_root=reuse_root,
        )
        out = mcts_result.output
        mcts_stats = out.stats
        
        mcts_elapsed_s = time.time() - mcts_start_time
        
        # Simulate remaining thinking time after MCTS if enabled
        if bool(settings.get("mcts_simulate_time", False)):
            remaining_time = engine_pred_time_s - mcts_elapsed_s
            if remaining_time > 0:
                time.sleep(remaining_time)
        
        return out, engine_stats, mcts_stats, mcts_result
    else:
        _fb, bh, rf = build_history_from_position(chess.Board(), moves_uci)
        
        # Use opening temperature during opening phase for more varied openings
        opening_length = int(settings.get("opening_length", 10))
        if current_ply < opening_length:
            effective_temperature = float(settings.get("opening_temperature", settings["temperature"]))
        else:
            effective_temperature = float(settings["temperature"])
        
        out = choose_move(
            model=model,
            board=board,
            board_history=bh,
            repetition_flags=rf,
            ctx=ctx,
            temperature=effective_temperature,
            top_p=float(settings["top_p"]),
            time_history_s=time_history_s,
            device=device,
            rng=rng,
        )

    return out, engine_stats, mcts_stats, None
