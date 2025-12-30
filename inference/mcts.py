from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Optional, Tuple

import numpy as np
import time
import torch
import chess

from inference.chessformer_policy import PolicyOutput
from inference.encoding import ContextOptions, HISTORY_LEN, canonicalize, encode_board, make_model_batch


_PROMO_INDEX = {chess.QUEEN: 0, chess.ROOK: 1, chess.BISHOP: 2, chess.KNIGHT: 3}


def _mirror_square(sq: int) -> int:
    return sq ^ 56


def _canonical_to_real_move(move: chess.Move, real_turn: chess.Color) -> chess.Move:
    if real_turn == chess.WHITE:
        return move
    return chess.Move(
        _mirror_square(move.from_square),
        _mirror_square(move.to_square),
        promotion=move.promotion,
    )


def _terminal_value_for_side_to_move(board: chess.Board) -> Optional[float]:
    """Returns value from perspective of side to move, or None if non-terminal."""
    if not board.is_game_over(claim_draw=True):
        return None

    outcome = board.outcome(claim_draw=True)
    if outcome is None or outcome.winner is None:
        return 0.0

    return 1.0 if outcome.winner == board.turn else -1.0


def _history_from_board(board: chess.Board, *, history_len: int = HISTORY_LEN) -> Tuple[list[list[int]], list[int]]:
    """Build (board_history, repetition_flags) newest-first, matching training."""
    b = board.copy(stack=True)

    boards: list[chess.Board] = []  # newest-first, canonicalized
    for _ in range(history_len):
        boards.append(canonicalize(b).copy(stack=False))
        if not b.move_stack:
            break
        b.pop()

    board_history = [encode_board(x) for x in boards]
    while len(board_history) < history_len:
        board_history.append([0] * 64)
    board_history = board_history[:history_len]

    seen: dict[str, int] = {}
    flags: list[int] = []
    for bb in boards:
        key = bb.board_fen()
        seen[key] = seen.get(key, 0) + 1
        flags.append(1 if seen[key] >= 2 else 0)

    while len(flags) < history_len:
        flags.insert(0, 0)
    flags = flags[:history_len]

    return board_history, flags


def _ctx_for_side_to_move(base: ContextOptions, *, root_turn: chess.Color, node_turn: chess.Color, halfmove_clock: int) -> ContextOptions:
    """Return a ContextOptions aligned to the side-to-move at the node.

    `base` is assumed to be correct for the root side-to-move.
    When the node side-to-move flips, we must swap active/opponent features.
    """

    if node_turn == root_turn:
        return ContextOptions(
            active_elo=base.active_elo,
            opponent_elo=base.opponent_elo,
            active_clock_s=base.active_clock_s,
            opponent_clock_s=base.opponent_clock_s,
            active_inc_s=base.active_inc_s,
            opponent_inc_s=base.opponent_inc_s,
            halfmove_clock=halfmove_clock,
        )

    return ContextOptions(
        active_elo=base.opponent_elo,
        opponent_elo=base.active_elo,
        active_clock_s=base.opponent_clock_s,
        opponent_clock_s=base.active_clock_s,
        active_inc_s=base.opponent_inc_s,
        opponent_inc_s=base.active_inc_s,
        halfmove_clock=halfmove_clock,
    )


def _time_bin_mid_seconds(bin_idx: torch.Tensor, *, active_clock_s: torch.Tensor) -> torch.Tensor:
    """Map time-bin indices to seconds using the same sqrt-scaling as training.

    Training encoding uses: scaled = sqrt(time_ratio) in [0,1], discretized into 256 bins.
    Bin midpoint in scaled-space is (i+0.5)/256, so time_ratio = scaled^2.
    """

    scaled_mid = (bin_idx + 0.5) / 256.0
    return (scaled_mid * scaled_mid) * torch.clamp(active_clock_s, min=1e-6)


def _evaluate_position(
    *,
    model: torch.nn.Module,
    board: chess.Board,
    ctx: ContextOptions,
    root_turn: chess.Color,
    time_history_s: list[float] | None,
    device: torch.device,
    contempt: float = 0.0,
) -> Tuple[Dict[chess.Move, float], float, float]:
    """Returns (priors over legal real moves, value for side-to-move, expected ponder time seconds)."""

    board_history, repetition_flags = _history_from_board(board)

    eff_ctx = _ctx_for_side_to_move(ctx, root_turn=root_turn, node_turn=board.turn, halfmove_clock=int(board.halfmove_clock))

    batch = make_model_batch(
        board=board,
        board_history=board_history,
        repetition_flags=repetition_flags,
        time_history_s=time_history_s,
        ctx=eff_ctx,
        device=device,
    )

    with torch.inference_mode():
        with torch.autocast(device_type=device.type, enabled=(device.type == "cuda")):
            move_logits, _v_raw, v_cls, _v_err, t_logits, _ss_logits, promo_logits = model(batch, return_promo=True)

    move_logits = move_logits[0]  # (4098,)
    promo_p = torch.softmax(promo_logits[0].float(), dim=-1)  # (8,4)

    # NOTE: The model's WDL head is ordered as [Loss, Draw, Win].
    wdl = torch.softmax(v_cls[0].float(), dim=-1)
    # Apply contempt: penalize draws to avoid drawish positions when ahead
    # value = P(win) - P(loss) - contempt * P(draw)
    value = float((wdl[2] - wdl[0] - contempt * wdl[1]).item())

    # Deterministic expected ponder time (seconds) for the side-to-move at this node.
    # This is used to fill simulated time_history entries downstream.
    t_probs = torch.softmax(t_logits[0].float(), dim=-1)  # (256,)
    bin_idx = torch.arange(t_probs.shape[0], device=t_probs.device, dtype=torch.float32)
    active_clock = torch.tensor(float(eff_ctx.active_clock_s), device=t_probs.device, dtype=torch.float32)
    bin_seconds = _time_bin_mid_seconds(bin_idx, active_clock_s=active_clock)
    exp_time_s = float(torch.sum(t_probs * bin_seconds).item())

    real_turn = board.turn
    canonical = canonicalize(board)

    logits: list[float] = []
    moves: list[chess.Move] = []

    for mv in canonical.legal_moves:
        base_idx = mv.from_square * 64 + mv.to_square
        logit = float(move_logits[base_idx].item())
        if mv.promotion is not None and 56 <= mv.to_square <= 63:
            file_idx = mv.to_square - 56
            p_idx = _PROMO_INDEX.get(mv.promotion, 0)
            promo_prob = float(promo_p[file_idx, p_idx].item())
            logit += float(np.log(max(1e-8, promo_prob)))

        real_mv = _canonical_to_real_move(mv, real_turn)
        moves.append(real_mv)
        logits.append(logit)

    if not moves:
        return {}, value, exp_time_s

    x = np.array(logits, dtype=np.float64)
    x = x - float(np.max(x))
    probs = np.exp(x)
    probs = probs / float(np.sum(probs) + 1e-12)

    priors: Dict[chess.Move, float] = {mv: float(p) for mv, p in zip(moves, probs)}
    return priors, value, exp_time_s


def _evaluate_positions_batch(
    *,
    model: torch.nn.Module,
    boards: list[chess.Board],
    ctx: ContextOptions,
    root_turn: chess.Color,
    time_histories_s: list[list[float] | None],
    device: torch.device,
    contempt: float = 0.0,
) -> Tuple[list[Dict[chess.Move, float]], list[float], list[float]]:
    """Batched version of `_evaluate_position`.

    Returns:
        priors_list: list of dicts mapping real legal moves -> prob
        values: list of values from perspective of side-to-move at each board
    """

    if len(boards) != len(time_histories_s):
        raise ValueError("boards and time_histories_s must have same length")
    if not boards:
        return [], []

    batches: list[dict[str, torch.Tensor]] = []
    canonicals: list[chess.Board] = []
    real_turns: list[chess.Color] = []

    for b, th in zip(boards, time_histories_s):
        board_history, repetition_flags = _history_from_board(b)
        eff_ctx = _ctx_for_side_to_move(ctx, root_turn=root_turn, node_turn=b.turn, halfmove_clock=int(b.halfmove_clock))
        batches.append(
            make_model_batch(
                board=b,
                board_history=board_history,
                repetition_flags=repetition_flags,
                time_history_s=th,
                ctx=eff_ctx,
                device=device,
            )
        )
        real_turns.append(b.turn)
        canonicals.append(canonicalize(b))

    merged: dict[str, torch.Tensor] = {}
    keys = list(batches[0].keys())
    for k in keys:
        merged[k] = torch.cat([bb[k] for bb in batches], dim=0)

    with torch.inference_mode():
        with torch.autocast(device_type=device.type, enabled=(device.type == "cuda")):
            move_logits, _v_raw, v_cls, _v_err, t_logits, _ss_logits, promo_logits = model(merged, return_promo=True)

    promo_p = torch.softmax(promo_logits.float(), dim=-1)  # (B,8,4)
    wdl = torch.softmax(v_cls.float(), dim=-1)  # (B,3) ordered [L,D,W]
    # Apply contempt: value = P(win) - P(loss) - contempt * P(draw)
    values = (wdl[:, 2] - wdl[:, 0] - contempt * wdl[:, 1]).detach().cpu().numpy().astype(np.float64).tolist()

    # Deterministic expected ponder time for each board (seconds).
    # Need per-board active_clock_s after side-to-move alignment.
    # eff_ctx is already used in make_model_batch; recompute active clocks here from ctx + turns.
    active_clocks: list[float] = []
    for b in boards:
        eff_ctx = _ctx_for_side_to_move(ctx, root_turn=root_turn, node_turn=b.turn, halfmove_clock=int(b.halfmove_clock))
        active_clocks.append(float(eff_ctx.active_clock_s))

    t_probs = torch.softmax(t_logits.float(), dim=-1)  # (B,256)
    bin_idx = torch.arange(t_probs.shape[-1], device=t_probs.device, dtype=torch.float32).view(1, -1)
    active_clock_t = torch.tensor(active_clocks, device=t_probs.device, dtype=torch.float32).view(-1, 1)
    bin_seconds = _time_bin_mid_seconds(bin_idx, active_clock_s=active_clock_t)
    exp_times = torch.sum(t_probs * bin_seconds, dim=-1).detach().cpu().numpy().astype(np.float64).tolist()

    priors_list: list[Dict[chess.Move, float]] = []

    for i, canonical in enumerate(canonicals):
        logits: list[float] = []
        moves: list[chess.Move] = []

        for mv in canonical.legal_moves:
            base_idx = mv.from_square * 64 + mv.to_square
            logit = float(move_logits[i, base_idx].item())
            if mv.promotion is not None and 56 <= mv.to_square <= 63:
                file_idx = mv.to_square - 56
                p_idx = _PROMO_INDEX.get(mv.promotion, 0)
                promo_prob = float(promo_p[i, file_idx, p_idx].item())
                logit += float(np.log(max(1e-8, promo_prob)))

            real_mv = _canonical_to_real_move(mv, real_turns[i])
            moves.append(real_mv)
            logits.append(logit)

        if not moves:
            priors_list.append({})
            continue

        x = np.array(logits, dtype=np.float64)
        x = x - float(np.max(x))
        probs = np.exp(x)
        probs = probs / float(np.sum(probs) + 1e-12)
        priors_list.append({mv: float(p) for mv, p in zip(moves, probs)})

    return priors_list, [float(v) for v in values], [float(t) for t in exp_times]


@dataclass
class MCTSSettings:
    simulations: int = 256
    c_puct: float = 1.5
    max_depth: int = 128
    max_children: int = 64
    root_dirichlet_alpha: float = 0.0
    root_exploration_frac: float = 0.0
    final_temperature: float = 0.0
    final_top_p: float = 0.90  # Top-p (nucleus) sampling for final move selection

    # Contempt: adjusts how draws are valued relative to wins/losses.
    # value = P(win) - P(loss) - contempt * P(draw)
    # Positive contempt penalizes draws, making the engine prefer sharper positions
    # when it has an advantage. Typical values: 0.0 (neutral), 0.1-0.3 (slight contempt).
    # Leela uses ~0.1 by default when playing weaker opponents.
    contempt: float = 0.15

    # First Play Urgency (FPU): for unvisited children, use (parent_q - fpu_reduction)
    # as the value term (in the *parent player's* perspective) instead of implicitly using 0.0.
    # Set to 0.0 to keep legacy behavior.
    fpu_reduction: float = 0.20

    # Leaf parallelism
    leaf_batch_size: int = 1
    virtual_loss: float = 1.0

    # Tree reuse: if True, preserve tree across searches for the same game
    tree_reuse: bool = False


@dataclass
class MCTSResult:
    """Extended result from MCTS search, including tree for potential reuse."""
    output: "PolicyOutput"
    root: "_Node"
    root_fen: str  # FEN of the position where search was run
    chosen_move: Optional[chess.Move]  # The move that was chosen


class _Node:
    __slots__ = ("prior", "visit_count", "value_sum", "children", "expanded", "pred_time_s")

    def __init__(self, prior: float) -> None:
        self.prior = float(prior)
        self.visit_count = 0
        self.value_sum = 0.0
        self.children: Dict[chess.Move, _Node] = {}
        self.expanded = False
        self.pred_time_s = 0.0

    def q(self) -> float:
        return 0.0 if self.visit_count == 0 else self.value_sum / self.visit_count


def find_subtree_for_position(
    last_root: Optional["_Node"],
    last_root_fen: str,
    last_chosen_move: Optional[chess.Move],
    current_board: chess.Board,
) -> Optional["_Node"]:
    """Find a reusable subtree from the previous search.
    
    After we play `last_chosen_move` and opponent plays their response,
    we want to find the grandchild node corresponding to:
    last_root -> our move -> opponent's move
    
    Returns the subtree root if found, None otherwise.
    """
    if last_root is None or last_chosen_move is None:
        return None
    
    if not last_root.expanded or not last_root.children:
        return None
    
    # Find our move's child
    if last_chosen_move not in last_root.children:
        return None
    
    our_child = last_root.children[last_chosen_move]
    
    if not our_child.expanded or not our_child.children:
        return None
    
    # The current position should match after our move + opponent's move
    # We need to figure out what opponent's move was by comparing positions
    # Try each of our_child's children and see if any matches current_board
    for opp_move, grandchild in our_child.children.items():
        # Reconstruct what the position would be after this opponent move
        # by checking if grandchild's legal moves match current position
        if grandchild.expanded and grandchild.children:
            # Check if the grandchild's children match current legal moves
            grandchild_moves = set(grandchild.children.keys())
            current_legal = set(current_board.legal_moves)
            # If there's good overlap, this is likely the right subtree
            if grandchild_moves and grandchild_moves & current_legal:
                overlap = len(grandchild_moves & current_legal) / max(1, len(current_legal))
                if overlap > 0.5:  # At least 50% overlap
                    return grandchild
    
    # Alternative: use FEN matching if we track FENs in nodes
    # For now, return None if we can't confidently identify the subtree
    return None


def find_subtree_by_move_sequence(
    last_root: Optional["_Node"],
    moves_since_last_search: list[chess.Move],
) -> Optional["_Node"]:
    """Find a reusable subtree by following a sequence of moves from the last root.
    
    Args:
        last_root: The root node from the previous search
        moves_since_last_search: List of moves played since the last search
                                 (typically [our_move, opponent_move])
    
    Returns the subtree root if found, None otherwise.
    """
    if last_root is None or not moves_since_last_search:
        return None
    
    node = last_root
    for move in moves_since_last_search:
        if not node.expanded or not node.children:
            return None
        if move not in node.children:
            return None
        node = node.children[move]
    
    # Return the node only if it has been expanded (has some search info)
    if node.expanded:
        return node
    return None


def _select_child(node: _Node, *, c_puct: float, fpu_reduction: float) -> Tuple[chess.Move, _Node]:
    assert node.children
    sqrt_n = np.sqrt(max(1, node.visit_count))

    parent_q = node.q()
    use_fpu = float(fpu_reduction) > 0.0

    best_move: Optional[chess.Move] = None
    best_child: Optional[_Node] = None
    best_score = -1e30

    for mv, child in node.children.items():
        u = c_puct * child.prior * (sqrt_n / (1 + child.visit_count))
        # `child.q()` is stored from the perspective of the player *to move at the child*.
        # At the parent, we choose moves for the current player, so we negate.
        if use_fpu and child.visit_count == 0:
            # FPU uses a pessimistic initial value for unvisited children.
            # parent_q is from the parent's side-to-move; that's the value perspective we
            # want for move selection at this node.
            v_term = float(parent_q - fpu_reduction)
        else:
            v_term = float(-child.q())

        score = v_term + u
        if score > best_score:
            best_score = score
            best_move = mv
            best_child = child

    assert best_move is not None and best_child is not None
    return best_move, best_child


def _apply_root_noise(
    priors: Dict[chess.Move, float],
    *,
    alpha: float,
    frac: float,
    rng: np.random.Generator,
) -> Dict[chess.Move, float]:
    if frac <= 0.0 or alpha <= 0.0 or len(priors) <= 1:
        return priors

    moves = list(priors.keys())
    p = np.array([priors[m] for m in moves], dtype=np.float64)
    p = p / float(p.sum() + 1e-12)

    noise = rng.dirichlet([alpha] * len(moves))
    mixed = (1.0 - frac) * p + frac * noise
    mixed = mixed / float(mixed.sum() + 1e-12)

    return {m: float(pp) for m, pp in zip(moves, mixed)}


def _prune_priors(priors: Dict[chess.Move, float], *, max_children: int) -> Dict[chess.Move, float]:
    if max_children <= 0 or len(priors) <= max_children:
        return priors

    items = sorted(priors.items(), key=lambda kv: kv[1], reverse=True)[:max_children]
    s = float(sum(p for _, p in items))
    if s <= 0:
        return {mv: 1.0 / len(items) for mv, _ in items}
    return {mv: float(p / s) for mv, p in items}


def mcts_choose_move(
    *,
    model: torch.nn.Module,
    board: chess.Board,
    ctx: ContextOptions,
    time_history_s: list[float] | None,
    device: torch.device,
    settings: MCTSSettings,
    rng: Optional[np.random.Generator] = None,
    stop_check: Callable[[], bool] | None = None,
    claim_draw: bool = False,
    progress_callback: Callable[[dict], None] | None = None,
    reuse_root: Optional["_Node"] = None,
) -> "MCTSResult":
    """PUCT MCTS using (policy priors, WDL value) from the model.
    
    If `reuse_root` is provided and matches the current position's children,
    it will be used as the starting point instead of building a fresh tree.
    
    Returns an MCTSResult containing the PolicyOutput, root node, and chosen move
    for potential tree reuse in subsequent searches.
    """

    if rng is None:
        rng = np.random.default_rng()

    # Match `inference/app.py` behavior: keep playing in claimable-draw positions.
    if board.is_game_over(claim_draw=bool(claim_draw)):
        return MCTSResult(
            output=PolicyOutput(move=None, policy_prob=1.0),
            root=_Node(1.0),
            root_fen=board.fen(),
            chosen_move=None,
        )

    root_turn = board.turn
    root_fen = board.fen()

    # Check if we can reuse the provided root
    reused = False
    if reuse_root is not None and reuse_root.expanded and reuse_root.children:
        # Verify that the reuse_root's children match legal moves in this position
        legal_moves = set(board.legal_moves)
        reuse_moves = set(reuse_root.children.keys())
        # Allow reuse if there's substantial overlap (some children may have been pruned)
        if reuse_moves & legal_moves:
            root = reuse_root
            reused = True
            # Get root value from existing tree (estimate from children)
            if root.visit_count > 0:
                root_value = root.q()
            else:
                root_value = 0.0
            root_time_s = root.pred_time_s
        else:
            reuse_root = None

    if not reused:
        root = _Node(1.0)

        # Root expansion (only if not reusing)
        priors, root_value, root_time_s = _evaluate_position(
            model=model,
            board=board,
            ctx=ctx,
            root_turn=root_turn,
            time_history_s=time_history_s,
            device=device,
            contempt=float(settings.contempt),
        )
        root.pred_time_s = float(root_time_s)

        # Immediate mate check (cheap tactical sanity)
        for mv in board.legal_moves:
            b2 = board.copy(stack=False)
            b2.push(mv)
            tv = _terminal_value_for_side_to_move(b2)
            if tv is not None and tv == -1.0:
                # After our move, opponent-to-move is losing => we delivered mate.
                return MCTSResult(
                    output=PolicyOutput(move=mv, policy_prob=1.0),
                    root=root,
                    root_fen=root_fen,
                    chosen_move=mv,
                )

        priors = _prune_priors(priors, max_children=int(settings.max_children))
        priors = _apply_root_noise(
            priors,
            alpha=float(settings.root_dirichlet_alpha),
            frac=float(settings.root_exploration_frac),
            rng=rng,
        )

        for mv, p in priors.items():
            root.children[mv] = _Node(p)
        root.expanded = True
    else:
        # When reusing, still check for immediate mate
        for mv in board.legal_moves:
            b2 = board.copy(stack=False)
            b2.push(mv)
            tv = _terminal_value_for_side_to_move(b2)
            if tv is not None and tv == -1.0:
                return MCTSResult(
                    output=PolicyOutput(move=mv, policy_prob=1.0),
                    root=root,
                    root_fen=root_fen,
                    chosen_move=mv,
                )

    # Run simulations
    sims = max(1, int(settings.simulations))
    max_depth = max(1, int(settings.max_depth))
    leaf_batch_size = max(1, int(getattr(settings, "leaf_batch_size", 1)))
    vloss = float(getattr(settings, "virtual_loss", 1.0))

    # Lightweight diagnostics counters (minimal overhead)
    done = 0
    node_count = 1 + len(root.children)  # count the root + immediate root children
    start_time = time.time()
    while done < sims:
        if stop_check is not None and stop_check():
            break

        batch_n = min(leaf_batch_size, sims - done)

        # Select up to `batch_n` leaves while applying virtual loss to discourage collisions.
        leaves: list[tuple[list[_Node], _Node, chess.Board, list[float] | None, Optional[float]]] = []
        for _ in range(batch_n):
            if stop_check is not None and stop_check():
                break
            node = root
            b = board.copy(stack=True)
            th = list(time_history_s) if time_history_s is not None else None
            path: list[_Node] = [node]

            depth = 0
            while node.expanded and node.children and depth < max_depth:
                mv, child = _select_child(
                    node,
                    c_puct=float(settings.c_puct),
                    fpu_reduction=float(getattr(settings, "fpu_reduction", 0.0)),
                )
                b.push(mv)
                if th is not None:
                    th = [float(getattr(node, "pred_time_s", 0.0))] + th[:-1]
                node = child
                path.append(node)
                depth += 1

                tv = _terminal_value_for_side_to_move(b)
                if tv is not None:
                    break

            tv = _terminal_value_for_side_to_move(b)

            # Apply virtual loss along the selected path (from each node's side-to-move perspective).
            # This makes subsequent selections in this batch less likely to pick the same line.
            for n in path:
                n.visit_count += 1
                n.value_sum += -vloss

            leaves.append((path, node, b, th, tv))

        # Evaluate all non-terminal leaves in a single forward pass.
        eval_boards: list[chess.Board] = []
        eval_th: list[list[float] | None] = []
        eval_map: list[int] = []
        for i, (_path, _node, b, th, tv) in enumerate(leaves):
            if tv is None:
                eval_boards.append(b)
                eval_th.append(th)
                eval_map.append(i)

        eval_priors: list[Dict[chess.Move, float]] = []
        eval_values: list[float] = []
        eval_times: list[float] = []
        if eval_boards:
            eval_priors, eval_values, eval_times = _evaluate_positions_batch(
                model=model,
                boards=eval_boards,
                ctx=ctx,
                root_turn=root_turn,
                time_histories_s=eval_th,
                device=device,
                contempt=float(settings.contempt),
            )

        # Commit expansions + backups, removing virtual loss first.
        eval_idx = 0
        for i, (path, leaf_node, _b, _th, tv) in enumerate(leaves):
            if tv is not None:
                leaf_value = float(tv)
            else:
                priors2 = eval_priors[eval_idx]
                leaf_value = float(eval_values[eval_idx])
                leaf_time_s = float(eval_times[eval_idx])
                eval_idx += 1

                leaf_node.pred_time_s = leaf_time_s

                priors2 = _prune_priors(priors2, max_children=int(settings.max_children))
                if not leaf_node.expanded:
                    for mv2, p2 in priors2.items():
                        leaf_node.children[mv2] = _Node(p2)
                    leaf_node.expanded = True
                    node_count += len(priors2)

            # Remove the virtual loss applied during selection.
            for n in path:
                n.visit_count -= 1
                n.value_sum -= -vloss

            # Backup real evaluation (flip perspective each ply)
            v = leaf_value
            for n in reversed(path):
                n.visit_count += 1
                n.value_sum += v
                v = -v

        if not leaves:
            break

        done += len(leaves)
        
        # Call progress callback with current stats
        if progress_callback is not None and root.children:
            try:
                progress_stats = []
                for m in root.children:
                    child = root.children[m]
                    progress_stats.append({
                        "move": m.uci(),
                        "visits": child.visit_count,
                        "q": -child.q() if child.visit_count > 0 else 0.0,
                        "prior": child.prior
                    })
                progress_stats.sort(key=lambda x: x["visits"], reverse=True)
                progress_callback({
                    "done": done,
                    "total": sims,
                    "root_value": root_value,
                    "children": progress_stats[:20]  # Limit to top 20 for efficiency
                })
            except Exception:
                pass  # Progress callback must never break MCTS

    if not root.children:
        return MCTSResult(
            output=PolicyOutput(move=None, policy_prob=1.0),
            root=root,
            root_fen=root_fen,
            chosen_move=None,
        )

    # Pick move from visit counts
    moves = list(root.children.keys())
    visits = np.array([root.children[m].visit_count for m in moves], dtype=np.float64)

    # Collect stats
    children_stats = []
    for m in moves:
        child = root.children[m]
        # child.q() is value for the player to move at the child node (opponent of root).
        # We negate it to get value for root player.
        children_stats.append({
            "move": m.uci(),
            "visits": child.visit_count,
            "q": -child.q(),
            "prior": child.prior
        })
    
    # Sort by visits descending
    children_stats.sort(key=lambda x: x["visits"], reverse=True)
    
    stats = {
        "root_value": root_value,
        "children": children_stats
    }
    # Add lightweight MCTS diagnostics
    try:
        elapsed = max(1e-6, time.time() - start_time)
        stats["mcts_nodes"] = int(node_count)
        stats["mcts_nodes_per_s"] = float(node_count) / float(elapsed)
    except Exception:
        # Be conservative: diagnostics must never raise
        stats["mcts_nodes"] = 0
        stats["mcts_nodes_per_s"] = 0.0

    total = float(visits.sum())
    if total <= 0:
        mv = moves[int(np.argmax([root.children[m].prior for m in moves]))]
        return MCTSResult(
            output=PolicyOutput(move=mv, policy_prob=1.0 / len(moves), stats=stats),
            root=root,
            root_fen=root_fen,
            chosen_move=mv,
        )

    if float(settings.final_temperature) <= 0.0:
        idx = int(np.argmax(visits))
        mv = moves[idx]
        return MCTSResult(
            output=PolicyOutput(move=mv, policy_prob=float(visits[idx] / total), stats=stats),
            root=root,
            root_fen=root_fen,
            chosen_move=mv,
        )

    t = float(settings.final_temperature)
    top_p = float(getattr(settings, "final_top_p", 1.0))
    
    # Temperature-adjusted distribution from visit counts.
    w = np.power(visits / total, 1.0 / max(1e-6, t))

    # Defensively sanitize the probability vector: replace NaN/inf, clamp negatives,
    # and fall back to using visit-based weights or uniform sampling if necessary.
    w = np.nan_to_num(w, nan=0.0, posinf=0.0, neginf=0.0)
    w[w < 0.0] = 0.0

    w_sum = float(w.sum())
    if not np.isfinite(w_sum) or w_sum <= 0.0:
        # Fallback: prefer non-negative visit counts if available, else uniform.
        vpos = np.nan_to_num(visits, nan=0.0, posinf=0.0, neginf=0.0)
        vpos[vpos < 0.0] = 0.0
        vsum = float(vpos.sum())
        if np.isfinite(vsum) and vsum > 0.0:
            w = vpos / vsum
        else:
            # Last resort: uniform over moves
            w = np.ones_like(visits, dtype=np.float64) / float(len(visits))
    else:
        w = w / w_sum

    # Apply top-p (nucleus) sampling: keep only moves with cumulative prob <= top_p
    if 0.0 < top_p < 1.0:
        sorted_indices = np.argsort(w)[::-1]  # descending order
        sorted_probs = w[sorted_indices]
        cumsum = np.cumsum(sorted_probs)
        # Keep moves up to and including the one that crosses top_p threshold
        cutoff_idx = np.searchsorted(cumsum, top_p, side='right')
        cutoff_idx = max(1, min(cutoff_idx + 1, len(w)))  # keep at least 1, include threshold-crossing move
        mask = np.zeros_like(w, dtype=bool)
        mask[sorted_indices[:cutoff_idx]] = True
        w = np.where(mask, w, 0.0)
        w = w / w.sum()  # renormalize

    idx = int(rng.choice(len(moves), p=w))
    mv = moves[idx]
    return MCTSResult(
        output=PolicyOutput(move=mv, policy_prob=float(w[idx]), stats=stats),
        root=root,
        root_fen=root_fen,
        chosen_move=mv,
    )
