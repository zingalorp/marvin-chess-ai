"""
Chessformer - Human-like chess transformer model.

Architecture features:
- Per-token input encoding (board history + metadata per square)
- Absolute position embeddings (per-square learned embeddings)
- Policy head with 1/sqrt(d_model) scaling
- Legal move masking
- Resign/flag as policy outputs (indices 4096, 4097)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint as gradient_checkpoint

# Special policy indices
RESIGN_MOVE_INDEX = 4096
FLAG_MOVE_INDEX = 4097
NUM_POLICY_OUTPUTS = 4098

# Classification head dimensions
NUM_TIME_BINS = 256  # Time prediction as 256-class classification
NUM_VALUE_CLASSES = 3  # WDL: Win/Draw/Loss
NUM_SQUARES = 64


class Mish(nn.Module):
    """Mish activation: x * tanh(softplus(x)). Better gradient flow than ReLU/SiLU."""
    def forward(self, x):
        return F.mish(x)


# =============================================================================
# MODEL CONFIGURATIONS
# =============================================================================
# All configs use token-based conditioning (7 tokens prepended to 64 square tokens)
# for ELO, opponent ELO, time control, urgency, increment, and move timing.
# Per-square encoding handles ONLY board state (piece history, EP, castling, rep flags).

# Large config (~49M params)
CONFIG_LARGE = {
    "d_model": 608,
    "n_layers": 16,
    "n_heads": 19,
    "d_head": 32,
    "d_ff": 608,              # 1.0x d_model (consistent with Small/Tiny)
    "dropout": 0.1,
    "max_rel_dist": 7,
    "history_len": 8,
    "num_piece_types": 13,
    "num_tc_cats": 3,
    "embedding_ffn": True,
    "smolgen": True,
    "smolgen_hidden": 256,
    "smolgen_per_head": 128,
}

# Small config (~23M params)
CONFIG_SMALL = {
    "d_model": 448,
    "n_layers": 12,
    "n_heads": 14,
    "d_head": 32,
    "d_ff": 448,
    "dropout": 0.1,
    "max_rel_dist": 7,
    "history_len": 8,
    "num_piece_types": 13,
    "num_tc_cats": 3,
    "embedding_ffn": True,
    "smolgen": True,
    "smolgen_hidden": 256,
    "smolgen_per_head": 256,
}

# Tiny config (~5M params)
CONFIG_TINY = {
    "d_model": 256,
    "n_layers": 6,
    "n_heads": 8,
    "d_head": 32,
    "d_ff": 256,
    "dropout": 0.1,
    "max_rel_dist": 7,
    "history_len": 8,
    "num_piece_types": 13,
    "num_tc_cats": 3,
    "embedding_ffn": True,
    "smolgen": True,
    "smolgen_hidden": 128,
    "smolgen_per_head": 64,
}

# Small-NoTime config (~23M params, same architecture as Small but without any time context)
# Removes all time-related conditioning: TC category, remaining clock, increment,
# my last move time, opponent last move time. Only ELO conditioning token is kept.
CONFIG_SMALL_NOTIME = {
    "d_model": 448,
    "n_layers": 12,
    "n_heads": 14,
    "d_head": 32,
    "d_ff": 448,
    "dropout": 0.1,
    "max_rel_dist": 7,
    "history_len": 8,
    "num_piece_types": 13,
    "num_tc_cats": 3,
    "embedding_ffn": True,
    "smolgen": True,
    "smolgen_hidden": 256,
    "smolgen_per_head": 256,
    "no_time_context": True,
}

# Small-NoTime-NoHist config (~23M params, same as Small-NoTime but without board history)
# No time context AND no board history — only the current position is seen.
# Ablation to measure the contribution of move-history information.
CONFIG_SMALL_NOTIME_NOHIST = {
    "d_model": 448,
    "n_layers": 12,
    "n_heads": 14,
    "d_head": 32,
    "d_ff": 448,
    "dropout": 0.1,
    "max_rel_dist": 7,
    "history_len": 1,
    "num_piece_types": 13,
    "num_tc_cats": 3,
    "embedding_ffn": True,
    "smolgen": True,
    "smolgen_hidden": 256,
    "smolgen_per_head": 256,
    "no_time_context": True,
    "no_board_history": True,
}

class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization."""
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x):
        rms = torch.sqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return x / rms * self.scale


# ============================================================================
# Token-Based Conditioning
# ============================================================================

# Bins for log-scaled time: 16 bins covering 0-1800s (30 min)
# Boundaries at approximately: 0, 1, 2, 4, 7, 12, 20, 35, 60, 100, 180, 300, 500, 900, 1800
NUM_TIME_LOG_BINS = 16
NUM_ELO_ANCHORS = 14  # 1200-2500 in 100 ELO increments
NUM_TC_CATEGORIES = 3  # Blitz, Rapid, Classical
NUM_INC_CATEGORIES = 5  # 0, 1, 2, 5, 10+
NUM_CONDITIONING_TOKENS = 7  # ELO, OPP_ELO, TC, URGENCY, INC, MY_TIME, OPP_TIME
NUM_CONDITIONING_TOKENS_ELO_ONLY = 2  # ELO + OPP_ELO only (no time context)


def log_bin_time(seconds: torch.Tensor, num_bins: int = NUM_TIME_LOG_BINS) -> torch.Tensor:
    """
    Bin time values logarithmically. Returns bin indices in [0, num_bins-1].
    
    Covers 0 to ~1800 seconds (30 min) with denser bins at lower times.
    Uses log1p for smooth handling of zero.
    
    ONNX-compatible: uses torch.clamp with min/max tensors to avoid type mismatch.
    """
    # log1p(1800) ≈ 7.5, so divide by 7.5 and scale to bins
    log_time = torch.log1p(torch.clamp(seconds, min=0.0))
    # Normalize to [0, 1] range, then to bin indices
    normalized = log_time / 7.5  # Max expected log1p value
    # Use explicit min/max for ONNX compatibility
    bins = (torch.clamp(normalized, min=0.0, max=0.9999) * num_bins).long()
    return bins


def bin_increment(inc: torch.Tensor) -> torch.Tensor:
    """
    Bin increment values into 5 categories: 0, 1, 2, 5, 10+.
    Returns indices 0-4.
    
    ONNX-compatible: uses arithmetic instead of torch.tensor() in traced context.
    """
    # Categories: 0->0, 1->1, 2->2, 3-9->3, 10+->4
    # Use arithmetic masking for ONNX compatibility
    bins = torch.zeros_like(inc)
    bins = bins + (inc == 1).to(inc.dtype)  # 1 -> 1
    bins = bins + 2 * (inc == 2).to(inc.dtype)  # 2 -> 2
    bins = bins + 3 * ((inc >= 3) & (inc < 10)).to(inc.dtype)  # 3-9 -> 3
    bins = bins + 4 * (inc >= 10).to(inc.dtype)  # 10+ -> 4
    return bins.long()


class TokenConditioningEncoder(nn.Module):
    """
    Encode global context as explicit tokens prepended to the square sequence.
    
    Tokens:
    1. [ELO_TOKEN] - Interpolated embedding for active player skill level
    2. [OPP_ELO_TOKEN] - Interpolated embedding for opponent skill level
    3. [TC_TOKEN] - Time control category (Blitz, Rapid, Classical)
    4. [URGENCY_TOKEN] - Log-binned remaining time
    5. [INC_TOKEN] - Increment category (0, 1, 2, 5, 10+)
    6. [MY_LAST_TIME] - Log-binned time spent on my last move
    7. [OPP_LAST_TIME] - Log-binned time spent on opponent's last move
    
    This approach lets the model attend to conditioning information directly 
    through self-attention.
    """
    def __init__(self, d_model: int):
        super().__init__()
        self.d_model = d_model
        
        # ELO: Interpolated embedding with anchor points
        # Anchors at 1200-2500 in 100 ELO increments (matches dataset range)
        self.elo_anchors = nn.Parameter(
            torch.tensor([1200.0, 1300.0, 1400.0, 1500.0, 1600.0, 1700.0, 1800.0,
                          1900.0, 2000.0, 2100.0, 2200.0, 2300.0, 2400.0, 2500.0]),
            requires_grad=False
        )
        self.elo_embeddings = nn.Embedding(NUM_ELO_ANCHORS, d_model)
        
        # Opponent ELO: Separate interpolated embedding (same anchor scheme)
        self.opp_elo_embeddings = nn.Embedding(NUM_ELO_ANCHORS, d_model)
        
        # TC: Categorical embedding (Blitz=0, Rapid=1, Classical=2)
        self.tc_embedding = nn.Embedding(NUM_TC_CATEGORIES, d_model)
        
        # Urgency: Log-binned remaining time
        self.urgency_embedding = nn.Embedding(NUM_TIME_LOG_BINS, d_model)
        
        # Increment: Categorical (0, 1, 2, 5, 10+)
        self.inc_embedding = nn.Embedding(NUM_INC_CATEGORIES, d_model)
        
        # My last move time: Log-binned
        self.my_time_embedding = nn.Embedding(NUM_TIME_LOG_BINS, d_model)
        
        # Opponent's last move time: Log-binned
        self.opp_time_embedding = nn.Embedding(NUM_TIME_LOG_BINS, d_model)
        
        # Position embeddings for the 6 conditioning tokens
        self.token_pos_embedding = nn.Embedding(NUM_CONDITIONING_TOKENS, d_model)
        
        # Pre-computed token position indices
        self.register_buffer(
            "token_pos_ids",
            torch.arange(NUM_CONDITIONING_TOKENS, dtype=torch.long),
            persistent=False
        )
    
    def _interpolate_elo(self, elo: torch.Tensor, embedding_table: nn.Embedding = None) -> torch.Tensor:
        """
        Interpolate ELO embedding from anchor points.
        
        Args:
            elo: (B,) raw ELO values
            embedding_table: which embedding to use (defaults to self.elo_embeddings)
        Returns:
            (B, d_model) interpolated embeddings
        """
        if embedding_table is None:
            embedding_table = self.elo_embeddings
        
        B = elo.shape[0]
        device = elo.device
        
        # Clamp ELO to anchor range
        elo_clamped = elo.clamp(self.elo_anchors[0], self.elo_anchors[-1])
        
        # Find which segment each ELO falls into
        # For each ELO, find the largest anchor <= elo
        anchors = self.elo_anchors  # (NUM_ELO_ANCHORS,)
        
        # Broadcast: (B, 1) vs (NUM_ELO_ANCHORS,) -> (B, NUM_ELO_ANCHORS)
        below_mask = elo_clamped.unsqueeze(1) >= anchors.unsqueeze(0)
        
        # Index of lower anchor (last True in each row)
        lower_idx = below_mask.sum(dim=1) - 1  # (B,)
        lower_idx = lower_idx.clamp(0, NUM_ELO_ANCHORS - 2)  # Ensure valid upper idx
        upper_idx = lower_idx + 1
        
        # Get anchor values
        lower_anchor = anchors[lower_idx]  # (B,)
        upper_anchor = anchors[upper_idx]  # (B,)
        
        # Interpolation weight
        t = (elo_clamped - lower_anchor) / (upper_anchor - lower_anchor + 1e-6)  # (B,)
        t = t.clamp(0, 1).unsqueeze(1)  # (B, 1)
        
        # Get embeddings and interpolate
        lower_emb = embedding_table(lower_idx)  # (B, d_model)
        upper_emb = embedding_table(upper_idx)  # (B, d_model)
        
        return (1 - t) * lower_emb + t * upper_emb  # (B, d_model)
    
    def forward(
        self,
        player_elo: torch.Tensor,      # (B,) raw ELO
        opp_elo: torch.Tensor,         # (B,) raw opponent ELO
        tc_cat: torch.Tensor,           # (B,) 0=Blitz, 1=Rapid, 2=Classical
        remaining_time: torch.Tensor,   # (B,) seconds remaining
        increment: torch.Tensor,        # (B,) increment in seconds
        my_last_time: torch.Tensor,     # (B,) seconds spent on my last move
        opp_last_time: torch.Tensor,    # (B,) seconds spent on opponent's last move
    ) -> torch.Tensor:
        """
        Generate conditioning token embeddings.
        
        Returns:
            (B, 7, d_model) - 7 conditioning tokens
        """
        B = player_elo.shape[0]
        
        # 1. ELO token (interpolated)
        elo_token = self._interpolate_elo(player_elo)  # (B, d_model)
        
        # 2. Opponent ELO token (interpolated, separate embedding table)
        opp_elo_token = self._interpolate_elo(opp_elo, self.opp_elo_embeddings)  # (B, d_model)
        
        # 3. TC token (categorical)
        tc_token = self.tc_embedding(tc_cat)  # (B, d_model)
        
        # 4. Urgency token (log-binned remaining time)
        urgency_bins = log_bin_time(remaining_time)
        urgency_token = self.urgency_embedding(urgency_bins)  # (B, d_model)
        
        # 5. Increment token (categorical)
        inc_bins = bin_increment(increment)
        inc_token = self.inc_embedding(inc_bins)  # (B, d_model)
        
        # 6. My last move time token (log-binned)
        my_time_bins = log_bin_time(my_last_time)
        my_time_token = self.my_time_embedding(my_time_bins)  # (B, d_model)
        
        # 7. Opponent's last move time token (log-binned)
        opp_time_bins = log_bin_time(opp_last_time)
        opp_time_token = self.opp_time_embedding(opp_time_bins)  # (B, d_model)
        
        # Stack tokens: (B, 7, d_model)
        tokens = torch.stack([
            elo_token, opp_elo_token, tc_token, urgency_token,
            inc_token, my_time_token, opp_time_token
        ], dim=1)
        
        # Add position embeddings
        tokens = tokens + self.token_pos_embedding(self.token_pos_ids)  # broadcast over B
        
        return tokens


class EloOnlyConditioningEncoder(nn.Module):
    """
    Conditioning encoder with ELO + opponent ELO — no time context at all.
    
    Used for the no-time-context model variant. Two tokens [ELO_TOKEN, OPP_ELO_TOKEN]
    are prepended to the 64 square tokens (total sequence length = 66).
    No time control, remaining clock, increment, or move timing information.
    """
    def __init__(self, d_model: int):
        super().__init__()
        self.d_model = d_model
        self.num_tokens = NUM_CONDITIONING_TOKENS_ELO_ONLY  # 2
        
        # ELO: Interpolated embedding with anchor points (same as TokenConditioningEncoder)
        self.elo_anchors = nn.Parameter(
            torch.tensor([1200.0, 1300.0, 1400.0, 1500.0, 1600.0, 1700.0, 1800.0,
                          1900.0, 2000.0, 2100.0, 2200.0, 2300.0, 2400.0, 2500.0]),
            requires_grad=False
        )
        self.elo_embeddings = nn.Embedding(NUM_ELO_ANCHORS, d_model)
        
        # Opponent ELO: Separate interpolated embedding (same anchor scheme)
        self.opp_elo_embeddings = nn.Embedding(NUM_ELO_ANCHORS, d_model)
        
        # Position embeddings for the 2 conditioning tokens
        self.token_pos_embedding = nn.Embedding(NUM_CONDITIONING_TOKENS_ELO_ONLY, d_model)
        
        self.register_buffer(
            "token_pos_ids",
            torch.arange(NUM_CONDITIONING_TOKENS_ELO_ONLY, dtype=torch.long),
            persistent=False
        )
    
    def _interpolate_elo(self, elo: torch.Tensor, embedding_table: nn.Embedding = None) -> torch.Tensor:
        """Interpolate ELO embedding from anchor points."""
        if embedding_table is None:
            embedding_table = self.elo_embeddings
        B = elo.shape[0]
        elo_clamped = elo.clamp(self.elo_anchors[0], self.elo_anchors[-1])
        anchors = self.elo_anchors
        below_mask = elo_clamped.unsqueeze(1) >= anchors.unsqueeze(0)
        lower_idx = below_mask.sum(dim=1) - 1
        lower_idx = lower_idx.clamp(0, NUM_ELO_ANCHORS - 2)
        upper_idx = lower_idx + 1
        lower_anchor = anchors[lower_idx]
        upper_anchor = anchors[upper_idx]
        t = (elo_clamped - lower_anchor) / (upper_anchor - lower_anchor + 1e-6)
        t = t.clamp(0, 1).unsqueeze(1)
        lower_emb = embedding_table(lower_idx)
        upper_emb = embedding_table(upper_idx)
        return (1 - t) * lower_emb + t * upper_emb
    
    def forward(self, player_elo: torch.Tensor, opp_elo: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Generate conditioning token embeddings (ELO + opponent ELO only).
        
        Extra kwargs (tc_cat, remaining_time, etc.) are accepted and ignored
        for interface compatibility with TokenConditioningEncoder.
        
        Returns:
            (B, 2, d_model) - two conditioning tokens [ELO, OPP_ELO]
        """
        elo_token = self._interpolate_elo(player_elo)  # (B, d_model)
        opp_elo_token = self._interpolate_elo(opp_elo, self.opp_elo_embeddings)  # (B, d_model)
        tokens = torch.stack([elo_token, opp_elo_token], dim=1)  # (B, 2, d_model)
        tokens = tokens + self.token_pos_embedding(self.token_pos_ids)
        return tokens


class AttentionPooling(nn.Module):
    """
    Attention-based pooling that learns which squares matter for value/time prediction.
    
    Unlike mean pooling which weights all 64 squares equally, this learns to
    focus on tactically relevant squares (e.g., the hanging queen, not empty squares).
    
    Optionally conditioned on global context (ELO might focus on different features).
    """
    def __init__(self, d_model, context_dim=None):
        super().__init__()
        self.d_model = d_model
        self.use_context = context_dim is not None
        
        if self.use_context:
            # Query is generated from context
            self.query_proj = nn.Linear(context_dim, d_model)
        else:
            # Learned query vector
            self.query = nn.Parameter(torch.randn(d_model) * 0.02)
        
        self.key_proj = nn.Linear(d_model, d_model, bias=False)
        self.scale = 1.0 / math.sqrt(d_model)
    
    def forward(self, x, context=None):
        """
        Args:
            x: (B, 64, d_model) - token representations
            context: (B, context_dim) - optional global context
        Returns:
            pooled: (B, d_model) - weighted combination of tokens
        """
        B = x.shape[0]
        keys = self.key_proj(x)  # (B, 64, d_model)
        
        if self.use_context and context is not None:
            query = self.query_proj(context)  # (B, d_model)
        else:
            query = self.query.unsqueeze(0).expand(B, -1)  # (B, d_model)
        
        # Compute attention scores
        attn_scores = (keys * query.unsqueeze(1)).sum(dim=-1) * self.scale  # (B, 64)
        attn_weights = F.softmax(attn_scores, dim=-1)  # (B, 64)
        
        # Weighted sum
        pooled = (x * attn_weights.unsqueeze(-1)).sum(dim=1)  # (B, d_model)
        return pooled


class PaperValueHead(nn.Module):
    """
    Paper implementation:
    1. Project tokens to d_val=32 (bottleneck)
    2. Flatten (preserve spatial structure)
    3. Project to value embedding
    """
    def __init__(self, d_model, d_val=32):
        super().__init__()
        self.bottleneck = nn.Linear(d_model, d_val, bias=False)
        self.flatten_dim = NUM_SQUARES * d_val
        
        # Value prediction (scalar)
        self.final_proj = nn.Sequential(
            nn.Linear(self.flatten_dim, 128),
            Mish(),
            nn.Linear(128, 1)
        )
        
        # WDL Classification
        self.wdl_proj = nn.Sequential(
            nn.Linear(self.flatten_dim, 128),
            Mish(),
            nn.Linear(128, NUM_VALUE_CLASSES)
        )

    def forward(self, x):
        """
        Args:
            x: (B, 64, d_model)
        Returns:
            val: (B, 1) scalar value
            wdl: (B, 3) WDL logits
        """
        b = self.bottleneck(x)      # (B, 64, 32)
        flat = b.flatten(1)         # (B, 2048)
        val = self.final_proj(flat)
        wdl = self.wdl_proj(flat)
        return val, wdl


class SwiGLU(nn.Module):
    """Swish-Gated Linear Unit FFN."""
    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.w1 = nn.Linear(d_model, d_ff, bias=False)
        self.w2 = nn.Linear(d_model, d_ff, bias=False)
        self.w3 = nn.Linear(d_ff, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.dropout(self.w3(F.silu(self.w1(x)) * self.w2(x)))


class RelativePositionGenerator(nn.Module):
    """Generate relative position indices for chess board."""
    def __init__(self, max_dist=7):
        super().__init__()
        self.max_dist = max_dist
        self.num_buckets = (2 * max_dist + 1) ** 2
        
        # Precompute relative indices
        indices = torch.zeros(64, 64, dtype=torch.long)
        for i in range(64):
            for j in range(64):
                ri, ci = i // 8, i % 8
                rj, cj = j // 8, j % 8
                dr = max(-max_dist, min(max_dist, rj - ri))
                dc = max(-max_dist, min(max_dist, cj - ci))
                bucket = (dr + max_dist) * (2 * max_dist + 1) + (dc + max_dist)
                indices[i, j] = bucket
        self.register_buffer("indices", indices)

    def forward(self):
        return self.indices


class Smolgen(nn.Module):
    """
    Smolgen: Dynamic attention bias generation based on position content.
    
    From Leela Chess Zero:
    1. Compress the current representation into a small vector (e.g., 256d)
    2. For each head, generate supplemental attention logits (64x64)
    
    This allows the model to dynamically adjust which squares attend to which,
    based on the current position (open vs closed, piece placement, etc.)
    """
    def __init__(self, d_model, n_heads, hidden_size=256, per_head_size=256):
        super().__init__()
        self.n_heads = n_heads
        self.hidden_size = hidden_size
        self.per_head_size = per_head_size
        
        # Step 1: Compress position into a global vector
        # Project each token to a smaller dimension, then flatten and compress
        self.token_compress = nn.Linear(d_model, 32, bias=False)
        self.global_compress = nn.Sequential(
            nn.Linear(64 * 32, hidden_size),
            RMSNorm(hidden_size),
        )
        
        # Step 2: Generate per-head attention biases using a single batched projection
        # Project from hidden_size to (n_heads * per_head_size) in one operation
        # This replaces the old ModuleList approach for better torch.compile compatibility
        self.head_proj = nn.Linear(hidden_size, n_heads * per_head_size, bias=False)
        
        # Shared projection from per_head_size to 64x64 attention logits
        # This is shared across all heads (as per Leela's description)
        self.attn_logit_proj = nn.Linear(per_head_size, 64 * 64, bias=False)
        
        # Flag to track if we've converted from legacy format
        self._converted_from_legacy = False
    
    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs):
        """Handle loading from old checkpoint format with head_projections ModuleList."""
        # Check if this is an old-format checkpoint (has head_projections.0.weight but no head_proj.weight)
        legacy_key_0 = f"{prefix}head_projections.0.weight"
        new_key = f"{prefix}head_proj.weight"
        
        if legacy_key_0 in state_dict and new_key not in state_dict:
            # Convert from legacy ModuleList format to new batched format
            # Old format: head_projections.{i}.weight with shape (per_head_size, hidden_size)
            # New format: head_proj.weight with shape (n_heads * per_head_size, hidden_size)
            legacy_weights = []
            for i in range(self.n_heads):
                key = f"{prefix}head_projections.{i}.weight"
                if key in state_dict:
                    legacy_weights.append(state_dict.pop(key))
                    # Add to unexpected_keys removal
                else:
                    # Missing head projection - this shouldn't happen in valid checkpoints
                    error_msgs.append(f"Missing legacy weight: {key}")
                    break
            
            if len(legacy_weights) == self.n_heads:
                # Stack all head weights: (n_heads, per_head_size, hidden_size)
                # Then reshape to (n_heads * per_head_size, hidden_size)
                stacked = torch.cat(legacy_weights, dim=0)  # (n_heads * per_head_size, hidden_size)
                state_dict[new_key] = stacked
                self._converted_from_legacy = True
        
        # Call parent implementation
        super()._load_from_state_dict(state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs)
        
    def forward(self, x):
        """
        Args:
            x: (B, 64, d_model) - token representations
        Returns:
            attn_bias: (B, n_heads, 64, 64) - dynamic attention biases
        """
        B = x.shape[0]
        
        # Compress tokens: (B, 64, d_model) -> (B, 64, 32)
        compressed = self.token_compress(x)
        
        # Flatten and compress to global: (B, 64*32) -> (B, hidden_size)
        global_repr = self.global_compress(compressed.flatten(1))
        
        # Generate all head representations in one go: (B, hidden_size) -> (B, n_heads * per_head_size)
        all_head_repr = self.head_proj(global_repr)
        
        # Reshape to (B, n_heads, per_head_size)
        all_head_repr = all_head_repr.view(B, self.n_heads, self.per_head_size)
        
        # Apply shared projection to all heads: (B, n_heads, per_head_size) -> (B, n_heads, 64*64)
        # Using einsum for batched linear: weight is (per_head_size, 64*64)
        head_logits = torch.matmul(all_head_repr, self.attn_logit_proj.weight.T)
        
        # Reshape to (B, n_heads, 64, 64)
        return head_logits.view(B, self.n_heads, 64, 64)


class ShawAttention(nn.Module):
    """Self-attention with Shaw relative position encoding and optional smolgen."""
    def __init__(self, d_model, n_heads, d_head, num_rel_pos, dropout=0.1, bias=False):
        super().__init__()
        self.n_heads = n_heads
        self.d_head = d_head
        self.scale = 1.0 / math.sqrt(d_head)
        
        self.q_proj = nn.Linear(d_model, d_model, bias=bias)
        self.k_proj = nn.Linear(d_model, d_model, bias=bias)
        self.v_proj = nn.Linear(d_model, d_model, bias=bias)
        self.o_proj = nn.Linear(d_model, d_model, bias=bias)

        self.q_norm = RMSNorm(d_head)
        self.k_norm = RMSNorm(d_head)
        
        self.rel_k_emb = nn.Embedding(num_rel_pos, d_head)
        self.rel_v_emb = nn.Embedding(num_rel_pos, d_head)
        
        self.dropout = nn.Dropout(dropout)

        # Debug/inference: populated with attention probabilities from the most recent forward.
        # Shape: (B, n_heads, 64, 64)
        self.last_attn_probs = None

        # Debug/inference: populated with the smolgen dynamic bias logits from the most recent forward.
        # Shape: (B, n_heads, 64, 64)
        self.last_smolgen_bias = None

    def forward(self, x, rel_indices, smolgen_bias=None):
        B, L, D = x.shape
        H, d_k = self.n_heads, self.d_head
        
        q = self.q_proj(x).view(B, L, H, d_k)
        k = self.k_proj(x).view(B, L, H, d_k)

        q = self.q_norm(q).transpose(1, 2)
        k = self.k_norm(k).transpose(1, 2)
        v = self.v_proj(x).view(B, L, H, d_k).transpose(1, 2)
        
        r_k = self.rel_k_emb(rel_indices)
        r_v = self.rel_v_emb(rel_indices)
        
        content_score = torch.matmul(q, k.transpose(-2, -1))
        rel_score = torch.einsum('bhid,ijd->bhij', q, r_k)
        
        attn_scores = (content_score + rel_score) * self.scale
        
        # Add smolgen dynamic attention bias if provided
        if smolgen_bias is not None:
            attn_scores = attn_scores + smolgen_bias
        
        attn_probs = self.dropout(F.softmax(attn_scores, dim=-1))

        # Stash for inference-time visualization only (not during training to avoid graph breaks)
        if not self.training:
            self.last_smolgen_bias = smolgen_bias.detach() if smolgen_bias is not None else None
            self.last_attn_probs = attn_probs.detach()
        
        content_out = torch.matmul(attn_probs, v)
        rel_out = torch.einsum('bhij,ijd->bhid', attn_probs, r_v)
        
        output = (content_out + rel_out).transpose(1, 2).contiguous().view(B, L, D)
        return self.o_proj(output)


class TransformerBlock(nn.Module):
    """Transformer block with Mish activation and bias=False for attention."""
    def __init__(self, config, num_rel_pos):
        super().__init__()
        d_model = config['d_model']
        
        self.norm1 = RMSNorm(d_model)
        self.norm2 = RMSNorm(d_model)
        
        # Bias=False for QKV (Paper)
        self.attn = ShawAttention(
            d_model, config['n_heads'], config['d_head'],
            num_rel_pos, config['dropout'], bias=False
        )
        
        # SwiGLU FFN for improved convergence
        self.ff = SwiGLU(d_model, config['d_ff'], config['dropout'])

    def forward(self, x, rel_indices, smolgen_bias=None):
        # Pre-LN structure
        x = x + self.attn(self.norm1(x), rel_indices=rel_indices, smolgen_bias=smolgen_bias)
        x = x + self.ff(self.norm2(x))
        return x


class PerSquareInputEncoder(nn.Module):
    """
    Per-square encoder with clean separation of concerns.
    
    This encoder handles ONLY board state information:
    - Board history (8 positions × 13 piece types = 104 dims)
    - En passant flag (1 dim, per-square)
    - Castling rights (4 dims, broadcast)
    - Repetition flags (8 dims, broadcast)
    
    All global/game context (ELO, clocks, increment, TC, move timing) is handled
    by TokenConditioningEncoder as prepended tokens, NOT duplicated here.
    
    Total input: 104 + 1 + 4 + 8 = 117 dims per square
    """
    def __init__(self, config):
        super().__init__()
        self.history_len = config['history_len']
        self.num_piece_types = config['num_piece_types']
        d_model = config['d_model']
        
        # Input dimension per token (board state only)
        # 8 * 13 (piece one-hots) + 1 (ep) + 4 (castling) + 8 (rep flags)
        self.board_dim = self.history_len * self.num_piece_types + 1  # 105
        self.state_dim = 4 + self.history_len  # castling + rep = 12
        
        # Project board features (piece positions + EP)
        self.board_proj = nn.Linear(self.board_dim, d_model, bias=False)
        
        # Project state features (castling + repetition) with smaller init
        self.state_proj = nn.Linear(self.state_dim, d_model, bias=False)
        
        # Per-square absolute position embedding
        self.square_emb = nn.Embedding(64, d_model)
        
        # Avoid per-forward torch.arange allocations
        self.register_buffer("square_ids", torch.arange(64, dtype=torch.long), persistent=False)
        
        # Per-square scale and bias (paper's "multiply by learned offset")
        self.square_scale = nn.Parameter(torch.ones(64, d_model))
        self.square_bias = nn.Parameter(torch.zeros(64, d_model))
        
        # Leela-style: optional FFN after embedding
        self.use_embedding_ffn = config.get('embedding_ffn', False)
        if self.use_embedding_ffn:
            self.emb_ffn = nn.Sequential(
                nn.Linear(d_model, d_model * 4),
                nn.GELU(),
                nn.Linear(d_model * 4, d_model),
            )
            self.emb_norm = RMSNorm(d_model)
        
        # Initialize: full-strength board, smaller state
        nn.init.normal_(self.board_proj.weight, std=0.02)
        nn.init.normal_(self.state_proj.weight, std=0.01)
        self.board_proj._no_reinit = True
        self.state_proj._no_reinit = True

    def forward(self, board_history, rep_flags, castling, ep_mask):
        """
        Args:
            board_history: (B, 8, 64) int64 - piece codes per square per history step
            rep_flags: (B, 8) float - repetition flags per history step
            castling: (B, 4) float - castling rights
            ep_mask: (B, 64) float - 1.0 at EP square, else 0
        
        Returns:
            (B, 64, d_model) token embeddings
        """
        B = board_history.shape[0]
        
        # 1. One-hot encode board history: (B, 8, 64, 13) -> (B, 64, 104)
        board_onehot = F.one_hot(board_history, num_classes=self.num_piece_types).float()
        board_onehot = board_onehot.permute(0, 2, 1, 3).reshape(B, 64, -1)
        
        # 2. EP mask per-square: (B, 64, 1)
        ep_per_square = ep_mask.unsqueeze(-1)
        
        # 3. Broadcast rep_flags to all squares: (B, 64, 8)
        rep_broadcast = rep_flags.unsqueeze(1).expand(-1, 64, -1)
        
        # 4. Broadcast castling to all squares: (B, 64, 4)
        castle_broadcast = castling.unsqueeze(1).expand(-1, 64, -1)
        
        # Combine inputs
        board_input = torch.cat([board_onehot, ep_per_square], dim=-1)  # (B, 64, 105)
        state_input = torch.cat([castle_broadcast, rep_broadcast], dim=-1)  # (B, 64, 12)
        
        # Project to d_model
        x = self.board_proj(board_input) + self.state_proj(state_input)
        
        # Add absolute position embedding
        x = x + self.square_emb(self.square_ids)
        
        # Apply per-square scale and bias
        x = x * self.square_scale + self.square_bias
        
        # Leela-style: FFN after embedding
        if self.use_embedding_ffn:
            x = x + self.emb_ffn(self.emb_norm(x))
        
        return x


class Chessformer(nn.Module):
    """
    Chessformer model with token-based conditioning.
    
    By default, uses 7 conditioning tokens prepended to 64 square tokens for:
    - Active player ELO (interpolated embedding)
    - Opponent ELO (interpolated embedding, separate weights)
    - Time control category (Blitz/Rapid/Classical)
    - Urgency (remaining time)
    - Increment category
    - My last move time
    - Opponent's last move time
    
    With config['no_time_context'] = True, uses 2 conditioning tokens (ELO + OPP_ELO),
    removing all time-related context for a purely position-based model.
    
    Per-square encoding handles only board state (piece history, EP, castling, rep flags).
    All global/game context is provided via the conditioning tokens.
    """
    def __init__(self, config):
        super().__init__()
        self.config = config
        d_model = config['d_model']
        self.no_time_context = config.get('no_time_context', False)
        self.no_board_history = config.get('no_board_history', False)
        
        # Token-based conditioning
        if self.no_time_context:
            # ELO + OPP_ELO: 2 tokens prepended to 64 square tokens = 66 total
            self.token_conditioning = EloOnlyConditioningEncoder(d_model)
            self.num_cond_tokens = NUM_CONDITIONING_TOKENS_ELO_ONLY
        else:
            # Full: 7 tokens prepended to 64 square tokens = 71 total
            self.token_conditioning = TokenConditioningEncoder(d_model)
            self.num_cond_tokens = NUM_CONDITIONING_TOKENS
        
        self.total_seq_len = self.num_cond_tokens + NUM_SQUARES  # 65 or 70
        
        # Gradient checkpointing for memory-efficient training
        self.gradient_checkpointing = config.get('gradient_checkpointing', False)
        
        # Input encoder - board state only, globals handled by token conditioning
        self.input_encoder = PerSquareInputEncoder(config)
        
        # Transformer body - using TransformerBlock with Mish activation
        self.rel_pos_gen = RelativePositionGenerator(config['max_rel_dist'])
        
        # Extended relative position for total_seq_len tokens
        # Conditioning tokens use the max distance bucket for all positions
        L = self.total_seq_len
        ext_indices = torch.zeros(L, L, dtype=torch.long)
        # Fill in square-to-square relative positions
        ext_indices[self.num_cond_tokens:, self.num_cond_tokens:] = self.rel_pos_gen.indices
        # All other positions (involving conditioning tokens) use max bucket
        max_bucket = self.rel_pos_gen.num_buckets - 1
        ext_indices[:self.num_cond_tokens, :] = max_bucket
        ext_indices[:, :self.num_cond_tokens] = max_bucket
        self.register_buffer("extended_rel_indices", ext_indices)
        
        self.layers = nn.ModuleList([
            TransformerBlock(config, self.rel_pos_gen.num_buckets)
            for _ in range(config['n_layers'])
        ])
        self.norm_f = RMSNorm(d_model)
        
        # Smolgen: dynamic attention biases (shared across all layers)
        self.use_smolgen = config.get('smolgen', False)
        if self.use_smolgen:
            self.smolgen = Smolgen(
                d_model=d_model,
                n_heads=config['n_heads'],
                hidden_size=config.get('smolgen_hidden', 256),
                per_head_size=config.get('smolgen_per_head', 256),
            )
        
        # --- Output Heads ---
        
        # Policy head: produces (B, 4098) logits
        # 4096 for regular moves + resign + flag
        self.policy_query = nn.Linear(d_model, d_model, bias=False)
        self.policy_key = nn.Linear(d_model, d_model, bias=False)
        self.policy_scale = 1.0 / math.sqrt(d_model)  # Scaling per paper
        
        # Special action tokens (resign, flag) - learned embeddings
        self.resign_key = nn.Parameter(torch.randn(d_model) * 0.02)
        self.flag_key = nn.Parameter(torch.randn(d_model) * 0.02)
        
        # Promotion head: per destination square
        # Paper's approach: additive bias from key vectors at promotion rank
        self.promo_bias_proj = nn.Linear(d_model, 4)  # Q, R, B, N promotions
        
        # Aux: Start Square Head (Human Mimicry Booster)
        # Predicts which of the 64 squares the piece moves FROM
        self.start_square_head = nn.Linear(d_model, 1)
        
        # Value head (Paper Style - flattened spatial + MLP)
        self.value_head = PaperValueHead(d_model)

        # Value error head to model volatility
        self.value_error_pool = AttentionPooling(d_model, context_dim=None)
        self.value_error_head = nn.Sequential(
            nn.Linear(d_model, 256),
            Mish(),
            nn.Linear(256, 1),
        )
        
        # Time head: 256-bin classification with attention pooling
        # Time distribution is multimodal (instant moves, short thinks, long thinks)
        # Omitted for no_time_context models (ablation study — no time info in or out)
        if not self.no_time_context:
            self.time_pooling = AttentionPooling(d_model, context_dim=None)
            self.time_head = nn.Sequential(
                nn.Linear(d_model, 256, bias=False),
                Mish(),
                nn.Linear(256, NUM_TIME_BINS),  # logits for 256 time bins
            )
        
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if getattr(module, "_no_reinit", False):
            return
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, batch, return_promo=False):
        """
        Args:
            batch: dict with keys from dataset
            return_promo: if True, also return promotion logits
        
        Returns:
            move_logits: (B, 4098) policy logits (masked if legal_mask provided)
            value_out: (B, 1) win probability
            value_error_out: (B, 1) predicted squared error / volatility proxy
            time_out: (B, 1) predicted time in normalized units
            promo_logits: (B, 8, 4) promotion logits for dest squares 56-63 (optional)
        """
        # Extract inputs from batch
        board_history = batch['board_history']
        rep_flags = batch['rep_flags']
        castling = batch['castling']
        ep_mask = batch['ep_mask']
        scalars = batch['scalars']
        legal_mask = batch.get('legal_mask', None)
        
        B = board_history.shape[0]
        device = board_history.device
        
        # Strip history if model only uses current position
        if self.no_board_history:
            board_history = board_history[:, :1, :]  # keep only current position
            rep_flags = rep_flags[:, :1]              # keep only current rep flag
        
        # Encode square inputs - board state only
        x = self.input_encoder(board_history, rep_flags, castling, ep_mask)
        
        # Token conditioning: prepend conditioning tokens to the sequence
        # ELO: denormalize from scalars (was (elo - 1900) / 700)
        player_elo_raw = scalars[:, 0] * 700 + 1900  # (B,)
        # Opponent ELO: denormalize from scalars (was (elo - 1900) / 700)
        opp_elo_raw = scalars[:, 1] * 700 + 1900  # (B,)
        
        if self.no_time_context:
            # ELO + OPP_ELO conditioning — no time information at all
            cond_tokens = self.token_conditioning(
                player_elo=player_elo_raw,
                opp_elo=opp_elo_raw,
            )  # (B, 2, d_model)
        else:
            time_history = batch['time_history']
            tc_cat = batch['tc_cat']
            
            # TC category is already in the batch
            # tc_cat: (B,) with values 0=Blitz, 1=Rapid, 2=Classical
            
            # Remaining time: denormalize from scalars (was log1p(seconds) / 10)
            # Use exp(x) - 1 instead of expm1 for ONNX compatibility
            remaining_time_raw = torch.exp(scalars[:, 3] * 10) - 1  # (B,)
            
            # Increment: denormalize from scalars (was inc / 30)
            increment_raw = (scalars[:, 5] * 30).long()  # (B,)
            
            # My last move time: time_history[:, 1] (index 1 is my last move, index 0 is opponent's last)
            # Denormalize from /60 and handle edge cases
            my_last_time_raw = time_history[:, 1] * 60  # (B,) in seconds
            
            # Opponent's last move time: time_history[:, 0]
            opp_last_time_raw = time_history[:, 0] * 60  # (B,) in seconds
            
            # Generate conditioning token embeddings
            cond_tokens = self.token_conditioning(
                player_elo=player_elo_raw,
                opp_elo=opp_elo_raw,
                tc_cat=tc_cat,
                remaining_time=remaining_time_raw,
                increment=increment_raw,
                my_last_time=my_last_time_raw,
                opp_last_time=opp_last_time_raw,
            )  # (B, 7, d_model)
        
        # Prepend conditioning tokens to square tokens
        NC = self.num_cond_tokens
        L = self.total_seq_len
        x = torch.cat([cond_tokens, x], dim=1)  # (B, L, d_model)
        
        # Compute smolgen bias once (shared across layers)
        # Smolgen is designed for 64x64 square attention, apply only to square-to-square
        smolgen_bias = None
        if self.use_smolgen:
            # Compute smolgen on square tokens only
            square_tokens = x[:, NC:, :]  # (B, 64, d_model)
            smolgen_bias_64 = self.smolgen(square_tokens)  # (B, n_heads, 64, 64)
            
            # Extend to LxL with zeros for conditioning token interactions
            n_heads = smolgen_bias_64.shape[1]
            smolgen_bias = torch.zeros(
                B, n_heads, L, L, 
                device=device, dtype=smolgen_bias_64.dtype
            )
            smolgen_bias[:, :, NC:, NC:] = smolgen_bias_64
        
        # Transformer layers
        for layer in self.layers:
            if self.gradient_checkpointing and self.training:
                # use_reentrant=False is required for torch.compile compatibility
                x = gradient_checkpoint(
                    layer, x, self.extended_rel_indices, smolgen_bias,
                    use_reentrant=False
                )
            else:
                x = layer(x, rel_indices=self.extended_rel_indices, smolgen_bias=smolgen_bias)
        x = self.norm_f(x)  # (B, L, d_model)
        
        # Extract square tokens for output heads
        x_squares = x[:, NC:, :]  # (B, 64, d_model)
        
        # --- Policy Head ---
        pol_q = self.policy_query(x_squares)  # (B, 64, d_model)
        pol_k = self.policy_key(x_squares)    # (B, 64, d_model)
        
        # Move logits: (B, 64, 64) -> (B, 4096)
        move_logits = torch.matmul(pol_q, pol_k.transpose(1, 2)) * self.policy_scale
        move_logits = move_logits.view(B, -1)
        
        # Add resign and flag logits
        # Score = mean query dot product with special keys
        mean_q = pol_q.mean(dim=1)  # (B, d_model)
        resign_logit = (mean_q * self.resign_key).sum(dim=-1, keepdim=True) * self.policy_scale
        flag_logit = (mean_q * self.flag_key).sum(dim=-1, keepdim=True) * self.policy_scale
        
        # Concatenate: (B, 4098)
        move_logits = torch.cat([move_logits, resign_logit, flag_logit], dim=-1)
        
        # Apply legal move mask if provided (with fix for 4096 vs 4098 mask size)
        if legal_mask is not None:
            mask_dim = legal_mask.shape[1]
            if mask_dim == 4096:
                # Append True for Resign/Flag (always legal to resign/flag in training)
                extras = torch.ones(B, 2, dtype=torch.bool, device=legal_mask.device)
                full_mask = torch.cat([legal_mask, extras], dim=1)
            else:
                full_mask = legal_mask
            move_logits = move_logits.masked_fill(~full_mask, float('-inf'))
        
        # --- Promotion Head ---
        # Get key vectors for promotion rank (squares 56-63)
        promo_keys = pol_k[:, 56:64, :]  # (B, 8, d_model)
        promo_logits = self.promo_bias_proj(promo_keys)  # (B, 8, 4)
        
        # --- Aux: Start Square Head ---
        # Predicts which of the 64 squares the piece moves FROM
        start_square_logits = self.start_square_head(x_squares).squeeze(-1)  # (B, 64)
        
        # --- Value and Time Heads ---
        # Value uses PaperValueHead (flattened spatial structure)
        value_out, value_cls_out = self.value_head(x_squares)  # (B, 1), (B, 3)

        # Value error models volatility
        value_error_feat = self.value_error_pool(x_squares)
        value_error_out = self.value_error_head(value_error_feat)
        
        # Time uses attention pooling (absent for no_time_context models)
        if self.no_time_context:
            time_cls_out = None
        else:
            time_pooled = self.time_pooling(x_squares)    # (B, d_model)
            time_cls_out = self.time_head(time_pooled)     # (B, 256) time bin logits
        
        if return_promo:
            return move_logits, value_out, value_cls_out, value_error_out, time_cls_out, start_square_logits, promo_logits
        return move_logits, value_out, value_cls_out, value_error_out, time_cls_out, start_square_logits


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    # Create dummy batch for testing
    B = 2
    dummy_batch = {
        'board_history': torch.randint(0, 13, (B, 8, 64)),
        'time_history': torch.randn(B, 8),
        'rep_flags': torch.randint(0, 2, (B, 8)).float(),
        'castling': torch.randint(0, 2, (B, 4)).float(),
        'ep_mask': torch.zeros(B, 64),
        'scalars': torch.randn(B, 8),
        'tc_cat': torch.randint(0, 3, (B,)),
        'legal_mask': torch.ones(B, NUM_POLICY_OUTPUTS, dtype=torch.bool),
    }
    
    # Test all configs
    print("=" * 60)
    print("MODEL CONFIGS")
    print("=" * 60)
    
    for name, config in [("TINY", CONFIG_TINY), ("SMALL", CONFIG_SMALL), ("LARGE", CONFIG_LARGE), ("SMALL_NOTIME", CONFIG_SMALL_NOTIME), ("SMALL_NOTIME_NOHIST", CONFIG_SMALL_NOTIME_NOHIST)]:
        print(f"\n--- CONFIG_{name} ---")
        model = Chessformer(config)
        print(f"Parameters: {count_parameters(model):,}")
        out = model(dummy_batch)
        print(f"Forward pass: OK (move_logits={out[0].shape})")