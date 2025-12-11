"""
Chessformer v2 - Model architecture aligned with LEVEL_UP_PLAN.md

Key changes from v1:
- Per-token input encoding (board history + metadata concatenated per square)
- Absolute position embeddings (per-square learned embeddings)
- Policy head with 1/sqrt(d_model) scaling
- Legal move masking
- Resign/flag as policy outputs (indices 4096, 4097)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

# Special policy indices
RESIGN_MOVE_INDEX = 4096
FLAG_MOVE_INDEX = 4097
NUM_POLICY_OUTPUTS = 4098

# --- Configuration ---
CONFIG_V2_SMALL = {
    "d_model": 384,
    "n_layers": 8,
    "n_heads": 12,
    "d_head": 32,
    "d_ff": 1536,             # 4x d_model (standard transformer)
    "dropout": 0.1,
    "max_rel_dist": 7,
    "history_len": 8,
    "num_piece_types": 13,    # 0=empty, 1-6=white pieces, 7-12=black pieces
    "num_tc_cats": 3,         # bullet, blitz, rapid+
    "embedding_ffn": False,   # Leela-style post-embedding FFN
}

# Deep+wide config with 1x FFN: tests Leela's hypothesis at ~20M params
# Same capacity as SMALL but trades FFN width for more layers and wider d_model
CONFIG_V2_DEEP = {
    "d_model": 448,           # Wider than SMALL (384 -> 448)
    "n_layers": 14,           # Deeper than SMALL (8 -> 14)
    "n_heads": 14,            # 448 / 32 = 14
    "d_head": 32,
    "d_ff": 448,              # 1x d_model (Leela-style)
    "dropout": 0.1,
    "max_rel_dist": 7,
    "history_len": 8,
    "num_piece_types": 13,
    "num_tc_cats": 3,
    "embedding_ffn": False,   # No embedding FFN, capacity from depth/width
}

# Leela-style config: 1x FFN but with post-embedding FFN layer
CONFIG_V2_LEELA = {
    "d_model": 384,
    "n_layers": 8,
    "n_heads": 48,            # More heads with smaller d_head
    "d_head": 8,              # Leela uses small head depths
    "d_ff": 384,              # 1x d_model (Leela-style)
    "dropout": 0.1,
    "max_rel_dist": 7,
    "history_len": 8,
    "num_piece_types": 13,
    "num_tc_cats": 3,
    "embedding_ffn": True,    # Add FFN after embedding (Leela-style)
    "smolgen": False,
}

# Smolgen config: 1x FFN + dynamic attention biases (full Leela approach)
# This allows 1x FFN to work by making attention position-content-aware
# Sized to ~20M params for fair comparison with CONFIG_V2_SMALL
CONFIG_V2_SMOLGEN = {
    "d_model": 448,           # Wider to compensate for 1x FFN
    "n_layers": 10,           # 10 layers (between 8 and 12)
    "n_heads": 14,            # 448 / 32 = 14
    "d_head": 32,
    "d_ff": 448,              # 1x d_model
    "dropout": 0.1,
    "max_rel_dist": 7,
    "history_len": 8,
    "num_piece_types": 13,
    "num_tc_cats": 3,
    "embedding_ffn": True,    # Leela uses both embedding FFN and smolgen
    "smolgen": True,          # Enable smolgen dynamic attention biases
    "smolgen_hidden": 256,    # Compressed position representation size
    "smolgen_per_head": 256,  # Per-head intermediate size
}

CONFIG_V2_LARGE = {
    "d_model": 768,
    "n_layers": 12,
    "n_heads": 12,
    "d_head": 64,
    "d_ff": 3072,             # 4x d_model (standard transformer)
    "dropout": 0.1,
    "max_rel_dist": 7,
    "history_len": 8,
    "num_piece_types": 13,
    "num_tc_cats": 3,
    "embedding_ffn": False,
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
        
        # Step 2: Generate per-head attention biases
        # Each head gets its own projection from hidden to per_head_size
        self.head_projections = nn.ModuleList([
            nn.Linear(hidden_size, per_head_size, bias=False)
            for _ in range(n_heads)
        ])
        
        # Shared projection from per_head_size to 64x64 attention logits
        # This is shared across all heads (as per Leela's description)
        self.attn_logit_proj = nn.Linear(per_head_size, 64 * 64, bias=False)
        
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
        
        # Generate per-head attention biases
        attn_biases = []
        for head_proj in self.head_projections:
            # (B, hidden_size) -> (B, per_head_size) -> (B, 64*64)
            head_repr = head_proj(global_repr)
            head_logits = self.attn_logit_proj(head_repr)
            attn_biases.append(head_logits.view(B, 64, 64))
        
        # Stack: (B, n_heads, 64, 64)
        return torch.stack(attn_biases, dim=1)


class ShawAttention(nn.Module):
    """Self-attention with Shaw relative position encoding and optional smolgen."""
    def __init__(self, d_model, n_heads, d_head, num_rel_pos, dropout=0.1):
        super().__init__()
        self.n_heads = n_heads
        self.d_head = d_head
        self.scale = 1.0 / math.sqrt(d_head)
        
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.o_proj = nn.Linear(d_model, d_model, bias=False)
        
        self.rel_k_emb = nn.Embedding(num_rel_pos, d_head)
        self.rel_v_emb = nn.Embedding(num_rel_pos, d_head)
        
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, rel_indices, smolgen_bias=None):
        B, L, D = x.shape
        H, d_k = self.n_heads, self.d_head
        
        q = self.q_proj(x).view(B, L, H, d_k).transpose(1, 2)
        k = self.k_proj(x).view(B, L, H, d_k).transpose(1, 2)
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
        
        content_out = torch.matmul(attn_probs, v)
        rel_out = torch.einsum('bhij,ijd->bhid', attn_probs, r_v)
        
        output = (content_out + rel_out).transpose(1, 2).contiguous().view(B, L, D)
        return self.o_proj(output)


class TransformerBlock(nn.Module):
    def __init__(self, config, num_rel_pos):
        super().__init__()
        self.norm1 = RMSNorm(config['d_model'])
        self.attn = ShawAttention(
            config['d_model'], config['n_heads'], config['d_head'], 
            num_rel_pos, config['dropout']
        )
        self.norm2 = RMSNorm(config['d_model'])
        self.ff = SwiGLU(config['d_model'], config['d_ff'], config['dropout'])

    def forward(self, x, rel_indices, smolgen_bias=None):
        x = x + self.attn(self.norm1(x), rel_indices=rel_indices, smolgen_bias=smolgen_bias)
        x = x + self.ff(self.norm2(x))
        return x


class PerSquareInputEncoder(nn.Module):
    """
    Encode per-square input features following the paper's approach.
    
    Per-token input = [
        8 × one-hot(piece_code, 13)  # 8 board states × 13 types = 104 dims
        + 1 × en_passant_flag        # 1 if this is the EP target square
        + 8 × time_history           # 8 floats for move times
        + 8 × repetition_flags       # 8 binary flags
        + 4 × castling_rights        # 4 binary flags
        + 8 × scalars                # normalized scalar features
    ] = 104 + 1 + 8 + 8 + 4 + 8 = 133 dims per token
    """
    def __init__(self, config):
        super().__init__()
        self.history_len = config['history_len']
        self.num_piece_types = config['num_piece_types']
        
        # Input dimension per token
        # 8 * 13 (piece one-hots) + 1 (ep) + 8 (time) + 8 (rep) + 4 (castle) + 8 (scalars)
        self.input_dim = (self.history_len * self.num_piece_types + 1 + 
                         self.history_len + self.history_len + 4 + 8)
        
        # Project to d_model
        self.input_proj = nn.Linear(self.input_dim, config['d_model'], bias=False)
        
        # Per-square absolute position embedding
        self.square_emb = nn.Embedding(64, config['d_model'])
        
        # Per-square scale and bias (paper's "multiply by learned offset")
        self.square_scale = nn.Parameter(torch.ones(64, config['d_model']))
        self.square_bias = nn.Parameter(torch.zeros(64, config['d_model']))
        
        # TC category embedding (added to all squares)
        self.tc_emb = nn.Embedding(config['num_tc_cats'], config['d_model'])
        
        # Leela-style: optional FFN after embedding
        # "We also add an FFN layer after the embedding so the model can make use of this information"
        self.use_embedding_ffn = config.get('embedding_ffn', False)
        if self.use_embedding_ffn:
            d_model = config['d_model']
            # Use 4x expansion for the embedding FFN (this is where capacity goes)
            self.emb_ffn = nn.Sequential(
                nn.Linear(d_model, d_model * 4),
                nn.GELU(),
                nn.Linear(d_model * 4, d_model),
            )
            self.emb_norm = RMSNorm(d_model)

    def forward(self, board_history, time_history, rep_flags, castling, 
                ep_mask, scalars, tc_cat):
        """
        Args:
            board_history: (B, 8, 64) int64 - piece codes per square per history step
            time_history: (B, 8) float - normalized time per historical move
            rep_flags: (B, 8) float - repetition flags per history step
            castling: (B, 4) float - castling rights
            ep_mask: (B, 64) float - 1.0 at EP square, else 0
            scalars: (B, 8) float - normalized scalar features
            tc_cat: (B,) long - time control category
        
        Returns:
            (B, 64, d_model) token embeddings
        """
        B = board_history.shape[0]
        device = board_history.device
        
        # 1. One-hot encode board history: (B, 8, 64, 13)
        board_onehot = F.one_hot(board_history, num_classes=self.num_piece_types).float()
        # Reshape to (B, 64, 8*13) - flatten history dimension per square
        board_onehot = board_onehot.permute(0, 2, 1, 3).reshape(B, 64, -1)
        
        # 2. Broadcast time_history to all squares: (B, 64, 8)
        time_hist_broadcast = time_history.unsqueeze(1).expand(-1, 64, -1)
        
        # 3. Broadcast rep_flags to all squares: (B, 64, 8)
        rep_broadcast = rep_flags.unsqueeze(1).expand(-1, 64, -1)
        
        # 4. Broadcast castling to all squares: (B, 64, 4)
        castle_broadcast = castling.unsqueeze(1).expand(-1, 64, -1)
        
        # 5. EP mask is already per-square: (B, 64, 1)
        ep_per_square = ep_mask.unsqueeze(-1)
        
        # 6. Broadcast scalars to all squares: (B, 64, 8)
        scalars_broadcast = scalars.unsqueeze(1).expand(-1, 64, -1)
        
        # Concatenate all features: (B, 64, input_dim)
        token_input = torch.cat([
            board_onehot,        # (B, 64, 104)
            ep_per_square,       # (B, 64, 1)
            time_hist_broadcast, # (B, 64, 8)
            rep_broadcast,       # (B, 64, 8)
            castle_broadcast,    # (B, 64, 4)
            scalars_broadcast,   # (B, 64, 8)
        ], dim=-1)
        
        # Project to d_model
        x = self.input_proj(token_input)  # (B, 64, d_model)
        
        # Add absolute position embedding
        square_ids = torch.arange(64, device=device)
        x = x + self.square_emb(square_ids)
        
        # Apply per-square scale and bias
        x = x * self.square_scale + self.square_bias
        
        # Add TC category embedding (broadcast to all squares)
        tc_emb = self.tc_emb(tc_cat)  # (B, d_model)
        x = x + tc_emb.unsqueeze(1)
        
        # Leela-style: FFN after embedding to let model make use of full board info
        if self.use_embedding_ffn:
            x = x + self.emb_ffn(self.emb_norm(x))
        
        return x


class ChessformerV2(nn.Module):
    """
    Chessformer v2 with LEVEL_UP_PLAN architecture changes.
    """
    def __init__(self, config):
        super().__init__()
        self.config = config
        d_model = config['d_model']
        
        # Input encoder
        self.input_encoder = PerSquareInputEncoder(config)
        
        # Transformer body
        self.rel_pos_gen = RelativePositionGenerator(config['max_rel_dist'])
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
        
        # Value head
        self.value_head = nn.Sequential(
            nn.Linear(d_model, 256, bias=False),
            nn.SiLU(),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )
        
        # Time head (regression instead of bins for v2)
        self.time_head = nn.Sequential(
            nn.Linear(d_model, 256, bias=False),
            nn.SiLU(),
            nn.Linear(256, 1),
            nn.Softplus()  # Ensure positive output
        )
        
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, batch, return_promo=False):
        """
        Args:
            batch: dict with keys from dataset_v2
            return_promo: if True, also return promotion logits
        
        Returns:
            move_logits: (B, 4098) policy logits (masked if legal_mask provided)
            value_out: (B, 1) win probability
            time_out: (B, 1) predicted time in normalized units
            promo_logits: (B, 8, 4) promotion logits for dest squares 56-63 (optional)
        """
        # Extract inputs from batch
        board_history = batch['board_history']
        time_history = batch['time_history']
        rep_flags = batch['rep_flags']
        castling = batch['castling']
        ep_mask = batch['ep_mask']
        scalars = batch['scalars']
        tc_cat = batch['tc_cat']
        legal_mask = batch.get('legal_mask', None)
        
        B = board_history.shape[0]
        device = board_history.device
        
        # Encode inputs
        x = self.input_encoder(
            board_history, time_history, rep_flags, castling,
            ep_mask, scalars, tc_cat
        )
        
        # Compute smolgen bias once (shared across layers)
        smolgen_bias = None
        if self.use_smolgen:
            smolgen_bias = self.smolgen(x)
        
        # Transformer layers
        rel_indices = self.rel_pos_gen()
        for layer in self.layers:
            x = layer(x, rel_indices=rel_indices, smolgen_bias=smolgen_bias)
        x = self.norm_f(x)  # (B, 64, d_model)
        
        # --- Policy Head ---
        pol_q = self.policy_query(x)  # (B, 64, d_model)
        pol_k = self.policy_key(x)    # (B, 64, d_model)
        
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
        
        # Apply legal move mask if provided
        if legal_mask is not None:
            # For resign/flag: always allow during training (they're targets, not options)
            # But we set them to large negative unless they're the actual target
            # Actually, we want to mask them during normal move prediction
            # Let's set them to -inf unless they're explicitly legal
            move_logits = move_logits.masked_fill(~legal_mask, float('-inf'))
        
        # --- Promotion Head ---
        # Get key vectors for promotion rank (squares 56-63)
        promo_keys = pol_k[:, 56:64, :]  # (B, 8, d_model)
        promo_logits = self.promo_bias_proj(promo_keys)  # (B, 8, 4)
        
        # --- Value and Time Heads ---
        x_pooled = x.mean(dim=1)  # (B, d_model)
        value_out = self.value_head(x_pooled)  # (B, 1)
        time_out = self.time_head(x_pooled)    # (B, 1)
        
        if return_promo:
            return move_logits, value_out, time_out, promo_logits
        return move_logits, value_out, time_out


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    # Test the model
    model = ChessformerV2(CONFIG_V2_SMALL)
    print(f"Total Trainable Parameters: {count_parameters(model):,}")
    
    # Create dummy batch
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
    
    move_logits, value_out, time_out = model(dummy_batch)
    print(f"Move logits: {move_logits.shape}")  # (2, 4098)
    print(f"Value: {value_out.shape}")           # (2, 1)
    print(f"Time: {time_out.shape}")             # (2, 1)
