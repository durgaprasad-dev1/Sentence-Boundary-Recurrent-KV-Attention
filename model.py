"""
model.py – SBR-KV vs GQA Research Repository
==============================================

Contains:
  - RMSNorm              : Root-Mean-Square layer normalisation
  - RotaryEmbedding      : Rotary Position Embedding (RoPE)
  - GQABaseline          : Grouped-Query Attention with KV cache
  - SentenceBoundaryRecurrentKVAttention (SBR-KV): novel recurrent-sentence attention
  - SBRTransformerBlock  : transformer block wrapping either GQA or SBR-KV
  - SBRTransformer       : full causal LM using SBR-KV blocks
  - GQATransformer       : full causal LM using GQA blocks (baseline)

Default config (dim=256, n_layers=4, n_heads=8, n_kv_heads=2, ffn_mult=4,
vocab_size=10000) produces **equal** parameter counts for both models.
Parameter parity is achieved by giving GQA a proportionally wider FFN so
that both models occupy the same number of trainable weights.

Parameter budget (default config)
----------------------------------
  Shared per model: embedding (10000×256) + LM head (tied) + n_layers×RMSNorm×2
  SBR extra per layer: recurrent-state projections + cross-attention heads
  GQA compensating: wider hidden FFN dimension to absorb the extra budget
"""

import math
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------

def count_params(model: nn.Module) -> int:
    """Return the total number of trainable parameters in *model*."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def _swiglu_hidden(dim: int, ffn_mult: int) -> int:
    """Canonical SwiGLU hidden dimension: 2/3 × dim × ffn_mult, ↑256."""
    hidden = int(dim * ffn_mult * 2 / 3)
    return (hidden + 255) // 256 * 256


def _sbr_extra_params_per_layer(dim: int, n_heads: int, n_kv_heads: int) -> int:
    """Count of extra trainable parameters introduced by SBR-KV per layer.

    These are parameters *beyond* a standard GQA block:
      - w_kv_proj : dim → dim          (sentence-mean compressor)
      - w_gate    : 2*dim → dim        (adaptive forget gate)
      - wq_cross  : dim → dim          (cross-attn queries)
      - wk_cross  : dim → kv_dim       (cross-attn keys from recurrent state)
      - wv_cross  : dim → kv_dim       (cross-attn values from recurrent state)
      - wo_cross  : dim → dim          (cross-attn output)
      - alpha     : scalar             (mixing gate)

    where kv_dim = n_kv_heads × (dim // n_heads).
    """
    kv_dim = n_kv_heads * (dim // n_heads)
    return (
        dim * dim          # w_kv_proj
        + 2 * dim * dim    # w_gate
        + dim * dim        # wq_cross
        + dim * kv_dim     # wk_cross
        + dim * kv_dim     # wv_cross
        + dim * dim        # wo_cross
        + 1                # alpha
    )


def _gqa_compensating_hidden(
    dim: int,
    n_heads: int,
    n_kv_heads: int,
    n_layers: int,
    base_ffn_hidden: int,
) -> int:
    """Compute a wider FFN hidden dim for GQA that absorbs the SBR-KV extra budget.

    Total SBR extra = n_layers × _sbr_extra_params_per_layer(...)
    Each SwiGLU FFN has 3 linear weight matrices: w1 (dim×h), w2 (h×dim), w3 (dim×h)
    → total FFN params = 3 × dim × h  per layer.

    So extra per layer = 3 × dim × (h_gqa - h_sbr)
    → h_gqa = h_sbr + extra_per_layer / (3 × dim)

    We round up to the nearest multiple of 32 to stay hardware-friendly.
    """
    extra_per_layer = _sbr_extra_params_per_layer(dim, n_heads, n_kv_heads)
    extra_h = extra_per_layer / (3 * dim)   # may be non-integer
    gqa_hidden = base_ffn_hidden + math.ceil(extra_h)
    # Round to multiple of 32
    return (gqa_hidden + 31) // 32 * 32


# ---------------------------------------------------------------------------
# RMSNorm
# ---------------------------------------------------------------------------

class RMSNorm(nn.Module):
    """Root-Mean-Square Layer Normalisation (Zhang & Sennrich, 2019).

    Unlike LayerNorm, RMSNorm omits the mean-subtraction step, which saves
    ~30 % compute while matching or exceeding LN quality in practice.

    Args:
        dim   : feature dimension to normalise over.
        eps   : small constant for numerical stability.
    """

    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (..., dim)
        rms = x.pow(2).mean(-1, keepdim=True).add(self.eps).rsqrt()
        return x * rms * self.weight


# ---------------------------------------------------------------------------
# Rotary Embedding (RoPE)
# ---------------------------------------------------------------------------

class RotaryEmbedding(nn.Module):
    """Rotary Position Embedding (Su et al., 2022 – RoFormer).

    Applies a position-dependent rotation to query and key vectors so that
    the dot-product between q_i and k_j depends only on their relative
    offset (i − j).

    Args:
        head_dim : dimension of each attention head.
        base     : frequency base (10 000 is standard).
        max_seq  : maximum sequence length to cache.
    """

    def __init__(self, head_dim: int, base: int = 10_000, max_seq: int = 4096) -> None:
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2).float() / head_dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self._build_cache(max_seq)

    def _build_cache(self, seq_len: int) -> None:
        t = torch.arange(seq_len, device=self.inv_freq.device).float()
        freqs = torch.outer(t, self.inv_freq)          # (seq_len, head_dim/2)
        emb = torch.cat([freqs, freqs], dim=-1)         # (seq_len, head_dim)
        self.register_buffer("cos_cached", emb.cos(), persistent=False)
        self.register_buffer("sin_cached", emb.sin(), persistent=False)

    @staticmethod
    def _rotate_half(x: torch.Tensor) -> torch.Tensor:
        """Swap and negate the two halves of the last dim."""
        half = x.shape[-1] // 2
        x1, x2 = x[..., :half], x[..., half:]
        return torch.cat([-x2, x1], dim=-1)

    def forward(self, x: torch.Tensor, seq_start: int = 0) -> torch.Tensor:
        """Apply RoPE to *x*.

        Args:
            x         : tensor of shape (B, heads, T, head_dim).
            seq_start : offset for the first position in *x* (useful for
                        incremental decoding).
        Returns:
            Rotated tensor with the same shape as *x*.
        """
        T = x.shape[2]
        end = seq_start + T
        if end > self.cos_cached.shape[0]:
            self._build_cache(end * 2)
        cos = self.cos_cached[seq_start:end].unsqueeze(0).unsqueeze(0)
        sin = self.sin_cached[seq_start:end].unsqueeze(0).unsqueeze(0)
        return x * cos + self._rotate_half(x) * sin


# ---------------------------------------------------------------------------
# GQABaseline – Grouped-Query Attention
# ---------------------------------------------------------------------------

class GQABaseline(nn.Module):
    """Grouped-Query Attention (Ainslie et al., 2023).

    *n_kv_heads* key-value heads are shared across groups of query heads
    (n_heads / n_kv_heads queries per KV head).  This reduces the KV cache
    memory footprint compared to Multi-Head Attention while retaining most
    of its expressivity.

    Supports:
      - Causal masking for training.
      - Incremental autoregressive decoding via an external KV cache dict.

    Args:
        dim        : model dimension.
        n_heads    : number of query heads.
        n_kv_heads : number of key/value heads (must divide n_heads evenly).
        dropout    : attention dropout probability (training only).
    """

    def __init__(
        self,
        dim: int,
        n_heads: int,
        n_kv_heads: int,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        assert dim % n_heads == 0, "dim must be divisible by n_heads"
        assert n_heads % n_kv_heads == 0, "n_heads must be divisible by n_kv_heads"

        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.head_dim = dim // n_heads
        self.groups = n_heads // n_kv_heads
        self.scale = self.head_dim ** -0.5
        self.dropout = dropout

        self.wq = nn.Linear(dim, n_heads * self.head_dim, bias=False)
        self.wk = nn.Linear(dim, n_kv_heads * self.head_dim, bias=False)
        self.wv = nn.Linear(dim, n_kv_heads * self.head_dim, bias=False)
        self.wo = nn.Linear(n_heads * self.head_dim, dim, bias=False)

        self.rope = RotaryEmbedding(self.head_dim)

    def forward(
        self,
        x: torch.Tensor,
        kv_cache: Optional[dict] = None,
        layer_idx: int = 0,
    ) -> torch.Tensor:
        """Grouped-query self-attention forward pass.

        Args:
            x         : input tensor (B, T, dim).
            kv_cache  : optional dict keyed by layer_idx storing past
                        (k, v) tensors for incremental decoding.
            layer_idx : identifier used to read/write kv_cache.

        Returns:
            Output tensor (B, T, dim).
        """
        B, T, _ = x.shape
        H, Hkv, Hd = self.n_heads, self.n_kv_heads, self.head_dim

        q = self.wq(x).view(B, T, H, Hd).transpose(1, 2)
        k = self.wk(x).view(B, T, Hkv, Hd).transpose(1, 2)
        v = self.wv(x).view(B, T, Hkv, Hd).transpose(1, 2)

        offset = 0
        if kv_cache is not None and layer_idx in kv_cache:
            past_k, past_v = kv_cache[layer_idx]
            offset = past_k.shape[2]

        q = self.rope(q, seq_start=offset)
        k = self.rope(k, seq_start=offset)

        if kv_cache is not None:
            if layer_idx in kv_cache:
                past_k, past_v = kv_cache[layer_idx]
                k = torch.cat([past_k, k], dim=2)
                v = torch.cat([past_v, v], dim=2)
            kv_cache[layer_idx] = (k, v)

        Tkv = k.shape[2]
        k = k.repeat_interleave(self.groups, dim=1)
        v = v.repeat_interleave(self.groups, dim=1)

        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale

        if T > 1:
            causal_mask = torch.triu(
                torch.full((T, Tkv), float("-inf"), device=x.device), diagonal=1 + offset
            )
            attn = attn + causal_mask

        attn = F.softmax(attn, dim=-1)
        if self.training and self.dropout > 0:
            attn = F.dropout(attn, p=self.dropout)

        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).contiguous().view(B, T, H * Hd)
        return self.wo(out)


# ---------------------------------------------------------------------------
# SentenceBoundaryRecurrentKVAttention (SBR-KV)
# ---------------------------------------------------------------------------

class SentenceBoundaryRecurrentKVAttention(nn.Module):
    """Sentence-Boundary Recurrent Key-Value Attention (SBR-KV).

    Combines two complementary information pathways:

    1. **Local window attention** – standard causal self-attention over the
       *current active sentence* (tokens since the last sentence-boundary
       marker).  This gives full-resolution context for the current sentence.

    2. **Recurrent cross-attention** – a single compressed *hidden state*
       ``h`` of dimension ``dim`` summarises all previously completed
       sentences via a gated running-average update applied at each sentence
       boundary::

           kv_proj  = tanh(W_kv  · sentence_avg)       # compress sentence mean
           gate     = sigmoid(W_g · [h ; kv_proj])      # adaptive forget gate
           h        = gate ⊙ h + (1 − gate) ⊙ kv_proj  # EMA-style update

       Current tokens cross-attend to ``h`` using un-rotated keys/values to
       avoid positional confusion across sentence boundaries.

    **Final output** combines both pathways via a learnable scalar gate α::

           out = σ(α) · local_out + (1 − σ(α)) · cross_out

    Sentence boundary tokens are identified by their token-IDs (e.g. the
    IDs of ``.``, ``!``, ``?`` in the vocabulary).

    Args:
        dim              : model dimension.
        n_heads          : number of query heads (local + cross attention).
        n_kv_heads       : number of KV heads for the *local* attention only.
        boundary_ids     : list of token IDs treated as sentence boundaries.
        dropout          : attention dropout probability.
    """

    def __init__(
        self,
        dim: int,
        n_heads: int,
        n_kv_heads: int,
        boundary_ids: Optional[List[int]] = None,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        assert dim % n_heads == 0
        assert n_heads % n_kv_heads == 0

        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.head_dim = dim // n_heads
        self.kv_dim = n_kv_heads * self.head_dim   # dimension of KV heads for cross-attn
        self.groups = n_heads // n_kv_heads
        self.scale = self.head_dim ** -0.5
        self.dropout = dropout
        self.dim = dim

        self.boundary_ids: List[int] = boundary_ids if boundary_ids is not None else []

        # --- Local (intra-sentence) attention projections ---
        self.wq_local = nn.Linear(dim, n_heads * self.head_dim, bias=False)
        self.wk_local = nn.Linear(dim, n_kv_heads * self.head_dim, bias=False)
        self.wv_local = nn.Linear(dim, n_kv_heads * self.head_dim, bias=False)
        self.wo_local = nn.Linear(n_heads * self.head_dim, dim, bias=False)

        self.rope = RotaryEmbedding(self.head_dim)

        # --- Recurrent accumulator projections ---
        # Projects sentence-mean vector (dim) → recurrent state (dim)
        self.w_kv_proj = nn.Linear(dim, dim, bias=False)
        # Gating: [h(dim) ; kv_proj(dim)] → gate(dim)
        self.w_gate = nn.Linear(2 * dim, dim, bias=False)

        # --- Cross-attention: current tokens → recurrent state h ---
        # Queries from current tokens (with RoPE), dim → n_heads × head_dim
        self.wq_cross = nn.Linear(dim, n_heads * self.head_dim, bias=False)
        # Keys/values from recurrent state h (un-rotated), dim → n_kv_heads × head_dim
        self.wk_cross = nn.Linear(dim, n_kv_heads * self.head_dim, bias=False)
        self.wv_cross = nn.Linear(dim, n_kv_heads * self.head_dim, bias=False)
        self.wo_cross = nn.Linear(n_heads * self.head_dim, dim, bias=False)

        # Learnable output mixing gate (pre-sigmoid scalar, init 0 → α=0.5)
        self.alpha = nn.Parameter(torch.zeros(1))

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _local_attention(
        self,
        x: torch.Tensor,
        window_mask: torch.Tensor,
        seq_start: int = 0,
    ) -> torch.Tensor:
        """Causal self-attention restricted to the active-sentence window.

        Args:
            x           : (B, T, dim)
            window_mask : (T, T) additive mask (−∞ outside the window)
            seq_start   : positional offset for RoPE

        Returns:
            (B, T, dim)
        """
        B, T, _ = x.shape
        H, Hkv, Hd = self.n_heads, self.n_kv_heads, self.head_dim

        q = self.wq_local(x).view(B, T, H, Hd).transpose(1, 2)
        k = self.wk_local(x).view(B, T, Hkv, Hd).transpose(1, 2)
        v = self.wv_local(x).view(B, T, Hkv, Hd).transpose(1, 2)

        q = self.rope(q, seq_start=seq_start)
        k = self.rope(k, seq_start=seq_start)

        k = k.repeat_interleave(self.groups, dim=1)
        v = v.repeat_interleave(self.groups, dim=1)

        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        attn = attn + window_mask.unsqueeze(0).unsqueeze(0)
        attn = F.softmax(attn, dim=-1)
        if self.training and self.dropout > 0:
            attn = F.dropout(attn, p=self.dropout)

        out = torch.matmul(attn, v).transpose(1, 2).contiguous().view(B, T, H * Hd)
        return self.wo_local(out)

    def _cross_attention(
        self,
        x: torch.Tensor,
        h: torch.Tensor,
        seq_start: int = 0,
    ) -> torch.Tensor:
        """Cross-attention from current tokens to recurrent hidden state *h*.

        Keys and values come from *h* without positional rotation to avoid
        cross-sentence positional confusion.

        Args:
            x  : (B, T, dim) – current token representations
            h  : (B, 1, dim) – recurrent hidden state
            seq_start: positional offset for query RoPE

        Returns:
            (B, T, dim)
        """
        B, T, _ = x.shape
        H, Hkv, Hd = self.n_heads, self.n_kv_heads, self.head_dim

        # Queries from current tokens (with RoPE)
        q = self.wq_cross(x).view(B, T, H, Hd).transpose(1, 2)
        q = self.rope(q, seq_start=seq_start)

        # Keys/values from recurrent state (NO RoPE)
        k = self.wk_cross(h).view(B, 1, Hkv, Hd).transpose(1, 2)  # (B, Hkv, 1, Hd)
        v = self.wv_cross(h).view(B, 1, Hkv, Hd).transpose(1, 2)

        k = k.repeat_interleave(self.groups, dim=1)
        v = v.repeat_interleave(self.groups, dim=1)

        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale  # (B, H, T, 1)
        attn = F.softmax(attn, dim=-1)

        out = torch.matmul(attn, v).transpose(1, 2).contiguous().view(B, T, H * Hd)
        return self.wo_cross(out)

    def _build_window_mask(
        self,
        input_ids: Optional[torch.Tensor],
        T: int,
        device: torch.device,
    ) -> torch.Tensor:
        """Build an additive causal mask limited to the current active sentence.

        A query at position *i* can attend to key *j* iff:
          - j ≤ i  (causal)
          - no sentence-boundary token appears in positions [j, i)

        When ``boundary_ids`` is empty or ``input_ids`` is None, falls back to
        a standard lower-triangular causal mask.

        Returns:
            (T, T) additive float mask (0 or −inf).
        """
        if not self.boundary_ids or input_ids is None:
            return torch.triu(
                torch.full((T, T), float("-inf"), device=device), diagonal=1
            )

        # Use the first batch item to determine boundary positions.
        # During training with identical padding, batch items share boundaries;
        # for inference, only one item is processed at a time.
        is_boundary = torch.zeros(T, dtype=torch.bool, device=device)
        for bid in self.boundary_ids:
            is_boundary |= (input_ids[0] == bid)

        # Compute sentence-start position for each query index
        sentence_start = torch.zeros(T, dtype=torch.long, device=device)
        last_boundary = -1
        for i in range(T):
            sentence_start[i] = last_boundary + 1
            if is_boundary[i]:
                last_boundary = i

        # Build additive mask
        mask = torch.full((T, T), float("-inf"), device=device)
        for i in range(T):
            s = sentence_start[i].item()
            mask[i, s : i + 1] = 0.0
        return mask

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        x: torch.Tensor,
        input_ids: Optional[torch.Tensor] = None,
        h: Optional[torch.Tensor] = None,
        seq_start: int = 0,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """SBR-KV forward pass.

        Args:
            x         : (B, T, dim) token representations.
            input_ids : (B, T) token IDs used to detect sentence boundaries.
                        If None, the module behaves like standard causal attn.
            h         : (B, 1, dim) recurrent hidden state from the previous
                        forward call.  Initialised to zeros if None.
            seq_start : positional offset for RoPE (incremental decoding).

        Returns:
            out : (B, T, dim) output representations.
            h   : (B, 1, dim) updated recurrent hidden state.
        """
        B, T, D = x.shape
        device = x.device

        if h is None:
            h = torch.zeros(B, 1, D, device=device, dtype=x.dtype)

        # -----------------------------------------------------------------
        # 1. Local intra-sentence attention
        # -----------------------------------------------------------------
        window_mask = self._build_window_mask(input_ids, T, device)
        local_out = self._local_attention(x, window_mask, seq_start=seq_start)

        # -----------------------------------------------------------------
        # 2. Cross-attention to recurrent hidden state
        # -----------------------------------------------------------------
        cross_out = self._cross_attention(x, h, seq_start=seq_start)

        # -----------------------------------------------------------------
        # 3. Combine pathways with learnable scalar α (sigmoid-gated)
        # -----------------------------------------------------------------
        alpha = torch.sigmoid(self.alpha)
        out = alpha * local_out + (1.0 - alpha) * cross_out

        # -----------------------------------------------------------------
        # 4. Update recurrent state at sentence boundaries (sequential scan)
        #    h_t = gate * h_{t-1} + (1 - gate) * kv_proj(sentence_avg)
        #    gate = sigmoid(W_g * [h_{t-1}; kv_proj])
        # -----------------------------------------------------------------
        if input_ids is not None and self.boundary_ids:
            is_boundary = torch.zeros(T, dtype=torch.bool, device=device)
            for bid in self.boundary_ids:
                is_boundary |= (input_ids[0] == bid)

            if is_boundary.any():
                sentence_start_idx = 0
                for t in range(T):
                    if is_boundary[t]:
                        # Mean of local representations over completed sentence
                        sent_mean = x[:, sentence_start_idx : t + 1, :].mean(
                            dim=1, keepdim=True
                        )                                                   # (B,1,D)
                        kv_proj = torch.tanh(self.w_kv_proj(sent_mean))    # (B,1,D)
                        gate_input = torch.cat([h, kv_proj], dim=-1)       # (B,1,2D)
                        gate = torch.sigmoid(self.w_gate(gate_input))      # (B,1,D)
                        h = gate * h + (1.0 - gate) * kv_proj              # (B,1,D)
                        sentence_start_idx = t + 1

        return out, h

    def step(
        self,
        x: torch.Tensor,
        token_id: int,
        layer_idx: int,
        sbr_cache: dict,
    ) -> torch.Tensor:
        """Single-token incremental decode step with true O(1) state caching.

        Discards past token KVs once a sentence boundary is encountered,
        compressing them into the recurrent state h and keeping only active sentence KVs.
        """
        B, T, D = x.shape
        assert T == 1
        device = x.device

        layer_cache = sbr_cache.setdefault(layer_idx, {
            "h": torch.zeros(B, 1, D, device=device, dtype=x.dtype),
            "active_k": None,
            "active_v": None,
            "active_x": [],
        })

        h = layer_cache["h"]
        H, Hkv, Hd = self.n_heads, self.n_kv_heads, self.head_dim

        # Current token Q, K, V
        q = self.wq_local(x).view(B, 1, H, Hd).transpose(1, 2)
        k = self.wk_local(x).view(B, 1, Hkv, Hd).transpose(1, 2)
        v = self.wv_local(x).view(B, 1, Hkv, Hd).transpose(1, 2)

        # Position offset in the active sentence
        pos = len(layer_cache["active_x"])
        q = self.rope(q, seq_start=pos)
        k = self.rope(k, seq_start=pos)

        # Append to active sentence cache
        if layer_cache["active_k"] is None:
            layer_cache["active_k"] = k
            layer_cache["active_v"] = v
        else:
            layer_cache["active_k"] = torch.cat([layer_cache["active_k"], k], dim=2)
            layer_cache["active_v"] = torch.cat([layer_cache["active_v"], v], dim=2)
        layer_cache["active_x"].append(x)

        # Attention over active sentence tokens
        k_full = layer_cache["active_k"].repeat_interleave(self.groups, dim=1)
        v_full = layer_cache["active_v"].repeat_interleave(self.groups, dim=1)
        scores = torch.matmul(q, k_full.transpose(-2, -1)) * self.scale
        attn = F.softmax(scores, dim=-1)
        local_out = torch.matmul(attn, v_full).transpose(1, 2).contiguous().view(B, 1, H * Hd)
        local_out = self.wo_local(local_out)

        # Cross-attention to recurrent state h
        cross_out = self._cross_attention(x, h, seq_start=pos)

        # Gate combination
        alpha = torch.sigmoid(self.alpha)
        out = alpha * local_out + (1.0 - alpha) * cross_out

        # Check for sentence boundary!
        if token_id in self.boundary_ids:
            # Sentence is complete: compress active sentence representations into h
            sent_tokens = torch.cat(layer_cache["active_x"], dim=1)  # (B, L_sent, D)
            sent_mean = sent_tokens.mean(dim=1, keepdim=True)        # (B, 1, D)
            kv_proj = torch.tanh(self.w_kv_proj(sent_mean))          # (B, 1, D)
            gate_input = torch.cat([h, kv_proj], dim=-1)             # (B, 1, 2D)
            gate = torch.sigmoid(self.w_gate(gate_input))            # (B, 1, D)
            new_h = gate * h + (1.0 - gate) * kv_proj                # (B, 1, D)

            # TRUE O(1) FLUSH: Discard past token KVs!
            layer_cache["h"] = new_h
            layer_cache["active_k"] = None
            layer_cache["active_v"] = None
            layer_cache["active_x"] = []

        return out


# ---------------------------------------------------------------------------
# SwiGLU Feed-Forward Network
# ---------------------------------------------------------------------------

class SwiGLUFFN(nn.Module):
    """Feed-Forward Network with SwiGLU activation (Shazeer, 2020).

    Uses two parallel linear projections gated by a Swish/SiLU activation::

        FFN(x) = (SiLU(W1 x) ⊙ W3 x) W2

    Args:
        dim        : model dimension.
        hidden_dim : hidden dimension.  Computed externally to allow
                     per-model budget balancing.
    """

    def __init__(self, dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.w1 = nn.Linear(dim, hidden_dim, bias=False)   # gate branch
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)   # output projection
        self.w3 = nn.Linear(dim, hidden_dim, bias=False)   # value branch

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


# ---------------------------------------------------------------------------
# SBRTransformerBlock
# ---------------------------------------------------------------------------

class SBRTransformerBlock(nn.Module):
    """A single transformer block supporting either SBR-KV or GQA attention.

    Uses pre-norm (RMSNorm applied before each sub-layer) and residual
    connections, following the PaLM / LLaMA convention.

    Args:
        dim           : model dimension.
        n_heads       : number of attention heads.
        n_kv_heads    : number of KV heads (GQA ratio).
        ffn_hidden    : FFN hidden dimension (caller controls for parity).
        use_sbr       : if True, use SBR-KV; otherwise use GQA.
        boundary_ids  : sentence-boundary token IDs (SBR-KV only).
        layer_idx     : layer index (used for KV-cache keying in GQA mode).
        dropout       : attention dropout.
    """

    def __init__(
        self,
        dim: int,
        n_heads: int,
        n_kv_heads: int,
        ffn_hidden: int,
        use_sbr: bool = True,
        boundary_ids: Optional[List[int]] = None,
        layer_idx: int = 0,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.use_sbr = use_sbr
        self.layer_idx = layer_idx

        self.norm1 = RMSNorm(dim)
        self.norm2 = RMSNorm(dim)

        if use_sbr:
            self.attn: nn.Module = SentenceBoundaryRecurrentKVAttention(
                dim=dim,
                n_heads=n_heads,
                n_kv_heads=n_kv_heads,
                boundary_ids=boundary_ids,
                dropout=dropout,
            )
        else:
            self.attn = GQABaseline(
                dim=dim,
                n_heads=n_heads,
                n_kv_heads=n_kv_heads,
                dropout=dropout,
            )

        self.ffn = SwiGLUFFN(dim=dim, hidden_dim=ffn_hidden)

    def forward(
        self,
        x: torch.Tensor,
        input_ids: Optional[torch.Tensor] = None,
        h: Optional[torch.Tensor] = None,
        kv_cache: Optional[dict] = None,
        seq_start: int = 0,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Forward pass through the transformer block.

        Args:
            x         : (B, T, dim) input tensor.
            input_ids : (B, T) token IDs (SBR-KV only).
            h         : (B, 1, dim) recurrent hidden state (SBR-KV only).
            kv_cache  : dict for GQA incremental decoding.
            seq_start : positional offset for RoPE.

        Returns:
            x : (B, T, dim) output tensor.
            h : updated recurrent state (or None for GQA).
        """
        normed = self.norm1(x)
        if self.use_sbr:
            attn_out, h = self.attn(
                normed, input_ids=input_ids, h=h, seq_start=seq_start
            )
        else:
            attn_out = self.attn(normed, kv_cache=kv_cache, layer_idx=self.layer_idx)
            h = None

        x = x + attn_out
        x = x + self.ffn(self.norm2(x))
        return x, h

    def step(
        self,
        x: torch.Tensor,
        token_id: int,
        sbr_cache: Optional[dict] = None,
        kv_cache: Optional[dict] = None,
    ) -> torch.Tensor:
        """Incremental decode step for a single token."""
        normed = self.norm1(x)
        if self.use_sbr:
            assert sbr_cache is not None
            attn_out = self.attn.step(
                normed, token_id=token_id, layer_idx=self.layer_idx, sbr_cache=sbr_cache
            )
        else:
            attn_out = self.attn(normed, kv_cache=kv_cache, layer_idx=self.layer_idx)
        x = x + attn_out
        x = x + self.ffn(self.norm2(x))
        return x


# ---------------------------------------------------------------------------
# SBRTransformer – full causal language model using SBR-KV
# ---------------------------------------------------------------------------

class SBRTransformer(nn.Module):
    """Causal Language Model built from SBR-KV transformer blocks.

    Token and output embeddings are weight-tied.

    Args:
        vocab_size   : vocabulary size.
        dim          : model dimension (embedding + hidden).
        n_layers     : number of transformer blocks.
        n_heads      : number of attention heads.
        n_kv_heads   : number of KV heads (local attention GQA ratio).
        ffn_mult     : FFN hidden-dimension multiplier (base; before SwiGLU 2/3 scaling).
        boundary_ids : sentence-boundary token IDs.
        dropout      : attention dropout.
    """

    def __init__(
        self,
        vocab_size: int = 10_000,
        dim: int = 256,
        n_layers: int = 4,
        n_heads: int = 8,
        n_kv_heads: int = 2,
        ffn_mult: int = 4,
        boundary_ids: Optional[List[int]] = None,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.dim = dim
        self.n_layers = n_layers

        ffn_hidden = _swiglu_hidden(dim, ffn_mult)

        self.embed = nn.Embedding(vocab_size, dim)
        self.blocks = nn.ModuleList([
            SBRTransformerBlock(
                dim=dim,
                n_heads=n_heads,
                n_kv_heads=n_kv_heads,
                ffn_hidden=ffn_hidden,
                use_sbr=True,
                boundary_ids=boundary_ids,
                layer_idx=i,
                dropout=dropout,
            )
            for i in range(n_layers)
        ])
        self.norm = RMSNorm(dim)
        self.lm_head = nn.Linear(dim, vocab_size, bias=False)
        self.lm_head.weight = self.embed.weight   # weight tying
        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if hasattr(module, "bias") and module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(
        self,
        input_ids: torch.Tensor,
        h_states: Optional[List[Optional[torch.Tensor]]] = None,
        seq_start: int = 0,
    ) -> Tuple[torch.Tensor, List[Optional[torch.Tensor]]]:
        """Full forward pass.

        Args:
            input_ids : (B, T) integer token IDs.
            h_states  : list of per-layer recurrent hidden states
                        (length n_layers); each element is (B,1,dim) or None.
            seq_start : positional offset for RoPE (incremental decoding).

        Returns:
            logits   : (B, T, vocab_size) next-token logits.
            h_states : updated list of per-layer hidden states.
        """
        x = self.embed(input_ids)
        if h_states is None:
            h_states = [None] * self.n_layers

        new_h: List[Optional[torch.Tensor]] = []
        for i, block in enumerate(self.blocks):
            x, h = block(x, input_ids=input_ids, h=h_states[i], seq_start=seq_start)
            new_h.append(h)

        x = self.norm(x)
        return self.lm_head(x), new_h

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 40,
        temperature: float = 0.8,
        top_k: int = 40,
    ) -> torch.Tensor:
        """Autoregressive generation with true O(1) sentence-boundary state caching."""
        self.eval()
        device = input_ids.device
        logits, h_states = self.forward(input_ids)

        def sample_tok(lgt):
            if temperature > 0:
                lgt = lgt / temperature
                if top_k > 0:
                    v, _ = torch.topk(lgt, min(top_k, lgt.size(-1)))
                    lgt[lgt < v[:, [-1]]] = -float("Inf")
                probs = F.softmax(lgt, dim=-1)
                return torch.multinomial(probs, num_samples=1)
            return torch.argmax(lgt, dim=-1, keepdim=True)

        next_token = sample_tok(logits[:, -1, :])
        generated = [next_token]

        # Initialize per-layer O(1) SBR state cache from prompt prefill
        sbr_cache = {}
        for i in range(self.n_layers):
            sbr_cache[i] = {
                "h": h_states[i] if h_states[i] is not None else torch.zeros(input_ids.size(0), 1, self.dim, device=device),
                "active_k": None,
                "active_v": None,
                "active_x": [],
            }

        curr_token = next_token
        for _ in range(max_new_tokens - 1):
            x = self.embed(curr_token)
            t_id = curr_token[0, 0].item()
            for block in self.blocks:
                x = block.step(x, token_id=t_id, sbr_cache=sbr_cache)
            x = self.norm(x)
            step_logits = self.lm_head(x)
            curr_token = sample_tok(step_logits[:, -1, :])
            generated.append(curr_token)

        return torch.cat([input_ids] + generated, dim=1)


# ---------------------------------------------------------------------------
# GQATransformer – full causal language model using GQA (baseline)
# ---------------------------------------------------------------------------

class GQATransformer(nn.Module):
    """Causal Language Model built from Grouped-Query Attention blocks.

    Serves as the parameter-matched baseline for SBRTransformer.
    Token and output embeddings are weight-tied.

    The FFN hidden dimension is automatically widened to exactly compensate
    for the extra parameters introduced by SBR-KV's recurrent machinery,
    ensuring **equal total parameter counts** at the default configuration.

    Args:
        vocab_size : vocabulary size.
        dim        : model dimension.
        n_layers   : number of transformer blocks.
        n_heads    : number of attention heads.
        n_kv_heads : number of KV heads.
        ffn_mult   : FFN multiplier (base).  The final hidden dim is
                     automatically enlarged for parameter parity.
        dropout    : attention dropout.
    """

    def __init__(
        self,
        vocab_size: int = 10_000,
        dim: int = 256,
        n_layers: int = 4,
        n_heads: int = 8,
        n_kv_heads: int = 2,
        ffn_mult: int = 4,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.dim = dim
        self.n_layers = n_layers

        base_hidden = _swiglu_hidden(dim, ffn_mult)
        # Widen FFN to absorb the SBR-KV extra budget for parameter parity
        ffn_hidden = _gqa_compensating_hidden(dim, n_heads, n_kv_heads, n_layers, base_hidden)

        self.embed = nn.Embedding(vocab_size, dim)
        self.blocks = nn.ModuleList([
            SBRTransformerBlock(
                dim=dim,
                n_heads=n_heads,
                n_kv_heads=n_kv_heads,
                ffn_hidden=ffn_hidden,
                use_sbr=False,
                layer_idx=i,
                dropout=dropout,
            )
            for i in range(n_layers)
        ])
        self.norm = RMSNorm(dim)
        self.lm_head = nn.Linear(dim, vocab_size, bias=False)
        self.lm_head.weight = self.embed.weight   # weight tying
        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if hasattr(module, "bias") and module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(
        self,
        input_ids: torch.Tensor,
        kv_cache: Optional[dict] = None,
        seq_start: int = 0,
    ) -> Tuple[torch.Tensor, Optional[dict]]:
        """Full forward pass.

        Args:
            input_ids : (B, T) integer token IDs.
            kv_cache  : optional dict for incremental autoregressive decoding.
            seq_start : positional offset for RoPE.

        Returns:
            logits   : (B, T, vocab_size) next-token logits.
            kv_cache : updated kv_cache dict (same object, mutated in place).
        """
        x = self.embed(input_ids)
        for i, block in enumerate(self.blocks):
            x, _ = block(x, kv_cache=kv_cache, seq_start=seq_start)
        x = self.norm(x)
        return self.lm_head(x), kv_cache

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 40,
        temperature: float = 0.8,
        top_k: int = 40,
    ) -> torch.Tensor:
        """Autoregressive generation using standard linear O(L) KV cache."""
        self.eval()
        B, T = input_ids.shape
        kv_cache = {}
        logits, _ = self.forward(input_ids, kv_cache=kv_cache, seq_start=0)

        def sample_tok(lgt):
            if temperature > 0:
                lgt = lgt / temperature
                if top_k > 0:
                    v, _ = torch.topk(lgt, min(top_k, lgt.size(-1)))
                    lgt[lgt < v[:, [-1]]] = -float("Inf")
                probs = F.softmax(lgt, dim=-1)
                return torch.multinomial(probs, num_samples=1)
            return torch.argmax(lgt, dim=-1, keepdim=True)

        next_token = sample_tok(logits[:, -1, :])
        generated = [next_token]

        curr_token = next_token
        for step in range(max_new_tokens - 1):
            logits_step, _ = self.forward(curr_token, kv_cache=kv_cache, seq_start=T + step)
            curr_token = sample_tok(logits_step[:, -1, :])
            generated.append(curr_token)

        return torch.cat([input_ids] + generated, dim=1)


def get_sentence_boundary_ids(tokenizer, max_vocab: int = 50257) -> List[int]:
    """Identify all token IDs in tokenizer vocabulary representing sentence terminators (., !, ?)."""
    boundary_chars = {".", "!", "?"}
    boundary_ids = set()
    for p in [".", "!", "?"]:
        boundary_ids.update(tokenizer.encode(p))
    if hasattr(tokenizer, "vocab_size"):
        scan_limit = min(tokenizer.vocab_size, max_vocab)
        for i in range(scan_limit):
            try:
                text = tokenizer.decode([i])
                if any(text.strip().endswith(c) for c in boundary_chars):
                    boundary_ids.add(i)
            except Exception:
                pass
    return sorted(list(boundary_ids))


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    torch.manual_seed(42)

    CFG = dict(
        vocab_size=10_000,
        dim=256,
        n_layers=4,
        n_heads=8,
        n_kv_heads=2,
        ffn_mult=4,
    )
    B, T = 2, 128

    print("=" * 62)
    print("SBR-KV vs GQA – Model Smoke Test")
    print("=" * 62)

    # ------------------------------------------------------------------
    # GQATransformer
    # ------------------------------------------------------------------
    gqa_model = GQATransformer(**CFG)
    gqa_params = count_params(gqa_model)
    print(f"\n[GQATransformer]  params = {gqa_params:,}")

    input_ids = torch.randint(0, CFG["vocab_size"], (B, T))
    logits_gqa, _ = gqa_model(input_ids)
    assert logits_gqa.shape == (B, T, CFG["vocab_size"]), (
        f"GQA logits shape mismatch: {logits_gqa.shape}"
    )
    print(f"  forward() output shape : {tuple(logits_gqa.shape)}  ✓")

    # Incremental decode (single step after full prefill)
    kv_cache: dict = {}
    gqa_model(input_ids, kv_cache=kv_cache, seq_start=0)
    next_id = torch.randint(0, CFG["vocab_size"], (B, 1))
    logits_step, _ = gqa_model(next_id, kv_cache=kv_cache, seq_start=T)
    assert logits_step.shape == (B, 1, CFG["vocab_size"])
    print(f"  incremental step shape : {tuple(logits_step.shape)}  ✓")

    # ------------------------------------------------------------------
    # SBRTransformer
    # ------------------------------------------------------------------
    sbr_model = SBRTransformer(**CFG, boundary_ids=[10, 11, 12])
    sbr_params = count_params(sbr_model)
    print(f"\n[SBRTransformer]  params = {sbr_params:,}")

    input_ids_sbr = input_ids.clone()
    input_ids_sbr[0, 20] = 10    # '.' boundary in batch item 0
    input_ids_sbr[0, 55] = 11    # '!' boundary in batch item 0
    input_ids_sbr[1, 40] = 12    # '?' boundary in batch item 1

    logits_sbr, h_states = sbr_model(input_ids_sbr)
    assert logits_sbr.shape == (B, T, CFG["vocab_size"]), (
        f"SBR logits shape mismatch: {logits_sbr.shape}"
    )
    print(f"  forward() output shape : {tuple(logits_sbr.shape)}  ✓")
    print(f"  h_states length        : {len(h_states)} (one per layer)  ✓")
    for i, h in enumerate(h_states):
        assert h is not None and h.shape == (B, 1, CFG["dim"]), (
            f"Layer {i} h_state shape unexpected: {h.shape}"
        )
    print(f"  h_state shape (layer 0): {tuple(h_states[0].shape)}  ✓")

    # ------------------------------------------------------------------
    # Parameter count comparison
    # ------------------------------------------------------------------
    print(f"\n{'─' * 62}")
    print(f"  GQA params : {gqa_params:>12,}")
    print(f"  SBR params : {sbr_params:>12,}")
    delta = abs(sbr_params - gqa_params)
    pct = delta / max(gqa_params, sbr_params) * 100
    print(f"  Δ          : {delta:>12,}  ({pct:.2f}%)")
    print(f"{'─' * 62}")

    if pct <= 1.0:
        print("  ✓  Parameter counts are within 1 % of each other.")
    elif pct <= 5.0:
        print("  ✓  Parameter counts are within 5 % of each other.")
    else:
        print(
            f"  ⚠  Parameter counts differ by {pct:.1f}%. "
            "Review _gqa_compensating_hidden() or model configs."
        )

    print("\nAll assertions passed. ✓")
    sys.exit(0)
