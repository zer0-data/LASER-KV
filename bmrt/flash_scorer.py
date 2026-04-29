"""
Flash Attention compatible exact scoring.

Computes per-layer attention scores post-hoc by:
1. Capturing hidden_states (query positions only) via forward pre-hooks
2. Recomputing Q from captured hidden_states (q_proj + RoPE)
3. Using K from KV cache (already post-RoPE)
4. Computing softmax(Q @ K^T / sqrt(d)) and accumulating across layers

This produces scores equivalent to the eager AttentionScoreAccumulator
without requiring attention weight materialisation.
"""

import math
import torch
from typing import Optional, List, Tuple, Dict


class FlashAttentionScorer:
    """
    Computes exact attention scores when using Flash Attention backend.

    Instead of intercepting attention weights during the forward pass
    (impossible with Flash Attention), this captures the hidden_states
    fed to each attention layer and recomputes Q post-hoc.  Combined
    with K from the KV cache the full softmax-normalised attention
    scores can be reconstructed per layer, giving an equivalent signal
    to the eager-mode AttentionScoreAccumulator.
    """

    def __init__(self):
        self._hooks: List = []
        self._captured_hidden: Dict[int, torch.Tensor] = {}
        self._query_len: int = 0
        self._is_active: bool = False
        self._num_layers: int = 0
        # Cached model reference for score computation
        self._model = None
        # Final accumulated scores (same format as AccumulatorScorer)
        self.accumulated_scores: Optional[torch.Tensor] = None

    # ------------------------------------------------------------------
    # Hook management
    # ------------------------------------------------------------------

    def install_hooks(self, model):
        """Register forward pre-hooks on every attention layer to
        capture hidden_states for query positions only."""
        self.remove_hooks()
        self._model = model

        layers = self._get_attention_layers(model)
        self._num_layers = len(layers)

        for layer_idx, attn_module in enumerate(layers):
            handle = attn_module.register_forward_pre_hook(
                self._make_pre_hook(layer_idx),
                with_kwargs=True,
            )
            self._hooks.append(handle)

        print(f"  [FlashScorer] Installed {len(self._hooks)} hidden-state capture hooks")

    def remove_hooks(self):
        for h in self._hooks:
            h.remove()
        self._hooks.clear()
        self._captured_hidden.clear()

    # ------------------------------------------------------------------
    # Block lifecycle (mirrors AttentionScoreAccumulator API)
    # ------------------------------------------------------------------

    def start_block(self, query_len: int):
        """Prepare for a new block.  ``query_len`` tells the hooks how
        many trailing positions of hidden_states to keep."""
        self._captured_hidden.clear()
        self._query_len = query_len
        self._is_active = True
        self.accumulated_scores = None

    def finish_block(self):
        self._captured_hidden.clear()
        self._is_active = False
        self.accumulated_scores = None

    # ------------------------------------------------------------------
    # Post-forward score computation
    # ------------------------------------------------------------------

    @torch.no_grad()
    def compute_scores(
        self,
        model,
        kv_cache,               # legacy tuple format: tuple of (K, V) per layer
        candidate_indices: List[int],
        query_position_ids: torch.Tensor,  # [query_len] absolute positions for RoPE
        score_history: bool,
        prefix_kv_len: int,
        block_len: int,
        prev_tail_len: int,
    ) -> torch.Tensor:
        """Compute accumulated attention scores for *candidate_indices*
        across all transformer layers.

        Returns
        -------
        accumulated_scores : Tensor of shape ``[scored_window_len]``
            Same semantic as ``AttentionScoreAccumulator.accumulated_scores[0]``.
        """
        if not self._captured_hidden:
            print("  [FlashScorer] WARNING: No hidden states captured — returning empty scores")
            self.accumulated_scores = None
            return torch.zeros(0)

        device = query_position_ids.device
        layers = self._get_attention_layers(model)
        num_layers = len(layers)

        # Determine scoring window (same logic as accumulator.py:62-71)
        if score_history:
            score_start = 0
            score_end = prefix_kv_len + block_len
        else:
            score_start = prefix_kv_len - prev_tail_len
            score_end = prefix_kv_len + block_len

        scored_window_len = score_end - score_start
        if scored_window_len <= 0:
            print(f"  [FlashScorer] WARNING: scored_window_len={scored_window_len} <= 0"
                  f" (score_start={score_start}, score_end={score_end})")
            self.accumulated_scores = None
            return torch.zeros(0, device=device)

        accumulated = torch.zeros(scored_window_len, device=device, dtype=torch.float32)

        # Ensure position_ids is [1, Q]
        if query_position_ids.dim() == 1:
            query_position_ids = query_position_ids.unsqueeze(0)

        # ---- Precompute things that are identical across layers ----
        # Resolve rotary_emb module once (location moved in transformers 4.45+:
        # used to be on each self_attn, now lives on model.model)
        rope = self._resolve_rotary_emb(model, layers[0])

        # Causal mask: depends only on positions, not on layer
        # query_position_ids: [1, Q], key_positions: [W]
        # mask[q, k] = True if query_pos[q] >= key_pos[k]
        key_positions = torch.arange(score_start, score_end, device=device)
        causal_mask = query_position_ids.squeeze(0).unsqueeze(-1) >= key_positions.unsqueeze(0)
        causal_mask = causal_mask.view(1, 1, 1, query_position_ids.shape[1], scored_window_len)
        # ^ broadcastable over [B, kv_heads, groups, Q, W]

        # cos/sin: depend on positions + rope module, not on Q values.
        # Build with a dummy Q tensor to query the right shape/dtype/device.
        head_dim = kv_cache[0][0].shape[3]
        sample_hidden = next(iter(self._captured_hidden.values()))
        dummy_q = torch.empty(
            (1, 1, 1, head_dim),
            dtype=sample_hidden.dtype, device=sample_hidden.device,
        )
        cos, sin = rope(dummy_q, query_position_ids)

        print(f"  [FlashScorer] Computing scores: {num_layers} layers, "
              f"Q={self._query_len}, scored_window=[{score_start}:{score_end}]")

        for layer_idx in range(num_layers):
            if layer_idx not in self._captured_hidden:
                print(f"  [FlashScorer] WARNING: Missing hidden states for layer {layer_idx}")
                continue

            attn = layers[layer_idx]
            hidden_q = self._captured_hidden[layer_idx]  # [1, Q, hidden_dim]

            try:
                layer_score = self._score_layer(
                    attn, hidden_q, kv_cache[layer_idx],
                    cos, sin, causal_mask,
                    score_start, score_end,
                )
                accumulated.add_(layer_score.to(accumulated.device))
            except Exception as e:
                print(f"  [FlashScorer] ERROR scoring layer {layer_idx}: {e}")
                import traceback
                traceback.print_exc()
                continue

        # Store in same format as accumulator: [1, scored_window_len]
        self.accumulated_scores = accumulated.unsqueeze(0)
        print(f"  [FlashScorer] Done. Score shape={self.accumulated_scores.shape}, "
              f"min={accumulated.min().item():.4f}, max={accumulated.max().item():.4f}")
        return accumulated

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _make_pre_hook(self, layer_idx: int):
        """Return a forward-pre-hook that captures the last ``query_len``
        positions of the hidden_states tensor."""
        scorer = self

        def hook(module, args, kwargs):
            if not scorer._is_active:
                return
            # hidden_states is the first positional argument
            hidden_states = args[0] if args else kwargs.get("hidden_states")
            if hidden_states is None:
                return
            q_len = scorer._query_len
            if q_len > 0 and q_len <= hidden_states.shape[1]:
                scorer._captured_hidden[layer_idx] = hidden_states[:, -q_len:, :].detach()
            else:
                scorer._captured_hidden[layer_idx] = hidden_states.detach()

        return hook

    def _score_layer(
        self,
        attn_module,
        hidden_q: torch.Tensor,          # [1, Q, hidden_dim]
        kv_pair: Tuple[torch.Tensor, torch.Tensor],  # (K, V) each [1, num_kv_heads, total_kv, head_dim]
        cos: torch.Tensor,               # precomputed RoPE cos
        sin: torch.Tensor,               # precomputed RoPE sin
        causal_mask: torch.Tensor,       # [1, 1, 1, Q, W] bool
        score_start: int,
        score_end: int,
    ) -> torch.Tensor:
        """Compute attention scores for one layer.

        Returns a float32 tensor of shape ``[score_end - score_start]``.
        """
        from transformers.models.llama.modeling_llama import apply_rotary_pos_emb

        K_cache = kv_pair[0]  # [1, num_kv_heads, total_kv, head_dim]
        num_kv_heads = K_cache.shape[1]
        head_dim = K_cache.shape[3]

        # --- Recompute Q (post-projection + RoPE) ---
        q = attn_module.q_proj(hidden_q)                     # [1, Q, num_heads * head_dim]
        _, q_len, hidden_total = q.shape
        # Robust num_heads derivation (transformers 4.45+ removed `attn_module.num_heads`)
        num_heads = getattr(attn_module, 'num_heads', None) or (hidden_total // head_dim)
        q = q.view(1, q_len, num_heads, head_dim).transpose(1, 2)  # [1, H, Q, D]

        # apply_rotary_pos_emb expects (q, k, cos, sin) — we only use the q result
        q_rotated, _ = apply_rotary_pos_emb(q, q, cos, sin)  # [1, H, Q, D]

        # --- K from cache (already post-RoPE) ---
        K_window = K_cache[:, :, score_start:score_end, :]    # [1, kv_heads, W, D]
        num_groups = num_heads // num_kv_heads

        # --- Compute attention scores via GQA broadcast (no repeat_interleave copy) ---
        # Q: [1, H, Q, D] → [1, kv_heads, groups, Q, D]
        # K: [1, kv_heads, W, D] → [1, kv_heads, 1, W, D] (broadcasts over groups)
        scale = 1.0 / math.sqrt(head_dim)
        q_g = q_rotated.view(1, num_kv_heads, num_groups, q_len, head_dim)
        k_g = K_window.unsqueeze(2)  # [1, kv_heads, 1, W, D] — broadcast view, no copy
        # raw_scores: [1, kv_heads, groups, Q, W]
        raw_scores = torch.matmul(
            q_g.to(k_g.dtype),
            k_g.transpose(-2, -1),
        ) * scale

        # Apply causal mask (precomputed, broadcasts over kv_heads/groups)
        raw_scores = raw_scores.masked_fill(~causal_mask, float('-inf'))

        # Softmax in fp32 for numerical stability, then collapse heads/queries
        attn_weights = torch.softmax(raw_scores.float(), dim=-1)  # fp32
        # Replace NaN from all-masked rows with 0
        attn_weights = attn_weights.nan_to_num(0.0)

        # Sum over (batch, kv_heads, groups, queries) → [W]
        layer_score = attn_weights.sum(dim=(0, 1, 2, 3))

        return layer_score

    @staticmethod
    def _resolve_rotary_emb(model, sample_attn_module):
        """Locate the rotary_emb module.

        - transformers <= 4.44: lives on each `self_attn` layer
        - transformers >= 4.45: moved to `model.model.rotary_emb` (shared)
        """
        rope = getattr(sample_attn_module, 'rotary_emb', None)
        if rope is not None:
            return rope
        inner = getattr(model, 'model', None)
        if inner is not None and hasattr(inner, 'rotary_emb'):
            return inner.rotary_emb
        raise RuntimeError(
            "[FlashScorer] Could not locate rotary_emb module on either "
            "self_attn (transformers <=4.44) or model.model (>=4.45)."
        )

    @staticmethod
    def _get_attention_layers(model):
        """Return a list of attention modules (one per transformer layer)."""
        # Llama-family models
        if hasattr(model, 'model') and hasattr(model.model, 'layers'):
            return [layer.self_attn for layer in model.model.layers]
        # Fallback: try to find attention modules generically
        attn_modules = []
        for name, module in model.named_modules():
            if 'self_attn' in name and not any(x in name for x in ['proj', 'norm', 'rotary']):
                # Only leaf self_attn modules
                if hasattr(module, 'q_proj'):
                    attn_modules.append(module)
        if attn_modules:
            return attn_modules
        raise RuntimeError(
            "[FlashScorer] Could not find attention layers. "
            "Expected model.model.layers[*].self_attn for Llama-family models."
        )
