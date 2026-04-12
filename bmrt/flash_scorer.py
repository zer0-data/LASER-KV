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
                    query_position_ids, score_start, score_end, device,
                )
                accumulated.add_(layer_score)
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
        query_position_ids: torch.Tensor, # [1, Q]
        score_start: int,
        score_end: int,
        device: torch.device,
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
        bsz, q_len, _ = q.shape
        num_heads = attn_module.num_heads
        q = q.view(bsz, q_len, num_heads, head_dim).transpose(1, 2)  # [1, H, Q, D]

        # RoPE — need cos/sin from the rotary embedding module
        cos, sin = attn_module.rotary_emb(q, query_position_ids)
        # apply_rotary_pos_emb expects (q, k, cos, sin) but we only need Q
        # Pass q as both q and k; we only use the first return value.
        q_rotated, _ = apply_rotary_pos_emb(q, q, cos, sin)  # [1, H, Q, D]

        # --- K from cache (already post-RoPE) ---
        K_window = K_cache[:, :, score_start:score_end, :]    # [1, kv_heads, W, D]

        # GQA expansion: repeat K heads to match Q heads
        num_groups = num_heads // num_kv_heads
        if num_groups > 1:
            K_window = K_window.repeat_interleave(num_groups, dim=1)  # [1, H, W, D]

        # --- Compute attention scores ---
        scale = 1.0 / math.sqrt(head_dim)
        # raw_scores: [1, H, Q, W]
        raw_scores = torch.matmul(
            q_rotated.to(K_window.dtype),
            K_window.transpose(-2, -1),
        ) * scale

        # Apply causal mask: query positions should only attend to keys before them.
        # query_position_ids: [1, Q], key positions: score_start .. score_end-1
        key_positions = torch.arange(score_start, score_end, device=device)
        # causal_mask[q, k] = True if query_pos[q] >= key_pos[k]
        causal_mask = query_position_ids.squeeze(0).unsqueeze(-1) >= key_positions.unsqueeze(0)
        # Expand to [1, 1, Q, W] for broadcasting over heads
        causal_mask = causal_mask.unsqueeze(0).unsqueeze(0)
        raw_scores = raw_scores.masked_fill(~causal_mask, float('-inf'))

        # Softmax over key dimension (correctly normalised over full scored window)
        attn_weights = torch.softmax(raw_scores, dim=-1)  # [1, H, Q, W]
        # Replace NaN from all-masked rows with 0
        attn_weights = attn_weights.nan_to_num(0.0)

        # Sum over batch, heads, queries → per-key importance
        layer_score = attn_weights.float().sum(dim=(0, 1, 2))  # [W]

        return layer_score

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
