import torch
from typing import List, Optional
from .base import BaseSelector
from ..accumulator import AttentionScoreAccumulator
from ..flash_scorer import FlashAttentionScorer

class ExactSelector(BaseSelector):
    """
    Selects tokens by true attention scores accumulated across all layers.

    Supports two backends:
      - ``'eager'``: Intercepts attention weights via AttentionWrapper
        (requires ``attn_implementation='eager'``).
      - ``'flash'``: Recomputes Q post-hoc from captured hidden states
        and uses K from the KV cache.  Compatible with Flash Attention.

    The ``select()`` interface is identical for both backends.
    """

    def __init__(self, backend: str = 'eager'):
        self.backend = backend
        self.block_start = 0
        self.block_len = 0
        self.score_history = False
        self.prev_local_tail_len = 0

        if backend == 'eager':
            self.accumulator = AttentionScoreAccumulator()
            self.flash_scorer = None
        elif backend == 'flash':
            self.accumulator = None
            self.flash_scorer = FlashAttentionScorer()
        else:
            raise ValueError(f"ExactSelector: unknown backend '{backend}'. Use 'eager' or 'flash'.")

        print(f"  [ExactSelector] backend={backend}")

    def setup(self, model):
        if self.backend == 'eager':
            self.accumulator.wrap_model(model)
        else:
            self.flash_scorer.install_hooks(model)

    def cleanup(self):
        if self.backend == 'eager':
            self.accumulator.unwrap_model(None)
        else:
            self.flash_scorer.remove_hooks()

    def prepare_block(self, total_seq_len, block_len, query_len, prefix_len,
                      prefix_in_kv, score_history=False, prev_local_tail_len=0):
        if self.backend == 'eager':
            self.accumulator.start_block_with_prefix(
                total_seq_len=total_seq_len,
                prefix_len=prefix_len,
                block_len=block_len,
                query_len=query_len,
                prefix_in_kv_cache=prefix_in_kv,
                score_history=score_history,
                prev_local_tail_len=prev_local_tail_len
            )
        else:
            self.flash_scorer.start_block(query_len=query_len)

        # Shared bookkeeping (needed by select())
        self.block_start = prefix_in_kv - prev_local_tail_len
        self.block_len = block_len + prev_local_tail_len
        self.score_history = score_history
        self.prev_local_tail_len = prev_local_tail_len

    def post_forward_score(self, model, kv_cache, candidate_indices,
                           query_position_ids, score_history,
                           prefix_kv_len, block_len, prev_tail_len):
        """Called by the processor after the model forward pass (flash only).

        Computes attention scores from captured hidden states + KV cache.
        After this call ``self.flash_scorer.accumulated_scores`` is populated
        and ``select()`` can read it.
        """
        if self.backend != 'flash':
            return  # eager mode scores during forward via accumulator hooks
        self.flash_scorer.compute_scores(
            model=model,
            kv_cache=kv_cache,
            candidate_indices=candidate_indices,
            query_position_ids=query_position_ids,
            score_history=score_history,
            prefix_kv_len=prefix_kv_len,
            block_len=block_len,
            prev_tail_len=prev_tail_len,
        )

    def finish_block(self):
        if self.backend == 'eager':
            self.accumulator.finish_block()
        else:
            self.flash_scorer.finish_block()

    # ------------------------------------------------------------------
    # Selection (same logic for both backends)
    # ------------------------------------------------------------------

    def select(
        self,
        query_ids: torch.Tensor,
        query_vectors: torch.Tensor,
        candidate_vectors: torch.Tensor,
        candidate_indices: List[int],
        budget: int,
        **kwargs
    ) -> List[int]:

        # Get accumulated scores from whichever backend produced them
        if self.backend == 'eager':
            if self.accumulator.accumulated_scores is None:
                print("  [ExactSelector] WARNING: No accumulated scores (eager) — returning empty")
                return []
            full_scores = self.accumulator.accumulated_scores[0]
        else:
            if self.flash_scorer.accumulated_scores is None:
                print("  [ExactSelector] WARNING: No accumulated scores (flash) — returning empty")
                return []
            full_scores = self.flash_scorer.accumulated_scores[0]

        # Map candidate absolute indices to relative positions in the scored window
        if self.score_history:
            filtered_candidates = candidate_indices
            relative_positions = candidate_indices
        else:
            block_start = self.block_start
            block_end = block_start + self.block_len
            relative_positions = []
            filtered_candidates = []
            for idx in candidate_indices:
                if block_start <= idx < block_end:
                    filtered_candidates.append(idx)
                    relative_positions.append(idx - block_start)
            if not relative_positions:
                return []

        cand_tensor = torch.tensor(relative_positions, device=full_scores.device, dtype=torch.long)

        # Bounds check for debugging on remote
        if cand_tensor.max().item() >= full_scores.shape[0]:
            print(f"  [ExactSelector] ERROR: relative index {cand_tensor.max().item()} "
                  f">= score length {full_scores.shape[0]}. "
                  f"block_start={self.block_start}, block_len={self.block_len}, "
                  f"score_history={self.score_history}")
            # Clamp to avoid crash, but scores will be wrong
            cand_tensor = cand_tensor.clamp(max=full_scores.shape[0] - 1)

        candidate_scores = full_scores[cand_tensor]

        if budget >= len(filtered_candidates):
            return sorted(filtered_candidates)

        _, top_rel_indices = torch.topk(candidate_scores, k=budget)
        selected_indices = [filtered_candidates[i] for i in top_rel_indices.cpu().tolist()]
        return sorted(selected_indices)
