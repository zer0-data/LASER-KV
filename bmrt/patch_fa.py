import torch
import transformers
import transformers.modeling_flash_attention_utils as fa_utils
from typing import Optional, Tuple
import contextlib
import warnings

# ---------------------------------------------------------------------------
# Version-tolerant detection of the position-id helper in transformers.
#
# Names across transformers versions (verified against upstream source):
#   - 4.44, 4.45, 4.46           : `prepare_fa2_from_position_ids` (public)
#   - main (unreleased, post-4.46): `_prepare_from_posids`          (private)
#
# Signature is identical in every version: (query, key, value, position_ids)
# returning (q, k, v, (cu_seq_lens_q, cu_seq_lens_k), (max_q, max_k)).
# ---------------------------------------------------------------------------
_CANDIDATE_NAMES = ("prepare_fa2_from_position_ids", "_prepare_from_posids")
_PATCH_ATTR = next((n for n in _CANDIDATE_NAMES if hasattr(fa_utils, n)), None)

if _PATCH_ATTR is None:
    warnings.warn(
        "bmrt.patch_fa: could not find a known position-id helper in "
        f"transformers {transformers.__version__}. Expected one of "
        f"{_CANDIDATE_NAMES}. Flash-Attention forward with disjoint "
        "position_ids will NOT be patched; the varlen path may split the "
        "sequence on gaps and produce incorrect attention. To silence "
        "this, install a supported transformers version."
    )
    _original_prepare_from_posids = None
else:
    _original_prepare_from_posids = getattr(fa_utils, _PATCH_ATTR)
    print(
        f"[patch_fa] Using transformers {transformers.__version__}; "
        f"patching `{_PATCH_ATTR}`."
    )


def _patched_prepare_from_posids(
    query, key, value, position_ids, *args, **kwargs
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Tuple[torch.Tensor, torch.Tensor], Tuple[int, int]]:
    """
    Patched version of the position-id helper that forces single-sequence
    behavior when batch_size is 1, regardless of gaps in position_ids.

    Without this, the varlen flash-attention path interprets gaps in
    position_ids as sequence boundaries and splits the (already compressed)
    BMRT sequence into multiple sub-sequences, corrupting attention.
    """

    # Check if we have a single sequence (Batch Size == 1)
    if isinstance(position_ids, torch.Tensor) and position_ids.dim() == 2 and position_ids.shape[0] == 1:
        seq_len = position_ids.shape[1]

        # Manually construct cumulative sequence lengths for a SINGLE sequence.
        # Format: [0, seq_len]
        cu_seq_lens = torch.tensor([0, seq_len], device=position_ids.device, dtype=torch.int32)
        max_length = seq_len

        # Reshape (flatten) q, k, v for Flash Attention Varlen if they are 4D
        # shape: [1, seq, heads, dim] -> [seq, heads, dim]
        if query.dim() == 4:
            query = query.squeeze(0)
        if key.dim() == 4:
            key = key.squeeze(0)
        if value.dim() == 4:
            value = value.squeeze(0)

        # Return flattened q, k, v + new lengths
        return query, key, value, (cu_seq_lens, cu_seq_lens), (max_length, max_length)

    # Fallback to original (multi-batch or non-tensor position_ids)
    if _original_prepare_from_posids is None:
        raise RuntimeError(
            "bmrt.patch_fa: cannot fall back to the original helper because "
            "no compatible position-id helper was found in this transformers "
            f"version ({transformers.__version__})."
        )
    return _original_prepare_from_posids(query, key, value, position_ids, *args, **kwargs)


@contextlib.contextmanager
def ApplyFlashAttentionPatch():
    """
    Context manager that temporarily patches transformers.modeling_flash_attention_utils
    to handle disjoint position_ids in single-batch inference.

    If the installed transformers version does not expose a known helper name,
    the context manager becomes a no-op (a warning was emitted at import time).
    """
    if _PATCH_ATTR is None:
        # No-op: nothing to patch on this transformers version.
        yield
        return

    try:
        # Apply Patch
        setattr(fa_utils, _PATCH_ATTR, _patched_prepare_from_posids)
        yield
    finally:
        # Restore Original
        setattr(fa_utils, _PATCH_ATTR, _original_prepare_from_posids)
