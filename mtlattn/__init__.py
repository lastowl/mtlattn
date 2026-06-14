"""
mtlattn: fused variable-length attention (forward only) for Apple Silicon.

Drop-in equivalents for the flash_attn varlen entry points used by
TRELLIS.2-family sparse transformers, running on PyTorch MPS tensors with no
padding and no materialized attention matrix.
"""

import math

import torch

from . import _C

__all__ = [
    "varlen_attention",
    "flash_attn_varlen_qkvpacked_func",
    "flash_attn_varlen_kvpacked_func",
    "sdpa",
    "replace_sdpa",
    "restore_sdpa",
]


def varlen_attention(q, k, v, cu_seqlens_q, cu_seqlens_kv, max_seqlen_q, scale=None,
                     causal=False, window=0):
    """q, k, v: [M, H, D] MPS tensors (row-strided views OK).
    cu_seqlens_*: int32 [B+1] on MPS. Returns [Mq, H, D].
    causal: if True, query i attends key j iff j <= i + (kv_len - q_len).
    window: if >0, sliding window — query i attends only its last `window`
        keys (relative to the causal diagonal). Mistral-style SWA is
        causal=True with window=W."""
    if scale is None:
        scale = 1.0 / math.sqrt(q.shape[-1])
    return _C.varlen_attention(
        q, k, v,
        cu_seqlens_q.int(), cu_seqlens_kv.int(),
        int(max_seqlen_q), float(scale), bool(causal), int(window),
    )


def _window_from_kwargs(window, kwargs):
    """Map flash_attn's window_size=(left, right) to our `window` (left+1
    keys). right is expected <= 0 (causal SWA); a positive right is ignored."""
    ws = kwargs.get("window_size")
    if ws is not None and ws[0] is not None and ws[0] >= 0:
        return int(ws[0]) + 1
    return window


def flash_attn_varlen_qkvpacked_func(qkv, cu_seqlens, max_seqlen, softmax_scale=None,
                                     causal=False, window=0, **kwargs):
    """qkv: [M, 3, H, D]. Mirrors flash_attn's API (forward only)."""
    q, k, v = qkv.unbind(dim=1)  # row-strided views; kernel supports them
    return varlen_attention(q, k, v, cu_seqlens, cu_seqlens, max_seqlen, softmax_scale,
                            causal=causal, window=_window_from_kwargs(window, kwargs))


def flash_attn_varlen_kvpacked_func(
    q, kv, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k,
    softmax_scale=None, causal=False, window=0, **kwargs,
):
    """q: [Mq, H, D]; kv: [Mkv, 2, H, D]. Mirrors flash_attn's API (forward only)."""
    k, v = kv.unbind(dim=1)
    return varlen_attention(q, k, v, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, softmax_scale,
                            causal=causal, window=_window_from_kwargs(window, kwargs))


# ---- Drop-in scaled_dot_product_attention for PyTorch/MPS ----

_ORIG_SDPA = None
# Below this query length, native MPS SDPA is competitive and supports
# autograd, so we don't intercept. mtlattn wins above it: large/ragged
# sequences, and where native SDPA pads, OOMs, or hits the >2^32 MPSGraph bug.
_SDPA_MIN_SEQLEN = 1024


def sdpa(query, key, value, attn_mask=None, dropout_p=0.0, is_causal=False, scale=None):
    """Dense [B, H, N, D] scaled_dot_product_attention backed by mtlattn's
    varlen kernel (equal-length batch -> trivial cu_seqlens). Forward only.
    Supports GQA (kv heads < q heads). attn_mask/dropout are not supported."""
    B, Hq, Nq, D = query.shape
    Hkv, Nkv = key.shape[1], key.shape[2]
    if scale is None:
        scale = 1.0 / math.sqrt(D)
    # [B, H, N, D] -> [B*N, H, D]. contiguous() before view so head stride == D
    # even when B==1 (a plain reshape would keep the permuted strides there).
    q = query.permute(0, 2, 1, 3).contiguous().view(B * Nq, Hq, D)
    k = key.permute(0, 2, 1, 3).contiguous().view(B * Nkv, Hkv, D)
    v = value.permute(0, 2, 1, 3).contiguous().view(B * Nkv, Hkv, D)
    cu_q = torch.arange(0, (B + 1) * Nq, Nq, dtype=torch.int32, device=query.device)
    cu_kv = torch.arange(0, (B + 1) * Nkv, Nkv, dtype=torch.int32, device=query.device)
    out = varlen_attention(q, k, v, cu_q, cu_kv, Nq, scale, causal=is_causal)
    return out.reshape(B, Nq, Hq, D).permute(0, 2, 1, 3).contiguous()


def _can_handle(query, key, value, attn_mask, dropout_p, is_causal, min_seqlen):
    if query.device.type != "mps" or query.dim() != 4:
        return False
    if dropout_p != 0.0 or attn_mask is not None:   # arbitrary masks: not yet
        return False
    if query.dtype not in (torch.float16, torch.bfloat16, torch.float32):
        return False
    if query.shape[-1] > 128 or key.shape[-1] > 128:
        return False
    if key.shape[1] == 0 or query.shape[1] % key.shape[1] != 0:  # GQA divisibility
        return False
    # is_causal with q_len != kv_len: our end-aligned convention may differ
    # from native top-left; stay safe and let native handle it.
    if is_causal and query.shape[2] != key.shape[2]:
        return False
    # No backward pass: never intercept when autograd needs gradients.
    if torch.is_grad_enabled() and (query.requires_grad or key.requires_grad or value.requires_grad):
        return False
    # Route by size: below the threshold native MPS SDPA is competitive.
    return query.shape[2] >= min_seqlen


def replace_sdpa(min_seqlen=_SDPA_MIN_SEQLEN):
    """Monkey-patch torch.nn.functional.scaled_dot_product_attention so any
    PyTorch/HF model transparently uses mtlattn on the large/ragged forward
    calls it wins on, falling back to native SDPA otherwise (small shapes,
    autograd/training, attn_mask, unsupported dtype/head_dim). Idempotent.
    Note: only affects callers that reach F.scaled_dot_product_attention at
    call time (most HF attention does); modules holding a direct import of the
    symbol are unaffected."""
    global _ORIG_SDPA
    if _ORIG_SDPA is None:
        _ORIG_SDPA = torch.nn.functional.scaled_dot_product_attention
    orig = _ORIG_SDPA

    def patched(query, key, value, attn_mask=None, dropout_p=0.0, is_causal=False,
                scale=None, **kwargs):
        if _can_handle(query, key, value, attn_mask, dropout_p, is_causal, min_seqlen):
            return sdpa(query, key, value, attn_mask, dropout_p, is_causal, scale)
        return orig(query, key, value, attn_mask=attn_mask, dropout_p=dropout_p,
                    is_causal=is_causal, scale=scale, **kwargs)

    torch.nn.functional.scaled_dot_product_attention = patched


def restore_sdpa():
    """Undo replace_sdpa()."""
    global _ORIG_SDPA
    if _ORIG_SDPA is not None:
        torch.nn.functional.scaled_dot_product_attention = _ORIG_SDPA
        _ORIG_SDPA = None
