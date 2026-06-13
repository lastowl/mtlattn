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
]


def varlen_attention(q, k, v, cu_seqlens_q, cu_seqlens_kv, max_seqlen_q, scale=None):
    """q, k, v: [M, H, D] MPS tensors (row-strided views OK).
    cu_seqlens_*: int32 [B+1] on MPS. Returns [Mq, H, D]."""
    if scale is None:
        scale = 1.0 / math.sqrt(q.shape[-1])
    return _C.varlen_attention(
        q, k, v,
        cu_seqlens_q.int(), cu_seqlens_kv.int(),
        int(max_seqlen_q), float(scale),
    )


def flash_attn_varlen_qkvpacked_func(qkv, cu_seqlens, max_seqlen, softmax_scale=None, **kwargs):
    """qkv: [M, 3, H, D]. Mirrors flash_attn's API (forward only)."""
    q, k, v = qkv.unbind(dim=1)  # row-strided views; kernel supports them
    return varlen_attention(q, k, v, cu_seqlens, cu_seqlens, max_seqlen, softmax_scale)


def flash_attn_varlen_kvpacked_func(
    q, kv, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k,
    softmax_scale=None, **kwargs,
):
    """q: [Mq, H, D]; kv: [Mkv, 2, H, D]. Mirrors flash_attn's API (forward only)."""
    k, v = kv.unbind(dim=1)
    return varlen_attention(q, k, v, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, softmax_scale)
