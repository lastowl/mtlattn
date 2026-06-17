"""
mtlattn: fused variable-length attention (forward + backward) for Apple Silicon.

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


class _VarlenAttnFn(torch.autograd.Function):
    """Differentiable varlen attention. Forward requests the LSE and saves it;
    backward returns dQ, dK, dV. `bias` (optional additive attn_mask) is treated
    as a constant — it is applied in both forward and backward but not
    differentiated (no dbias)."""

    @staticmethod
    def forward(ctx, q, k, v, cu_q, cu_kv, max_q, scale, causal, window, bias):
        lse = torch.empty(q.shape[0], q.shape[1], dtype=torch.float32, device=q.device)
        out = _C.varlen_attention(q, k, v, cu_q, cu_kv, max_q, scale, causal, window, lse, bias)
        ctx.save_for_backward(q, k, v, out, lse, cu_q, cu_kv)
        ctx.scale, ctx.causal, ctx.window, ctx.bias = scale, causal, window, bias
        return out

    @staticmethod
    def backward(ctx, dout):
        q, k, v, out, lse, cu_q, cu_kv = ctx.saved_tensors
        dQ, dK, dV = _C.varlen_attention_bwd(
            q.contiguous(), k.contiguous(), v.contiguous(),
            out.contiguous(), dout.contiguous(), lse, cu_q, cu_kv,
            ctx.scale, ctx.causal, ctx.window, ctx.bias)
        return dQ, dK, dV, None, None, None, None, None, None, None


def _prep_bias(attn_bias):
    """Normalize an additive bias to the kernel's contract: fp32, last-dim
    contiguous. Raises if it requires grad (a differentiable mask is unsupported;
    the bias is applied but not differentiated)."""
    if attn_bias is None:
        return None
    if attn_bias.requires_grad:
        raise NotImplementedError(
            "mtlattn: a differentiable attn_bias (requires_grad=True) is not supported; "
            "detach it or treat the mask as constant")
    if attn_bias.dtype != torch.float32:
        attn_bias = attn_bias.to(torch.float32)
    if attn_bias.dim() != 3 or attn_bias.stride(-1) != 1:
        attn_bias = attn_bias.contiguous()
    return attn_bias


def varlen_attention(q, k, v, cu_seqlens_q, cu_seqlens_kv, max_seqlen_q, scale=None,
                     causal=False, window=0, attn_bias=None):
    """q, k, v: [M, H, D] MPS tensors (row-strided views OK).
    cu_seqlens_*: int32 [B+1] on MPS. Returns [Mq, H, D].
    causal: if True, query i attends key j iff j <= i + (kv_len - q_len).
    window: if >0, sliding window — query i attends only its last `window`
        keys (relative to the causal diagonal). Mistral-style SWA is
        causal=True with window=W.
    attn_bias: optional additive mask [total_q, H or 1, max_kv] fp32, added to
        the logits before softmax (logit = scale*(q·k) + bias). dim1==1
        broadcasts across heads. Indexed by global query row and seq-local key.
        MPP-only (macOS 26.2+, fp16/bf16, head_dim 64/96/128/256); raises on the
        simdgroup fallback. Applied but not differentiated.

    Differentiable: if any of q/k/v requires grad, routes through the backward
    kernel (training). Otherwise uses the fast inference path (MPP on M5)."""
    if scale is None:
        scale = 1.0 / math.sqrt(q.shape[-1])
    cu_q, cu_kv = cu_seqlens_q.int(), cu_seqlens_kv.int()
    bias = _prep_bias(attn_bias)
    if torch.is_grad_enabled() and (q.requires_grad or k.requires_grad or v.requires_grad):
        return _VarlenAttnFn.apply(q, k, v, cu_q, cu_kv, int(max_seqlen_q),
                                   float(scale), bool(causal), int(window), bias)
    return _C.varlen_attention(
        q, k, v, cu_q, cu_kv, int(max_seqlen_q), float(scale), bool(causal), int(window),
        None, bias,
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
    """qkv: [M, 3, H, D]. Mirrors flash_attn's API; differentiable (routes
    through varlen_attention's autograd, so qkv.grad flows)."""
    q, k, v = qkv.unbind(dim=1)  # row-strided views; kernel supports them
    return varlen_attention(q, k, v, cu_seqlens, cu_seqlens, max_seqlen, softmax_scale,
                            causal=causal, window=_window_from_kwargs(window, kwargs))


def flash_attn_varlen_kvpacked_func(
    q, kv, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k,
    softmax_scale=None, causal=False, window=0, **kwargs,
):
    """q: [Mq, H, D]; kv: [Mkv, 2, H, D]. Mirrors flash_attn's API; differentiable."""
    k, v = kv.unbind(dim=1)
    return varlen_attention(q, k, v, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, softmax_scale,
                            causal=causal, window=_window_from_kwargs(window, kwargs))


# ---- Drop-in scaled_dot_product_attention for PyTorch/MPS ----

_ORIG_SDPA = None
# Below this query length, native MPS SDPA is competitive and supports
# autograd, so we don't intercept. mtlattn wins above it: large/ragged
# sequences, and where native SDPA pads, OOMs, or hits the >2^32 MPSGraph bug.
_SDPA_MIN_SEQLEN = 1024


def _sdpa_mask_to_bias(attn_mask, B, Hq, Nq, Nkv):
    """Convert a dense SDPA attn_mask to the varlen additive-bias layout
    [B*Nq, H or 1, Nkv] fp32. Accepts a [Nq,Nkv] or broadcastable [b,h,Nq,Nkv]
    mask, bool (True=keep -> 0, False -> -inf) or float (already additive). Keeps
    the head dim at 1 when the mask is head-broadcast (the kernel reads it with a
    zero head stride). Raises NotImplementedError for shapes it can't map."""
    m = attn_mask
    if m.dim() == 2:                       # [Nq, Nkv]
        m = m.view(1, 1, Nq, Nkv)
    elif m.dim() == 4:                     # [b in {1,B}, h in {1,Hq}, Nq, Nkv]
        if m.shape[2] != Nq or m.shape[3] != Nkv:
            raise NotImplementedError("mtlattn: attn_mask query/key dims must match")
    else:
        raise NotImplementedError("mtlattn: attn_mask must be 2D or 4D")
    if m.dtype == torch.bool:
        m = torch.zeros(m.shape, dtype=torch.float32, device=m.device).masked_fill_(
            ~m, float("-inf"))
    else:
        m = m.to(torch.float32)
    b, h = m.shape[0], m.shape[1]
    if b == 1 and B > 1:
        m = m.expand(B, h, Nq, Nkv)
    elif b != B:
        raise NotImplementedError("mtlattn: attn_mask batch dim must be 1 or B")
    if h != 1 and h != Hq:
        raise NotImplementedError("mtlattn: attn_mask head dim must be 1 or num_heads")
    # [B, h, Nq, Nkv] -> [B, Nq, h, Nkv] -> [B*Nq, h, Nkv] (last dim contiguous)
    return m.permute(0, 2, 1, 3).contiguous().reshape(B * Nq, h, Nkv)


def sdpa(query, key, value, attn_mask=None, dropout_p=0.0, is_causal=False, scale=None):
    """Dense [B, H, N, D] scaled_dot_product_attention backed by mtlattn's
    varlen kernel (equal-length batch -> trivial cu_seqlens). Forward only.
    Supports GQA (kv heads < q heads). An attn_mask (bool or additive float,
    [Nq,Nkv] or broadcastable [B,H,Nq,Nkv]) is applied as an additive bias
    (MPP-only). dropout is not supported."""
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
    bias = _sdpa_mask_to_bias(attn_mask, B, Hq, Nq, Nkv) if attn_mask is not None else None
    out = varlen_attention(q, k, v, cu_q, cu_kv, Nq, scale, causal=is_causal, attn_bias=bias)
    return out.reshape(B, Nq, Hq, D).permute(0, 2, 1, 3).contiguous()


def _basic_ok(query, key, value, dropout_p, min_seqlen):
    if query.device.type != "mps" or query.dim() != 4:
        return False
    if dropout_p != 0.0:
        return False
    if query.dtype not in (torch.float16, torch.bfloat16, torch.float32):
        return False
    if query.shape[-1] > 128 or key.shape[-1] > 128:
        return False
    if key.shape[1] == 0 or query.shape[1] % key.shape[1] != 0:  # GQA divisibility
        return False
    # No backward pass: never intercept when autograd needs gradients.
    if torch.is_grad_enabled() and (query.requires_grad or key.requires_grad or value.requires_grad):
        return False
    # Route by size: below the threshold native MPS SDPA is competitive.
    return query.shape[2] >= min_seqlen


def _try_padding_varlen(query, key, value, attn_mask, is_causal, scale):
    """If attn_mask is a self-attention key-padding mask (per-batch, query- and
    head-independent, with the valid keys a contiguous prefix), pack the valid
    tokens and run varlen — exactly what cu_seqlens is for, so padded compute is
    skipped. Returns the dense [B,H,N,D] result (padded query rows zeroed, which
    are ignored downstream) or None if the mask isn't a convertible padding mask.
    """
    B, Hq, Nq, D = query.shape
    Hkv, Nkv = key.shape[1], key.shape[2]
    if Nq != Nkv:                                  # token-level padding => self-attn
        return None
    keep = attn_mask if attn_mask.dtype == torch.bool else (attn_mask > -1e30)
    try:
        keep = keep.expand(B, Hq, Nq, Nkv)
    except RuntimeError:
        return None
    kv_keep = keep[:, 0, 0, :]                      # [B, Nkv]
    if not torch.equal(keep, kv_keep[:, None, None, :].expand(B, Hq, Nq, Nkv)):
        return None                                # depends on query/head: not padding
    idx = torch.arange(Nkv, device=query.device)
    L = kv_keep.sum(dim=1)                          # valid length per batch
    if not torch.equal(kv_keep, idx[None, :] < L[:, None]):
        return None                                # not a contiguous prefix
    Ls = L.to(torch.int64).tolist()
    total = int(sum(Ls))
    if total == 0:
        return query.new_zeros(B, Hq, Nq, D)
    cu = torch.zeros(B + 1, dtype=torch.int32, device=query.device)
    cu[1:] = torch.cumsum(L.to(torch.int32), 0)
    qp, kp = query.new_empty(total, Hq, D), key.new_empty(total, Hkv, D)
    vp = value.new_empty(total, Hkv, D)
    off = 0
    for b in range(B):
        Lb = Ls[b]
        if Lb:
            qp[off:off + Lb] = query[b, :, :Lb, :].transpose(0, 1)
            kp[off:off + Lb] = key[b, :, :Lb, :].transpose(0, 1)
            vp[off:off + Lb] = value[b, :, :Lb, :].transpose(0, 1)
            off += Lb
    op = varlen_attention(qp, kp, vp, cu, cu, max(Ls), scale, causal=is_causal)
    out = query.new_zeros(B, Hq, Nq, D)
    off = 0
    for b in range(B):
        Lb = Ls[b]
        if Lb:
            out[b, :, :Lb, :] = op[off:off + Lb].transpose(0, 1)
            off += Lb
    return out


def replace_sdpa(min_seqlen=_SDPA_MIN_SEQLEN):
    """Monkey-patch torch.nn.functional.scaled_dot_product_attention so any
    PyTorch/HF model transparently uses mtlattn on the large/ragged forward
    calls it wins on, falling back to native SDPA otherwise (small shapes,
    autograd/training, unsupported dtype/head_dim). A self-attention key-padding
    attn_mask is converted to varlen; any other mask falls back. Idempotent.
    Note: only affects callers that reach F.scaled_dot_product_attention at call
    time (most HF attention does); modules holding a direct import are unaffected."""
    global _ORIG_SDPA
    if _ORIG_SDPA is None:
        _ORIG_SDPA = torch.nn.functional.scaled_dot_product_attention
    orig = _ORIG_SDPA

    def patched(query, key, value, attn_mask=None, dropout_p=0.0, is_causal=False,
                scale=None, **kwargs):
        if _basic_ok(query, key, value, dropout_p, min_seqlen):
            sc = scale if scale is not None else 1.0 / math.sqrt(query.shape[-1])
            if attn_mask is None:
                # end-aligned causal only well-defined when q_len == kv_len here.
                if not (is_causal and query.shape[2] != key.shape[2]):
                    return sdpa(query, key, value, None, dropout_p, is_causal, scale)
            else:
                r = _try_padding_varlen(query, key, value, attn_mask, is_causal, sc)
                if r is not None:
                    return r
                # General additive/bool mask -> additive-bias path (MPP-only).
                # Falls back to native SDPA if MPP is unavailable or the mask
                # shape can't be mapped to the varlen bias layout.
                try:
                    return sdpa(query, key, value, attn_mask, dropout_p, is_causal, scale)
                except (NotImplementedError, RuntimeError):
                    pass
        return orig(query, key, value, attn_mask=attn_mask, dropout_p=dropout_p,
                    is_causal=is_causal, scale=scale, **kwargs)

    torch.nn.functional.scaled_dot_product_attention = patched


def restore_sdpa():
    """Undo replace_sdpa()."""
    global _ORIG_SDPA
    if _ORIG_SDPA is not None:
        torch.nn.functional.scaled_dot_product_attention = _ORIG_SDPA
        _ORIG_SDPA = None
