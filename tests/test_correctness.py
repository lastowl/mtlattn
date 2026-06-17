"""Exactness tests: mtlattn vs per-sequence fp32 SDPA reference."""

import math
import os
import sys

import torch

import mtlattn


def ref_varlen(q, k, v, q_lens, kv_lens, scale, causal=False, window=0, bias=None):
    """Per-sequence fp32 SDPA on CPU. causal: key j seen by query i iff
    j <= i + (kv_len - q_len) (flash_attn end-aligned convention). window>0:
    additionally j > i + (kv_len - q_len) - window (sliding window). bias:
    optional additive [Mq, H, max_kv] fp32 added to the logits before softmax,
    indexed by global query row, head, and seq-local key."""
    out = torch.empty_like(q, dtype=torch.float32)
    qo = kvo = 0
    for ql, kl in zip(q_lens, kv_lens):
        qi = q[qo:qo + ql].float().permute(1, 0, 2)   # [H, L, D]
        ki = k[kvo:kvo + kl].float().permute(1, 0, 2)
        vi = v[kvo:kvo + kl].float().permute(1, 0, 2)
        a = (qi @ ki.transpose(-1, -2)) * scale
        if bias is not None:
            a = a + bias[qo:qo + ql, :, :kl].float().permute(1, 0, 2)  # [H, ql, kl]
        i = torch.arange(ql)[:, None]
        j = torch.arange(kl)[None, :]
        coff = kl - ql
        mask = torch.ones(ql, kl, dtype=torch.bool)
        if causal:
            mask &= (j <= i + coff)
        if window > 0:
            mask &= (j > i + coff - window)
        if causal or window > 0:
            a = a.masked_fill(~mask, float("-inf"))
        out[qo:qo + ql] = (a.softmax(-1) @ vi).permute(1, 0, 2)
        qo += ql
        kvo += kl
    return out


def cu(lens):
    t = torch.tensor([0] + list(lens), dtype=torch.int32)
    return torch.cumsum(t, 0).int().to("mps")


def run_case(name, q_lens, kv_lens, H, D, dtype, atol, causal=False, window=0):
    torch.manual_seed(0)
    Mq, Mkv = sum(q_lens), sum(kv_lens)
    q = torch.randn(Mq, H, D, dtype=dtype)
    k = torch.randn(Mkv, H, D, dtype=dtype)
    v = torch.randn(Mkv, H, D, dtype=dtype)
    scale = 1.0 / math.sqrt(D)

    ref = ref_varlen(q, k, v, q_lens, kv_lens, scale, causal, window)
    out = mtlattn.varlen_attention(
        q.to("mps"), k.to("mps"), v.to("mps"),
        cu(q_lens), cu(kv_lens), max(q_lens), scale, causal=causal, window=window,
    ).cpu().float()

    err = (out - ref).abs().max().item()
    ok = err < atol
    print(f"{name}: max_err={err:.2e} (atol={atol}) {'OK' if ok else 'FAIL'}")
    return ok


def run_packed_case(dtype, atol):
    """qkv-packed and kv-packed wrapper forms (strided views)."""
    torch.manual_seed(1)
    lens = [37, 256, 1, 1023]
    M, H, D = sum(lens), 12, 128
    scale = 1.0 / math.sqrt(D)

    qkv = torch.randn(M, 3, H, D, dtype=dtype)
    q, k, v = qkv.unbind(1)
    ref = ref_varlen(q, k, v, lens, lens, scale)
    out = mtlattn.flash_attn_varlen_qkvpacked_func(
        qkv.to("mps"), cu(lens), max(lens)
    ).cpu().float()
    e1 = (out - ref).abs().max().item()

    q_lens, kv_lens = [64, 512], [300, 100]
    q2 = torch.randn(sum(q_lens), H, D, dtype=dtype)
    kv = torch.randn(sum(kv_lens), 2, H, D, dtype=dtype)
    k2, v2 = kv.unbind(1)
    ref2 = ref_varlen(q2, k2, v2, q_lens, kv_lens, scale)
    out2 = mtlattn.flash_attn_varlen_kvpacked_func(
        q2.to("mps"), kv.to("mps"), cu(q_lens), cu(kv_lens),
        max(q_lens), max(kv_lens),
    ).cpu().float()
    e2 = (out2 - ref2).abs().max().item()

    ok = e1 < atol and e2 < atol
    print(f"packed[{dtype}]: qkv_err={e1:.2e} kv_err={e2:.2e} {'OK' if ok else 'FAIL'}")
    return ok


def run_gqa_case(name, q_lens, kv_lens, Hq, Hkv, D, dtype, atol, causal=False):
    """GQA/MQA: k/v have Hkv heads, Hq a multiple of Hkv. Reference expands
    kv heads (repeat_interleave) so it reduces to the standard varlen ref."""
    torch.manual_seed(0)
    Mq, Mkv = sum(q_lens), sum(kv_lens)
    g = Hq // Hkv
    q = torch.randn(Mq, Hq, D, dtype=dtype)
    k = torch.randn(Mkv, Hkv, D, dtype=dtype)
    v = torch.randn(Mkv, Hkv, D, dtype=dtype)
    scale = 1.0 / math.sqrt(D)

    ref = ref_varlen(q, k.repeat_interleave(g, 1), v.repeat_interleave(g, 1),
                     q_lens, kv_lens, scale, causal)
    out = mtlattn.varlen_attention(
        q.to("mps"), k.to("mps"), v.to("mps"),
        cu(q_lens), cu(kv_lens), max(q_lens), scale, causal=causal,
    ).cpu().float()
    err = (out - ref).abs().max().item()
    ok = err < atol
    print(f"{name}: max_err={err:.2e} (atol={atol}) {'OK' if ok else 'FAIL'}")
    return ok


def run_sdpa_case(name, B, Hq, Hkv, N, D, dtype, atol, causal=False):
    """Dense [B,H,N,D] mtlattn.sdpa() adapter vs the fp32 varlen reference."""
    torch.manual_seed(0)
    q = torch.randn(B, Hq, N, D, dtype=dtype)
    k = torch.randn(B, Hkv, N, D, dtype=dtype)
    v = torch.randn(B, Hkv, N, D, dtype=dtype)
    scale = 1.0 / math.sqrt(D)
    g = Hq // Hkv
    qf = q.permute(0, 2, 1, 3).reshape(B * N, Hq, D)
    kf = k.repeat_interleave(g, 1).permute(0, 2, 1, 3).reshape(B * N, Hq, D)
    vf = v.repeat_interleave(g, 1).permute(0, 2, 1, 3).reshape(B * N, Hq, D)
    ref = ref_varlen(qf, kf, vf, [N] * B, [N] * B, scale, causal)
    ref = ref.reshape(B, N, Hq, D).permute(0, 2, 1, 3)
    out = mtlattn.sdpa(q.to("mps"), k.to("mps"), v.to("mps"), is_causal=causal).cpu().float()
    err = (out - ref.float()).abs().max().item()
    ok = err < atol
    print(f"{name}: max_err={err:.2e} (atol={atol}) {'OK' if ok else 'FAIL'}")
    return ok


def run_padding_case(name, B, H, N, D, Ls, dtype, atol, causal=False):
    """replace_sdpa() converting a self-attention key-padding mask to varlen,
    vs the fp32 varlen reference over the packed valid tokens."""
    import torch.nn.functional as F
    torch.manual_seed(0)
    q = torch.randn(B, H, N, D, dtype=dtype)
    k = torch.randn(B, H, N, D, dtype=dtype)
    v = torch.randn(B, H, N, D, dtype=dtype)
    scale = 1.0 / math.sqrt(D)
    pack = lambda x: torch.cat([x[b, :, :Ls[b], :].transpose(0, 1) for b in range(B)], 0)
    ref = ref_varlen(pack(q), pack(k), pack(v), Ls, Ls, scale, causal)
    mask = torch.zeros(B, 1, 1, N, dtype=torch.bool)
    for b, Lb in enumerate(Ls):
        mask[b, 0, 0, :Lb] = True
    mtlattn.replace_sdpa(min_seqlen=1)
    try:
        out = F.scaled_dot_product_attention(q.to("mps"), k.to("mps"), v.to("mps"),
                                             attn_mask=mask.to("mps"), is_causal=causal).cpu().float()
    finally:
        mtlattn.restore_sdpa()
    outp = torch.cat([out[b, :, :Ls[b], :].transpose(0, 1) for b in range(B)], 0)
    err = (outp - ref.float()).abs().max().item()
    ok = err < atol
    print(f"{name}: max_err={err:.2e} (atol={atol}) {'OK' if ok else 'FAIL'}")
    return ok


def run_bwd_case(name, q_lens, kv_lens, Hq, Hkv, D, dtype, atol, causal=False, window=0):
    """Gradients from mtlattn's backward vs a differentiable per-sequence ref."""
    torch.manual_seed(0)
    g = Hq // Hkv
    scale = 1.0 / math.sqrt(D)
    Mq, Mk = sum(q_lens), sum(kv_lens)
    q = torch.randn(Mq, Hq, D, dtype=dtype)
    k = torch.randn(Mk, Hkv, D, dtype=dtype)
    v = torch.randn(Mk, Hkv, D, dtype=dtype)
    qm = q.to("mps").requires_grad_(); km = k.to("mps").requires_grad_(); vm = v.to("mps").requires_grad_()
    out = mtlattn.varlen_attention(qm, km, vm, cu(q_lens), cu(kv_lens), max(q_lens),
                                   causal=causal, window=window)
    dO = torch.randn_like(out)
    out.backward(dO)

    # differentiable fp32 reference (kv heads expanded for GQA)
    qr = q.float().requires_grad_()
    kr = k.repeat_interleave(g, 1).float().requires_grad_()
    vr = v.repeat_interleave(g, 1).float().requires_grad_()
    outs, qo, ko = [], 0, 0
    for ql, kl in zip(q_lens, kv_lens):
        qi = qr[qo:qo + ql].permute(1, 0, 2)
        ki = kr[ko:ko + kl].permute(1, 0, 2)
        vi = vr[ko:ko + kl].permute(1, 0, 2)
        a = (qi @ ki.transpose(-1, -2)) * scale
        if causal or window:
            i = torch.arange(ql)[:, None]; j = torch.arange(kl)[None, :]; coff = kl - ql
            m = torch.ones(ql, kl, dtype=torch.bool)
            if causal:
                m &= (j <= i + coff)
            if window:
                m &= (j > i + coff - window)
            a = a.masked_fill(~m, float("-inf"))
        outs.append((a.softmax(-1) @ vi).permute(1, 0, 2)); qo += ql; ko += kl
    torch.cat(outs, 0).backward(dO.cpu().float())
    gk = kr.grad.view(Mk, Hkv, g, D).sum(2)   # fold expanded-head grads back
    gv = vr.grad.view(Mk, Hkv, g, D).sum(2)
    eq = (qm.grad.cpu().float() - qr.grad).abs().max().item()
    ek = (km.grad.cpu().float() - gk).abs().max().item()
    ev = (vm.grad.cpu().float() - gv).abs().max().item()
    ok = max(eq, ek, ev) < atol
    print(f"{name}: dQ={eq:.2e} dK={ek:.2e} dV={ev:.2e} (atol={atol}) {'OK' if ok else 'FAIL'}")
    return ok


def _make_bias(Mq, H, max_kv, broadcast, seed=7):
    """Additive bias [Mq, H or 1, max_kv] fp32, and its per-head-expanded form
    for the reference. Mix of smooth and a few -inf-ish entries to exercise the
    softmax max/normalize with a real mask."""
    torch.manual_seed(seed)
    hb = 1 if broadcast else H
    b = torch.randn(Mq, hb, max_kv, dtype=torch.float32)
    b[:, :, ::7] -= 30.0  # strongly-suppressed keys (near -inf after softmax)
    ref = b.expand(Mq, H, max_kv).contiguous() if broadcast else b
    return b, ref


def run_bias_case(name, q_lens, kv_lens, H, D, dtype, atol, causal=False,
                  window=0, broadcast=False):
    """Forward with an additive attn_mask (per-head or head-broadcast)."""
    torch.manual_seed(0)
    Mq, Mkv = sum(q_lens), sum(kv_lens)
    max_kv = max(kv_lens)
    q = torch.randn(Mq, H, D, dtype=dtype)
    k = torch.randn(Mkv, H, D, dtype=dtype)
    v = torch.randn(Mkv, H, D, dtype=dtype)
    scale = 1.0 / math.sqrt(D)
    b, bref = _make_bias(Mq, H, max_kv, broadcast)
    ref = ref_varlen(q, k, v, q_lens, kv_lens, scale, causal, window, bias=bref)
    out = mtlattn.varlen_attention(
        q.to("mps"), k.to("mps"), v.to("mps"), cu(q_lens), cu(kv_lens),
        max(q_lens), scale, causal=causal, window=window, attn_bias=b.to("mps"),
    ).cpu().float()
    err = (out - ref).abs().max().item()
    ok = err < atol
    print(f"{name}: max_err={err:.2e} (atol={atol}) {'OK' if ok else 'FAIL'}")
    return ok


def run_bias_bwd_case(name, q_lens, kv_lens, Hq, Hkv, D, dtype, atol, causal=False, broadcast=False):
    """Gradients (dQ/dK/dV) with a constant additive attn_mask vs a
    differentiable fp32 reference (bias held constant)."""
    torch.manual_seed(0)
    g = Hq // Hkv
    scale = 1.0 / math.sqrt(D)
    Mq, Mk = sum(q_lens), sum(kv_lens)
    max_kv = max(kv_lens)
    q = torch.randn(Mq, Hq, D, dtype=dtype)
    k = torch.randn(Mk, Hkv, D, dtype=dtype)
    v = torch.randn(Mk, Hkv, D, dtype=dtype)
    b, bref = _make_bias(Mq, Hq, max_kv, broadcast)
    qm = q.to("mps").requires_grad_(); km = k.to("mps").requires_grad_(); vm = v.to("mps").requires_grad_()
    out = mtlattn.varlen_attention(qm, km, vm, cu(q_lens), cu(kv_lens), max(q_lens),
                                   causal=causal, attn_bias=b.to("mps"))
    dO = torch.randn_like(out)
    out.backward(dO)

    qr = q.float().requires_grad_()
    kr = k.repeat_interleave(g, 1).float().requires_grad_()
    vr = v.repeat_interleave(g, 1).float().requires_grad_()
    outs, qo, ko = [], 0, 0
    for ql, kl in zip(q_lens, kv_lens):
        qi = qr[qo:qo + ql].permute(1, 0, 2)
        ki = kr[ko:ko + kl].permute(1, 0, 2)
        vi = vr[ko:ko + kl].permute(1, 0, 2)
        a = (qi @ ki.transpose(-1, -2)) * scale
        a = a + bref[qo:qo + ql, :, :kl].permute(1, 0, 2)
        if causal:
            i = torch.arange(ql)[:, None]; j = torch.arange(kl)[None, :]; coff = kl - ql
            a = a.masked_fill(~(j <= i + coff), float("-inf"))
        outs.append((a.softmax(-1) @ vi).permute(1, 0, 2)); qo += ql; ko += kl
    torch.cat(outs, 0).backward(dO.cpu().float())
    gk = kr.grad.view(Mk, Hkv, g, D).sum(2)
    gv = vr.grad.view(Mk, Hkv, g, D).sum(2)
    eq = (qm.grad.cpu().float() - qr.grad).abs().max().item()
    ek = (km.grad.cpu().float() - gk).abs().max().item()
    ev = (vm.grad.cpu().float() - gv).abs().max().item()
    ok = max(eq, ek, ev) < atol
    print(f"{name}: dQ={eq:.2e} dK={ek:.2e} dV={ev:.2e} (atol={atol}) {'OK' if ok else 'FAIL'}")
    return ok


def run_sdpa_mask_case(name, B, H, N, D, dtype, atol, bool_mask=False, broadcast_head=False):
    """Dense sdpa() with an attn_mask (float additive or bool) vs native fp32 SDPA."""
    import torch.nn.functional as F
    torch.manual_seed(0)
    q = torch.randn(B, H, N, D, dtype=dtype)
    k = torch.randn(B, H, N, D, dtype=dtype)
    v = torch.randn(B, H, N, D, dtype=dtype)
    hb = 1 if broadcast_head else H
    if bool_mask:
        m = torch.rand(B, hb, N, N) > 0.3
        m[..., 0] = True                    # guarantee >=1 visible key per row
    else:
        m = torch.randn(B, hb, N, N, dtype=torch.float32)
    out = mtlattn.sdpa(q.to("mps"), k.to("mps"), v.to("mps"), attn_mask=m.to("mps")).cpu().float()
    ref = F.scaled_dot_product_attention(q.float(), k.float(), v.float(), attn_mask=m).float()
    err = (out - ref).abs().max().item()
    ok = err < atol
    print(f"{name}: max_err={err:.2e} (atol={atol}) {'OK' if ok else 'FAIL'}")
    return ok


def main():
    results = []
    # dtype -> tolerance (inputs are random N(0,1); fp16/bf16 storage rounding
    # dominates; accumulation is fp32 in both kernel and reference)
    for dtype, atol in [(torch.float32, 1e-4), (torch.float16, 5e-3), (torch.bfloat16, 3e-2)]:
        results.append(run_case(f"self[{dtype}] ragged", [5, 128, 33, 1000], [5, 128, 33, 1000], 12, 128, dtype, atol))
        results.append(run_case(f"cross[{dtype}]", [64, 7, 300], [128, 1, 900], 12, 128, dtype, atol))
        results.append(run_packed_case(dtype, atol))
    # odd head dims (D < 128, including non-multiple-of-4)
    results.append(run_case("self D=64", [100, 200], [100, 200], 16, 64, torch.float16, 5e-3))
    results.append(run_case("self D=80", [77, 13], [77, 13], 8, 80, torch.float16, 5e-3))
    # many tiny windows (windowed-attention shape)
    torch.manual_seed(2)
    wlens = torch.randint(1, 65, (5000,)).tolist()
    results.append(run_case("5000 windows", wlens, wlens, 12, 128, torch.float16, 5e-3))
    # single long sequence
    results.append(run_case("1x16384", [16384], [16384], 12, 128, torch.float16, 5e-3))

    # head_dim=64 on the MPP path (seqlen >= the MPP gate so it isn't the
    # simdgroup fallback). matmul2d is dimension-general; D=64 covers many LLMs.
    results.append(run_case("D=64 MPP full", [2048], [2048], 12, 64, torch.float16, 5e-3))
    results.append(run_case("D=64 MPP causal", [4096], [4096], 12, 64, torch.float16, 5e-3, causal=True))
    results.append(run_case("D=64 MPP bf16", [1500], [1500], 8, 64, torch.bfloat16, 3e-2, causal=True))
    results.append(run_case("D=64 MPP SWA", [2048], [2048], 12, 64, torch.float16, 5e-3, causal=True, window=128))
    results.append(run_case("D=96 MPP causal", [2048], [2048], 12, 96, torch.float16, 5e-3, causal=True))
    results.append(run_case("D=96 MPP bf16", [1500], [1500], 8, 96, torch.bfloat16, 3e-2))
    # head_dim > 128 (256) — MPP-only (no simdgroup kernel that large), so only
    # exercise it where the MPP path is available (macOS 26.2+, not MTLATTN_NO_MPP).
    if mtlattn._C.mpp_available() and not os.environ.get("MTLATTN_NO_MPP"):
        results.append(run_case("D=256 MPP full", [2048], [2048], 8, 256, torch.float16, 5e-3))
        results.append(run_case("D=256 MPP causal", [1024], [1024], 12, 256, torch.float16, 5e-3, causal=True))
        results.append(run_case("D=256 MPP bf16", [1500], [1500], 8, 256, torch.bfloat16, 4e-2, causal=True))
    # head_dim not on the fast path (<=128) falls back to simdgroup, still correct
    results.append(run_case("D=80 simdgroup", [200, 700], [200, 700], 8, 80, torch.float16, 5e-3, causal=True))

    # causal masking. D in {64,128} large -> MPP path, small/odd-D -> simdgroup;
    # ragged batches and a cross (q_len<kv_len, cached-decode) offset included.
    results.append(run_case("causal self D=128", [1024], [1024], 12, 128, torch.float16, 5e-3, causal=True))
    results.append(run_case("causal self bf16", [777], [777], 8, 128, torch.bfloat16, 3e-2, causal=True))
    results.append(run_case("causal ragged D=128", [33, 256, 1000], [33, 256, 1000], 12, 128, torch.float16, 5e-3, causal=True))
    results.append(run_case("causal cross (cached)", [64], [320], 8, 128, torch.float16, 5e-3, causal=True))
    results.append(run_case("causal self D=64", [200, 50], [200, 50], 16, 64, torch.float16, 5e-3, causal=True))

    # GQA / MQA (grouped-query KV). D=128 -> MPP path, D=64 -> simdgroup;
    # combined with causal and a cross (cached-decode) shape.
    results.append(run_gqa_case("GQA 12->4 D=128", [512], [512], 12, 4, 128, torch.float16, 5e-3))
    results.append(run_gqa_case("MQA 8->1 D=128", [33, 300], [33, 300], 8, 1, 128, torch.float16, 5e-3))
    results.append(run_gqa_case("GQA 8->2 causal", [400], [400], 8, 2, 128, torch.float16, 5e-3, causal=True))
    results.append(run_gqa_case("MQA cross cached", [16], [512], 8, 1, 128, torch.float16, 5e-3, causal=True))
    results.append(run_gqa_case("GQA 8->2 D=64", [200, 50], [200, 50], 8, 2, 64, torch.float16, 5e-3))

    # Sliding-window / local attention. Mistral SWA = causal + window; also a
    # ragged batch, a window >= seqlen (no-op), non-causal banded, and D=64.
    results.append(run_case("SWA causal w=128 D=128", [1024], [1024], 12, 128, torch.float16, 5e-3, causal=True, window=128))
    results.append(run_case("SWA causal ragged", [600, 50, 333], [600, 50, 333], 8, 128, torch.float16, 5e-3, causal=True, window=64))
    results.append(run_case("SWA w>=len (no-op)", [200], [200], 8, 128, torch.float16, 5e-3, causal=True, window=512))
    results.append(run_case("SWA non-causal band", [256], [256], 8, 128, torch.float16, 5e-3, causal=False, window=48))
    results.append(run_case("SWA D=64", [300, 40], [300, 40], 16, 64, torch.float16, 5e-3, causal=True, window=80))

    # Backward pass (gradients) — full / causal / GQA / window / varlen / dtypes.
    results.append(run_bwd_case("bwd full fp32", [256], [256], 8, 8, 128, torch.float32, 1e-4))
    results.append(run_bwd_case("bwd causal fp32", [256], [256], 8, 8, 128, torch.float32, 1e-4, causal=True))
    results.append(run_bwd_case("bwd GQA+causal", [256], [256], 8, 2, 128, torch.float32, 1e-4, causal=True))
    results.append(run_bwd_case("bwd window", [512], [512], 8, 8, 128, torch.float32, 1e-4, causal=True, window=64))
    results.append(run_bwd_case("bwd varlen GQA", [128, 300], [128, 300], 12, 4, 128, torch.float32, 1e-4, causal=True))
    results.append(run_bwd_case("bwd fp16 causal", [256], [256], 8, 8, 128, torch.float16, 5e-3, causal=True))
    # fp16/bf16 D=128 backward exercises the matmul2d (MPP) path, incl. packed ragged.
    results.append(run_bwd_case("bwd fp16 varlen", [128, 300, 64], [128, 300, 64], 8, 8, 128, torch.float16, 5e-3, causal=True))
    results.append(run_bwd_case("bwd fp16 varlen GQA", [200, 50, 333], [200, 50, 333], 12, 4, 128, torch.float16, 6e-3, causal=True))
    results.append(run_bwd_case("bwd fp16 window", [512], [512], 8, 8, 128, torch.float16, 5e-3, causal=True, window=64))
    results.append(run_bwd_case("bwd bf16 varlen", [180, 90], [180, 90], 8, 8, 128, torch.bfloat16, 4e-2, causal=True))
    results.append(run_bwd_case("bwd fp16 D=64", [200, 50], [200, 50], 8, 8, 64, torch.float16, 5e-3, causal=True))
    results.append(run_bwd_case("bwd fp16 D=96 GQA", [256], [256], 8, 2, 96, torch.float16, 6e-3, causal=True))
    if mtlattn._C.mpp_available() and not os.environ.get("MTLATTN_NO_MPP"):
        results.append(run_bwd_case("bwd fp16 D=256", [512], [512], 8, 8, 256, torch.float16, 6e-3, causal=True))

    # Arbitrary additive attn_mask (MPP-only). Forward + backward, per-head and
    # head-broadcast, with/without causal, ragged varlen, fp16/bf16, D 64/96/128/256.
    if mtlattn._C.mpp_available() and not os.environ.get("MTLATTN_NO_MPP"):
        results.append(run_bias_case("bias fwd perhead D128", [1536], [1536], 8, 128, torch.float16, 5e-3))
        results.append(run_bias_case("bias fwd broadcast D128", [1536], [1536], 8, 128, torch.float16, 5e-3, broadcast=True))
        results.append(run_bias_case("bias fwd causal D128", [1536], [1536], 12, 128, torch.float16, 5e-3, causal=True))
        results.append(run_bias_case("bias fwd ragged varlen", [200, 50, 1300], [200, 50, 1300], 8, 128, torch.float16, 5e-3))
        results.append(run_bias_case("bias fwd cross", [64], [900], 8, 128, torch.float16, 5e-3))
        results.append(run_bias_case("bias fwd SWA", [1536], [1536], 8, 128, torch.float16, 5e-3, causal=True, window=128))
        results.append(run_bias_case("bias fwd bf16", [1500], [1500], 8, 128, torch.bfloat16, 3e-2, broadcast=True))
        results.append(run_bias_case("bias fwd D64", [2048], [2048], 12, 64, torch.float16, 5e-3))
        results.append(run_bias_case("bias fwd D96 causal", [2048], [2048], 8, 96, torch.float16, 5e-3, causal=True))
        results.append(run_bias_case("bias fwd D256", [1024], [1024], 8, 256, torch.float16, 6e-3))
        results.append(run_bias_bwd_case("bias bwd perhead D128", [1280], [1280], 8, 8, 128, torch.float16, 5e-3))
        results.append(run_bias_bwd_case("bias bwd broadcast D128", [1280], [1280], 8, 8, 128, torch.float16, 5e-3, broadcast=True))
        results.append(run_bias_bwd_case("bias bwd causal D128", [1280], [1280], 12, 12, 128, torch.float16, 5e-3, causal=True))
        results.append(run_bias_bwd_case("bias bwd GQA causal", [1024], [1024], 8, 2, 128, torch.float16, 6e-3, causal=True))
        results.append(run_bias_bwd_case("bias bwd ragged", [128, 700], [128, 700], 8, 8, 128, torch.float16, 5e-3))
        results.append(run_bias_bwd_case("bias bwd D64 bf16", [1280], [1280], 8, 8, 64, torch.bfloat16, 4e-2))

    # The SDPA adapter cases go through torch's own MPS ops (permute/contiguous
    # -> a runtime-compiled transpose shader). Set MTLATTN_SKIP_SDPA to skip them
    # where torch can't JIT-compile MPS shaders (e.g. headless CI runners, which
    # still run our pre-compiled metallibs fine — the kernel cases above pass).
    if os.environ.get("MTLATTN_SKIP_SDPA"):
        print("(skipping sdpa/padding adapter cases — MTLATTN_SKIP_SDPA set)")
    else:
        # Dense scaled_dot_product_attention adapter (mtlattn.sdpa): MHA, GQA, causal.
        results.append(run_sdpa_case("sdpa MHA B2", 2, 8, 8, 1024, 128, torch.float16, 5e-3))
        results.append(run_sdpa_case("sdpa GQA B1", 1, 8, 2, 2048, 128, torch.float16, 5e-3))
        results.append(run_sdpa_case("sdpa causal", 2, 12, 12, 512, 128, torch.float16, 5e-3, causal=True))
        # replace_sdpa key-padding mask -> varlen conversion (full and causal).
        results.append(run_padding_case("pad mask", 4, 8, 1024, 128, [1024, 800, 500, 1024], torch.float16, 5e-3))
        results.append(run_padding_case("pad mask causal", 3, 12, 1024, 128, [1024, 700, 400], torch.float16, 5e-3, causal=True))
        # sdpa() with a general attn_mask -> additive-bias path (MPP-only).
        if mtlattn._C.mpp_available() and not os.environ.get("MTLATTN_NO_MPP"):
            results.append(run_sdpa_mask_case("sdpa float mask perhead", 2, 8, 1024, 128, torch.float16, 5e-3))
            results.append(run_sdpa_mask_case("sdpa float mask broadcast", 2, 8, 1024, 128, torch.float16, 5e-3, broadcast_head=True))
            results.append(run_sdpa_mask_case("sdpa bool mask", 2, 8, 1024, 128, torch.float16, 5e-3, bool_mask=True))
            results.append(run_sdpa_mask_case("sdpa bool mask broadcast", 1, 12, 2048, 128, torch.float16, 5e-3, bool_mask=True, broadcast_head=True))
    # outlier channels (real transformer activations spike to 1e2-1e3; QK
    # partial sums must not overflow — caught a NaN bug in half fragments)
    torch.manual_seed(3)
    q = torch.randn(2048, 12, 128, dtype=torch.bfloat16); q[:, :, 5] *= 300
    k = torch.randn(2048, 12, 128, dtype=torch.bfloat16); k[:, :, 5] *= 300
    v = torch.randn(2048, 12, 128, dtype=torch.bfloat16)
    out = mtlattn.varlen_attention(q.to("mps"), k.to("mps"), v.to("mps"),
                                   cu([1024, 1024]), cu([1024, 1024]), 1024)
    ref = ref_varlen(q, k, v, [1024, 1024], [1024, 1024], 1.0 / math.sqrt(128))
    # With 300x outliers the softmax is near-argmax over scores spanning
    # thousands: arithmetic-order noise legitimately flips near-tied winners,
    # so tolerance is loose; the hard assertion is NaN/inf-freedom (the
    # actual half-fragment failure mode this guards against).
    err = (out.cpu().float() - ref).abs().max().item()
    ok = not out.isnan().any().item() and not out.isinf().any().item() and err < 1e-1
    print(f"outlier channels bf16: max_err={err:.2e} nan={out.isnan().any().item()} {'OK' if ok else 'FAIL'}")
    results.append(ok)

    print("ALL PASS" if all(results) else "FAILURES PRESENT")
    sys.exit(0 if all(results) else 1)


if __name__ == "__main__":
    main()
