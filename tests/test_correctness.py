"""Exactness tests: mtlattn vs per-sequence fp32 SDPA reference."""

import math
import sys

import torch

import mtlattn


def ref_varlen(q, k, v, q_lens, kv_lens, scale):
    """Per-sequence fp32 SDPA on CPU."""
    out = torch.empty_like(q, dtype=torch.float32)
    qo = kvo = 0
    for ql, kl in zip(q_lens, kv_lens):
        qi = q[qo:qo + ql].float().permute(1, 0, 2)   # [H, L, D]
        ki = k[kvo:kvo + kl].float().permute(1, 0, 2)
        vi = v[kvo:kvo + kl].float().permute(1, 0, 2)
        a = (qi @ ki.transpose(-1, -2)) * scale
        out[qo:qo + ql] = (a.softmax(-1) @ vi).permute(1, 0, 2)
        qo += ql
        kvo += kl
    return out


def cu(lens):
    t = torch.tensor([0] + list(lens), dtype=torch.int32)
    return torch.cumsum(t, 0).int().to("mps")


def run_case(name, q_lens, kv_lens, H, D, dtype, atol):
    torch.manual_seed(0)
    Mq, Mkv = sum(q_lens), sum(kv_lens)
    q = torch.randn(Mq, H, D, dtype=dtype)
    k = torch.randn(Mkv, H, D, dtype=dtype)
    v = torch.randn(Mkv, H, D, dtype=dtype)
    scale = 1.0 / math.sqrt(D)

    ref = ref_varlen(q, k, v, q_lens, kv_lens, scale)
    out = mtlattn.varlen_attention(
        q.to("mps"), k.to("mps"), v.to("mps"),
        cu(q_lens), cu(kv_lens), max(q_lens), scale,
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
