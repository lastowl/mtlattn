"""mtlattn quickstart — the main entry points, each checked against a
reference. Run on an Apple Silicon Mac with PyTorch MPS:

    python examples/quickstart.py
"""

import math

import torch
import torch.nn.functional as F

import mtlattn


def _ref(q, k, v, scale, causal=False, window=0):
    # q,k,v: [N, H, D] single sequence; returns [N, H, D]
    N, _, D = q.shape
    L = k.shape[0]
    a = (q.permute(1, 0, 2).float() @ k.permute(1, 0, 2).float().transpose(-1, -2)) * scale
    if causal or window:
        i = torch.arange(N)[:, None]
        j = torch.arange(L)[None, :]
        coff = L - N
        m = torch.ones(N, L, dtype=torch.bool)
        if causal:
            m &= j <= i + coff
        if window:
            m &= j > i + coff - window
        a = a.masked_fill(~m.to(a.device), float("-inf"))
    return (a.softmax(-1) @ v.permute(1, 0, 2).float()).permute(1, 0, 2)


def main():
    assert torch.backends.mps.is_available(), "needs PyTorch MPS"
    dev, dt = "mps", torch.float16
    torch.manual_seed(0)
    H, D = 12, 128
    scale = 1.0 / math.sqrt(D)

    def cu(*lens):
        return torch.tensor([0, *torch.tensor(lens).cumsum(0).tolist()],
                            dtype=torch.int32, device=dev)

    # 1. Variable-length (ragged) attention — three packed sequences, no padding.
    lens = [512, 37, 1000]
    M = sum(lens)
    q = torch.randn(M, H, D, dtype=dt, device=dev)
    k = torch.randn(M, H, D, dtype=dt, device=dev)
    v = torch.randn(M, H, D, dtype=dt, device=dev)
    out = mtlattn.varlen_attention(q, k, v, cu(*lens), cu(*lens), max(lens))
    print(f"1. varlen (ragged {lens}):           out {tuple(out.shape)}")

    # 2. Causal self-attention.
    N = 1024
    q = torch.randn(N, H, D, dtype=dt, device=dev)
    out = mtlattn.varlen_attention(q, q, q, cu(N), cu(N), N, causal=True)
    err = (out.float() - _ref(q, q, q, scale, causal=True)).abs().max().item()
    print(f"2. causal (N={N}):                    max_err {err:.1e}")

    # 3. GQA — 12 query heads, 4 KV heads.
    kv = torch.randn(N, 4, D, dtype=dt, device=dev)
    out = mtlattn.varlen_attention(q, kv, kv, cu(N), cu(N), N, causal=True)
    print(f"3. GQA (12 q-heads, 4 kv-heads):      out {tuple(out.shape)}")

    # 4. Sliding window (Mistral-style SWA = causal + window).
    out = mtlattn.varlen_attention(q, q, q, cu(N), cu(N), N, causal=True, window=256)
    err = (out.float() - _ref(q, q, q, scale, causal=True, window=256)).abs().max().item()
    print(f"4. sliding window (W=256):            max_err {err:.1e}")

    # 5. Drop-in for scaled_dot_product_attention — any model uses mtlattn.
    mtlattn.replace_sdpa(min_seqlen=512)
    qd = torch.randn(2, H, N, D, dtype=dt, device=dev)
    o_mtl = F.scaled_dot_product_attention(qd, qd, qd, is_causal=True)
    mtlattn.restore_sdpa()
    o_ref = F.scaled_dot_product_attention(qd, qd, qd, is_causal=True)
    err = (o_mtl.float() - o_ref.float()).abs().max().item()
    print(f"5. replace_sdpa() vs native SDPA:     max_err {err:.1e}")

    print("\nok")


if __name__ == "__main__":
    main()
