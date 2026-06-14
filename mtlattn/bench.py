"""Benchmark CLI for mtlattn.

    python -m mtlattn.bench                       # default sweep, auto path
    python -m mtlattn.bench --paths both          # MPP and simdgroup side by side
    python -m mtlattn.bench --causal --vs-sdpa    # causal, compared to native SDPA
    python -m mtlattn.bench --sizes 1024 8192 --dtype fp16 --window 256

Reports per-call latency and effective TFLOPS (2*2*M*L*D*H, i.e. QK + PV).
The simdgroup path is forced with MTLATTN_NO_MPP for an apples-to-apples view
on machines that have the M5 MPP path (it is the path M1-M4 actually run).
"""

import argparse
import math
import os
import time

import torch

from . import varlen_attention

_DT = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}


def _pairs(M, causal, window):
    """Attended (query, key) pairs for square self-attention — so reported
    TFLOPS reflects work actually done, not full-attention-equivalent."""
    if window > 0:                       # causal-style sliding window
        return sum(min(i + 1, window) for i in range(M))
    if causal:
        return M * (M + 1) // 2
    return M * M


def _time(fn, iters, warmup=5):
    for _ in range(warmup):
        fn()
    torch.mps.synchronize()
    best = float("inf")
    for _ in range(4):
        t = time.perf_counter()
        for _ in range(iters):
            fn()
        torch.mps.synchronize()
        best = min(best, (time.perf_counter() - t) / iters)
    return best


def _bench_mtlattn(M, H, D, dtype, causal, window, force_simdgroup, iters):
    prev = os.environ.get("MTLATTN_NO_MPP")
    os.environ.pop("MTLATTN_NO_MPP", None)
    os.environ["MTLATTN_MPP_MIN"] = "16"
    if force_simdgroup:
        os.environ["MTLATTN_NO_MPP"] = "1"
    try:
        q = torch.randn(M, H, D, dtype=dtype, device="mps")
        cu = torch.tensor([0, M], dtype=torch.int32, device="mps")
        scale = 1.0 / math.sqrt(D)
        return _time(lambda: varlen_attention(q, q, q, cu, cu, M, scale,
                                              causal=causal, window=window), iters)
    finally:
        os.environ.pop("MTLATTN_NO_MPP", None)
        if prev is not None:
            os.environ["MTLATTN_NO_MPP"] = prev


def _bench_sdpa(M, H, D, dtype, causal, iters):
    q = torch.randn(1, H, M, D, dtype=dtype, device="mps")
    return _time(lambda: torch.nn.functional.scaled_dot_product_attention(
        q, q, q, is_causal=causal), iters)


def main(argv=None):
    ap = argparse.ArgumentParser(prog="mtlattn.bench", description="Benchmark mtlattn attention.")
    ap.add_argument("--sizes", type=int, nargs="+", default=[1024, 2048, 4096],
                    help="sequence lengths (square self-attention)")
    ap.add_argument("--heads", type=int, default=12)
    ap.add_argument("--head-dim", type=int, default=128)
    ap.add_argument("--dtype", default="bf16", choices=list(_DT))
    ap.add_argument("--causal", action="store_true")
    ap.add_argument("--window", type=int, default=0, help="sliding-window size (0 = off)")
    ap.add_argument("--paths", default="auto", choices=["auto", "simdgroup", "both"],
                    help="auto = MPP where available; both = MPP and simdgroup")
    ap.add_argument("--vs-sdpa", action="store_true", help="also time native MPS SDPA")
    ap.add_argument("--iters", type=int, default=10)
    args = ap.parse_args(argv)

    if not torch.backends.mps.is_available():
        raise SystemExit("mtlattn.bench: MPS not available")

    dtype = _DT[args.dtype]
    H, D = args.heads, args.head_dim
    paths = [("auto", False)] if args.paths == "auto" else \
            [("simdgroup", True)] if args.paths == "simdgroup" else \
            [("auto", False), ("simdgroup", True)]

    tag = f"{args.dtype}, H={H}, D={D}" + (", causal" if args.causal else "") + \
          (f", window={args.window}" if args.window else "")
    cols = [name for name, _ in paths] + (["sdpa"] if args.vs_sdpa else [])
    print(f"mtlattn.bench — {tag}")
    print("  " + "size".rjust(7) + "".join(c.rjust(22) for c in cols))
    for M in args.sizes:
        flops = 4 * _pairs(M, args.causal, args.window) * D * H  # QK + PV
        cells = []
        for _, force_sg in paths:
            s = _bench_mtlattn(M, H, D, dtype, args.causal, args.window, force_sg, args.iters)
            cells.append(f"{s*1e3:7.2f}ms {flops/s/1e12:6.2f}TF")
        if args.vs_sdpa:
            try:
                s = _bench_sdpa(M, H, D, dtype, args.causal, args.iters)
                cells.append(f"{s*1e3:7.2f}ms {flops/s/1e12:6.2f}TF")
            except Exception as e:  # SDPA OOMs / errors at large sizes
                cells.append(f"{'err':>16}")
        print("  " + f"{M:7d}" + "".join(c.rjust(22) for c in cells))


if __name__ == "__main__":
    main()
