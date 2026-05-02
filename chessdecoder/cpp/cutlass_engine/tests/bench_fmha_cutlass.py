"""Bench CUTLASS FMHA vs hand-rolled fmha_prefill at varying B, S.

Causal-mask only (J.2 first iter). Reports per-call latency in ms +
speedup ratio. Targets the conditions where the long-rollout cliff hits
the hand-rolled kernel.
"""

import math
import time

import torch
import _cutlass_decoder_cpp as ce


def _bench(B: int, S: int, NH: int, HD: int = 64, iters: int = 20):
    scale = 1.0 / math.sqrt(HD)

    torch.manual_seed(0)
    Q = torch.randn(B, S, NH, HD, device="cuda", dtype=torch.float16) * 0.1
    K = torch.randn(B, S, NH, HD, device="cuda", dtype=torch.float16) * 0.1
    V = torch.randn(B, S, NH, HD, device="cuda", dtype=torch.float16) * 0.1
    O = torch.zeros_like(Q)

    # CUTLASS path setup
    ws_bytes = ce.kernels.fmha_prefill_cutlass_workspace_bytes(B, S, NH, HD)
    lse_n = ce.kernels.fmha_prefill_cutlass_lse_elements(B, S, NH)
    workspace = torch.empty(max(1, ws_bytes), device="cuda", dtype=torch.uint8)
    lse = torch.empty(lse_n, device="cuda", dtype=torch.float32)

    # Hand-rolled path setup: each token in own block ⇒ behaves causally
    block_id = torch.arange(S, device="cuda", dtype=torch.int32).unsqueeze(0).expand(B, -1).contiguous()
    active = torch.ones(B, device="cuda", dtype=torch.int32)

    # warmup
    for _ in range(3):
        ce.kernels.fmha_prefill_cutlass_causal(
            Q.data_ptr(), K.data_ptr(), V.data_ptr(), O.data_ptr(),
            B, S, NH, HD, scale, workspace.data_ptr(), lse.data_ptr())
        ce.kernels.fmha_prefill_dispatch(
            Q.data_ptr(), K.data_ptr(), V.data_ptr(),
            block_id.data_ptr(), active.data_ptr(), O.data_ptr(),
            B, S, NH, HD, scale)
    torch.cuda.synchronize()

    # Bench CUTLASS
    t0 = time.perf_counter()
    for _ in range(iters):
        ce.kernels.fmha_prefill_cutlass_causal(
            Q.data_ptr(), K.data_ptr(), V.data_ptr(), O.data_ptr(),
            B, S, NH, HD, scale, workspace.data_ptr(), lse.data_ptr())
    torch.cuda.synchronize()
    t_cutlass = (time.perf_counter() - t0) / iters * 1000  # ms

    # Bench hand-rolled
    t0 = time.perf_counter()
    for _ in range(iters):
        ce.kernels.fmha_prefill_dispatch(
            Q.data_ptr(), K.data_ptr(), V.data_ptr(),
            block_id.data_ptr(), active.data_ptr(), O.data_ptr(),
            B, S, NH, HD, scale)
    torch.cuda.synchronize()
    t_hand = (time.perf_counter() - t0) / iters * 1000  # ms

    speedup = t_hand / t_cutlass
    print(f"  B={B:3d} S={S:4d}: cutlass={t_cutlass:7.3f}ms  hand={t_hand:7.3f}ms  speedup={speedup:5.2f}x")
    return t_cutlass, t_hand, speedup


def main():
    print("=== FMHA bench (causal, HD=64 FP16) ===")
    print(f"{'config':>20}  {'cutlass':>10}  {'hand':>10}  {'speedup':>8}")
    for B, S in [
        (16, 1024),
        (32, 1024),
        (64, 1024),
        (16, 2048),
        (64, 2048),
        (16, 4096),
        (64, 4096),
        (128, 4096),
        (256, 4096),
    ]:
        _bench(B, S, NH=16)


if __name__ == "__main__":
    main()
