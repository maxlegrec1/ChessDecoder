"""Smoke test for CUTLASS Blackwell FMHA wrapper (J.2 first iter).

Compares fmha_prefill_cutlass_causal against torch's scaled_dot_product_attention
with is_causal=True for HD=64 FP16. Pass criterion: max abs diff under FP16
noise threshold (~5e-3).
"""

import math

import torch
import _cutlass_decoder_cpp as ce


def _run_case(B: int, S: int, NH: int, HD: int = 64, scale: float | None = None):
    if scale is None:
        scale = 1.0 / math.sqrt(HD)

    torch.manual_seed(0)
    Q = torch.randn(B, S, NH, HD, device="cuda", dtype=torch.float16) * 0.1
    K = torch.randn(B, S, NH, HD, device="cuda", dtype=torch.float16) * 0.1
    V = torch.randn(B, S, NH, HD, device="cuda", dtype=torch.float16) * 0.1
    O = torch.zeros_like(Q)

    ws_bytes = ce.kernels.fmha_prefill_cutlass_workspace_bytes(B, S, NH, HD)
    lse_n = ce.kernels.fmha_prefill_cutlass_lse_elements(B, S, NH)
    workspace = torch.empty(max(1, ws_bytes), device="cuda", dtype=torch.uint8)
    lse = torch.empty(lse_n, device="cuda", dtype=torch.float32)

    ce.kernels.fmha_prefill_cutlass_causal(
        Q.data_ptr(), K.data_ptr(), V.data_ptr(), O.data_ptr(),
        B, S, NH, HD, scale, workspace.data_ptr(), lse.data_ptr())

    # torch reference: SDPA expects [B, H, S, D]
    Q_ref = Q.permute(0, 2, 1, 3).contiguous()
    K_ref = K.permute(0, 2, 1, 3).contiguous()
    V_ref = V.permute(0, 2, 1, 3).contiguous()
    O_ref = torch.nn.functional.scaled_dot_product_attention(
        Q_ref, K_ref, V_ref, is_causal=True, scale=scale)
    O_ref = O_ref.permute(0, 2, 1, 3).contiguous()

    diff = (O.float() - O_ref.float()).abs()
    max_diff = diff.max().item()
    mean_diff = diff.mean().item()
    print(f"  B={B} S={S} NH={NH} HD={HD}: max={max_diff:.4e} mean={mean_diff:.4e}")

    assert max_diff < 0.01, f"max diff {max_diff:.4e} exceeds tolerance"
    return max_diff, mean_diff


def main():
    print("=== CUTLASS FMHA causal smoke test ===")
    for B, S, NH in [
        (1, 256, 16),
        (4, 256, 16),
        (8, 512, 16),
        (16, 1024, 16),
        (32, 2048, 16),
        (64, 4096, 16),
    ]:
        _run_case(B, S, NH)
    print("All cases passed.")


if __name__ == "__main__":
    main()
