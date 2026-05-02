"""Debug: hardcoded apply_mask limit=1 should make output strongly
diverge from causal output. Run causal then block_aware on the same
inputs and check they differ — confirms apply_mask is being executed."""

import math
import torch
import _cutlass_decoder_cpp as ce


B, S, NH, HD = 1, 256, 16, 64
scale = 1.0 / math.sqrt(HD)

torch.manual_seed(0)
Q = torch.randn(B, S, NH, HD, device="cuda", dtype=torch.float16) * 0.1
K = torch.randn(B, S, NH, HD, device="cuda", dtype=torch.float16) * 0.1
V = torch.randn(B, S, NH, HD, device="cuda", dtype=torch.float16) * 0.1
O_causal = torch.zeros_like(Q)
O_block = torch.zeros_like(Q)

ws_bytes = ce.kernels.fmha_prefill_cutlass_workspace_bytes(B, S, NH, HD)
lse_n = ce.kernels.fmha_prefill_cutlass_lse_elements(B, S, NH)
workspace = torch.empty(max(1, ws_bytes), device="cuda", dtype=torch.uint8)
lse = torch.empty(lse_n, device="cuda", dtype=torch.float32)

ce.kernels.fmha_prefill_cutlass_causal(
    Q.data_ptr(), K.data_ptr(), V.data_ptr(), O_causal.data_ptr(),
    B, S, NH, HD, scale, workspace.data_ptr(), lse.data_ptr())

eff = torch.zeros(B, S, dtype=torch.int32, device="cuda")  # value irrelevant: hardcoded
ce.kernels.fmha_prefill_cutlass_block_aware(
    Q.data_ptr(), K.data_ptr(), V.data_ptr(), O_block.data_ptr(),
    eff.data_ptr(), S, B, S, NH, HD, scale,
    workspace.data_ptr(), lse.data_ptr())

diff = (O_causal.float() - O_block.float()).abs()
print(f"|O_causal - O_block_hardcoded_lim1|: max={diff.max().item():.4e}, mean={diff.mean().item():.4e}")
if diff.max().item() < 1e-3:
    print("FAIL: outputs look identical → apply_mask was NOT called for block_aware path")
else:
    print("OK: apply_mask is being executed (outputs differ as expected)")
