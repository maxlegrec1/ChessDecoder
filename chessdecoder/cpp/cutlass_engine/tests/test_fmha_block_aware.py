"""Parity test for CUTLASS FMHA block-aware causal mask (J.3).

Mask semantics (matching the model's prefix-mode attention):
   valid[b, q, k] = (block_id[b, q] == block_id[b, k]) || (k <= q)

Test inputs use synthetic block_ids partitioning each batch's S tokens into
roughly 4 blocks. We compare against a torch reference that builds the
explicit mask and runs SDPA with attn_mask.

Pass: max abs diff < 5e-3 (FP16 noise floor for full-mask attention).
"""

import math

import torch
import _cutlass_decoder_cpp as ce


def _make_block_ids(B: int, S: int, n_blocks: int = 4):
    block_size = S // n_blocks
    block_id = torch.zeros(B, S, dtype=torch.int32, device="cuda")
    for i in range(n_blocks):
        s = i * block_size
        e = (i + 1) * block_size if i < n_blocks - 1 else S
        block_id[:, s:e] = i
    return block_id


def _compute_effective_limit(block_id: torch.Tensor) -> torch.Tensor:
    """For each (b, q): effective_limit[b, q] = max(q, end_of_block(b, q))
    where end_of_block(b, q) = max{i : block_id[b, i] == block_id[b, q]}."""
    B, S = block_id.shape
    eff = torch.zeros(B, S, dtype=torch.int32, device=block_id.device)
    for b in range(B):
        for q in range(S):
            bq = int(block_id[b, q].item())
            # end of this block (largest k with block_id[b, k] == bq)
            same = (block_id[b] == bq).nonzero(as_tuple=False).flatten()
            end = int(same.max().item())
            eff[b, q] = max(q, end)
    return eff


def _run_case(B: int, S: int, NH: int, HD: int = 64):
    scale = 1.0 / math.sqrt(HD)

    torch.manual_seed(0)
    Q = torch.randn(B, S, NH, HD, device="cuda", dtype=torch.float16) * 0.1
    K = torch.randn(B, S, NH, HD, device="cuda", dtype=torch.float16) * 0.1
    V = torch.randn(B, S, NH, HD, device="cuda", dtype=torch.float16) * 0.1
    O = torch.zeros_like(Q)

    block_id = _make_block_ids(B, S, n_blocks=4)
    eff = _compute_effective_limit(block_id).contiguous()

    ws_bytes = ce.kernels.fmha_prefill_cutlass_workspace_bytes(B, S, NH, HD)
    lse_n = ce.kernels.fmha_prefill_cutlass_lse_elements(B, S, NH)
    workspace = torch.empty(max(1, ws_bytes), device="cuda", dtype=torch.uint8)
    lse = torch.empty(lse_n, device="cuda", dtype=torch.float32)

    ce.kernels.fmha_prefill_cutlass_block_aware(
        Q.data_ptr(), K.data_ptr(), V.data_ptr(), O.data_ptr(),
        eff.data_ptr(), S, B, S, NH, HD, scale,
        workspace.data_ptr(), lse.data_ptr())

    # torch reference: build the explicit mask
    # mask[b, q, k] = (block_id[b,q] == block_id[b,k]) || (k <= q)
    same_block = (block_id.unsqueeze(2) == block_id.unsqueeze(1))  # [B, S, S]
    causal = (torch.arange(S, device="cuda").unsqueeze(0) <=
              torch.arange(S, device="cuda").unsqueeze(1)).unsqueeze(0)  # [1, S, S]
    valid = same_block | causal  # [B, S, S]
    attn_mask = torch.where(valid, 0.0, float("-inf")).to(torch.float16)
    # SDPA expects mask of shape broadcastable to [B, H, S, S]
    attn_mask_bh = attn_mask.unsqueeze(1)  # [B, 1, S, S]

    Q_ref = Q.permute(0, 2, 1, 3).contiguous()
    K_ref = K.permute(0, 2, 1, 3).contiguous()
    V_ref = V.permute(0, 2, 1, 3).contiguous()
    O_ref = torch.nn.functional.scaled_dot_product_attention(
        Q_ref, K_ref, V_ref, attn_mask=attn_mask_bh, scale=scale)
    O_ref = O_ref.permute(0, 2, 1, 3).contiguous()

    diff = (O.float() - O_ref.float()).abs()
    max_diff = diff.max().item()
    mean_diff = diff.mean().item()
    print(f"  B={B} S={S} NH={NH}: max={max_diff:.4e} mean={mean_diff:.4e}")

    assert max_diff < 5e-3, f"max diff {max_diff:.4e} exceeds tolerance"
    return max_diff, mean_diff


def main():
    print("=== CUTLASS FMHA block-aware parity test ===")
    for B, S, NH in [
        (1, 256, 16),
        (4, 256, 16),
        (8, 512, 16),
        (16, 1024, 16),
        (4, 2048, 16),
    ]:
        _run_case(B, S, NH)
    print("All cases passed.")


if __name__ == "__main__":
    main()
