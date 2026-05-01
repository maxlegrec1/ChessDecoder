"""Per-kernel unit tests against PyTorch references.

Each test allocates torch tensors on CUDA, hands their data_ptr() to the
custom CUTLASS-engine kernel via the kernels submodule, runs a torch
reference computation, and compares.
"""

import math
import sys

import pytest
import torch

# Make the local engine importable (editable-install puts the .so in-place).
sys.path.insert(0, "/workspace/ChessDecoder/chessdecoder/cpp/cutlass_engine/src")

import _cutlass_decoder_cpp as ce  # noqa: E402


def _ptr(t: torch.Tensor) -> int:
    assert t.is_cuda
    return t.data_ptr()


def test_module_imports():
    assert hasattr(ce, "ThinkingEngine")
    assert hasattr(ce, "RolloutResult")
    assert hasattr(ce, "kernels")


# ---------- RMSNorm ----------


def _ref_rmsnorm(x: torch.Tensor, w: torch.Tensor, eps: float) -> torch.Tensor:
    # x [M, E], w [E]. FP32 reduce, FP16 out.
    x32 = x.float()
    rms = torch.rsqrt(x32.pow(2).mean(-1, keepdim=True) + eps)
    return (x32 * rms * w.float()).half()


@pytest.mark.parametrize("M, E", [(1, 1024), (64, 1024), (64 * 71, 1024), (8, 256)])
def test_rmsnorm_fp16(M, E):
    torch.manual_seed(0)
    x = torch.randn(M, E, dtype=torch.float16, device="cuda")
    w = torch.randn(E, dtype=torch.float16, device="cuda") * 0.5 + 1.0
    y = torch.empty_like(x)
    ce.kernels.rmsnorm_fp16(_ptr(x), _ptr(w), _ptr(y), M, E, 1e-6)
    ref = _ref_rmsnorm(x, w, 1e-6)
    err = (y.float() - ref.float()).abs().max().item()
    assert err < 2e-3, f"rmsnorm err {err} too high (M={M}, E={E})"


def test_rmsnorm_residual_fp16():
    M, E = 64, 1024
    torch.manual_seed(1)
    x = torch.randn(M, E, dtype=torch.float16, device="cuda") * 0.3
    r = torch.randn(M, E, dtype=torch.float16, device="cuda") * 0.3
    w = torch.randn(E, dtype=torch.float16, device="cuda") * 0.5 + 1.0
    y = torch.empty_like(x)
    out_r = torch.empty_like(x)
    ce.kernels.rmsnorm_residual_fp16(_ptr(x), _ptr(r), _ptr(w),
                                     _ptr(y), _ptr(out_r), M, E, 1e-6)
    ref_sum = (x.float() + r.float()).half()
    ref_y = _ref_rmsnorm(ref_sum, w, 1e-6)
    err_r = (out_r.float() - ref_sum.float()).abs().max().item()
    err_y = (y.float() - ref_y.float()).abs().max().item()
    assert err_r < 1e-3, f"residual sum err {err_r}"
    # Fused residual+normalize accumulates one extra fp16 round-trip vs the
    # split impl, so we allow slightly higher max-abs (still ~2 ULPs at this dr).
    assert err_y < 5e-3, f"normed err {err_y}"


# ---------- RoPE ----------


def _build_rope_table(max_seq: int, head_dim: int, base: float = 10000.0):
    half = head_dim // 2
    cos_t = torch.zeros(max_seq, half, dtype=torch.float32, device="cuda")
    sin_t = torch.zeros(max_seq, half, dtype=torch.float32, device="cuda")
    for j in range(half):
        theta = base ** (-(2.0 * j) / head_dim)
        for p in range(max_seq):
            cos_t[p, j] = math.cos(p * theta)
            sin_t[p, j] = math.sin(p * theta)
    return cos_t, sin_t


def _ref_rope(x: torch.Tensor, pos: torch.Tensor, cos_t, sin_t):
    """Apply interleaved RoPE to [M, NH, HD]."""
    M, NH, HD = x.shape
    out = x.clone().float()
    half = HD // 2
    for m in range(M):
        p = int(pos[m].item())
        c = cos_t[p]
        s = sin_t[p]
        for h in range(NH):
            for j in range(half):
                a = out[m, h, 2 * j].item()
                b = out[m, h, 2 * j + 1].item()
                out[m, h, 2 * j]     = a * c[j] - b * s[j]
                out[m, h, 2 * j + 1] = a * s[j] + b * c[j]
    return out.half()


def test_rope_fp16():
    M, NH, HD = 4, 2, 64
    rope_max = 32
    cos_t, sin_t = _build_rope_table(rope_max, HD)
    Q = torch.randn(M, NH, HD, dtype=torch.float16, device="cuda")
    K = torch.randn(M, NH, HD, dtype=torch.float16, device="cuda")
    pos = torch.tensor([0, 5, 17, 31], dtype=torch.int32, device="cuda")
    Q_ref = _ref_rope(Q.clone(), pos, cos_t, sin_t)
    K_ref = _ref_rope(K.clone(), pos, cos_t, sin_t)
    ce.kernels.rope_apply_qk_fp16(_ptr(Q), _ptr(K), _ptr(pos),
                                  _ptr(cos_t), _ptr(sin_t),
                                  M, NH, HD, rope_max)
    err_q = (Q.float() - Q_ref.float()).abs().max().item()
    err_k = (K.float() - K_ref.float()).abs().max().item()
    assert err_q < 5e-3, f"RoPE Q err {err_q}"
    assert err_k < 5e-3, f"RoPE K err {err_k}"


# ---------- SwiGLU ----------


def test_swiglu_fp16():
    M, d_ff = 8, 256
    torch.manual_seed(2)
    gate = torch.randn(M, d_ff, dtype=torch.float16, device="cuda") * 0.5
    up   = torch.randn(M, d_ff, dtype=torch.float16, device="cuda") * 0.5
    gate_up = torch.cat([gate, up], dim=-1).contiguous()
    y = torch.empty(M, d_ff, dtype=torch.float16, device="cuda")
    ce.kernels.swiglu_fp16(_ptr(gate_up), _ptr(y), M, d_ff)
    ref = (torch.nn.functional.silu(gate.float()) * up.float()).half()
    err = (y.float() - ref.float()).abs().max().item()
    assert err < 5e-3, f"swiglu err {err}"


def test_mish_inplace_fp16():
    N = 4096
    torch.manual_seed(3)
    x = torch.randn(N, dtype=torch.float16, device="cuda")
    ref = (x.float() * torch.tanh(torch.nn.functional.softplus(x.float()))).half()
    ce.kernels.mish_inplace_fp16(_ptr(x), N)
    err = (x.float() - ref.float()).abs().max().item()
    assert err < 5e-3, f"mish err {err}"


# ---------- Argmax sampler ----------


def test_argmax_fp16():
    B, V = 32, 1924
    torch.manual_seed(4)
    logits = torch.randn(B, V, dtype=torch.float16, device="cuda")
    out = torch.empty(B, dtype=torch.int32, device="cuda")
    ce.kernels.argmax_fp16(_ptr(logits), _ptr(out), B, V)
    ref = logits.argmax(dim=-1).int()
    assert torch.equal(out, ref), f"argmax mismatch: {(out != ref).sum().item()} rows"


# ---------- GEMM (FP16) ----------


def test_gemm_fp16():
    M, N, K = 64, 4096, 1024
    torch.manual_seed(5)
    A = torch.randn(M, K, dtype=torch.float16, device="cuda") * 0.1
    Bw = torch.randn(N, K, dtype=torch.float16, device="cuda") * 0.1   # [out, in]
    C = torch.empty(M, N, dtype=torch.float16, device="cuda")
    ce.kernels.gemm_fp16(_ptr(A), _ptr(Bw), _ptr(C), M, N, K)
    ref = (A.float() @ Bw.float().t()).half()
    # FP16 GEMM with FP32 acc against naive FP32 matmul: tolerate fp16 cast noise.
    err = (C.float() - ref.float()).abs().max().item()
    assert err < 5e-2, f"gemm err {err} (rel to magnitudes ~{ref.abs().mean().item()})"


# ---------- FMHA decode (against PyTorch SDPA) ----------


def test_fmha_decode_against_sdpa():
    B, NH, HD = 2, 4, 64
    # Decode contract: cache has `past_len + 1` valid entries
    # (kv_scatter wrote the new one at index past_len; fmha reads up to past_len+1).
    past_len_val = 15  # 15 prior entries + 1 just-written = 16 valid total
    valid_count = past_len_val + 1
    max_seq = 64
    layer_idx = 0
    NL = 1

    torch.manual_seed(6)
    Q = torch.randn(B, NH, HD, dtype=torch.float16, device="cuda") * 0.5
    # Cache: [NL, B, NH, max_seq, HD]
    Kc = torch.randn(NL, B, NH, max_seq, HD, dtype=torch.float16, device="cuda") * 0.5
    Vc = torch.randn(NL, B, NH, max_seq, HD, dtype=torch.float16, device="cuda") * 0.5
    past_len = torch.tensor([past_len_val] * B, dtype=torch.int32, device="cuda")
    active = torch.ones(B, dtype=torch.int32, device="cuda")
    O = torch.zeros(B, NH, HD, dtype=torch.float16, device="cuda")

    scale = 1.0 / math.sqrt(HD)
    ce.kernels.fmha_decode_dispatch(
        _ptr(Q), _ptr(Kc), _ptr(Vc), _ptr(past_len), _ptr(active), _ptr(O),
        B, NH, HD, max_seq, layer_idx, scale)

    # Reference: torch SDPA with the truncated cache.
    K_used = Kc[layer_idx, :, :, :valid_count, :].float()  # [B, NH, P, HD]
    V_used = Vc[layer_idx, :, :, :valid_count, :].float()
    Q_f = Q.float().unsqueeze(2)  # [B, NH, 1, HD]
    scores = (Q_f @ K_used.transpose(-1, -2)) * scale  # [B, NH, 1, P]
    probs = torch.softmax(scores, dim=-1)
    ref = (probs @ V_used).squeeze(2).half()  # [B, NH, HD]

    err = (O.float() - ref.float()).abs().max().item()
    assert err < 2e-2, f"fmha_decode err {err}"
