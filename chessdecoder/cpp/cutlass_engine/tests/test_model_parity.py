"""End-to-end model-parity test: compare cutlass_engine forward_decode
against the Python ChessDecoder forward at S=1.

Builds a small ChessDecoder (configurable size), exports its weights via
the cutlass-engine exporter, instantiates ThinkingEngine, runs one decode
step over a batch of slots at varying past_len positions, and compares the
hidden state output.
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import pytest
import torch

sys.path.insert(0, "/workspace/ChessDecoder/chessdecoder/cpp/cutlass_engine/src")
sys.path.insert(0, str(Path(__file__).resolve().parent))

from chessdecoder.models.model import ChessDecoder  # noqa: E402

import _cutlass_decoder_cpp as ce  # noqa: E402
from export_for_cutlass import export_for_cutlass  # noqa: E402


def _ptr(t: torch.Tensor) -> int:
    return t.data_ptr()


# A small but realistic config for a fast end-to-end test.
SMALL_CFG = {
    "model": {
        "embed_dim": 128,
        "num_heads": 4,
        "num_layers": 2,
        "max_seq_len": 256,
        "d_ff": 512,
        "n_buckets": 100,
        "value_hidden_size": 128,
        "num_fourier_freq": 64,
        "wl_sigma": 0.4,
    }
}


def _build_small_model(seed: int = 0):
    torch.manual_seed(seed)
    mc = SMALL_CFG["model"]
    # vocab_size is determined by the vocab module; ChessDecoder takes it.
    from chessdecoder.models.vocab import vocab_size
    m = ChessDecoder(
        vocab_size=vocab_size,
        embed_dim=mc["embed_dim"],
        num_heads=mc["num_heads"],
        num_layers=mc["num_layers"],
        max_seq_len=mc["max_seq_len"],
        d_ff=mc["d_ff"],
        n_buckets=mc["n_buckets"],
        value_hidden_size=mc["value_hidden_size"],
        num_fourier_freq=mc["num_fourier_freq"],
        wl_sigma=mc["wl_sigma"],
    ).cuda().half()
    m.eval()
    return m


def _run_python_decode(m, ids: torch.Tensor, pos: torch.Tensor) -> torch.Tensor:
    """Reference: run the eager Python model over a single-token decode.

    For a fair comparison with the cutlass engine, we must include the new
    token's K in the attention. The simplest way is to feed the model the
    full prefix [0..pos] of cached tokens plus the new one — but that
    requires reconstructing the cache. Easier: feed a length-(pos+1) sequence
    where positions 0..pos-1 are arbitrary "filler" — *but* this only works
    if the engine's cache also contains those exact filler keys/values.

    Approach: we simulate a decode by running the FULL prefix in a single
    causal forward at sequence length pos+1, then taking the last position's
    hidden state. The cutlass engine, when seeded with the SAME prefix's K/V
    in the cache, should produce the same last-position hidden.

    Caller is responsible for first running prefill to populate the cache.

    For the simplest test, we run decode at past_len=0 with one fresh token:
    the model output at S=1 with no past should equal the cutlass engine
    output at past_len=0.
    """
    # input_pos is what RoPE uses; pass it explicitly to ensure consistency.
    with torch.no_grad():
        h = m(ids.unsqueeze(0).long() if ids.dim() == 1 else ids,  # [B, S]
              input_pos=pos.unsqueeze(0).long() if pos.dim() == 1 else pos,
              mask_type="causal")
    return h


def test_model_forward_decode_at_pos_zero():
    """Simplest case: past_len=0 across all slots. The cutlass engine reads
    [0, 1) of the cache (the just-written token) — equivalent to the Python
    model running with seq_len=1 and no prior context."""
    m = _build_small_model(seed=0)
    mc = SMALL_CFG["model"]
    E = mc["embed_dim"]
    B = 2

    with tempfile.TemporaryDirectory(prefix="cutlass_parity_") as td:
        export_dir = Path(td)
        export_for_cutlass(m, SMALL_CFG, export_dir)

        engine = ce.ThinkingEngine(
            backbone_pt="",  # unused
            weights_dir=str(export_dir / "weights"),
            vocab_json=str(export_dir / "vocab.json"),
            config_json=str(export_dir / "config.json"),
            batch_size=B,
        )

        ids = torch.tensor([42, 137], dtype=torch.int32, device="cuda")
        pos = torch.tensor([0, 0], dtype=torch.int32, device="cuda")
        active = torch.ones(B, dtype=torch.int32, device="cuda")
        past_len = torch.zeros(B, dtype=torch.int32, device="cuda")
        out_h = torch.zeros(B, E, dtype=torch.float16, device="cuda")

        engine.forward_decode_test(_ptr(ids), _ptr(pos), _ptr(active),
                                   _ptr(past_len), _ptr(out_h))

        # Reference: Python model at S=1.
        ref_h = _run_python_decode(m, ids.long(), pos.long())  # [1, B, S=1, E]? actually [B, S, E]
        # The Python forward returns [B, S, E]. ids is [B], we expect [B, 1, E].
        # Actually we passed ids.unsqueeze(0).long() if 1d — yields [1, B] which is wrong.
        # Let me just call directly here:
        with torch.no_grad():
            ids_2d = ids.long().unsqueeze(1)  # [B, 1]
            pos_2d = pos.long().unsqueeze(1)  # [B, 1]
            ref = m(ids_2d, input_pos=pos_2d, mask_type="causal")  # [B, 1, E]
        ref = ref.squeeze(1)  # [B, E]

        err = (out_h.float() - ref.float()).abs().max().item()
        rel = err / max(ref.float().abs().mean().item(), 1e-6)
        print(f"max-abs err: {err:.6f}  rel: {rel:.4f}")
        assert err < 5e-2, f"forward_decode err {err} too high (rel {rel})"
        assert past_len.cpu().tolist() == [1, 1]  # incremented for active slots


def test_model_forward_decode_with_kv_history():
    """Decode at past_len > 0: feed a sequence one token at a time, then
    compare the final hidden state against the Python model run on the full
    prefix (causal forward at S=N)."""
    m = _build_small_model(seed=1)
    mc = SMALL_CFG["model"]
    E = mc["embed_dim"]
    B = 2
    N = 5  # number of tokens to feed sequentially

    with tempfile.TemporaryDirectory(prefix="cutlass_parity_") as td:
        export_dir = Path(td)
        export_for_cutlass(m, SMALL_CFG, export_dir)

        engine = ce.ThinkingEngine(
            backbone_pt="", weights_dir=str(export_dir / "weights"),
            vocab_json=str(export_dir / "vocab.json"),
            config_json=str(export_dir / "config.json"),
            batch_size=B,
        )

        torch.manual_seed(7)
        # Generate N tokens for each slot.
        ids_seq = torch.randint(0, 1900, (B, N), dtype=torch.int32, device="cuda")

        # Drive the engine: feed one token at a time, advance past_len.
        active = torch.ones(B, dtype=torch.int32, device="cuda")
        past_len = torch.zeros(B, dtype=torch.int32, device="cuda")
        out_h = torch.zeros(B, E, dtype=torch.float16, device="cuda")
        last_eng = None
        for t in range(N):
            ids_t = ids_seq[:, t].contiguous()
            pos_t = torch.tensor([t] * B, dtype=torch.int32, device="cuda")
            engine.forward_decode_test(ids_t.data_ptr(), pos_t.data_ptr(),
                                       active.data_ptr(), past_len.data_ptr(),
                                       out_h.data_ptr())
            last_eng = out_h.clone()

        # Reference: python model run on the whole prefix [B, N] in one shot.
        with torch.no_grad():
            pos_full = torch.arange(N, device="cuda").long().unsqueeze(0).expand(B, -1)
            ref = m(ids_seq.long(), input_pos=pos_full, mask_type="causal")  # [B, N, E]
        ref_last = ref[:, -1, :].float()

        err = (last_eng.float() - ref_last).abs().max().item()
        rel = err / max(ref_last.abs().mean().item(), 1e-6)
        print(f"[kv-history N={N}] max-abs err: {err:.6f}  rel: {rel:.4f}")
        # Tolerance is wider here because errors compound across N steps.
        assert err < 1e-1, f"forward_decode KV err {err} too high (rel {rel})"
        # past_len should have advanced by N for active slots.
        assert past_len.cpu().tolist() == [N] * B
