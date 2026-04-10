"""Verify C++ batched engine log-probs match PyTorch log-probs.

Runs ~1000 FENs through the C++ batched inference engine (which now returns
per-move log-probs) and through the PyTorch model's prefix forward pass.
Compares the log-probs position-by-position.

CPU-resident design: all rollout tensors live on CPU RAM after generation.
The PyTorch forward pass runs on small mini-batches — each mini-batch is
moved to GPU, the forward runs, per-move log-probs are gathered and copied
back to CPU, and the mini-batch GPU tensors are dropped before the next one.

The C++ engine runs in its own subprocess (via generate_rollouts), so GPU
memory is fully released before the PyTorch model is loaded.

Usage:
    uv run python scripts/verify_cpp_log_probs.py \\
        --num-fens 1000 --mini-batch 4 \\
        --export-dir exports/export_282k \\
        --ckpt checkpoints/finetune-thinking-v1_20260320_205453/checkpoint_step_282000.pt \\
        --config chessdecoder/rl/config.yaml
"""

import argparse
import time

import torch

from chessdecoder.dataloader.sampling import load_pretrain_positions
from chessdecoder.models.model import ChessDecoder
from chessdecoder.models.vocab import vocab_size
from chessdecoder.rl.config import GRPOConfig
from chessdecoder.rl.log_probs import compute_current_log_probs
from chessdecoder.rl.rollout import generate_rollouts
from chessdecoder.rl.sequence import parse_rollout


def _batch_to_device(batch: dict, device: torch.device) -> dict:
    return {k: v.to(device) if isinstance(v, torch.Tensor) else v
            for k, v in batch.items()}


def _stack_parsed(parsed_list: list[dict]) -> dict:
    """CPU-side stack of parsed rollouts (no device transfer)."""
    keys = ["input_ids", "block_id", "wl_positions", "d_positions",
            "wl_values", "d_values", "thinking_move_mask", "final_move_mask",
            "move_token_ids", "old_log_probs"]
    out = {}
    for k in keys:
        out[k] = torch.stack([p[k] for p in parsed_list])
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--num-fens", type=int, default=1000)
    parser.add_argument("--mini-batch", type=int, default=4)
    parser.add_argument("--export-dir", default="exports/export_282k")
    parser.add_argument("--ckpt", default="checkpoints/finetune-thinking-v1_20260320_205453/checkpoint_step_282000.pt")
    parser.add_argument("--config", default="chessdecoder/rl/config.yaml")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--inference-batch-size", type=int, default=32)
    parser.add_argument("--max-abs-tol", type=float, default=5e-2)
    parser.add_argument("--mean-abs-tol", type=float, default=5e-3)
    args = parser.parse_args()

    config = GRPOConfig.from_yaml(args.config)

    # ── 1. Load FENs (CPU) ────────────────────────────────────────────
    print(f"Loading {args.num_fens} FENs from {config.pretrain_parquet_dir} ...")
    positions = load_pretrain_positions(
        config.pretrain_parquet_dir, args.num_fens, seed=args.seed,
    )
    fens = [p["fen"] for p in positions][: args.num_fens]
    print(f"  Got {len(fens)} FENs")

    # ── 2. Run C++ rollouts in subprocess (GPU released when done) ────
    rollout_cfg = GRPOConfig(
        group_size=1,
        inference_batch_size=args.inference_batch_size,
        think_temperature=config.think_temperature,
        policy_temperature=config.policy_temperature,
        board_temperature=config.board_temperature,
    )
    print(f"Running C++ rollouts (inference_batch_size={args.inference_batch_size}) ...")
    t0 = time.time()
    grouped = generate_rollouts(args.export_dir, fens, rollout_cfg)
    t_rollout = time.time() - t0
    rollouts = [g[0] for g in grouped]  # flatten [B][1] → [B]
    total_tok = sum(r.num_tokens for r in rollouts)
    print(f"  Rollouts: {t_rollout:.1f}s, {total_tok} tok ({total_tok/t_rollout:.0f} tok/s)")

    n_moves_cpp = sum(len(r.move_log_probs) for r in rollouts)
    print(f"  Total move log-probs from C++: {n_moves_cpp}")
    if n_moves_cpp == 0:
        raise RuntimeError("C++ engine returned zero move log-probs — "
                           "check that the extension was rebuilt.")

    # ── 3. Parse rollouts into CPU tensors ─────────────────────────────
    max_seq_len = config.model["max_seq_len"]
    print(f"Parsing {len(rollouts)} rollouts (max_seq_len={max_seq_len}) ...")
    parsed = []
    for r in rollouts:
        parsed.append(parse_rollout(
            r.token_ids, r.wl_entries, r.d_entries, max_seq_len,
            move_log_probs=r.move_log_probs,
        ))
    all_batch_cpu = _stack_parsed(parsed)
    N = len(parsed)

    # Mask of move positions (CPU)
    move_mask_cpu = all_batch_cpu["thinking_move_mask"] | all_batch_cpu["final_move_mask"]
    cpp_lp_all = all_batch_cpu["old_log_probs"][move_mask_cpu]  # [N_moves]
    n_moves = move_mask_cpu.sum().item()
    print(f"  Parsed moves: {n_moves} (cpp={n_moves_cpp})")

    # ── 4. Load PyTorch model ──────────────────────────────────────────
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading PyTorch model from {args.ckpt} onto {device} ...")
    mc = config.model
    model = ChessDecoder(
        vocab_size=vocab_size,
        embed_dim=mc["embed_dim"],
        num_heads=mc["num_heads"],
        num_layers=mc["num_layers"],
        max_seq_len=mc["max_seq_len"],
        d_ff=mc.get("d_ff"),
        n_buckets=mc.get("n_buckets", 100),
        value_hidden_size=mc.get("value_hidden_size", 256),
        num_fourier_freq=mc.get("num_fourier_freq", 128),
        wl_sigma=mc.get("wl_sigma", 0.4),
    )
    state = torch.load(args.ckpt, map_location="cpu", weights_only=True)
    if "model_state_dict" in state:
        state = state["model_state_dict"]
    model.load_state_dict(state)
    model.to(device).eval()

    # ── 5. Mini-batched PyTorch forward (GPU peak stays flat) ──────────
    print(f"Running PyTorch forward in mini-batches of {args.mini_batch} ...")
    torch_lp_chunks = []
    t1 = time.time()
    with torch.no_grad():
        for start in range(0, N, args.mini_batch):
            end = min(start + args.mini_batch, N)
            mb_cpu = {k: v[start:end] for k, v in all_batch_cpu.items()}
            mb = _batch_to_device(mb_cpu, device)
            lp, _ = compute_current_log_probs(model, mb, use_amp=True)  # [mb, S]
            mask = mb["thinking_move_mask"] | mb["final_move_mask"]
            torch_lp_chunks.append(lp[mask].float().cpu())
            del mb, lp, mask
            if device.type == "cuda":
                torch.cuda.empty_cache()
            if (start // args.mini_batch) % 25 == 0:
                print(f"  {end}/{N}")
    t_fwd = time.time() - t1
    torch_lp_all = torch.cat(torch_lp_chunks)
    print(f"  PyTorch forward: {t_fwd:.1f}s")

    # ── 6. Compare ─────────────────────────────────────────────────────
    assert torch_lp_all.shape == cpp_lp_all.shape, \
        f"shape mismatch cpp={cpp_lp_all.shape} torch={torch_lp_all.shape}"

    diff = (torch_lp_all - cpp_lp_all).abs()
    max_err = diff.max().item()
    mean_err = diff.mean().item()
    median_err = diff.median().item()
    p99_err = diff.quantile(0.99).item()
    n_bad = (diff > 1e-2).sum().item()

    # Pearson correlation
    t_f = torch_lp_all - torch_lp_all.mean()
    c_f = cpp_lp_all - cpp_lp_all.mean()
    corr = (t_f * c_f).sum() / (t_f.norm() * c_f.norm() + 1e-12)

    print()
    print("=" * 60)
    print("Log-prob comparison: C++ batched engine vs PyTorch")
    print("=" * 60)
    print(f"  N moves               : {n_moves}")
    print(f"  max |Δ|               : {max_err:.6f}")
    print(f"  mean |Δ|              : {mean_err:.6f}")
    print(f"  median |Δ|            : {median_err:.6f}")
    print(f"  P99 |Δ|               : {p99_err:.6f}")
    print(f"  Pearson r             : {corr.item():.6f}")
    print(f"  moves with |Δ| > 1e-2 : {n_bad} ({100 * n_bad / n_moves:.2f}%)")
    print()

    # Show worst 5 mismatches
    worst = torch.topk(diff, k=min(5, n_moves))
    print("Worst mismatches:")
    for rank, (err, idx) in enumerate(zip(worst.values, worst.indices)):
        print(f"  #{rank}: |Δ|={err.item():.6f}  "
              f"cpp={cpp_lp_all[idx].item():.6f}  torch={torch_lp_all[idx].item():.6f}")
    print()

    pass_max = max_err < args.max_abs_tol
    pass_mean = mean_err < args.mean_abs_tol
    if pass_max and pass_mean:
        print(f"PASS: max |Δ|={max_err:.2e} < {args.max_abs_tol:.0e}, "
              f"mean |Δ|={mean_err:.2e} < {args.mean_abs_tol:.0e}")
        return 0
    else:
        print(f"FAIL: max |Δ|={max_err:.2e} (tol {args.max_abs_tol:.0e}), "
              f"mean |Δ|={mean_err:.2e} (tol {args.mean_abs_tol:.0e})")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
