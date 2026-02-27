"""Test CUDA graph correctness by comparing graph vs non-graph outputs."""
import torch

def main():
    model_path = "export/backbone.pt"

    print("Loading model...")
    model = torch.jit.load(model_path, map_location="cuda")
    model.eval()

    NL = 12
    NH = 16
    HD = 64
    E = 1024
    MAX_LEN = 512

    opts_int = dict(dtype=torch.int64, device="cuda")
    opts_fp16 = dict(dtype=torch.float16, device="cuda")
    opts_fp32 = dict(dtype=torch.float32, device="cuda")
    opts_bool = dict(dtype=torch.bool, device="cuda")

    # === Step 1: Prefill 69 tokens ===
    print("\nStep 1: Prefill 69 tokens...")
    seq_len = 69
    ids = torch.randint(0, 1968, (1, seq_len), **opts_int)
    pos = torch.arange(seq_len, **opts_int).unsqueeze(0)
    mask = torch.full((1, 1, seq_len, seq_len), -1e9, **opts_fp32)
    for i in range(seq_len):
        mask[0, 0, i, :i+1] = 0.0
    past_k = torch.zeros(NL, 1, NH, 0, HD, **opts_fp16)
    past_v = torch.zeros(NL, 1, NH, 0, HD, **opts_fp16)
    ov = torch.zeros(1, seq_len, **opts_fp16)
    om = torch.zeros(1, seq_len, **opts_bool)

    with torch.no_grad():
        h, pk, pv = model(ids, pos, mask, past_k, past_v, ov, om)

    prefill_k = pk
    prefill_v = pv

    # === Step 2: Setup CUDA graph ===
    g_ids = torch.zeros(1, 1, **opts_int)
    g_pos = torch.zeros(1, 1, **opts_int)
    g_mask = torch.full((1, 1, 1, MAX_LEN + 1), -1e9, **opts_fp32)
    g_ov = torch.zeros(1, 1, **opts_fp16)
    g_om = torch.zeros(1, 1, **opts_bool)
    g_buf_k = torch.zeros(NL, 1, NH, MAX_LEN, HD, **opts_fp16)
    g_buf_v = torch.zeros(NL, 1, NH, MAX_LEN, HD, **opts_fp16)

    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        for _ in range(3):
            model(g_ids, g_pos, g_mask, g_buf_k, g_buf_v, g_ov, g_om)
        stream.synchronize()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.stream(stream):
        graph.capture_begin()
        g_out_h, g_out_pk, g_out_pv = model(g_ids, g_pos, g_mask, g_buf_k, g_buf_v, g_ov, g_om)
        graph.capture_end()
    stream.synchronize()

    # === Step 3: Run 200 steps comparing graph vs non-graph ===
    print("\nStep 3: 200 sequential replays...")

    # Graph path state
    g_buf_k[:, :, :, :69, :] = prefill_k
    g_buf_v[:, :, :, :69, :] = prefill_v
    g_mask.fill_(-1e9)
    g_mask[0, 0, 0, :69] = 0.0
    g_mask[0, 0, 0, MAX_LEN] = 0.0  # new token always at MAX_LEN after cat

    # Non-graph path state (dynamic-sized KV cache)
    ng_past_k = prefill_k.clone()
    ng_past_v = prefill_v.clone()

    g_cache_len = 69
    max_diffs = []

    for step in range(200):
        tok_id = torch.randint(0, 1968, (1,)).item()
        position = 69 + step

        # Non-graph reference
        ref_mask = torch.zeros(1, 1, 1, ng_past_k.size(3) + 1, **opts_fp32)
        ref_ids = torch.tensor([[tok_id]], **opts_int)
        ref_pos = torch.tensor([[position]], **opts_int)
        ref_ov = torch.zeros(1, 1, **opts_fp16)
        ref_om = torch.zeros(1, 1, **opts_bool)

        with torch.no_grad():
            h_ref, pk_ref, pv_ref = model(ref_ids, ref_pos, ref_mask, ng_past_k, ng_past_v, ref_ov, ref_om)
        ng_past_k = pk_ref
        ng_past_v = pv_ref

        # Graph path
        g_ids[0, 0] = tok_id
        g_pos[0, 0] = position

        graph.replay()
        torch.cuda.synchronize()

        # Copy new KV from position MAX_LEN to cache_len
        g_buf_k[:, :, :, g_cache_len, :] = g_out_pk[:, :, :, MAX_LEN, :]
        g_buf_v[:, :, :, g_cache_len, :] = g_out_pv[:, :, :, MAX_LEN, :]
        g_mask[0, 0, 0, g_cache_len] = 0.0
        g_cache_len += 1

        diff = (g_out_h[0, 0].float().cpu() - h_ref[0, 0].float().cpu()).abs()
        max_diffs.append(diff.max().item())

    # Stats
    import statistics
    print(f"  Max diff range: [{min(max_diffs):.6e}, {max(max_diffs):.6e}]")
    print(f"  Mean max diff: {statistics.mean(max_diffs):.6e}")
    print(f"  Median max diff: {statistics.median(max_diffs):.6e}")
    print(f"  Steps with diff > 0.01: {sum(1 for d in max_diffs if d > 0.01)}/{len(max_diffs)}")
    print(f"  Steps with diff > 0.05: {sum(1 for d in max_diffs if d > 0.05)}/{len(max_diffs)}")
    print(f"  Steps with exact match (0.0): {sum(1 for d in max_diffs if d == 0.0)}/{len(max_diffs)}")

    # Print first 10 and every 20th
    for i in range(min(10, len(max_diffs))):
        print(f"  Step {i}: {max_diffs[i]:.6e}")
    for i in range(20, len(max_diffs), 20):
        print(f"  Step {i}: {max_diffs[i]:.6e}")

if __name__ == "__main__":
    main()
