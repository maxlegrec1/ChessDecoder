"""Generate the Q_ref reward table: deep-search Q for sampled corpus roots.

Roots come from corpus shards [:-1] (the last shard stays held out for the
eval suite), sampled with probability biased by |root_q - orig_q| (search-
sensitive positions; quiet roots give zero-variance GRPO groups).

Output parquet shards in agent_data/qref/. Runs forever as a systemd unit;
ctrl-c / stop safe (whole chunks only).

  uv run python scripts/gen_qref.py [--sims 800] [--batch 384] [--chunk 4096]
"""
import argparse
import glob
import os

import chess
import numpy as np
import pandas as pd

OUT_DIR = "agent_data/qref"
SHARD_GLOB = "/mnt/2tb_2/decoder/parquet_files_decoder/*.parquet"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sims", type=int, default=800)
    ap.add_argument("--batch", type=int, default=384)
    ap.add_argument("--chunk", type=int, default=4096)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    from chessdecoder.agent.rl.oracle_engine import OracleEngine
    from chessdecoder.agent.rl.qref import search_batch_cpp as search_batch

    os.makedirs(OUT_DIR, exist_ok=True)
    engine = OracleEngine()
    rng = np.random.default_rng(args.seed)
    shards = sorted(glob.glob(SHARD_GLOB))[:-1]      # last shard = held out
    seen: set[str] = set()
    for f in sorted(glob.glob(f"{OUT_DIR}/qref_*.parquet")):
        seen.update(x.rsplit(" ", 2)[0]
                    for x in pd.read_parquet(f, columns=["fen"]).fen)
    print(f"resuming with {len(seen)} roots already done", flush=True)

    counter = len(glob.glob(f"{OUT_DIR}/qref_*.parquet"))
    rows = []
    while True:
        shard = shards[rng.integers(len(shards))]
        df = pd.read_parquet(shard, columns=["fen", "best_move", "root_q",
                                             "orig_q"])
        w = (0.05 + (df.root_q - df.orig_q).abs()).fillna(0.0)
        idx = rng.choice(len(df), size=min(50_000, len(df)), replace=False,
                         p=(w / w.sum()).values)
        df = df.iloc[idx]
        batch_boards, batch_meta = [], []
        for fen, bm, rq, oq in df.itertuples(index=False):
            key = fen.rsplit(" ", 2)[0]
            if key in seen:
                continue
            seen.add(key)
            try:
                b = chess.Board(fen)
            except Exception:
                continue
            if b.is_game_over() or not b.is_valid():
                continue
            batch_boards.append(b)
            batch_meta.append((bm, float(rq), float(oq)))
            if len(batch_boards) == args.batch:
                res = search_batch(engine, batch_boards, sims=args.sims)
                for r, (cbm, crq, coq) in zip(res, batch_meta):
                    rows.append(dict(
                        fen=r.fen, moves=r.moves, q=r.q, visits=r.visits,
                        oracle_greedy=r.oracle_greedy,
                        search_best=r.search_best,
                        search_sensitive=r.search_best != r.oracle_greedy,
                        corpus_best=cbm, corpus_root_q=crq,
                        corpus_orig_q=coq, sims=args.sims))
                batch_boards, batch_meta = [], []
                if len(rows) >= args.chunk:
                    out = f"{OUT_DIR}/qref_{counter:05d}.parquet"
                    pd.DataFrame(rows).to_parquet(out)
                    sens = np.mean([r["search_sensitive"] for r in rows])
                    print(f"wrote {out} ({len(rows)} roots, "
                          f"{sens:.1%} search-sensitive, total "
                          f"{len(seen)})", flush=True)
                    rows, counter = [], counter + 1


if __name__ == "__main__":
    main()
