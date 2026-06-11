"""Build eval suites (once) and print the L0/L1/L2 table for a checkpoint.

  uv run python scripts/eval_agent_rl.py --build-suites
  uv run python scripts/eval_agent_rl.py --ckpt checkpoints/agent_grpo_v1/grpo_2000.pt
"""
import argparse
import os

from chessdecoder.agent.rl.eval import SUITE_PATH, build_suites, eval_model

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--build-suites", action="store_true")
    ap.add_argument("--ckpt")
    ap.add_argument("--ks", default="0,4,16")
    ap.add_argument("--max-roots", type=int, default=None)
    a = ap.parse_args()
    if a.build_suites:
        df = build_suites()
        print(df.groupby("suite").size())
    if a.ckpt:
        assert os.path.exists(SUITE_PATH), "run --build-suites first"
        ks = tuple(int(x) for x in a.ks.split(","))
        t = eval_model(a.ckpt, ks=ks, max_roots=a.max_roots)
        print(t.to_string(index=False))
