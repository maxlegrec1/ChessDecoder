#!/usr/bin/env python3
"""Parse training logs into loss/step curves and compare runs at matched steps.

Usage:
  uv run python experiments/curve.py LOG [LOG2 ...]              # summary table
  uv run python experiments/curve.py --at 5000,10000,20000 LOG  # values @ steps
  uv run python experiments/curve.py --cmp BASE.log EXP.log      # exp vs base

Logs are the train.py stdout (tqdm + "Step N: loss ... move_acc=... wdl_acc=...").
"""
import re
import sys

STEP_RE = re.compile(
    r"Step (\d+): loss ([\d.]+) \(pol ([\d.]+) wdl ([\d.]+)\).*?"
    r"move_acc=([\d.]+) wdl_acc=([\d.]+)")
KEYS = ["step", "loss", "pol", "wdl", "move_acc", "wdl_acc"]


def parse(path):
    rows = []
    with open(path, errors="ignore") as f:
        for line in f.read().replace("\r", "\n").split("\n"):
            m = STEP_RE.search(line)
            if m:
                rows.append(dict(zip(KEYS, [int(m.group(1))] +
                                     [float(m.group(i)) for i in range(2, 7)])))
    return rows


def nearest(rows, step):
    return min(rows, key=lambda r: abs(r["step"] - step)) if rows else None


def fmt(r):
    return (f"step {r['step']:>6} | loss {r['loss']:7.3f} | pol {r['pol']:6.3f} "
            f"| wdl {r['wdl']:6.3f} | move_acc {r['move_acc']:.3f} "
            f"| wdl_acc {r['wdl_acc']:.3f}")


def main():
    args = sys.argv[1:]
    at = None
    if args and args[0] == "--at":
        at = [int(x) for x in args[1].split(",")]
        args = args[2:]
    cmp = False
    if args and args[0] == "--cmp":
        cmp = True
        args = args[1:]

    runs = {p: parse(p) for p in args}
    for p, rows in runs.items():
        if not rows:
            print(f"{p}: no Step lines found"); continue
        print(f"\n== {p}  ({len(rows)} points, last step {rows[-1]['step']}) ==")
        steps = at or [rows[len(rows)//4]["step"], rows[len(rows)//2]["step"],
                       rows[-1]["step"]]
        for s in steps:
            r = nearest(rows, s)
            if r:
                print("  " + fmt(r))

    if cmp and len(args) == 2:
        base, exp = parse(args[0]), parse(args[1])
        common = sorted(set(r["step"] for r in base) & set(r["step"] for r in exp))
        print(f"\n== {args[1]} (exp)  vs  {args[0]} (base) ==")
        for s in (at or common[::max(1, len(common)//10)]):
            rb, re_ = nearest(base, s), nearest(exp, s)
            if rb and re_:
                dl = re_["loss"] - rb["loss"]
                dm = re_["move_acc"] - rb["move_acc"]
                flag = "  <-- exp better" if dl < 0 else ""
                print(f"  step {s:>6} | Δloss {dl:+.3f} | Δmove_acc {dm:+.3f}{flag}")


if __name__ == "__main__":
    main()
