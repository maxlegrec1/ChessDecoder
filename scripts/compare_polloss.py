#!/usr/bin/env python3
"""Compare smoothed train pol-loss between two training logs over a step window.

Pol-loss (averaged over the window, n~200/4k-steps at log_every=5) is the
low-variance discriminator the architecture study uses — val move_acc at
val_batches=64 is far noisier. Prints both means, the delta, and a z-score.

  uv run python scripts/compare_polloss.py A.log B.log --lo 28000 --hi 32000
"""
import argparse, re, math


def pols(log, lo, hi):
    vals = []
    txt = open(log).read()
    for m in re.finditer(r"Step (\d+): loss [\d.]+ \(pol ([\d.]+)", txt):
        s = int(m.group(1))
        if lo <= s < hi:
            vals.append(float(m.group(2)))
    return vals


def stats(v):
    n = len(v); mu = sum(v) / n
    sd = math.sqrt(sum((x - mu) ** 2 for x in v) / (n - 1)) if n > 1 else 0.0
    return n, mu, sd, sd / math.sqrt(n)          # n, mean, std, SE


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("a"); ap.add_argument("b")
    ap.add_argument("--label-a", default="A"); ap.add_argument("--label-b", default="B")
    ap.add_argument("--lo", type=int, default=28000); ap.add_argument("--hi", type=int, default=32000)
    x = ap.parse_args()
    va, vb = pols(x.a, x.lo, x.hi), pols(x.b, x.lo, x.hi)
    if not va or not vb:
        print(f"insufficient data: {x.label_a} n={len(va)}  {x.label_b} n={len(vb)}")
        return
    na, ma, sda, sea = stats(va)
    nb, mb, sdb, seb = stats(vb)
    delta = mb - ma
    se = math.sqrt(sea ** 2 + seb ** 2)
    z = delta / se if se else 0.0
    print(f"  window [{x.lo}, {x.hi})  (smoothed train pol-loss)")
    print(f"  {x.label_a:24} n={na:4d}  mean={ma:.4f}  SE={sea:.4f}")
    print(f"  {x.label_b:24} n={nb:4d}  mean={mb:.4f}  SE={seb:.4f}")
    better = x.label_b if delta < 0 else x.label_a
    print(f"  delta ({x.label_b} - {x.label_a}) = {delta:+.4f}   z = {z:+.2f}   "
          f"-> {better} better by {abs(delta):.4f} ({abs(z):.1f}sigma)")


if __name__ == "__main__":
    main()
