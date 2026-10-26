"""Table 1: pair-runs rejected by the news-only test and by re-asking (binary, rank), in the S games and the
planted-coin (C1) games, with one-sided 95% Clopper-Pearson bounds. News-only = llm/analyse.py logic, test only
(gamma above its within-news shuffle q95), binary move "raised or not". Run from the repo root -> results/table1.csv"""
import csv, glob, re, sys
import numpy as np
from scipy.stats import beta
sys.path.insert(0, "llm"); sys.path.insert(0, ".")
import analyse


def cp_bounds(k, n, a=0.05):
    """One-sided 95% Clopper-Pearson: lower bound (for 'caught') and upper bound (for 'accused')."""
    lo = beta.ppf(a, k, n - k + 1) if k > 0 else 0.0
    hi = beta.ppf(1 - a, k + 1, n - k) if k < n else 1.0
    return lo, hi


def news_only(cond):
    """{pair: [reject per run]} for the news-only test on one condition's runs."""
    out = {}
    for d in sorted(glob.glob(f"llm/work/repo/results/*_pilot-{cond}-*")):
        news, asks = analyse.load(d)
        keep, P = analyse.binarise(asks)
        seed = int(re.search(r"-(\d+)$", d).group(1))
        for a, b, g, q, m, f in analyse.detect3(news[keep], P, np.random.default_rng([seed, ord(cond[0])])):
            out.setdefault(f"{a + 1}{b + 1}", []).append(int(g > q))
    return out


def from_csv(path, cond, col="reject"):
    out = {}
    for r in csv.DictReader(open(path)):
        if r["condition"] == cond: out.setdefault(r["A"] + r["B"], []).append(int(r[col]))
    return out


def cells(rej_S, rej_C1):
    """(k, n) for: S all pairs; C1 planted pair (1,2); C1 other pairs."""
    s = sum(sum(v) for v in rej_S.values()), sum(len(v) for v in rej_S.values())
    p = sum(rej_C1["12"]), len(rej_C1["12"])
    o = sum(sum(v) for k, v in rej_C1.items() if k != "12"), sum(len(v) for k, v in rej_C1.items() if k != "12")
    return s, p, o


if __name__ == "__main__":
    assert abs(cp_bounds(0, 9)[1] - (1 - 0.05 ** (1 / 9))) < 1e-9 and abs(cp_bounds(3, 3)[0] - 0.05 ** (1 / 3)) < 1e-9
    rows = {"news-only test": cells(news_only("S"), news_only("C1")),
            "re-asking, binary (pre-registered)": cells(from_csv("results/replay.csv", "S"), from_csv("results/replay.csv", "C1")),
            "re-asking, rank (addendum 1)": cells(from_csv("results/replay_rank.csv", "S"), from_csv("results/replay_rank.csv", "C1"))}
    with open("results/table1.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["test", "cell", "rejected", "of", "lower95_one_sided", "upper95_one_sided"])
        for test, cs in rows.items():
            for cell, (k, n) in zip(["S games, all pairs", "planted games, planted pair (1,2)", "planted games, other pairs"], cs):
                lo, hi = cp_bounds(k, n)
                w.writerow([test, cell, k, n, round(lo, 3), round(hi, 3)])
                print(f"{test:<36} {cell:<36} {k}/{n}  lower {lo:.0%}  upper {hi:.0%}")
    print("wrote results/table1.csv")
