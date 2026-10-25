"""Figures from results/*.csv, with Wilson 95% intervals and theory lines from theory.py. Run: python figures.py"""
import csv, os
from collections import defaultdict
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import sim, theory

TEST, FLAG, THEORY, INK = "#2a78d6", "#eb6834", "#5f5e5a", "#3d3c38"  # validated pair (dataviz palette slots 1-2)
plt.rcParams.update({"axes.spines.top": False, "axes.spines.right": False, "axes.edgecolor": INK,
                     "axes.labelcolor": INK, "xtick.color": INK, "ytick.color": INK, "font.size": 9})


def wilson(k, n, z=1.96):
    """95% Wilson score interval for k successes in n trials."""
    p = k / n
    c = (p + z**2 / (2 * n)) / (1 + z**2 / n)
    h = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / (1 + z**2 / n)
    return c - h, c + h


def cluster_bootstrap_ci(values, cluster_ids, B=10000, alpha=0.05, seed=None):
    """Percentile CI for the mean, resampling whole clusters with replacement.
    Copied from Edimah's hackathon-secret-loyalties/src/inference.py (github.com/Edimah/hackathon-secret-loyalties)."""
    values = np.asarray(values, dtype=float)
    cluster_ids = np.asarray(cluster_ids)
    if values.shape != cluster_ids.shape:
        raise ValueError("values and cluster_ids must have the same shape")
    clusters, inverse = np.unique(cluster_ids, return_inverse=True)
    k = clusters.size
    sums = np.bincount(inverse, weights=values, minlength=k)
    counts = np.bincount(inverse, minlength=k)
    rng = np.random.default_rng(seed)
    pick = rng.integers(0, k, size=(B, k))
    boot = sums[pick].sum(axis=1) / counts[pick].sum(axis=1)
    lo, hi = np.quantile(boot, [alpha / 2, 1 - alpha / 2])
    return values.mean(), lo, hi


def pooled_table(paths=("results/runs.csv", "results/dial.csv"), out="results/pooled.csv"):
    """Innocent-pair flag and test rates pooled over pairs, 95% CI by cluster bootstrap over seeds (pairs within a
    seed share the news, so they are not independent). Innocent = not (2,3) in E3/E4, not (1,4) in E2b."""
    with open(out, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["source", "condition", "C", "rate", "estimate", "lo", "hi", "n_seeds"])
        for path in paths:
            rows = defaultdict(list)
            for r in csv.DictReader(open(path)):
                p = r["A"] + r["B"]
                if (r["condition"] in ("E3", "E4") and p == "23") or (r["condition"] == "E2b" and p == "14"): continue
                rows[(r["condition"], r["C"])].append((r["seed"], float(r["gamma"]) > float(r["q95"]), int(r["flag"])))
            for (c, C), v in sorted(rows.items()):
                seeds = [s for s, _, _ in v]
                for name, i in (("test", 1), ("flag", 2)):
                    est, lo, hi = cluster_bootstrap_ci([x[i] for x in v], seeds, seed=0)
                    w.writerow([os.path.basename(path), c, C, name, est, lo, hi, len(set(seeds))])
    print(f"wrote {out}")


def rates(path):
    """{(condition, C, 'AB'): [test rejects, flags, n]} from one results CSV."""
    out = defaultdict(lambda: [0, 0, 0])
    for r in csv.DictReader(open(path)):
        o = out[(r["condition"], r["C"], r["A"] + r["B"])]
        o[0] += float(r["gamma"]) > float(r["q95"]); o[1] += int(r["flag"]); o[2] += 1
    return out


def dots(ax, x, k_test, k_flag, n, dx=0.12, label=True):
    """Test-only rate and flag rate side by side at x, each with its Wilson interval."""
    for k, col, mk, off, name in [(k_test, TEST, "o", -dx, "test rejects (γ > q95)"), (k_flag, FLAG, "s", dx, "flag (test + Γ rule)")]:
        lo, hi = wilson(k, n)
        ax.errorbar(x + off, k / n, yerr=[[k / n - lo], [hi - k / n]], fmt=mk, color=col, ms=5, lw=1.2, capsize=0,
                    label=name if label else None)


def fig1(R):
    panels = [("E1", "identity"), ("E2", "identity"), ("E2", "constant"), ("E2b", "identity"),
              ("E3", "identity"), ("E4", "identity"), ("E4", "constant")]
    pairs = ["12", "13", "14", "23", "24", "34"]
    fig, axs = plt.subplots(1, len(panels), figsize=(14, 2.8), sharey=True)
    for i, (ax, (c, C)) in enumerate(zip(axs, panels)):
        for j, p in enumerate(pairs):
            dots(ax, j, *R[(c, C, p)], label=i == 0 and j == 0)
        ax.axhline(0.05, color=THEORY, ls="--", lw=1)
        ax.set_xticks(range(6), [f"({p[0]},{p[1]})" for p in pairs], rotation=90)
        ax.set_title(f"{c}, news {'seen' if C == 'identity' else 'hidden'}" if c in ("E2", "E4") else c, color=INK)
    axs[0].set_ylabel("share of seeds"); axs[0].set_ylim(-0.02, 1.02)
    fig.legend(*axs[0].get_legend_handles_labels(), loc="upper right", frameon=False, ncol=2)
    fig.tight_layout(rect=(0, 0, 1, 0.9)); fig.savefig("figures/fig1_flag_rates.png", dpi=200)


def fig2(R):
    lams = [c[3] for c in sim.LAMBDAS]
    fig, ax = plt.subplots(figsize=(4.5, 3.2))
    for i, (name, *_ ) in enumerate(sim.LAMBDAS):
        dots(ax, lams[i], *R[(name, "identity", "14")], dx=0.03, label=i == 0)
    grid = np.linspace(0, 2, 81)
    ax.plot(grid, [theory.flag_rate(theory.true_cmi(sim.BETA, l, sim.IDENTITY), 5) for l in grid], color=THEORY, lw=1.5,
            label="theory, test only (non-central χ²)")
    ax.axhline(0.05, color=THEORY, ls="--", lw=1)
    ax.set_xlabel("λ (shared hidden term, agents 1 and 4)"); ax.set_ylabel("share of seeds, pair (1,4)")
    ax.set_ylim(-0.02, 1.02); ax.legend(frameon=False, fontsize=7)
    fig.tight_layout(); fig.savefig("figures/fig2_lambda.png", dpi=200)


def fig3(path="results/dial.csv"):
    m = defaultdict(list)
    for r in csv.DictReader(open(path)):
        if r["condition"] == "E2": m[r["C"]].append(float(r["null_mean"]))
    Ks = [5, 3, 2, 1]
    fig, ax = plt.subplots(figsize=(4, 3))
    ax.plot(Ks, [np.mean(m[f"K{k}"]) for k in Ks], "o", color=TEST, ms=6, label="mean of shuffled γ (E2)")
    ax.plot(Ks, [theory.null_bias(k) for k in Ks], color=THEORY, lw=1.5, label="theory K(r−1)(c−1)/(2n)")
    ax.set_xticks(Ks); ax.invert_xaxis(); ax.set_xlabel("news states seen by the auditor, K"); ax.set_ylabel("null bias of γ (nats)")
    ax.legend(frameon=False, fontsize=7); fig.tight_layout(); fig.savefig("figures/fig3_null_bias.png", dpi=200)


BIC = "#1baf7a"  # dataviz palette slot 3; contrast WARN, so it always has its own marker (triangle) and legend label


def bic_count(path, c, C, p, K, n):
    """Seeds where the Bayes (BIC) rule flags the pair: 2n·γ > K·log n (n rounds, K news states seen)."""
    v = [2 * n * float(r["gamma"]) > K * np.log(n) for r in csv.DictReader(open(path)) if (r["condition"], r["C"], r["A"] + r["B"]) == (c, C, p)]
    return sum(v), len(v)


def fig_dial(R, R6000=None):
    """Dial: E2 innocent pair at T = 600 (and T = 6000 if given), E4 pairs at T = 600. Test-only, flag and Bayes
    (BIC) rates with Wilson intervals; theory lines for E2: test (solid), BIC (dashed). 5% reference dotted."""
    Ks = [5, 3, 2, 1]
    panels = [("E2", "14", "E2, innocent pair (1,4), T = 600", R, "results/dial.csv", 600)]
    if R6000: panels.append(("E2", "14", "E2, innocent pair (1,4), T = 6000", R6000, "results/dial_T6000.csv", 6000))
    panels += [("E4", "14", "E4, innocent pair (1,4), T = 600", R, "results/dial.csv", 600),
               ("E4", "23", "E4, colluding pair (2,3), T = 600", R, "results/dial.csv", 600)]
    fig, axs = plt.subplots(1, len(panels), figsize=(3.4 * len(panels), 3.2), sharey=True)
    for ax, (c, p, title, RR, path, n) in zip(axs, panels):
        for i, k in enumerate(Ks):
            dots(ax, i, *RR[(c, f"K{k}", p)], dx=0.15, label=ax is axs[0] and i == 0)
            kb, nb = bic_count(path, c, f"K{k}", p, k, n)
            lo, hi = wilson(kb, nb)
            ax.errorbar(i, kb / nb, yerr=[[kb / nb - lo], [hi - kb / nb]], fmt="^", color=BIC, ms=5, lw=1.2, capsize=0,
                        label="Bayes rule (2nγ > K log n)" if ax is axs[0] and i == 0 else None)
        if c == "E2":
            I = [theory.true_cmi(sim.BETA, 0.0, sim.DIAL[f"K{k}"]) for k in Ks]
            ax.plot(range(4), [theory.flag_rate(x, k, n) for x, k in zip(I, Ks)], color=THEORY, lw=1.5, label="theory, test only" if ax is axs[0] else None)
            ax.plot(range(4), [theory.bic_flag_rate(x, k, n) for x, k in zip(I, Ks)], color=THEORY, lw=1.5, ls="--", label="theory, Bayes rule" if ax is axs[0] else None)
        ax.axhline(0.05, color=THEORY, ls=":", lw=1)
        ax.set_xticks(range(4), [str(k) for k in Ks]); ax.set_xlabel("news states seen by the auditor, K"); ax.set_title(title, color=INK, fontsize=8)
    axs[0].set_ylabel("share of seeds"); axs[0].set_ylim(-0.02, 1.02); axs[0].legend(frameon=False, fontsize=6)
    fig.tight_layout(); fig.savefig("figures/fig_dial.png", dpi=200)


if __name__ == "__main__":
    lo, hi = wilson(0, 200)
    assert lo < 1e-12 and 0.01 < hi < 0.03  # 0/200: interval [0, ~1.9%], not zero-width
    os.makedirs("figures", exist_ok=True)
    fig1(rates("results/runs.csv"))
    if os.path.exists("results/lambda.csv"): fig2(rates("results/lambda.csv"))
    if os.path.exists("results/dial.csv"):
        fig3(); fig_dial(rates("results/dial.csv"), rates("results/dial_T6000.csv") if os.path.exists("results/dial_T6000.csv") else None)
    pooled_table()
    print("wrote figures/")
