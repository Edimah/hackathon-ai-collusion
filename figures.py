"""Figures from results/*.csv, with Wilson 95% intervals and theory lines from theory.py. Run: python figures.py"""
import csv, os
from collections import defaultdict
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker
import sim, theory

TEST, FLAG, THEORY, INK, GREEN = "#2a78d6", "#eb6834", "#5f5e5a", "#3d3c38", "#2e9e62"  # validated pair (dataviz palette slots 1-2)
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


def pooled_table(paths=("results/runs.csv", "results/dial.csv", "results/dial_T6000.csv"), out="results/pooled.csv"):
    """Innocent-pair flag and test rates pooled over pairs, 95% CI by cluster bootstrap over seeds (pairs within a
    seed share the news, so they are not independent). Innocent = not (2,3) in E3/E4, not (1,4) in E2b."""
    with open(out, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["source", "condition", "C", "rate", "estimate", "lo", "hi", "n_seeds"])
        for path in paths:
            rows = defaultdict(list)
            if not os.path.exists(path): continue
            for r in csv.DictReader(open(path)):
                p = r["A"] + r["B"]
                if (r["condition"] in ("E3", "E4") and p == "23") or (r["condition"] == "E2b" and p == "14"): continue
                K = {"identity": 5, "constant": 1}.get(r["C"]) or int(r["C"][1:])  # news levels the auditor sees
                n = 6000 if "T6000" in path else 600
                rows[(r["condition"], r["C"])].append((r["seed"], float(r["gamma"]) > float(r["q95"]), int(r["flag"]),
                                                       2 * n * float(r["gamma"]) > K * np.log(n)))
            for (c, C), v in sorted(rows.items()):
                seeds = [x[0] for x in v]
                for name, i in (("test", 1), ("flag", 2), ("bayes", 3)):
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
    """Standard-test rate and ranking-rule rate side by side at x, each with its Wilson interval."""
    for k, col, mk, off, name in [(k_test, TEST, "o", -dx, "standard test"), (k_flag, FLAG, "s", dx, "standard test + ranking rule (Γ)")]:
        lo, hi = wilson(k, n)
        ax.errorbar(x + off, k / n, yerr=[[k / n - lo], [hi - k / n]], fmt=mk, color=col, ms=4.5, lw=1.1, capsize=0,
                    label=name if label else None)


def fig1(R):
    """Appendix: every condition and pair, 600 rounds, in plain condition names (two rows so it prints legibly)."""
    panels = [("E1", "identity", "prices ignore news,\nno planted pair (50 repeats)"), ("E2", "identity", "prices follow news,\nall five news levels"),
              ("E2", "constant", "prices follow news,\nno news information"), ("E2b", "identity", "hidden shared input\nfor sellers 1 and 4"),
              ("E3", "identity", "planted pair (2,3),\nprices ignore news"), ("E4", "identity", "planted pair (2,3),\nall five news levels"),
              ("E4", "constant", "planted pair (2,3),\nno news information")]
    pairs = ["12", "13", "14", "23", "24", "34"]
    fig, axs = plt.subplots(2, 4, figsize=(6.6, 4.6), sharey=True)
    for i, (ax, (c, C, title)) in enumerate(zip(axs.flat, panels)):
        for j, p in enumerate(pairs):
            dots(ax, j, *R[(c, C, p)], dx=0.15, label=i == 0 and j == 0)
        ax.axhline(0.05, color=THEORY, ls="--", lw=0.9)
        ax.set_xticks(range(6), [f"{p[0]},{p[1]}" for p in pairs], fontsize=6.5)
        ax.set_title(title, color=INK, fontsize=7)
        ax.tick_params(axis="y", labelsize=7)
    axs.flat[-1].axis("off")
    axs[0, 0].set_ylabel("share of repeats flagged", fontsize=7.5); axs[1, 0].set_ylabel("share of repeats flagged", fontsize=7.5)
    axs[0, 0].set_ylim(-0.02, 1.02)
    axs.flat[-1].legend(*axs[0, 0].get_legend_handles_labels(), loc="center", frameon=False, fontsize=7, title="seller pair (x-axis)\ndashed line: 5%", title_fontsize=7)
    fig.tight_layout(pad=0.3); fig.savefig("figures/fig1_flag_rates.png", dpi=250)


def fig2(R):
    """Appendix: hidden shared input of strength λ for sellers 1 and 4, complete record, 600 rounds."""
    lams = [c[3] for c in sim.LAMBDAS]
    fig, ax = plt.subplots(figsize=(4.3, 3.0))
    for i, (name, *_ ) in enumerate(sim.LAMBDAS):
        dots(ax, lams[i], *R[(name, "identity", "14")], dx=0.03, label=i == 0)
    grid = np.linspace(0, 2, 81)
    ax.plot(grid, [theory.flag_rate(theory.true_cmi(sim.BETA, l, sim.IDENTITY), 5) for l in grid], color=THEORY, lw=1.5,
            label="prediction from the model (standard test)")
    ax.axhline(0.05, color=THEORY, ls="--", lw=1)
    ax.set_xlabel("λ, strength of the unrecorded shared input"); ax.set_ylabel("share of repeats flagged, pair (1,4)")
    ax.set_ylim(-0.02, 1.02); ax.legend(frameon=False, fontsize=7)
    fig.tight_layout(); fig.savefig("figures/fig2_lambda.png", dpi=250)


def fig3(path="results/dial.csv"):
    """Appendix: average shuffled statistic (non-colluding pairs, 600 rounds) against K/(2n)."""
    m = defaultdict(list)
    for r in csv.DictReader(open(path)):
        if r["condition"] == "E2": m[r["C"]].append(float(r["null_mean"]))
    Ks = [5, 3, 2, 1]
    fig, ax = plt.subplots(figsize=(3.6, 2.8))
    ax.plot(Ks, [np.mean(m[f"K{k}"]) for k in Ks], "o", color=TEST, ms=6, label=f"average of 500 shuffled {I_HAT}")
    ax.plot(Ks, [theory.null_bias(k) for k in Ks], color=THEORY, lw=1.5, label="approximation $K/(2n)$")
    ax.set_xticks(Ks, ["5", "3", "2", "no news"]); ax.invert_xaxis()
    ax.set_xlabel("news categories used by the screen"); ax.set_ylabel(f"{I_HAT} under shuffling (nats)")
    ax.legend(frameon=False, fontsize=7); fig.tight_layout(); fig.savefig("figures/fig3_null_bias.png", dpi=250)


I_HAT = r"$\hat{I}$"
BIC = "#1baf7a"  # dataviz palette slot 3; contrast WARN, so it always has its own marker (triangle) and legend label


def fig_hero(path="results/pooled.csv"):
    """Main figure: non-colluding pairs wrongly flagged vs the news categories the auditor records (E2, six pairs pooled,
    cluster bootstrap by repeat). (a) standard test, (b) stricter BIC-based rule; data at 600 and 6,000 rounds, lines = prediction."""
    P = {(r["source"], r["C"], r["rate"]): r for r in csv.DictReader(open(path)) if r["condition"] == "E2"}
    Ks = [5, 3, 2, 1]
    I = [theory.true_cmi(sim.BETA, 0.0, sim.DIAL[f"K{k}"]) for k in Ks]
    fig, axs = plt.subplots(1, 2, figsize=(6.6, 3.1), sharey=True)
    for ax, (rate, pred, title) in zip(axs, [("test", theory.flag_rate, "(a) standard test, 5% level"),
                                             ("bayes", theory.bic_flag_rate, "(b) stricter BIC-based rule")]):
        for src, n, col, mk in [("dial.csv", 600, TEST, "o"), ("dial_T6000.csv", 6000, FLAG, "s")]:
            ax.plot(range(4), [pred(x, k, n) for x, k in zip(I, Ks)], color=col, lw=1.3, alpha=0.7)
            for i, k in enumerate(Ks):
                r = P.get((src, f"K{k}", rate))
                if r is None: continue
                e, lo, hi = float(r["estimate"]), float(r["lo"]), float(r["hi"])
                dx = 0.06 if n == 6000 else -0.06
                ax.errorbar(i + dx, e, yerr=[[e - lo], [hi - e]], fmt=mk, color=col, ms=5, lw=1.2, capsize=0,
                            label=f"{n:,} rounds (line: prediction)" if i == 0 else None)
                if 0.004 < e < 0.995:  # print the measured rate next to every point that is not 0% or 100%
                    ax.annotate(f"{e:.1%}", (i + dx, e), xytext=(7 if n == 6000 else -7, 8 if i == 0 else 0), textcoords="offset points",
                                ha="left" if n == 6000 else "right", va="center", fontsize=6.5, color=col)
        ax.axhline(0.05, color=THEORY, ls=":", lw=1, label="5% target" if ax is axs[0] else None)
        ax.set_xticks(range(4), ["5\n(all levels)", "3", "2", "no news"]); ax.set_xlabel("news categories used by the screen")
        ax.set_title(title, color=INK, fontsize=8.5)
    axs[0].set_ylabel("non-colluding pairs wrongly flagged"); axs[0].set_ylim(-0.03, 1.05)
    axs[0].yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0)); axs[0].legend(frameon=False, fontsize=7, loc="upper left")
    fig.tight_layout(); fig.savefig("figures/fig_hero.png", dpi=250)


def fig_method(out="figures/fig1_method.png"):
    """Figure 1: the two tests as flat box-and-arrow rows, drawn at page width so its text prints at about 8 pt."""
    from matplotlib.patches import FancyBboxPatch
    rows = [("STANDARD TEST", TEST, ["Log:\nnews and every\nseller's price", "Auditor's view:\nnews categories\nused by the screen",
                                      "Shuffle B's prices\n500 times within\neach category", f"{I_HAT}: estimated\nremaining\ndependence",
                                      f"Flag if {I_HAT} is\nstrictly above the\n95th percentile\nof the shuffles"]),
            ("RE-ASKING", FLAG, ["Recorded prompt\nof each seller,\nplanted note\nremoved", "Same model,\nsame prompt,\nasked 4 more times",
                                 "Rank original ask\namong usable\nre-asked answers", "Remove mean rank\nper news category;\nco-movement of\nthe two sellers",
                                 "Flag if strictly\nabove the 95th\npercentile of\n500 relabellings"])]
    fig, ax = plt.subplots(figsize=(6.6, 3.0)); ax.set_xlim(0, 6.6); ax.set_ylim(0, 3.0); ax.axis("off")
    w, gap = 1.08, 0.23
    for r, (label, col, boxes) in enumerate(rows):
        y = 2.05 - 1.45 * r
        ax.text(0.02, y + 0.55, label, fontsize=8, family="monospace", color=col, weight="bold")
        for i, t in enumerate(boxes):
            x = 0.02 + i * (w + gap)
            ax.add_patch(FancyBboxPatch((x, y - 0.42), w, 0.84, boxstyle="round,pad=0.01,rounding_size=0.06",
                                        fc="#fcfcfb", ec=col, lw=1.2))
            ax.text(x + w / 2, y, t, ha="center", va="center", fontsize=7.0, color=INK, linespacing=1.25)
            if i < len(boxes) - 1:
                ax.annotate("", xy=(x + w + gap - 0.02, y), xytext=(x + w + 0.02, y), arrowprops=dict(arrowstyle="->", color=INK, lw=1))
    fig.tight_layout(pad=0.2); fig.savefig(out, dpi=250)


if __name__ == "__main__":
    lo, hi = wilson(0, 200)
    assert lo < 1e-12 and 0.01 < hi < 0.03  # 0/200: interval [0, ~1.9%], not zero-width
    os.makedirs("figures", exist_ok=True)
    fig1(rates("results/runs.csv"))
    if os.path.exists("results/lambda.csv"): fig2(rates("results/lambda.csv"))
    if os.path.exists("results/dial.csv"):
        fig3()
    pooled_table(); fig_hero(); fig_method()
    print("wrote figures/")
