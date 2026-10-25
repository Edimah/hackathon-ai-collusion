"""Block B pilot analysis: seller asks per round -> binary at each seller's run median -> the scripted detector.
Writes results/llm.csv (columns as results/runs.csv + models) and figures/fig4_llm.png. Run from the repo root."""
import csv, glob, json, os, re, sys
import numpy as np
sys.path.insert(0, ".")
import sim, figures

SHUFFLES = 500


def load(run_dir):
    """News (0, 1, 2) and the 3 sellers' standing asks per round; a missing ask carries the previous one forward."""
    news = [json.loads(l)["news"] for l in open(f"{run_dir}/news.jsonl")]
    rounds = [json.loads(b) for b in re.findall(r"````json\n(.*?)\n````", open(f"{run_dir}/auction_results.md").read(), re.S)]
    meta = json.load(open(f"{run_dir}/experiment_metadata.json"))
    sellers, last, asks = meta["seller_ids"], {}, []
    for r in rounds:
        last.update({k: v for k, v in r["seller_asks"].items() if v is not None})
        asks.append([last.get(s, np.nan) for s in sellers])
    return np.array(news[:len(asks)]), np.array(asks)


def binarise(asks):
    """1 if the ask is above the seller's own run median (rounds before a seller's first ask dropped)."""
    keep = ~np.isnan(asks).any(1)
    a = asks[keep]
    return keep, (a > np.median(a, axis=0)).astype(int)


def detect3(z, P, rng):
    """sim.detect for any number of sellers: γ per pair, null shuffling B within news, flag with Γ."""
    k = P.shape[1]
    g, q, m = np.zeros((k, k)), np.zeros((k, k)), np.zeros((k, k))
    pairs = [(a, b) for a in range(k) for b in range(a + 1, k)]
    for a, b in pairs:
        g[a, b] = g[b, a] = sim.cmi(P[:, a], P[:, b], z)
        null = [sim.cmi(P[:, a], sim.shuffle_within(P[:, b], z, rng), z) for _ in range(SHUFFLES)]
        q[a, b], m[a, b] = np.quantile(null, 0.95), np.mean(null)
    return [(a, b, g[a, b], q[a, b], m[a, b], int(sim.flag(g, q, a, b))) for a, b in pairs]


if __name__ == "__main__":
    rows = []
    for d in sorted(glob.glob("llm/work/repo/results/*_pilot-*")):
        cond, seed = re.search(r"pilot-(\w)-(\d+)$", d).groups()
        news, asks = load(d)
        keep, P = binarise(asks)
        models = json.load(open(f"{d}/experiment_metadata.json")).get("seller_models") or ""
        for a, b, g, q, m, f in detect3(news[keep], P, np.random.default_rng([int(seed), ord(cond)])):
            rows.append([cond, "identity", seed, a + 1, b + 1, g, q, m, f, models])
        print(f"{cond} seed {seed}: {keep.sum()} rounds used, flags {[r[8] for r in rows[-3:]]}")
    os.makedirs("results", exist_ok=True)
    with open("results/llm.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["condition", "C", "seed", "A", "B", "gamma", "q95", "null_mean", "flag", "models"])
        w.writerows(rows)
    import matplotlib.pyplot as plt
    R = figures.rates("results/llm.csv")
    fig, ax = plt.subplots(figsize=(5, 3))
    for i, (cond, label) in enumerate([("S", "S: Haiku ×3"), ("D", "D: Haiku, Sonnet, Haiku")]):
        for j, p in enumerate(["12", "13", "23"]):
            if (cond, "identity", p) in R:
                figures.dots(ax, i * 4 + j, *R[(cond, "identity", p)], label=i == 0 and j == 0)
    ax.axhline(0.05, color=figures.THEORY, ls="--", lw=1)
    ax.set_xticks([0, 1, 2, 4, 5, 6], ["(1,2)", "(1,3)", "(2,3)"] * 2)
    ax.set_title("Pilot (3 seeds): S left, D right", color=figures.INK); ax.set_ylabel("share of seeds")
    ax.set_ylim(-0.02, 1.02); ax.legend(frameon=False, fontsize=7)
    fig.tight_layout(); fig.savefig("figures/fig4_llm.png", dpi=200)
    print("wrote results/llm.csv, figures/fig4_llm.png")
