"""Price paths of two pilot runs (S seed 0, C1 seed 1): each seller's ask per round, demand news as a strip,
C1 coin heads as shaded rounds. Run from the repo root: python llm/examples.py -> figures/fig6_llm_paths.png"""
import glob, json, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
sys.path.insert(0, "llm"); sys.path.insert(0, ".")
import analyse, figures as F

COLORS = [F.TEST, F.FLAG, F.BIC]
fig, axs = plt.subplots(2, 1, figsize=(9, 5.5), sharex=True)
for ax, (tag, title) in zip(axs, [("S-0", "S, seed 0: three Haiku sellers, no coupling"),
                                  ("C1-1", "C1, seed 1: sellers 1 and 2 share a private coin (shaded = heads: \"do not lower your ask\")")]):
    d = sorted(glob.glob(f"llm/work/repo/results/*_pilot-{tag}"))[0]
    rows = [json.loads(l) for l in open(f"{d}/news.jsonl")]
    news, asks = analyse.load(d)
    t = np.arange(1, len(asks) + 1)
    for i, h in enumerate(r.get("coin", 0) for r in rows):
        if h: ax.axvspan(i + 0.5, i + 1.5, color=F.INK, alpha=0.08, lw=0)
    for s in range(asks.shape[1]):
        ax.plot(t, asks[:, s], color=COLORS[s], lw=1.4, label=f"seller {s + 1}" + (" (coupled)" if tag.startswith("C1") and s < 2 else ""))
    ax.scatter(t, np.full(len(t), 79.6), c=[["#c3c2b7", "#8a8980", "#3d3c38"][n] for n in news], marker="|", s=60)
    ax.set_ylim(79.4, 84.5); ax.set_ylabel("ask ($)"); ax.set_title(title, color=F.INK, fontsize=8)
    ax.legend(frameon=False, fontsize=7, loc="upper right")
axs[1].set_xlabel("round (hour); bottom strip = demand news: light falling, mid steady, dark rising; seller cost = $80")
fig.tight_layout(); fig.savefig("figures/fig6_llm_paths.png", dpi=200)
print("wrote figures/fig6_llm_paths.png")
