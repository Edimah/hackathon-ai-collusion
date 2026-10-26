"""Story GIFs C and D in the triptych's style, then the story in order C -> D -> E. Run from the repo root:
python docs/gifs/story.py -> docs/gifs/C-observability-dial.gif, D-llm-pilot.gif, ALL-stitched.gif (C, D, E).
Numbers: C from results/pooled.csv (innocent pairs, test only, 600 rounds) and theory.py; D from results/table1.csv."""
import csv, os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.patches import Circle, FancyBboxPatch
from PIL import Image
sys.path.insert(0, "."); sys.path.insert(0, "docs/gifs")
import sim, theory
from triptych import GREY, EDGE, LINK, INK, MUTED, GREEN, AMBER, ease, mix, lens, FPS

W, H = 1440, 380
NEWS = ["#f1d9d3", "#e8b8ab", "#d9d9d6", "#b9cbe6", "#8eaedb"]  # falling -> rising
STAGES = [("K5", "5 levels (complete)", [[0], [1], [2], [3], [4]]), ("K3", "3 levels", [[0, 1], [2], [3, 4]]),
          ("K2", "2 levels", [[0, 1, 2], [3, 4]]), ("K1", "none", [[0, 1, 2, 3, 4]])]


def observed():
    P = {(r["source"], r["C"], r["rate"]): float(r["estimate"]) for r in csv.DictReader(open("results/pooled.csv")) if r["condition"] == "E2"}
    return [P[("dial.csv", k, "test")] for k, _, _ in STAGES]


def predicted():
    return [theory.flag_rate(theory.true_cmi(sim.BETA, 0.0, sim.DIAL[k]), int(k[1])) for k, _, _ in STAGES]


def ax_new():
    fig = plt.figure(figsize=(W / 100, H / 100), dpi=100)
    ax = fig.add_axes([0, 0, 1, 1]); return fig, ax


def frame_base(ax, label):
    ax.clear(); ax.set_xlim(0, W); ax.set_ylim(0, H); ax.set_aspect("equal"); ax.axis("off")
    ax.text(16, H - 22, label, family="monospace", fontsize=9, weight="bold", color=INK)


def scene_c(out="docs/gifs/C-observability-dial.gif", stage_s=2.8, hold_s=2.0):
    obs, pred = observed(), predicted()
    nS, nH = int(stage_s * FPS), int(hold_s * FPS)
    agents = {1: (150, 230), 2: (330, 230), 3: (150, 100), 4: (330, 100)}
    fig, ax = ax_new()

    def colours(stage):  # each news cell shows its group's mean colour in the auditor's record
        groups = STAGES[stage][2]; out = [None] * 5
        for g in groups:
            c = np.mean([matplotlib.colors.to_rgb(NEWS[i]) for i in g], axis=0)
            for i in g: out[i] = c
        return out

    def gaps(stage):  # gap after each cell: 0 inside a merged group
        groups = STAGES[stage][2]; same = set()
        for g in groups: same.update((i, i + 1) for i in g[:-1])
        return [0 if (i, i + 1) in same else 8 for i in range(4)]

    def frame(f):
        frame_base(ax, "WHAT THE AUDITOR RECORDS")
        s, t = (min(f // nS, 3), (f % nS) / nS) if f < 4 * nS else (3, 1.0)
        m = ease(t, 0.05, 0.35) if s > 0 and f < 4 * nS else 1.0  # merge transition into stage s
        prev = max(s - 1, 0)
        # left: what the sellers react to
        ax.text(60, 330, "news the sellers react to", fontsize=8, color=MUTED)
        for i in range(5):
            ax.add_patch(FancyBboxPatch((60 + i * 52, 290), 46, 26, boxstyle="round,pad=0,rounding_size=4", fc=NEWS[i], ec="none"))
            ax.text(83 + i * 52, 303, str(i + 1), ha="center", va="center", fontsize=8, color=INK)
        rate = obs[prev] + (obs[s] - obs[prev]) * m
        for i, j in [(1, 2), (3, 4), (1, 3), (2, 4), (1, 4), (2, 3)]:
            (x1, y1), (x2, y2) = agents[i], agents[j]
            ax.plot([x1, x2], [y1, y2], color=LINK, lw=1.3, zorder=1)
            ax.plot([x1, x2], [y1, y2], color=AMBER, lw=2.4, ls=(0, (4, 3)), alpha=rate, zorder=1)
        for i, (x, y) in agents.items():
            ax.add_patch(Circle((x, y), 22, fc=GREY, ec=EDGE, lw=1.3, zorder=3))
            ax.text(x, y, f"A{i}", ha="center", va="center", fontsize=8.5, color=INK, zorder=4)
        ax.text(240, 52, "4 innocent sellers, no agreement", ha="center", fontsize=8, color=MUTED)
        # middle: the auditor's record, cells merging
        c0, c1 = colours(prev), colours(s)
        g0, g1 = gaps(prev), gaps(s)
        x = 560
        ax.text(560, 268, "the auditor's record", fontsize=8, color=MUTED)
        for i in range(5):
            col = tuple(np.array(c0[i]) + (np.array(c1[i]) - np.array(c0[i])) * m)
            ax.add_patch(FancyBboxPatch((x, 210), 52, 34, boxstyle="round,pad=0,rounding_size=4", fc=col, ec="none"))
            if i < 4: x += 52 + g0[i] + (g1[i] - g0[i]) * m
        lens(ax, 690, 227, 1.0)
        ax.text(690, 160, STAGES[s][1], ha="center", fontsize=10, weight="bold", color=INK, alpha=m)
        # right: observed and predicted, a mini chart that fills as the record coarsens
        ax.text(1000, 330, "innocent pairs accused (test at 5%)", fontsize=8, color=MUTED)
        for st, al in ((prev, 1 - m), (s, m)) if s != prev else ((s, 1.0),):  # cross-fade: only measured values are shown
            ax.text(1000, 285, f"observed {obs[st]:.1%}", fontsize=16, weight="bold", color=AMBER if obs[st] > 0.08 else INK, alpha=al)
            ax.text(1000, 255, f"predicted from the model {pred[st]:.1%}", fontsize=9, color=INK, alpha=al)
        x0, y0, w, h = 1010, 70, 380, 140
        ax.plot([x0, x0 + w], [y0, y0], color=EDGE, lw=1); ax.plot([x0, x0 + w], [y0 + 0.05 * h] * 2, color=MUTED, lw=0.8, ls=":")
        ax.text(x0 + w, y0 + 0.05 * h + 4, "5% target", fontsize=6.5, color=MUTED, ha="right")
        for k in range(4):
            xk = x0 + 40 + k * 100
            ax.text(xk, y0 - 16, STAGES[k][1].split(" (")[0], ha="center", fontsize=7, color=MUTED)
            show = 1.0 if k < s else (m if k == s else 0.0)
            if show > 0:
                ax.plot([xk - 14, xk + 14], [y0 + pred[k] * h] * 2, color=INK, lw=1.6, alpha=show)
                ax.add_patch(Circle((xk, y0 + obs[k] * h * show), 6, fc=AMBER, ec="none", alpha=show))
        ax.add_patch(Circle((x0 + 10, 48 - 10), 5, fc=AMBER, ec="none"))
        ax.text(x0 + 22, 34, "observed", fontsize=7, color=MUTED)
        ax.plot([x0 + 90, x0 + 110], [38, 38], color=INK, lw=1.6); ax.text(x0 + 116, 34, "predicted", fontsize=7, color=MUTED)
        ax.text(W / 2, 12, "the less of the news the auditor records, the more innocent sellers look coordinated",
                ha="center", fontsize=8.5, color=INK)
        return []

    FuncAnimation(fig, frame, frames=4 * nS + nH).save(out, writer=PillowWriter(fps=FPS))
    plt.close(fig); print("wrote", out)


def scene_d(out="docs/gifs/D-llm-pilot.gif", dur_s=8.0):
    T1 = {(r["test"], r["cell"]): (int(r["rejected"]), int(r["of"])) for r in csv.DictReader(open("results/table1.csv"))}
    k, n = T1[("news-only test", "S games, all pairs")]
    N = int(dur_s * FPS)
    sellers = {1: (470, 245), 2: (650, 245), 3: (560, 125)}
    # real standing asks, S game seed 0, rounds 1, 4, 8, 30 (llm/analyse.py load)
    asks = {1: [95.50, 88.50, 81.00, 80.20], 2: [95.50, 87.00, 80.67, 80.13], 3: [91.50, 86.00, 81.50, 80.20]}
    ROUNDS = [1, 4, 8, 30]
    fig, ax = ax_new()

    def frame(f):
        t = f / (N - 1)
        frame_base(ax, "AI SELLERS · NEWS-ONLY TEST")
        a1, a2 = ease(t, 0.0, 0.15), ease(t, 0.08, 0.23)
        for (y, title, sub, a, col) in [(270, "demand news", "falling / steady / rising", a1, NEWS[3]),
                                        (130, "order book", "everyone's past asks", a2, GREY)]:
            ax.add_patch(FancyBboxPatch((60, y - 26), 190, 52, boxstyle="round,pad=0,rounding_size=6", fc=col, ec="none", alpha=a))
            ax.text(155, y + 6, title, ha="center", fontsize=9, color=INK, alpha=a)
            ax.text(155, y - 12, sub, ha="center", fontsize=7, color=MUTED, alpha=a)
            ax.annotate("", xy=(392, 180 + (y - 200) * 0.3), xytext=(252, y),
                        arrowprops=dict(arrowstyle="->", color=EDGE, lw=1.2, alpha=a, shrinkA=0, shrinkB=0))
        ga = ease(t, 0.12, 0.25)  # the sellers as one group: every seller sees both inputs
        ax.add_patch(FancyBboxPatch((392, 62), 340, 250, boxstyle="round,pad=0,rounding_size=10", fc="none", ec=LINK, lw=1.2, ls="--", alpha=ga))
        ax.text(562, 322, "every seller sees both", ha="center", fontsize=7.5, color=MUTED, alpha=ga)
        lk = ease(t, 0.5, 0.65)
        for i, j in [(1, 2), (1, 3), (2, 3)]:
            (x1, y1), (x2, y2) = sellers[i], sellers[j]
            ax.plot([x1, x2], [y1, y2], color=LINK, lw=1.3, zorder=1)
            ax.plot([x1, x2], [y1, y2], color=AMBER, lw=2.4, ls=(0, (4, 3)), alpha=lk, zorder=1)
        step = min(int(ease(t, 0.2, 0.45) * 3.999), 3)
        for i, (x, y) in sellers.items():
            ax.add_patch(Circle((x, y), 24, fc=GREY, ec=EDGE, lw=1.3, zorder=3))
            ax.text(x, y + 4, f"A{i}", ha="center", va="center", fontsize=8.5, color=INK, zorder=4)
            ax.text(x, y - 9, "Haiku", ha="center", va="center", fontsize=6, color=MUTED, zorder=4)
            ax.text(x, y + 34 if i < 3 else y - 40, f"ask ${asks[i][step]:.2f}", ha="center", fontsize=7.5, color=INK, alpha=ease(t, 0.2, 0.3))
        ax.text(562, 70, f"round {ROUNDS[step]}", ha="center", fontsize=7, color=MUTED, alpha=ease(t, 0.2, 0.3))
        # the auditor's log keeps the news and the asks, not the order book
        la = ease(t, 0.35, 0.5)
        ax.add_patch(FancyBboxPatch((800, 150), 230, 110, boxstyle="round,pad=0,rounding_size=6", fc="#ffffff", ec=EDGE, lw=1.2, alpha=la))
        ax.text(915, 238, "auditor's log", ha="center", fontsize=8, color=MUTED, alpha=la)
        ax.text(915, 210, "news  ✓     asks  ✓", ha="center", fontsize=9, color=INK, alpha=la)
        ax.text(915, 180, "order book  ✗", ha="center", fontsize=9, color=AMBER, alpha=la)
        s = ease(t, 0.45, 0.62)
        lens(ax, 860 + s * 110, 205, la * (1 - ease(t, 0.62, 0.72)))
        r = ease(t, 0.62, 0.75)
        ax.text(1080, 250, f"innocent pairs accused: {k} of {n}", fontsize=13, weight="bold", color=AMBER, alpha=r)
        ax.text(1080, 225, "3 games, no planted pair · pilot", fontsize=8, color=MUTED, alpha=r)
        ax.text(1080, 190, "every seller also reacts to the order book,", fontsize=8, color=INK, alpha=r)
        ax.text(1080, 174, "which the auditor's record leaves out", fontsize=8, color=INK, alpha=r)
        e = ease(t, 0.8, 0.92)
        ax.text(1080, 120, "but AI agents can be re-asked →", fontsize=13, weight="bold", color=GREEN, alpha=e)
        return []

    FuncAnimation(fig, frame, frames=N + int(1.5 * FPS)).save(out, writer=PillowWriter(fps=FPS))
    plt.close(fig); print("wrote", out)


def stitch(parts, out="docs/gifs/ALL-stitched.gif"):
    """Concatenate GIFs of the same size, keeping each frame's duration."""
    frames, durs = [], []
    for p in parts:
        im = Image.open(p)
        for i in range(im.n_frames):
            im.seek(i); frames.append(im.convert("RGB").copy()); durs.append(im.info.get("duration", 1000 // FPS))
    assert len({f.size for f in frames}) == 1, "all parts must have the same size"
    frames[0].save(out, save_all=True, append_images=frames[1:], duration=durs, loop=0, optimize=True)
    print("wrote", out, f"({len(frames)} frames)")


if __name__ == "__main__":
    o, p = observed(), predicted()
    assert abs(o[0] - 0.058) < 0.005 and abs(p[2] - 0.518) < 0.005, (o, p)  # matches results/pooled.csv and theory
    scene_c(); scene_d()
    stitch(["docs/gifs/C-observability-dial.gif", "docs/gifs/D-llm-pilot.gif", "docs/gifs/E-re-asking-triptych.gif"])
