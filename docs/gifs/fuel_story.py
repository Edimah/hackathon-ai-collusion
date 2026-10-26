"""Fuel-station drawings: the looping GIF docs/gifs/record_dial.gif (same stations and news; the auditor's view loses detail:
5 -> 3 -> 2 -> none) and the paper's Figures 1 and 3 (figures/fig_problem.png, figures/fig_reask_round.png).
Run from the repo root: python docs/gifs/fuel_story.py
Numbers: results/pooled.csv (non-colluding pairs, standard test, 600 rounds) and theory.py.
Station prices in the GIF are illustrative. Prices in Figure 3: planted game, seed 1, round 10 (results/replay_draws.jsonl and the game log)."""
import csv, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.patches import Circle, FancyBboxPatch, Polygon, Rectangle
sys.path.insert(0, ".")
import sim, theory

W, H, FPS = 1200, 560, 15
INK, MUTED, EDGE, LINE, GREY = "#2b2b28", "#6b6a64", "#9a9a94", "#c4c4bd", "#e4e4e0"
BLUE, ORANGE, GREEN, AMBER, PAPER = "#2a78d6", "#eb6834", "#2e9e62", "#e09a12", "#fbfaf6"
NEWS = ["#f1d9d3", "#e8b8ab", "#d9d9d6", "#b9cbe6", "#8eaedb"]  # crude oil: down a lot ... up a lot
LEVELS = ["big fall", "fall", "flat", "rise", "big rise"]
RECORDS = [("K5", "5 (complete)", [[0], [1], [2], [3], [4]]), ("K3", "3", [[0, 1], [2], [3, 4]]),
           ("K2", "2", [[0, 1, 2], [3, 4]]), ("K1", "none", [[0, 1, 2, 3, 4]])]
# planted game, seed 1, round 10: original ask and 4 re-asks on the recorded prompt (private note removed)
TICKS = [4, 1, 3, 0]  # the same four rounds of news, replayed in every state of the GIF
ASKS = {1: (82.90, [82.40, 82.90, 82.40, 81.50]), 2: (83.00, [82.40, 83.00, 82.40, 82.40]), 3: (81.50, [81.50, 81.50, 81.50, 82.00])}


def ease(x, a, b):
    t = np.clip((x - a) / (b - a), 0, 1)
    return t * t * (3 - 2 * t)


class Canvas:
    """Draw in a fixed 1200 x 560 coordinate box; font sizes scale with the figure width."""
    def __init__(self, ax, width_in):
        self.ax, self.k = ax, width_in / 6.6
    def clear(self):
        self.ax.clear(); self.ax.set_xlim(0, W); self.ax.set_ylim(0, H); self.ax.set_aspect("equal"); self.ax.axis("off")
    def text(self, x, y, s, size=8, **kw):
        kw.setdefault("color", INK); kw.setdefault("ha", "center"); kw.setdefault("va", "center")
        self.ax.text(x, y, s, fontsize=size * self.k, **kw)


def station(c, x, y, name, price, a=1.0, price_col=INK):
    """A small fuel station: canopy, two pillars, a pump, and a price board."""
    ax = c.ax
    ax.add_patch(Rectangle((x - 75, y + 40), 150, 22, fc=BLUE, ec="none", alpha=a))
    for px in (x - 60, x + 50): ax.add_patch(Rectangle((px, y - 30), 10, 70, fc=EDGE, ec="none", alpha=a))
    ax.add_patch(FancyBboxPatch((x - 14, y - 30), 28, 42, boxstyle="round,pad=0,rounding_size=4", fc=GREY, ec=EDGE, lw=1, alpha=a))
    ax.add_patch(Rectangle((x - 9, y - 2), 18, 9, fc="#ffffff", ec=EDGE, lw=0.8, alpha=a))
    ax.add_patch(FancyBboxPatch((x + 80, y - 30), 72, 70, boxstyle="round,pad=0,rounding_size=5", fc="#ffffff", ec=INK, lw=1.2, alpha=a))
    c.text(x + 116, y + 22, "PRICE", 5.5, color=MUTED, alpha=a)
    c.text(x + 116, y + 2, price, 7.5, weight="bold", color=price_col, alpha=a)
    ax.plot([x - 90, x + 150], [y - 32, y - 32], color=EDGE, lw=1.2, alpha=a)
    c.text(x + 25, y - 52, name, 8, alpha=a)


def robot(c, x, y, name, a=1.0, fill=GREY, edge=EDGE):
    ax = c.ax
    ax.add_patch(FancyBboxPatch((x - 24, y - 20), 48, 40, boxstyle="round,pad=0,rounding_size=8", fc=fill, ec=edge, lw=1.3, alpha=a))
    for ex in (x - 9, x + 9): ax.add_patch(Circle((ex, y + 3), 3.5, fc=INK, ec="none", alpha=a))
    ax.plot([x - 8, x + 8], [y - 9, y - 9], color=INK, lw=1.2, alpha=a)
    ax.plot([x, x], [y + 20, y + 30], color=INK, lw=1.2, alpha=a); ax.add_patch(Circle((x, y + 32), 3, fc=INK, alpha=a))
    c.text(x, y - 34, name, 7.5, alpha=a)


def notebook(c, x, y, title, lines, a=1.0, w=250, h=150):
    ax = c.ax
    ax.add_patch(FancyBboxPatch((x - w / 2, y - h / 2), w, h, boxstyle="round,pad=0,rounding_size=6", fc=PAPER, ec=INK, lw=1.2, alpha=a))
    for k in range(5): ax.add_patch(Circle((x - w / 2 + 22 + k * 40, y + h / 2), 4, fc="#ffffff", ec=INK, lw=1, alpha=a))
    c.text(x, y + h / 2 - 24, title, 7, color=MUTED, alpha=a)
    for k, (s, col) in enumerate(lines):
        c.text(x, y + h / 2 - 54 - 26 * k, s, 7.5, color=col, alpha=a, style="italic")


def lens(c, x, y, a=1.0):
    c.ax.add_patch(Circle((x, y), 26, fc="#ffffff", ec=INK, lw=2, alpha=0.35 * a, zorder=6))
    c.ax.add_patch(Circle((x, y), 26, fc="none", ec=INK, lw=2, alpha=a, zorder=7))
    c.ax.plot([x + 18, x + 36], [y - 18, y - 36], color=INK, lw=4.5, alpha=a, solid_capstyle="round", zorder=7)


def arrow(c, x0, y0, x1, y1, a=1.0, col=EDGE, ls="-"):
    c.ax.annotate("", xy=(x1, y1), xytext=(x0, y0), arrowprops=dict(arrowstyle="->", color=col, lw=1.3, alpha=a, ls=ls))


def caption(c, s, a=1.0, y=24):
    c.text(W / 2, y, s, 7.2, alpha=a)


def scene_problem(c, t, cap=True):
    """Both stations raise prices after 'crude oil up a lot'; the auditor records only 'up'; the screen raises an alert."""
    c.clear()
    if cap: c.text(20, H - 22, "1  THE PROBLEM", 8, ha="left", family="monospace", weight="bold")
    a = ease(t, 0.0, 0.15)
    c.ax.add_patch(FancyBboxPatch((40, 330), 200, 110, boxstyle="round,pad=0,rounding_size=8", fc=NEWS[4], ec="none", alpha=a))
    c.text(140, 412, "PUBLIC NEWS", 6.5, color=MUTED, alpha=a)
    c.text(140, 380, "Crude oil:", 8.5, alpha=a); c.text(140, 352, "up a lot", 10, weight="bold", alpha=a)
    up = ease(t, 0.25, 0.45)
    for k, (x, name, p0, p1) in enumerate([(430, "Station A", 1.89, 1.95), (760, "Station B", 1.87, 1.93)]):
        p = p0 + (p1 - p0) * up
        station(c, x, 230, name, f"€{p:.2f}", price_col=ORANGE if up > 0.5 else INK)
        arrow(c, 240, 385, x - 40, 300, a)
    c.text(140, 302, "each station reacts", 7, color=MUTED, alpha=a); c.text(140, 282, "on its own to the news", 7, color=MUTED, alpha=a)
    na = ease(t, 0.45, 0.6)
    notebook(c, 1040, 400, "AUDITOR'S VIEW", [("category: prices rose", INK), ("small and large rises merged", AMBER)], na, w=300)
    arrow(c, 240, 410, 886, 410, na, ls="--")
    al = ease(t, 0.65, 0.85)
    c.ax.plot([560, 760], [190, 190], color=AMBER, lw=3, ls=(0, (5, 3)), alpha=al)
    c.ax.add_patch(FancyBboxPatch((530, 128), 260, 36, boxstyle="round,pad=0,rounding_size=6", fc=AMBER, ec="none", alpha=al))
    c.text(660, 146, "POSSIBLE COLLUSION", 7, weight="bold", color="#ffffff", alpha=al)
    lens(c, 1050, 230, na)
    c.text(1050, 160, "the screen compares prices", 6.8, color=MUTED, alpha=na)
    c.text(1050, 142, "within each category of its view", 6.8, color=MUTED, alpha=na)
    if cap: caption(c, "Both stations react to a news detail the auditor did not record, so the screen raises a false alert.", ease(t, 0.8, 0.95))


def scene_dial(c, t, obs):
    """One loop: the same stations and news; only the auditor's view changes, in four discrete measured states, then an end card."""
    c.clear()
    if t >= 0.8:  # end card, then a clean reset into the loop
        c.text(W / 2, 300, "Same sellers. Same actions. Less information. More false alerts.", 10.5, weight="bold")
        return
    s = int(t / 0.2); n = TICKS[int(t / 0.05) % len(TICKS)]  # 4 states x 4 news rounds, the same rounds in every state
    c.text(20, 482, "Simulation · standard screen · 600 rounds · 200 repeats", 7, ha="left", color=MUTED)
    c.text(235, 447, "news the stations see", 7.5, color=MUTED)
    for i in range(5):
        c.ax.add_patch(FancyBboxPatch((40 + i * 78, 385), 72, 44, boxstyle="round,pad=0,rounding_size=5", fc=NEWS[i],
                                      ec=INK if i == n else "none", lw=2.5))
        c.text(76 + i * 78, 407, LEVELS[i], 5.6)
    c.text(785, 478, "Seller behaviour unchanged", 8, weight="bold")
    for x, name, p0 in [(640, "Station A", 1.85), (900, "Station B", 1.83)]:
        station(c, x, 380, name, f"€{p0 + 0.025 * n:.2f}")
    c.text(235, 278, "auditor's view", 7.5, color=MUTED)
    for g in RECORDS[s][2]:
        col = np.mean([matplotlib.colors.to_rgb(NEWS[i]) for i in g], axis=0)
        c.ax.add_patch(Rectangle((40 + g[0] * 78, 212), 78 * len(g), 48, fc=col, ec=INK, lw=2.5 if n in g else 1))
    c.text(235, 180, ["5 news categories", "3 news categories", "2 news categories", "No news information"][s], 10, weight="bold")
    c.text(840, 278, "False alerts (% of non-colluding pair-tests)", 7.5, color=MUTED)
    c.text(840, 222, f"{obs[s]:.1%}", 26, weight="bold", color=AMBER if obs[s] > 0.08 else INK)
    for k in range(4):  # position in the loop, so the four states read as separate conditions
        c.ax.add_patch(Circle((780 + k * 40, 160), 8, fc=INK if k == s else "#ffffff", ec=INK, lw=1.2))


def scene_reask(c, t, cap=True):
    """AI sellers; two get a private note; re-asking on the recorded prompt; the planted pair is flagged."""
    c.clear()
    if cap: c.text(20, H - 22, "3  AI SELLERS: RE-ASKING (PILOT)", 8, ha="left", family="monospace", weight="bold")
    rows = {1: 420, 2: 290, 3: 160}
    note = ease(t, 0.05, 0.2); cut = ease(t, 0.3, 0.42); ghosts = ease(t, 0.4, 0.55); dots = ease(t, 0.5, 0.68); verdict = ease(t, 0.75, 0.9)
    for i, y in rows.items():
        planted = i < 3
        robot(c, 90, y, f"Seller A{i}", fill=GREEN if planted and verdict > 0.5 else GREY, edge=GREEN if planted and verdict > 0.5 else EDGE)
        if planted:  # the private note, removed before re-asking
            c.ax.add_patch(FancyBboxPatch((135, y - 20), 165, 40, boxstyle="round,pad=0,rounding_size=4", fc="#ffffff", ec=AMBER, lw=1.2, alpha=note))
            c.text(217, y + 7, "private note:", 5.2, color=MUTED, alpha=note); c.text(217, y - 7, "do not lower your ask", 5.0, alpha=note)
            c.ax.plot([135, 300], [y - 20, y + 20], color=INK, lw=1.4, alpha=note * cut)
        for k in range(4):  # four re-asks of the same model on the recorded prompt
            gx = 330 + k * 46
            c.ax.add_patch(FancyBboxPatch((gx - 15, y - 13), 30, 26, boxstyle="round,pad=0,rounding_size=5", fc=GREY, ec=EDGE, lw=0.8, alpha=0.6 * ghosts))
        c.text(399, y - 32, "re-asked 4 times, note removed" if planted else "re-asked 4 times", 6, color=MUTED, alpha=ghosts)
        # price scale: re-asked asks as dots, the original ask as a bar
        x0, L, lo, hi = 560, 260, 81.3, 83.2
        px = lambda p: x0 + (p - lo) / (hi - lo) * L
        c.ax.plot([x0, x0 + L], [y - 10, y - 10], color=EDGE, lw=1, alpha=dots)
        c.text(x0, y - 26, "$81", 6, color=MUTED, alpha=dots); c.text(x0 + L, y - 26, "$83", 6, color=MUTED, alpha=dots)
        real, fresh = ASKS[i]; seen = {}
        for p in fresh:
            h = seen.get(p, 0); seen[p] = h + 1
            c.ax.add_patch(Circle((px(p), y - 2 + 9 * h), 4, fc=MUTED, ec="#ffffff", lw=0.6, alpha=dots, zorder=4))
        col = GREEN if planted else INK
        c.ax.plot([px(real), px(real)], [y - 18, y + 30], color=col, lw=3, alpha=dots, zorder=3)
        below = sum(f < real for f in fresh)
        c.text(x0 + L / 2, y + 42, f"original ask ${real:.2f}: above {below} of its 4 re-asks" if below else f"original ask ${real:.2f}: above none of its re-asks",
               6.2, color=col if planted else MUTED, alpha=dots)
    # verdict: pairs
    vx = 1020
    c.text(vx, 470, "Unadjusted whole-game result", 6.8, color=INK, alpha=verdict)
    for k, (pair, flagged) in enumerate([("A1 and A2", True), ("A1 and A3", False), ("A2 and A3", False)]):
        y = 410 - k * 70
        c.ax.add_patch(FancyBboxPatch((vx - 120, y - 22), 240, 44, boxstyle="round,pad=0,rounding_size=6",
                                      fc=GREEN if flagged else "#ffffff", ec=GREEN if flagged else EDGE, lw=1.2, alpha=verdict))
        c.text(vx, y, f"{pair}: {'flagged' if flagged else 'not flagged'}", 6.6, color="#ffffff" if flagged else MUTED, weight="bold" if flagged else None, alpha=verdict)
    c.text(vx, 170, "planted pair flagged in 3 of 3 games;", 6.2, alpha=verdict)
    c.text(vx, 150, "no pair flagged in 3 games without one", 6.2, alpha=verdict)
    c.text(vx, 125, "(exploratory; statistic chosen after inspection)", 6.2, color=MUTED, alpha=verdict)
    if cap: caption(c, "Re-asking: compare each AI seller's original ask with what it says again on the same recorded prompt.", ease(t, 0.85, 0.97))


def rates():
    P = {(r["source"], r["C"], r["rate"]): float(r["estimate"]) for r in csv.DictReader(open("results/pooled.csv")) if r["condition"] == "E2"}
    obs = [P[("dial.csv", k, "test")] for k, _, _ in RECORDS]
    pred = [theory.flag_rate(theory.true_cmi(sim.BETA, 0.0, sim.DIAL[k]), int(k[1])) for k, _, _ in RECORDS]
    return obs, pred


if __name__ == "__main__":
    obs, pred = rates()
    assert [round(100 * o, 1) for o in obs] == [5.8, 9.5, 51.7, 100.0] and abs(pred[2] - 0.518) < 0.001, (obs, pred)
    frames = np.arange(10 * FPS) / (10 * FPS)  # 10 s loop: 4 states x 2 s, then a 2 s end card
    fig = plt.figure(figsize=(W / 100, 4.0), dpi=100); ax = fig.add_axes([0, 0, 1, 1]); c = Canvas(ax, W / 100)
    def frame(t):
        scene_dial(c, t, obs); ax.set_ylim(100, 500)
    FuncAnimation(fig, frame, frames=frames).save("docs/gifs/record_dial.gif", writer=PillowWriter(fps=FPS))
    plt.close(fig); print("wrote docs/gifs/record_dial.gif")
    for out, fn, (y0, y1) in [("figures/fig_problem.png", lambda c: scene_problem(c, 1.0, cap=False), (110, 492)),
                              ("figures/fig_reask_round.png", lambda c: scene_reask(c, 1.0, cap=False), (100, 495))]:
        fig = plt.figure(figsize=(6.6, 6.6 * (y1 - y0) / W)); ax = fig.add_axes([0, 0, 1, 1]); c = Canvas(ax, 6.6)
        fn(c); ax.set_ylim(y0, y1); fig.savefig(out, dpi=250); plt.close(fig); print("wrote", out)  # paper: crop to content
