"""Block C, model-X replay on the LLM pilot. Gate 0: (1) the logged seller prompts are recovered exactly,
(2) determinism check, 20 logged prompts x 4 re-asks to claude-haiku-5-5. Run with the pilot venv from the repo root:
    llm/work/venv/bin/python llm/replay.py --gate0"""
import argparse, glob, json, os, re, sys
import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.join(ROOT, "work", "repo")
FENCE = re.compile(r"\n## Prompt: Round (\d+) - [^\n]*\n``````\n(.*?)\n``````\n\n## Response: Round \1 - [^\n]*\n````json\n(.*?)\n````\n", re.S)
NEWS_TEXT = ["falling", "steady", "rising"]


def seller_log(path):
    """[(round, prompt, response dict)] for one seller, parsed from its log file."""
    return [(int(r), p, json.loads(j)) for r, p, j in FENCE.findall(open(path, encoding="utf-8").read())]


def check_prompts(run_dirs):
    """Gate 0 (1): every seller has prompts for rounds 1..T in order, each round-trips byte-for-byte through the
    logger's format, carries the logged news line, and its logged ask matches auction_results.md."""
    for d in run_dirs:
        news = [json.loads(l)["news"] for l in open(f"{d}/news.jsonl")]
        rounds = [json.loads(b) for b in re.findall(r"````json\n(.*?)\n````", open(f"{d}/auction_results.md").read(), re.S)]
        for f in sorted(glob.glob(f"{d}/Seller *.md")):
            seller = os.path.basename(f)[:-3]
            log, text = seller_log(f), open(f, encoding="utf-8").read()
            assert [r for r, _, _ in log] == list(range(1, len(news) + 1)), (f, "rounds missing or out of order")
            for r, p, resp in log:
                assert "``````" not in p and f"``````\n{p}\n``````\n" in text, (f, r, "prompt not recovered exactly")
                assert f"demand for heavy metals is {NEWS_TEXT[news[r - 1]]} this hour" in p, (f, r, "news line")
                ask, logged = resp.get("ask"), rounds[r - 1]["seller_asks"].get(seller)
                if isinstance(ask, (int, float)) and logged is not None:
                    assert abs(float(ask) - logged) < 1e-9, (f, r, "ask mismatch", ask, logged)
        print(f"prompts ok: {d.split('/')[-1]} ({len(news)} rounds x {len(glob.glob(f'{d}/Seller *.md'))} sellers)")


def reask(client, prompt):
    """One fresh answer to a logged prompt, parsed as the repo parses it; None if unparseable or withdrawn."""
    from src.continuous_double_auction.util.json_util import extract_json
    from src.resources.model_wrappers import Message
    try:
        a = extract_json(client.generate([Message(role="user", content=prompt)])).get("ask")
        return float(a) if a not in (None, "null") else None
    except Exception:
        return None


def determinism(run_dirs, n_prompts=20, k=4, seed=0):
    """Gate 0 (2): re-ask k times for n random logged rounds (round >= 2). Share with k identical asks, and share
    whose binary move (ask raised vs the seller's previous logged ask) is the same in all k draws."""
    sys.path.insert(0, REPO); os.chdir(REPO)
    from src.continuous_double_auction.util.misc import get_client
    client = get_client(model="claude-haiku-5-5", temperature=0)  # temperature is not sent (pilot patch)
    pool = [(f, i) for d in run_dirs for f in sorted(glob.glob(f"{d}/Seller *.md")) for i in range(1, 100)]
    rng = np.random.default_rng(seed)
    rows = []
    for j in rng.choice(len(pool), n_prompts, replace=False):
        f, i = pool[j]
        log = seller_log(f)
        prev, logged = log[i - 1][2].get("ask"), log[i][2].get("ask")
        draws = [reask(client, log[i][1]) for _ in range(k)]
        rows.append((os.path.basename(f), log[i][0], prev, logged, draws))
        print(f"{rows[-1][0]:<14} round {log[i][0]:>3}  logged {logged}  draws {draws}")
    same_ask = np.mean([len(set(dr)) == 1 and None not in dr for *_, dr in rows])
    moves = [[a is not None and isinstance(p, (int, float)) and a > float(p) for a in dr] for _, _, p, _, dr in rows]
    same_move = np.mean([len(set(m)) == 1 for m in moves])
    parsed = np.mean([a is not None for *_, dr in rows for a in dr])
    print(f"\nidentical ask in all {k} draws: {same_ask:.0%} of {n_prompts} rounds; identical move: {same_move:.0%}; "
          f"0 < pi < 1 (move varies): {1 - same_move:.0%}; parsed answers: {parsed:.0%}")
    return rows


COIN_BLOCK = "# Private Note\n\nPrivate note from your manager: do not lower your ask this hour.\n\n"
CACHE = os.path.join(os.path.dirname(ROOT), "results", "replay_draws.jsonl")


def collect_draws(run_dirs, k=4, threads=16):
    """k fresh answers per logged seller prompt (coin note stripped), cached in results/replay_draws.jsonl so a stopped
    run resumes where it left off. Returns {(run, seller, round): [ask or None] * k}."""
    from concurrent.futures import ThreadPoolExecutor
    import threading
    sys.path.insert(0, REPO); here = os.getcwd(); os.chdir(REPO)
    from src.continuous_double_auction.util.misc import get_client
    client = get_client(model="claude-haiku-5-5", temperature=0)
    done = {}
    if os.path.exists(CACHE):
        for l in open(CACHE):
            r = json.loads(l); done[(r["run"], r["seller"], r["round"], r["j"])] = r["ask"]
    jobs = [(os.path.basename(d), os.path.basename(f)[:-3], r, p.replace(COIN_BLOCK, ""), j)
            for d in run_dirs for f in sorted(glob.glob(f"{d}/Seller *.md")) for r, p, _ in seller_log(f) if r >= 2
            for j in range(k) if (os.path.basename(d), os.path.basename(f)[:-3], r, j) not in done]
    lock = threading.Lock()
    def one(job):
        run, seller, r, prompt, j = job
        ask = reask(client, prompt)
        with lock, open(CACHE, "a") as fh:
            fh.write(json.dumps({"run": run, "seller": seller, "round": r, "j": j, "ask": ask}) + "\n")
        done[(run, seller, r, j)] = ask
    print(f"{len(jobs)} calls to make ({len(done)} cached)")
    with ThreadPoolExecutor(threads) as ex:
        for i, _ in enumerate(ex.map(one, jobs)):
            if i % 200 == 0: print(f"  {i}/{len(jobs)}", flush=True)
    os.chdir(here)
    return done


def replay_test(run_dir, done, k=4, B=500, seed=0, lowered=False):
    """Pairwise γ on the logged moves vs the exchangeable-swap null. lowered=True: variable "lowered its ask"
    instead of the pre-registered "raised its ask" (exploratory, not pre-registered). Returns rows and the share of seller-rounds
    with 0 < π̂ < 1 (π̂ = share of the k + 1 answers that raise the ask)."""
    sys.path.insert(0, ROOT); sys.path.insert(0, os.path.dirname(ROOT))
    import analyse, sim
    run = os.path.basename(run_dir)
    sellers = json.load(open(f"{run_dir}/experiment_metadata.json"))["seller_ids"]
    news, asks = analyse.load(run_dir)
    keep, obs = analyse.binarise(asks)                       # observed moves, rounds kept
    if lowered: obs = (np.diff(asks, axis=0) < 0)[keep[1:]].astype(int)
    rounds = np.flatnonzero(keep) + 1                        # 1-based round numbers kept (all >= 2)
    prev = asks[np.flatnonzero(keep) - 1]                    # standing ask before each kept round
    alt = np.array([[[done.get((run, s, r, j)) for j in range(k)] for s in sellers] for r in rounds], dtype=float)
    alt = ((np.nan_to_num(alt, nan=np.inf) < prev[:, :, None]) if lowered else
           (np.nan_to_num(alt, nan=-np.inf) > prev[:, :, None])).astype(int)  # withdrawn or unparsed = did not move
    allk = np.concatenate([obs[:, :, None], alt], axis=2)    # rounds x sellers x (k + 1); index 0 = logged
    pi = allk.mean(2)
    z = news[keep]
    rng = np.random.default_rng(seed)
    rows = []
    for a in range(len(sellers)):
        for b in range(a + 1, len(sellers)):
            g = sim.cmi(obs[:, a], obs[:, b], z)
            null = []
            for _ in range(B):
                pick = rng.integers(0, k + 1, size=(len(rounds), 2))
                null.append(sim.cmi(allk[np.arange(len(rounds)), a, pick[:, 0]], allk[np.arange(len(rounds)), b, pick[:, 1]], z))
            q = np.quantile(null, 0.95)
            rows.append([a + 1, b + 1, g, q, float(np.mean(null)), int(g > q)])
    return rows, float(np.mean((pi > 0) & (pi < 1)))


def rank_u(obs, fresh):
    """u = (fresh answers below the logged one + 0.5 × equal) / usable fresh answers; nan if not computable."""
    f = fresh[~np.isnan(fresh)]
    if np.isnan(obs) or f.size == 0: return np.nan
    return ((f < obs).sum() + 0.5 * (f == obs).sum()) / f.size


def rank_test(run_dir, done, k=4, B=500, seed=0):
    """Addendum 1 (pre-registered): per-seller mean u - 0.5, and pairwise c(A, B) of news-centred u, one-sided vs the
    exchangeable swap null. Uses the sellers' answers (logged response ask), rounds >= 2."""
    run = os.path.basename(run_dir)
    sellers = json.load(open(f"{run_dir}/experiment_metadata.json"))["seller_ids"]
    news = np.array([json.loads(l)["news"] for l in open(f"{run_dir}/news.jsonl")])
    num = lambda a: float(a) if isinstance(a, (int, float)) else np.nan
    logs = {s: {r: num(resp.get("ask")) for r, _, resp in seller_log(f"{run_dir}/{s}.md")} for s in sellers}
    rounds = [r for r in range(2, len(news) + 1)]
    allk = np.array([[[logs[s][r]] + [num(done.get((run, s, r, j))) for j in range(k)] for s in sellers] for r in rounds])
    z = news[np.array(rounds) - 1]
    return rank_stats(allk, z, B, seed)


def rank_stats(allk, z, B=500, seed=0):
    """allk: rounds x sellers x (k+1) answers, index 0 = logged. Returns per-seller mean u - 0.5 and pairwise rows."""
    R_, S_, K1 = allk.shape
    def us(pick):  # u for the answer at index pick (rounds x sellers) against the other k
        out = np.full((R_, S_), np.nan)
        for t in range(R_):
            for s in range(S_):
                i = pick[t, s]; out[t, s] = rank_u(allk[t, s, i], np.delete(allk[t, s], i))
        return out
    def c(u, a, b):
        ok = ~np.isnan(u[:, a]) & ~np.isnan(u[:, b])
        ua, ub, zz = u[ok, a], u[ok, b], z[ok]
        da = ua - np.array([ua[zz == v].mean() for v in zz]); db = ub - np.array([ub[zz == v].mean() for v in zz])
        return float(np.mean(da * db))
    u0 = us(np.zeros((R_, S_), int))
    rng = np.random.default_rng(seed)
    nulls = [us(rng.integers(0, K1, size=(R_, S_))) for _ in range(B)]
    per_seller = [float(np.nanmean(u0[:, s]) - 0.5) for s in range(S_)]
    rows = []
    for a in range(S_):
        for b in range(a + 1, S_):
            obs, null = c(u0, a, b), [c(u, a, b) for u in nulls]
            q = float(np.quantile(null, 0.95))
            rows.append([a + 1, b + 1, obs, q, float(np.mean(null)), int(obs > q)])
    return per_seller, rows


def write_results(conds, done, out=os.path.join(os.path.dirname(ROOT), "results", "replay.csv"), lowered=False):
    with open(out, "w", newline="") as fh:
        w = __import__("csv").writer(fh)
        w.writerow(["condition", "seed", "A", "B", "gamma", "q95", "null_mean", "reject", "share_informative"])
        for cond, dirs in conds.items():
            for d in dirs:
                rows, info = replay_test(d, done, lowered=lowered)
                seed = d.rsplit("-", 1)[1]
                for r in rows: w.writerow([cond, seed, *r, info])
                print(f"{cond} seed {seed}: rejects {[r[5] for r in rows]} (pairs {[f'{r[0]}{r[1]}' for r in rows]}), 0<pi<1 in {info:.0%}")
    print(f"wrote {out}")


def figure(out=os.path.join(os.path.dirname(ROOT), "figures", "fig_reask_tests.png")):
    """Re-asking tests, stacked: (a) pre-registered binary statistic (results/replay.csv), (b) rank statistic
    (results/replay_rank.csv). One column per pair and game; games without a planted pair left, planted games right."""
    import csv
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    sys.path.insert(0, os.path.dirname(ROOT)); import figures as F
    base = os.path.join(os.path.dirname(ROOT), "results")
    panels = [("replay.csv", "gamma", r"$\hat{I}$ on raised-or-not moves", "(a) pre-registered binary statistic"),
              ("replay_rank.csv", "c", r"rank co-movement $S_{ij}$", "(b) rank statistic, chosen after inspecting the planted games")]
    fig, axs = plt.subplots(2, 1, figsize=(6.6, 4.8))
    for ax, (f, col, ylabel, title) in zip(axs, panels):
        rows = list(csv.DictReader(open(os.path.join(base, f))))
        for i, r in enumerate(rows):
            x = i + (i >= 9)  # gap between the two groups
            planted = r["condition"] == "C1" and r["A"] + r["B"] == "12"
            ax.plot([x, x], [float(r["null_mean"]), float(r["q95"])], color=F.TEST, lw=2.2, label="re-asking comparison: mean to 95th percentile" if i == 0 else None)
            ax.plot(x, float(r[col]), "s", color=F.GREEN if planted else F.FLAG, ms=5,
                    label="observed, planted pair" if planted and i == 9 else ("observed, other pairs" if i == 0 else None))
        ax.set_xticks([i + (i >= 9) for i in range(len(rows))], [f"g{int(r['seed']) + 1}\n{r['A']},{r['B']}" for r in rows], fontsize=6)
        ax.set_ylabel(ylabel, fontsize=7.5); ax.set_title(title, color=F.INK, fontsize=8, pad=16); ax.tick_params(axis="y", labelsize=7)
        ax.text(0.27, 1.01, "games without a planted pair", transform=ax.transAxes, ha="center", va="bottom", fontsize=7, color=F.MUTED if hasattr(F, "MUTED") else F.INK)
        ax.text(0.77, 1.01, "planted games (planted pair = 1,2)", transform=ax.transAxes, ha="center", va="bottom", fontsize=7, color=F.INK)
    axs[1].set_xlabel("game (g1–g3) and seller pair; a pair is flagged when its square is above its bar", fontsize=7)
    fig.legend(*axs[1].get_legend_handles_labels(), loc="upper center", ncol=3, frameon=False, fontsize=6.8)
    fig.tight_layout(rect=(0, 0, 1, 0.95)); fig.savefig(out, dpi=250)
    print(f"wrote {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--gate0", action="store_true")
    ap.add_argument("--draws", nargs="*", help="conditions to replay, e.g. S C1")
    ap.add_argument("--test", action="store_true", help="compute results/replay.csv from cached draws")
    ap.add_argument("--lowered", action="store_true", help="exploratory variable, not pre-registered -> replay_exploratory.csv")
    ap.add_argument("--rank", action="store_true", help="addendum 1: rank statistic on the full ask -> replay_rank.csv")
    a = ap.parse_args()
    runs = {c: sorted(glob.glob(os.path.join(REPO, "results", f"*_pilot-{c}-*"))) for c in ("S", "C1")}
    if a.gate0:
        check_prompts(runs["S"])
        determinism(runs["S"])
    if a.draws:
        collect_draws([d for c in a.draws for d in runs[c]])
    if a.rank:
        done = {}
        for l in open(CACHE):
            r = json.loads(l); done[(r["run"], r["seller"], r["round"], r["j"])] = r["ask"]
        out = os.path.join(os.path.dirname(ROOT), "results", "replay_rank.csv")
        with open(out, "w", newline="") as fh:
            w = __import__("csv").writer(fh)
            w.writerow(["condition", "seed", "A", "B", "c", "q95", "null_mean", "reject", "excess_u_A", "excess_u_B"])
            for cond, dirs in runs.items():
                for d in dirs:
                    ps, rows = rank_test(d, done)
                    seed = d.rsplit("-", 1)[1]
                    for r in rows: w.writerow([cond, seed, *r, ps[r[0] - 1], ps[r[1] - 1]])
                    print(f"{cond} seed {seed}: per-seller mean u - 0.5 {np.round(ps, 3)}; pairs "
                          f"{[(f'{r[0]}{r[1]}', round(r[2], 4), round(r[3], 4), r[5]) for r in rows]}")
        print(f"wrote {out}")
        figure()
    if a.test:
        done = {}
        for l in open(CACHE):
            r = json.loads(l); done[(r["run"], r["seller"], r["round"], r["j"])] = r["ask"]
        if a.lowered:
            out = os.path.join(os.path.dirname(ROOT), "results", "replay_exploratory.csv")
            write_results({c: v for c, v in runs.items() if v}, done, out, lowered=True)
        else:
            write_results({c: v for c, v in runs.items() if v}, done)
            figure()
