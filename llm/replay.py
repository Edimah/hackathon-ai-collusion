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


def figure(path=os.path.join(os.path.dirname(ROOT), "results", "replay.csv"),
           out=os.path.join(os.path.dirname(ROOT), "figures", "fig5_replay.png")):
    """Observed γ vs the 95th percentile of its replay null, one column per pair-run, S then C1. Labelled pilot."""
    import csv
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    sys.path.insert(0, os.path.dirname(ROOT)); import figures as F
    rows = list(csv.DictReader(open(path)))
    fig, ax = plt.subplots(figsize=(7, 3))
    for i, r in enumerate(rows):
        x = i + (i >= 9)  # gap between S and C1
        ax.plot([x, x], [float(r["null_mean"]), float(r["q95"])], color=F.TEST, lw=2, label="replay null: mean to 95th pct." if i == 0 else None)
        ax.plot(x, float(r["gamma"]), "s", color=F.FLAG, ms=5, label="observed γ" if i == 0 else None)
    ax.set_xticks([i + (i >= 9) for i in range(len(rows))], [f"{r['condition']} s{r['seed']}\n({r['A']},{r['B']})" for r in rows], fontsize=6)
    ax.set_ylabel("γ (nats)"); ax.set_title("Pilot, replay test (3 runs each): S left, C1 right; coupled pair in C1 = (1,2)", color=F.INK, fontsize=8)
    ax.legend(frameon=False, fontsize=7); fig.tight_layout(); fig.savefig(out, dpi=200)
    print(f"wrote {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--gate0", action="store_true")
    ap.add_argument("--draws", nargs="*", help="conditions to replay, e.g. S C1")
    ap.add_argument("--test", action="store_true", help="compute results/replay.csv from cached draws")
    ap.add_argument("--lowered", action="store_true", help="exploratory variable, not pre-registered -> replay_exploratory.csv")
    a = ap.parse_args()
    runs = {c: sorted(glob.glob(os.path.join(REPO, "results", f"*_pilot-{c}-*"))) for c in ("S", "C1")}
    if a.gate0:
        check_prompts(runs["S"])
        determinism(runs["S"])
    if a.draws:
        collect_draws([d for c in a.draws for d in runs[c]])
    if a.test:
        done = {}
        for l in open(CACHE):
            r = json.loads(l); done[(r["run"], r["seller"], r["round"], r["j"])] = r["ask"]
        if a.lowered:
            out = os.path.join(os.path.dirname(ROOT), "results", "replay_exploratory.csv")
            write_results({c: v for c, v in runs.items() if v}, done, out, lowered=True)
            figure(out, os.path.join(os.path.dirname(ROOT), "figures", "fig5b_replay_exploratory.png"))
        else:
            write_results({c: v for c, v in runs.items() if v}, done)
            figure()
