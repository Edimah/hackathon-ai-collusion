"""Collusion detector (Bonjour et al. 2022) on scripted pricing agents. See README.md."""
import argparse, csv, itertools, os
import numpy as np

T, SEEDS, SHUFFLES = 600, 200, 500
BETA, KAPPA, LAM = 1.0, 0.2, 1.0  # set by Edimah, 24 Oct 
IDENTITY, CONSTANT = np.arange(6), np.zeros(6, int)  # C as a lookup table: N_obs = C[N]
DIAL = {"K5": IDENTITY, "K3": np.array([0, 0, 0, 1, 2, 2]), "K2": np.array([0, 0, 0, 0, 1, 1]), "K1": CONSTANT}  # adjacent merges: {12}{3}{45}, {123}{45}
PAIRS = list(itertools.combinations(range(4), 2))  # 6 pairs (A, B) with A < B, agents 0..3 = agents 1..4; one test per pair (Edimah, 25 Oct)


def simulate(rng, T, beta, kappa=0.0, lam=0.0):
    """One run: T x 5 int table, column 0 = news N_t in 1..5, columns 1..4 = prices (0 low, 1 high)."""
    N = rng.integers(1, 6, T)
    logit = np.repeat(beta * (N - 3.0)[:, None], 4, axis=1)
    logit[:, [0, 3]] += lam * rng.standard_normal(T)[:, None]  # shared hidden model term, agents 1 and 4
    p = (rng.random((T, 4)) < 1 / (1 + np.exp(-logit))).astype(int)
    copy, coin = rng.random(T) < kappa, rng.integers(0, 2, T)
    p[copy, 1] = p[copy, 2] = coin[copy]  # colluding pair = agents 2 and 3
    return np.column_stack([N, p])


def cmi(x, y, z):
    
    """Plug-in estimate of I(x; y | z) from counts, in nats (Bonjour et al. Eq. 5-6).
    x, y, z: 1-D int arrays of equal length (x, y in {0, 1}; z any small non-negative ints)."""
    
    I = 0.0                                         # cmi variable
    for v in np.unique(z):                          # each possible news value
        xs, ys = x[z == v], y[z == v]               # rounds with that news
        pz = len(xs) / len(x)                       # p̂(z = v)
        pxy = np.bincount(2 * xs + ys, minlength=4).reshape(2, 2) / len(xs)  # p̂(x, y | v), 2x2: codes each (price A, price B) combination as one integer 0..3; minlength=4 counts missing combinations as zero
        px, py = pxy.sum(1), pxy.sum(0)             # marginals p̂(x | v), p̂(y | v)
        nz = pxy > 0                                # skip empty cells: 0 log 0 = 0 by convention (log(0) would give nan and a warning)
        I += pz * np.sum(pxy[nz] * np.log(pxy[nz] / np.outer(px, py)[nz]))
    return I # conditional mutual information (mutual information of agents x and y, conditioned on news z)


def shuffle_within(y, z, rng):
    """Return a copy of y permuted at random within each value of z (z itself unchanged)."""

    y2 = y.copy()                                   # safeguard to never modify B's original column
    for v in np.unique(z):                          # each news value
        idx = np.flatnonzero(z == v)                # rounds with that news
        y2[idx] = y[rng.permutation(idx)]           # B's prices reshuffled among those rounds only
    return y2 # new shuffled column of prices for agent B (preserves the distribution of prices for each news value v)


def flag(gamma, q95, a, b):
    """Flag rule for the ordered pair (a, b).
    gamma[i, j] = γ(i; j), q95[i, j] = 95th percentile of the permutation null of γ(i; j).
    Γ(i; j) = γ(i; j) - max over k not in {i, j} of γ(k; j).
    Flag if γ(a; b) > q95[a, b] and Γ(a; b) > 0 and Γ(b; a) > 0."""
    others = [k for k in range(len(gamma)) if k not in (a, b)]   # any agent other than A and B
    G_ab = gamma[a, b] - gamma[others, b].max()         # Γ(a; b): does a explain b better than any third agent?
    G_ba = gamma[b, a] - gamma[others, a].max()         # Γ(b; a): the same, the other way round
    return gamma[a, b] > q95[a, b] and G_ab > 0 and G_ba > 0 # returns if we should flag the ordered pair (a, b) as "breaking competition law" (could be collusion or not, but data is flagged)



def detect(L, C, rng, shuffles):
    """γ, null 95th percentile (shuffling B) and null mean for the 6 pairs A < B, then the 6 flags."""
    z, P = C[L[:, 0]], L[:, 1:]
    gamma, q95, null_mean = np.zeros((4, 4)), np.zeros((4, 4)), np.zeros((4, 4))
    for a, b in PAIRS:
        gamma[a, b] = gamma[b, a] = cmi(P[:, a], P[:, b], z)  # symmetric; Γ needs both halves
        null = [cmi(P[:, a], shuffle_within(P[:, b], z, rng), z) for _ in range(shuffles)]
        q95[a, b], null_mean[a, b] = np.quantile(null, 0.95), np.mean(null)
    return gamma, q95, null_mean, {(a, b): flag(gamma, q95, a, b) for a, b in PAIRS}


def selftest():
    rng = np.random.default_rng(0)
    L = simulate(rng, 20000, beta=1.0, kappa=0.5, lam=0.0)
    assert L.shape == (20000, 5) and set(np.unique(L[:, 0])) == {1, 2, 3, 4, 5}
    assert np.mean(L[:, 2] == L[:, 3]) > 0.7 > np.mean(L[:, 1] == L[:, 4])  # pair (2,3) agrees more
    x, y, z = rng.integers(0, 2, 20000), rng.integers(0, 2, 20000), rng.integers(0, 5, 20000)
    assert 0 <= cmi(x, y, z) < 0.01                      # independent columns: γ ≈ 0
    assert abs(cmi(x, x, z) - np.log(2)) < 0.01          # y = x, fair coin: γ = H(x) = ln 2
    assert cmi(z % 2, z % 2, z) < 1e-12                  # common cause fully seen: γ = 0
    y2 = shuffle_within(y, z, rng)
    assert all(np.array_equal(np.sort(y2[z == v]), np.sort(y[z == v])) for v in range(5))
    assert not np.array_equal(y2, y) and cmi(x, shuffle_within(x, z, rng), z) < 0.01
    g, q = np.full((4, 4), 0.01), np.full((4, 4), 0.02)
    g[0, 1] = g[1, 0] = 0.1
    assert flag(g, q, 0, 1) and flag(g, q, 1, 0) and not flag(g, q, 2, 3)
    g[2, 1] = 0.2  # agent 3 explains agent 2 better than agent 1 does: Γ(1;2) < 0
    assert not flag(g, q, 0, 1)
    print("selftest: pass")


def show(seed=0, rows=8):
    """Print the first rows of one run per condition, so the data is concrete."""
    for name, beta, kappa, lam in CONDITIONS:
        L = simulate(np.random.default_rng(seed), T, beta, kappa, lam)
        print(f"{name} (beta={beta}, kappa={kappa}, lambda={lam})\n  N  p1 p2 p3 p4")
        for r in L[:rows]:
            print("  " + "  ".join(map(str, r)))


def run(T, seeds, shuffles, out="results/runs.csv", views=None, only=None, conds=None):
    """Every condition x auditor view x seed: one CSV row per pair with γ, null q95, null mean, flag.
    views: {name: C} for every condition (default: identity, plus constant for E2 and E4); only: condition names to run; conds: replaces CONDITIONS."""
    os.makedirs("results", exist_ok=True)
    with open(out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["condition", "C", "seed", "A", "B", "gamma", "q95", "null_mean", "flag"])
        for i, (name, beta, kappa, lam) in enumerate(conds or CONDITIONS):
            if only and name not in only: continue
            for cname, C in (views or ({"identity": IDENTITY, "constant": CONSTANT} if name in ("E2", "E4") else {"identity": IDENTITY})).items():
                for seed in range(min(seeds, 50) if name == "E1" else seeds):  # ponytail: E1 is a calibration test, at most 50 seeds
                    rng = np.random.default_rng([seed, i])  # same seed, same data under every C (dial K5/K1 = runs identity/constant)
                    g, q, m, fl = detect(simulate(rng, T, beta, kappa, lam), C, rng, shuffles)
                    for a, b in PAIRS:
                        w.writerow([name, cname, seed, a + 1, b + 1, g[a, b], q[a, b], m[a, b], int(fl[a, b])])
    print(f"wrote {out}")


CONDITIONS = [("E1", 0.0, 0.0, 0.0), ("E2", BETA, 0.0, 0.0), ("E2b", BETA, 0.0, LAM),
              ("E3", 0.0, KAPPA, 0.0), ("E4", BETA, KAPPA, 0.0)]
LAMBDAS = [("E2b-lambda%.2f" % l, BETA, 0.0, l) for l in np.arange(0, 2.01, 0.25)]  # Fig 2 sweep

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--test", action="store_true")
    ap.add_argument("--show", action="store_true")
    ap.add_argument("--debug", action="store_true", help="3 seeds, 100 rounds")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--dial", action="store_true", help="observability dial only")
    ap.add_argument("--lam", action="store_true", help="λ sweep only")
    a = ap.parse_args()
    if a.test: selftest()
    if a.show: show()
    if a.debug: run(100, 3, SHUFFLES, "results/debug.csv")
    if a.all: run(T, 100, SHUFFLES, "results/accept.csv"); run(T, SEEDS, SHUFFLES)  # acceptance checks (100 seeds), then the full grid
    if a.all or a.lam: run(T, SEEDS, SHUFFLES, "results/lambda.csv", views={"identity": IDENTITY}, conds=LAMBDAS)  # λ sweep
    if a.all or a.dial: run(T, SEEDS, SHUFFLES, "results/dial.csv", views=DIAL, only=("E2", "E4"))  # observability dial
