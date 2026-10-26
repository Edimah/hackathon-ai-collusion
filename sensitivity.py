"""Global sensitivity analysis (Sobol indices) of the predicted test false-positive rate, theory only (theory.py).
Inputs: β ~ U[0.25, 2], λ ~ U[0, 1.5], information level uniform on {5, 3, 2, none}, log10 n ~ U[2, log10 6000].
Pick-freeze estimators: Saltelli 2010 (first order), Jansen 1999 (total); 6,000 base samples, 200 bootstrap resamples.
Run: python sensitivity.py -> results/sobol.csv, figures/figA_sobol.png"""
import csv, os
import numpy as np
import sim, theory

NAMES = ["beta (news strength)", "lambda (unrecorded shared input)", "information level", "rounds n"]
LEVELS = [("K5", 5), ("K3", 3), ("K2", 2), ("K1", 1)]


def sample(rng, N):
    """N x 4 inputs on their own scales: beta, lambda, level index 0..3, log10 n."""
    return np.column_stack([rng.uniform(0.25, 2, N), rng.uniform(0, 1.5, N), rng.integers(0, 4, N),
                            rng.uniform(2, np.log10(6000), N)])


def model(X, nodes=24):
    """Predicted test rejection rate for a pair of innocent sellers sharing the hidden term with weight λ."""
    out = np.empty(len(X))
    for i, (b, l, lev, lg) in enumerate(X):
        name, K = LEVELS[int(lev)]
        out[i] = theory.flag_rate(theory.true_cmi(b, l, sim.DIAL[name], nodes), K, int(round(10 ** lg)))
    return out


def indices(fA, fB, fAB):
    """First-order (Saltelli) and total (Jansen) indices from pick-freeze outputs; fAB[:, i] = model(A with column i from B)."""
    V = np.var(np.concatenate([fA, fB]))
    S1 = np.mean(fB[:, None] * (fAB - fA[:, None]), axis=0) / V
    ST = 0.5 * np.mean((fA[:, None] - fAB) ** 2, axis=0) / V
    return S1, ST


def sobol(N=6000, B=200, seed=0, f=model, sampler=sample, d=4):
    rng = np.random.default_rng(seed)
    A, Bm = sampler(rng, N), sampler(rng, N)
    fA, fB = f(A), f(Bm)
    fAB = np.column_stack([f(np.where(np.arange(d) == i, Bm, A)) for i in range(d)])
    S1, ST = indices(fA, fB, fAB)
    boot = [indices(fA[k], fB[k], fAB[k]) for k in (rng.integers(0, N, N) for _ in range(B))]
    lo1, hi1 = np.quantile([b[0] for b in boot], [0.025, 0.975], axis=0)
    loT, hiT = np.quantile([b[1] for b in boot], [0.025, 0.975], axis=0)
    return S1, lo1, hi1, ST, loT, hiT


if __name__ == "__main__":
    # self-check on Y = x1 + 2 x2, x ~ U[0, 1]: S1 = ST = (0.2, 0.8)
    s1, _, _, st, _, _ = sobol(20000, 20, f=lambda X: X[:, 0] + 2 * X[:, 1], sampler=lambda r, n: r.uniform(0, 1, (n, 2)), d=2)
    assert np.allclose(s1, [0.2, 0.8], atol=0.03) and np.allclose(st, [0.2, 0.8], atol=0.03), (s1, st)
    S1, lo1, hi1, ST, loT, hiT = sobol()
    os.makedirs("results", exist_ok=True)
    with open("results/sobol.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["input", "first_order", "first_lo", "first_hi", "total", "total_lo", "total_hi"])
        for row in zip(NAMES, S1, lo1, hi1, ST, loT, hiT):
            w.writerow(row); print(f"{row[0]:<34} first {row[1]:.2f} [{row[2]:.2f}, {row[3]:.2f}]  total {row[4]:.2f} [{row[5]:.2f}, {row[6]:.2f}]")
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    import figures as F
    fig, ax = plt.subplots(figsize=(5.0, 2.6))
    y = np.arange(4)[::-1]
    ax.barh(y + 0.18, ST, 0.34, xerr=[ST - loT, hiT - ST], color=F.TEST, label="total index")
    ax.barh(y - 0.18, S1, 0.34, xerr=[S1 - lo1, hi1 - S1], color=F.FLAG, label="first-order index")
    labels = ["news sensitivity β", "unrecorded shared input λ", "auditor's view (news categories)", "number of rounds n"]
    ax.set_yticks(y, labels, fontsize=8); ax.set_xlabel("Sobol index of the predicted false-positive rate")
    ax.legend(frameon=False, fontsize=7); fig.tight_layout(); fig.savefig("figures/figA_sobol.png", dpi=250)
    print("wrote results/sobol.csv, figures/figA_sobol.png")
