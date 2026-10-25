"""Theory lines from the generative model (not fitted): exact CMI, predicted flag rate, null bias. See sim.py."""
import numpy as np
from scipy.stats import chi2, ncx2

R = C_ = 2  # price levels per agent


def true_cmi(beta, lam, C, nodes=80):
    """Exact I(p_i; p_j | C(N)) in nats for two innocent agents sharing the hidden term with weight lam
    (lam = 0: two plain innocents). N uniform on 1..5, m ~ N(0, 1) integrated by Gauss-Hermite."""
    m, w = np.polynomial.hermite_e.hermegauss(nodes)
    w = w / w.sum()
    joint = {}  # (N_obs) -> 2x2 table of P(x, y, N_obs)
    for n in range(1, 6):
        q = 1 / (1 + np.exp(-(beta * (n - 3) + lam * m)))  # P(high | n, m), same for both agents
        pq = np.array([[w @ ((1 - q) * (1 - q)), w @ ((1 - q) * q)], [w @ (q * (1 - q)), w @ (q * q)]]) / 5
        joint[C[n]] = joint.get(C[n], 0) + pq
    I = 0.0
    for t in joint.values():
        px, py, pz = t.sum(1), t.sum(0), t.sum()
        I += np.sum(t * np.log(t * pz / np.outer(px, py)))
    return I


def flag_rate(I, K, n=600, alpha=0.05):
    """Predicted rejection rate of the permutation test: 2n·Î ≈ non-central χ²_d(2nI), d = K(r-1)(c-1).
    shortcut: ignores the Γ > 0 conditions of the flag rule, so it is the test's power only."""
    d = K * (R - 1) * (C_ - 1)
    return 1 - ncx2.cdf(chi2.ppf(1 - alpha, d), d, 2 * n * I) if I > 0 else alpha


def null_bias(K, n=600):
    """Expected plug-in γ under the null: K(r-1)(c-1)/(2n)."""
    return K * (R - 1) * (C_ - 1) / (2 * n)


if __name__ == "__main__":
    ident = np.arange(6)
    assert true_cmi(1.0, 0.0, ident) < 1e-12                      # news seen, no hidden term: independent
    assert true_cmi(1.0, 0.0, np.zeros(6, int)) > 0.01            # news hidden: common cause shows up
    assert true_cmi(1.0, 1.0, ident) > true_cmi(1.0, 0.5, ident)  # stronger hidden term, more dependence
    assert abs(flag_rate(0.0, 5) - 0.05) < 1e-12 and abs(null_bias(5) - 5 / 1200) < 1e-15
    for lam in np.arange(0, 2.01, 0.25):
        I = true_cmi(1.0, lam, ident)
        print(f"lambda={lam:.2f}  I={I:.5f}  predicted flag rate={flag_rate(I, 5):.1%}")
