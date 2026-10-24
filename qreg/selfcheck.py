"""Fast invariants that must hold before any experiment result is believable.

    python -m qreg.selfcheck
"""

from __future__ import annotations

import numpy as np

from .decode import corner_error, estimate
from .evaluate import ransac_transform, warmup
from .features import gt_inlier_mask, match_candidates
from .qubo import FeatureConfig, QuboWeights, build_qubo, qubo_energy, to_ising
from .solvers import solve
from .synth import PairConfig, make_pair


def check_ising_equivalence(tol=1e-8):
    rng = np.random.default_rng(0)
    n = 12
    u = rng.normal(size=n)
    A = rng.normal(size=(n, n))
    J = np.triu(A, 1)
    J = J + J.T
    h, Js = to_ising(u, J, normalise=False)
    offs = []
    for _ in range(20):
        x = rng.integers(0, 2, n).astype(float)
        s = 2 * x - 1
        e_q = qubo_energy(u, J, x)
        e_i = h @ s + 0.5 * s @ Js @ s
        offs.append(e_q - e_i)
    assert np.ptp(offs) < 1e-8, f"Ising conversion is not a constant shift: {np.ptp(offs)}"
    return True


def check_pipeline():
    cfg = PairConfig(oblique_deg=30, photometric=0.7, seed=7)
    img1, img2, h_gt, _ = make_pair(cfg)
    assert img1.shape == img2.shape == (512, 512, 3)

    cand = match_candidates(img1, img2, n_max=160)
    assert 20 <= len(cand) <= 160, f"candidate count {len(cand)}"
    inl = gt_inlier_mask(cand, h_gt)
    assert 0.05 < inl.mean() <= 1.0, f"GT inlier fraction {inl.mean():.2f} is implausible"

    q = build_qubo(cand, QuboWeights())
    assert np.isfinite(q["u"]).all() and np.isfinite(q["J"]).all()
    assert np.allclose(q["J"], q["J"].T) and np.allclose(np.diag(q["J"]), 0)

    h, Js = to_ising(q["u"], q["J"])
    for m in ("greedy", "sa", "pt", "sqa"):
        res = solve(m, h, Js, budget=60_000, seed=1)
        assert set(np.unique(res["spins"])) <= {-1, 1}
        assert np.isfinite(res["energy"])
        H, n_used, rms, _ = estimate(cand, res["x"].astype(bool))
        assert n_used >= 4 and np.isfinite(rms)

    Hr = ransac_transform(cand)
    assert Hr is not None
    ce = corner_error(Hr, h_gt, img1.shape)
    ce_oracle = corner_error(estimate(cand, inl)[0], h_gt, img1.shape)
    assert np.isfinite(ce) and np.isfinite(ce_oracle)

    from .train import blackbox_grad, init_theta, prepare
    prep = prepare(cand, h_gt, img1.shape, FeatureConfig())
    g = blackbox_grad(init_theta(), prep, np.random.default_rng(0).normal(size=len(cand)),
                      method="pt", budget=60_000, lam=1.0)
    assert g.shape == (7,) and np.isfinite(g).all(), g
    return ce, ce_oracle, len(cand), inl.mean()


def main() -> int:
    check_ising_equivalence()
    print("ok  ising conversion is an exact constant shift")

    warmup()
    ce, ce_oracle, n, frac = check_pipeline()
    print(f"ok  pipeline: {n} candidates, GT inlier frac {frac:.2f}")
    print(f"    RANSAC corner err {ce:.2f}px   oracle(all GT inliers) {ce_oracle:.2f}px")
    print(f"    headroom available to a selector: {ce - ce_oracle:.2f}px")
    print("all checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
