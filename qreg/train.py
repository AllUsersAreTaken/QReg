"""Phase 2: learn the QUBO against registration quality (not against energy).

The energy is linear in theta by construction (see `qubo.assemble`):

    E(x; theta) = <theta_u, U(x)> + <theta_p, P(x)>

Two consequences matter:

1. The solver becomes a black box mapping theta -> argmin_x E(x; theta). Because
   the objective is a *linear* function of theta, derivatives can be pushed through
   it without touching the solver (see `blackbox_grad`).
2. The loss is registration error of the decoded transform, NOT the energy. That is
   the whole point: a hand-tuned energy has near-degenerate optima whose quality
   varies wildly (see `evaluate.energy_quality_rank_corr`), so minimising energy is
   the wrong objective.

`train_es` is the safe default for a handful of parameters: antithetic, rank-based
evolution strategies. `blackbox_grad` is the solver-in-the-loop alternative.
"""

from __future__ import annotations

import json
from dataclasses import dataclass

import numpy as np

from .decode import corner_error, estimate
from .features import Candidates, gt_inlier_mask, inject_outliers, match_candidates
from .qubo import FeatureConfig, QuboWeights, assemble, build_features, theta_from_weights, to_ising
from .solvers import solve

THETA_NAMES = ("app", "card_lin", "one2one", "geo_hinge", "geo_attr", "spread", "card_quad")
# Broad, asymmetric box: penalties are non-negative, the attraction channel is
# non-positive. The cardinality channels carry a large linear offset, hence the
# wide range; ES steps are scaled per channel (see `train_es`).
LOWER = np.array([0.0, -1e4, 0.0, 0.0, -1e4, 0.0, 0.0])
UPPER = np.array([1e3, 1e4, 1e3, 1e3, 0.0, 1e3, 1e4])


@dataclass
class Prepared:
    cand: Candidates
    feats: dict
    h_gt: np.ndarray
    shape: tuple
    inl: np.ndarray


def prepare(cand: Candidates, h_gt, shape, cfg: FeatureConfig) -> Prepared:
    return Prepared(cand, build_features(cand, cfg), h_gt, shape, gt_inlier_mask(cand, h_gt))


def prepare_synthetic(cfgs, n_max: int = 160, feat_cfg: FeatureConfig = FeatureConfig(),
                      inlier_frac: float | None = None, cand_total: int = 240):
    from .synth import make_pair

    out = []
    for c in cfgs:
        img1, img2, h_gt, _ = make_pair(c)
        cand = match_candidates(img1, img2, n_max=n_max)
        if inlier_frac is not None:
            cand = inject_outliers(cand, h_gt, inlier_frac,
                                   np.random.default_rng(c.seed + 991), total=cand_total)
        out.append(prepare(cand, h_gt, img1.shape, feat_cfg))
    return out


def registration_error(theta, prep: Prepared, *, method="pt", budget=300_000, seed=0,
                       solver_kw=None) -> float:
    u, J = assemble(theta[:2], theta[2:], prep.feats)
    h, Js = to_ising(u, J)
    res = solve(method, h, Js, budget=budget, seed=seed, **(solver_kw or {}))
    H, n_used, _, _ = estimate(prep.cand, res["x"].astype(bool))
    return corner_error(H, prep.h_gt, prep.shape)


def objective(theta, data, **kw) -> float:
    errs = np.array([registration_error(theta, p, **kw) for p in data])
    errs = np.clip(errs, 0.0, 1e4)
    return float(np.mean(np.log1p(errs)))


def train_es(theta0, train_data, val_data=None, *, iters=20, popsize=8, sigma=0.35,
             lr=0.25, method="pt", budget=300_000, seed=0, log=print,
             objective_fn=None) -> np.ndarray:
    """Antithetic rank-based ES on theta. Deterministic, needs no solver gradients.

    Steps are scaled per channel by |theta0| so that the cardinality offset (order
    1e2) is perturbed on the same relative scale as the unit-sized weights.

    `objective_fn(theta, data) -> float` lets other regimes (e.g. per-tile under relief)
    reuse this optimiser with their own loss.
    """
    fn = objective_fn or (lambda th, d: objective(th, d, method=method, budget=budget))
    rng = np.random.default_rng(seed)
    theta0 = np.clip(np.asarray(theta0, float), LOWER, UPPER)
    scale = np.maximum(np.abs(theta0), 1.0)
    theta = theta0.copy()
    best = fn(theta, train_data)
    best_theta = theta.copy()
    if log:
        v = fn(theta, val_data) if val_data else float("nan")
        log(f"init  train={best:.3f} val={v:.3f}" if val_data else f"init  train={best:.3f}")

    for it in range(iters):
        eps = rng.normal(size=(popsize, theta.size))
        deltas = np.concatenate([eps, -eps])
        cand_thetas = np.clip(theta + sigma * scale * deltas, LOWER, UPPER)
        scores = np.array([fn(t, train_data) for t in cand_thetas])

        ranks = np.empty(len(scores))
        ranks[np.argsort(scores)] = np.arange(len(scores)) - (len(scores) - 1) / 2
        ranks /= np.abs(ranks).max()
        grad = (ranks[:, None] * deltas).sum(0) / (len(scores) * sigma)
        theta = np.clip(theta - lr * scale * grad, LOWER, UPPER)

        cur = fn(theta, train_data)
        if cur < best:
            best, best_theta = cur, theta.copy()
        if log:
            msg = f"iter {it:02d} train={cur:.3f} best={best:.3f}"
            if val_data is not None:
                msg += f" val={fn(theta, val_data):.3f}"
            msg += "  " + " ".join(f"{n}={v:+.2f}" for n, v in zip(THETA_NAMES, theta))
            log(msg)
    return best_theta


def blackbox_grad(theta, prep: Prepared, grad_x, *, method="pt", budget=300_000,
                  seed=0, lam=1.0, solver_kw=None) -> np.ndarray:
    """Pogancic et al. (ICLR'20) backward pass for a black-box linear-objective solver.

    grad_x : dLoss/dx at the selected solution x*, shape (N,).
    Returns dLoss/dtheta.

    Implementation: perturb the solve with the loss direction, re-solve, and use the
    feature displacement. Plug `grad_x` from a differentiable decoder (or a finite
    difference on x) and this is a drop-in training loop around SQA.
    """
    u, J = assemble(theta[:2], theta[2:], prep.feats)
    h, Js = to_ising(u, J)
    x = solve(method, h, Js, budget=budget, seed=seed, **(solver_kw or {}))["x"].astype(bool)

    # The solver minimises <theta, phi(x)>. Perturb its input with the loss direction
    # on x: solve with an extra linear term lam * grad_x added to the unary part.
    u2 = u + lam * np.asarray(grad_x, float)
    h2, Js2 = to_ising(u2, J)
    x2 = solve(method, h2, Js2, budget=budget, seed=seed, **(solver_kw or {}))["x"].astype(bool)

    def phi(mask):
        idx = np.flatnonzero(mask)
        return np.concatenate([prep.feats["U"][idx].sum(0),
                               prep.feats["P"][np.ix_(idx, idx)].sum((0, 1))])

    return (phi(x2) - phi(x)) / lam


def save_theta(path, theta):
    with open(path, "w") as f:
        json.dump({n: float(v) for n, v in zip(THETA_NAMES, theta)}, f, indent=2)


def load_theta(path) -> np.ndarray:
    with open(path) as f:
        d = json.load(f)
    return np.array([d[n] for n in THETA_NAMES], float)


def init_theta(w: QuboWeights = QuboWeights(), n: int = 160) -> np.ndarray:
    tu, tp = theta_from_weights(w, n)
    return np.concatenate([tu, tp])


def main(argv=None) -> int:
    import argparse

    from .synth import PairConfig

    p = argparse.ArgumentParser(description="Train QUBO weights against registration error.")
    p.add_argument("--train-pairs", type=int, default=6)
    p.add_argument("--val-pairs", type=int, default=4)
    p.add_argument("--oblique", default="20,30,40")
    p.add_argument("--inlier-frac", type=float, default=None,
                   help="reshape candidates to this inlier fraction (low-inlier regime)")
    p.add_argument("--cand-total", type=int, default=240)
    p.add_argument("--iters", type=int, default=15)
    p.add_argument("--popsize", type=int, default=6)
    p.add_argument("--sigma", type=float, default=0.35)
    p.add_argument("--lr", type=float, default=0.25)
    p.add_argument("--method", default="pt", choices=["greedy", "sa", "pt", "sqa"])
    p.add_argument("--budget", type=int, default=300_000)
    p.add_argument("--out", default=None)
    a = p.parse_args(argv)

    angles = [float(x) for x in a.oblique.split(",")]
    mk = lambda n, off: [PairConfig(oblique_deg=ang, yaw_deg=0.35 * ang, photometric=0.6,
                                    seed=7000 + off + 100 * int(ang) + i)
                         for ang in angles for i in range(n)]
    train_data = prepare_synthetic(mk(a.train_pairs, 0), inlier_frac=a.inlier_frac,
                                   cand_total=a.cand_total)
    val_data = prepare_synthetic(mk(a.val_pairs, 500), inlier_frac=a.inlier_frac,
                                 cand_total=a.cand_total)

    theta0 = init_theta()
    base_tr, base_va = objective(theta0, train_data), objective(theta0, val_data)
    print(f"hand-tuned   train={base_tr:.3f} val={base_va:.3f}")

    theta = train_es(theta0, train_data, val_data, iters=a.iters, popsize=a.popsize,
                     sigma=a.sigma, lr=a.lr, method=a.method, budget=a.budget)
    print(f"learned      train={objective(theta, train_data, method=a.method, budget=a.budget):.3f} "
          f"val={objective(theta, val_data, method=a.method, budget=a.budget):.3f}")
    print("theta: " + " ".join(f"{n}={v:+.2f}" for n, v in zip(THETA_NAMES, theta)))
    if a.out:
        save_theta(a.out, theta)
        print(f"wrote {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
