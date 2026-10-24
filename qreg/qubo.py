"""QUBO construction for correspondence selection.

Energy over binary variables x_i (match i is selected):

    E(x) = sum_i u_i x_i  +  sum_{i<j} J_ij x_i x_j

u_i  : appearance / descriptor term            (negative = attractive)
J_ij : positive = repulsive (incompatible),    (same image keypoint, geometric)
       negative = attractive (compatible)

`weighths` is the parameter vector theta that the training stage will optimise.
Everything here is a *hand-tuned* template: the point of the project is to replace
the constants with learned functions of pair features.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict

import numpy as np

from .features import Candidates


@dataclass
class QuboWeights:
    w_app: float = 1.0        # weight on normalised descriptor distance
    lam_oo: float = 6.0       # one-to-one violation penalty (shared keypoint)
    lam_geo: float = 3.0      # hinge weight on geometric inconsistency
    lam_attr: float = 0.0     # optional attraction for consistent pairs
    lam_spread: float = 2.0   # repulsion between spatially close matches (coverage)
    tau_s: float = 0.25       # log-scale tolerance (natural units)
    tau_th: float = 0.20      # rotation tolerance (radians)
    slack: float = 1.0        # inconsistency allowed before the hinge fires
    lam_card: float = 2.0     # cardinality prior weight (0 = off)
    k_target: int | None = None
    k_frac: float = 0.30      # target density when k_target is None
    local_k: int = 8          # neighbourhood size for the local similarity prior

    def as_dict(self) -> dict:
        return asdict(self)


def _pdist(a: np.ndarray) -> np.ndarray:
    d = a[:, None, :] - a[None, :, :]
    return np.sqrt((d * d).sum(-1))


def pair_features(cand: Candidates, local_k: int = 8) -> dict:
    """Similarity-consistency features for every candidate pair.

    For pair (i, j) the two matches imply a scale ratio s_ij and a rotation th_ij.
    Both are compared against a *local* robust prior (median over the k nearest
    candidates in image 1), which tracks the slowly varying part of a homography
    without needing an explicit model fit.
    """
    n = len(cand)
    p1, p2 = cand.p1.astype(np.float64), cand.p2.astype(np.float64)
    d1 = _pdist(p1)
    d2 = _pdist(p2)

    k = min(local_k, max(n - 1, 1))
    d1n = d1 + np.where(np.eye(n, dtype=bool), np.inf, 0.0)
    nb = np.argsort(d1n, axis=1)[:, :k]

    v1 = p1[:, None, :] - p1[None, :, :]
    v2 = p2[:, None, :] - p2[None, :, :]
    cross = v1[..., 0] * v2[..., 1] - v1[..., 1] * v2[..., 0]
    dot = (v1 * v2).sum(-1)
    th = np.arctan2(cross, dot)

    with np.errstate(divide="ignore", invalid="ignore"):
        s = d2 / np.maximum(d1, 1e-6)

    rows = np.arange(n)[:, None]
    s_local = np.median(s[rows, nb], axis=1)
    th_local = np.arctan2(
        np.median(np.sin(th[rows, nb]), axis=1),
        np.median(np.cos(th[rows, nb]), axis=1),
    )

    def wrap(a):
        return np.abs(np.arctan2(np.sin(a), np.cos(a)))

    with np.errstate(divide="ignore", invalid="ignore"):
        cs = 0.5 * (np.abs(np.log(np.maximum(s, 1e-6) / np.maximum(s_local[:, None], 1e-6)))
                    + np.abs(np.log(np.maximum(s, 1e-6) / np.maximum(s_local[None, :], 1e-6))))
    cth = 0.5 * (wrap(th - th_local[:, None]) + wrap(th - th_local[None, :]))

    return {
        "d1": d1, "d2": d2, "s": s, "theta": th,
        "s_local": s_local, "theta_local": th_local,
        "c_scale": cs, "c_theta": cth, "nb": nb,
    }


def consistency_cost(cand: Candidates, w: QuboWeights) -> np.ndarray:
    """Dimensionless geometric inconsistency c_ij (>= 0, ~1 at the tolerance edge)."""
    f = pair_features(cand, w.local_k)
    c = f["c_scale"] / max(w.tau_s, 1e-6) + f["c_theta"] / max(w.tau_th, 1e-6)
    return np.nan_to_num(c, nan=0.0, posinf=50.0, neginf=0.0)


# ---------------------------------------------------------------------------
# Feature stack: E(x; theta) = <theta_u, U(x)> + <theta_p, P(x)>
#
# The energy is *linear in theta*. That property is what makes the QUBO
# trainable end-to-end through a black-box solver (Pogancic et al., ICLR'20):
# the solver is an argmin of a linear function of theta, so a perturb-and-resolve
# backward pass is available.  Non-linear hand-tuning (e.g. tau_s) lives in
# FeatureConfig and is *not* differentiated.
# ---------------------------------------------------------------------------

UNARY_NAMES = ("app", "card_lin")
PAIR_NAMES = ("one2one", "geo_hinge", "geo_attr", "spread", "card_quad")


@dataclass
class FeatureConfig:
    tau_s: float = 0.25
    tau_th: float = 0.20
    slack: float = 1.0
    local_k: int = 8
    k_frac: float = 0.30


def build_features(cand: Candidates, cfg: FeatureConfig = FeatureConfig()) -> dict:
    """Fixed feature maps. U: (N, 2) unary, P: (N, N, 5) pairwise."""
    n = len(cand)
    if n < 2:
        return {"U": np.zeros((n, len(UNARY_NAMES))),
                "P": np.zeros((n, n, len(PAIR_NAMES))),
                "c_geo": np.zeros((n, n)), "n": n}

    f = pair_features(cand, cfg.local_k)
    d = cand.dist.astype(np.float64)
    app = (d - d.mean()) / (d.std() + 1e-6)

    c = np.nan_to_num(f["c_scale"] / max(cfg.tau_s, 1e-6)
                      + f["c_theta"] / max(cfg.tau_th, 1e-6),
                      nan=0.0, posinf=50.0, neginf=0.0)
    hinge = np.maximum(0.0, c - cfg.slack)
    attr = np.maximum(0.0, cfg.slack - c)

    d1 = f["d1"]
    d1n = d1 + np.where(np.eye(n, dtype=bool), np.inf, 0.0)
    sigma = float(np.median(d1n.min(axis=1))) + 1e-6
    spread = np.exp(-0.5 * (d1 / sigma) ** 2)

    same = (cand.kp1[:, None] == cand.kp1[None, :]) | (cand.kp2[:, None] == cand.kp2[None, :])
    np.fill_diagonal(same, False)

    U = np.stack([app, np.ones(n)], axis=1)
    P = np.stack([same.astype(np.float64), hinge, attr, spread,
                  np.ones((n, n))], axis=2)
    return {"U": U, "P": P, "c_geo": c, "n": n}


def assemble(theta_u, theta_p, feats: dict) -> tuple[np.ndarray, np.ndarray]:
    """Linear-in-theta assembly of E = u.x + sum_{i<j} J_ij x_i x_j."""
    n = feats["n"]
    if n < 2:
        return np.zeros(n), np.zeros((n, n))
    u = feats["U"] @ np.asarray(theta_u, np.float64)
    J = feats["P"] @ np.asarray(theta_p, np.float64)
    np.fill_diagonal(J, 0.0)
    J = 0.5 * (J + J.T)
    return u, J


def theta_from_weights(w: QuboWeights, n: int) -> tuple[np.ndarray, np.ndarray]:
    """Map the interpretable hand-tuned weights onto the linear feature stack."""
    if w.lam_card > 0.0:
        k = float(w.k_target) if w.k_target is not None else max(4.0, w.k_frac * n)
        card_u, card_p = w.lam_card * (1.0 - 2.0 * k), 2.0 * w.lam_card
    else:
        card_u, card_p = 0.0, 0.0
    tu = np.array([w.w_app, card_u])
    tp = np.array([w.lam_oo, w.lam_geo, -w.lam_attr, w.lam_spread, card_p])
    return tu, tp


def build_qubo(cand: Candidates, w: QuboWeights, theta=None) -> dict:
    """Assemble the QUBO. Returns u, J (symmetric, zero diagonal), and diagnostics."""
    n = len(cand)
    cfg = FeatureConfig(tau_s=w.tau_s, tau_th=w.tau_th, slack=w.slack,
                        local_k=w.local_k, k_frac=w.k_frac)
    feats = build_features(cand, cfg)
    if theta is None:
        tu, tp = theta_from_weights(w, n)
    else:
        th = np.asarray(theta, np.float64)
        tu, tp = (th[:2], th[2:]) if th.ndim == 1 else theta
    u, J = assemble(tu, tp, feats)
    return {"u": u, "J": J, "c_geo": feats["c_geo"], "n": n,
            "feats": feats, "theta": (np.asarray(tu), np.asarray(tp))}


def to_ising(u: np.ndarray, J: np.ndarray, normalise: bool = True) -> tuple[np.ndarray, np.ndarray]:
    """Convert E = u.x + sum_{i<j} J_ij x_i x_j (J symmetric, zero diag) with x=(s+1)/2.

    Returns (h, Js) with E = h.s + 0.5 * s^T Js s (constants dropped).
    """
    u = np.asarray(u, float)
    J = np.asarray(J, float)
    h = 0.5 * u + 0.25 * J.sum(axis=1)
    Js = 0.25 * J
    if normalise:
        nz_h = h[h != 0]
        nz_j = Js[np.triu_indices_from(Js, 1)]
        scale = np.sqrt(np.mean(nz_h ** 2) if nz_h.size else 1.0) + \
            np.sqrt(np.mean(nz_j ** 2) if nz_j.size else 1.0)
        scale = max(scale, 1e-9)
        h, Js = h / scale, Js / scale
    return h, Js


def qubo_energy(u: np.ndarray, J: np.ndarray, x: np.ndarray) -> float:
    """E = u.x + sum_{i<j} J_ij x_i x_j  (0.5 factor because J is symmetric)."""
    return float(u @ x + 0.5 * x @ J @ x)
