"""Evaluation harness: fixed SIFT front-end, pluggable correspondence selectors.

Reported per method: corner error vs ground truth, success@px, wall clock, and the
number of attempted spin flips (so cost is auditable).  The `energy_quality_rank_corr`
column is the diagnostic that motivates the whole project: if a hand-tuned QUBO's
final energy does not predict registration error, then optimising that energy is the
wrong objective and the energy itself has to be learned against registration quality.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import cv2
import numpy as np

from .decode import corner_error, estimate, estimate_irls, overlap_ncc
from .features import Candidates, gt_inlier_mask, inject_outliers, match_candidates, spatial_isotropy
from .qubo import QuboWeights, build_qubo, to_ising
from .solvers import solve

QUBO_METHODS = ("greedy", "sa", "pt", "sqa")
ALL_METHODS = ("ransac",) + QUBO_METHODS


def ransac_transform(cand: Candidates, thresh: float = 3.0):
    if len(cand) < 4:
        return None
    H, _ = cv2.findHomography(cand.p1.astype(np.float64), cand.p2.astype(np.float64),
                              cv2.RANSAC, thresh, maxIters=5000, confidence=0.995)
    return H


def qubo_selected(cand: Candidates, w: QuboWeights, method: str, budget: int, seed: int,
                  solver_kw: dict | None = None, theta=None):
    q = build_qubo(cand, w, theta=theta)
    if q["n"] < 2:
        return None, float("nan"), 0
    h, J = to_ising(q["u"], q["J"])
    res = solve(method, h, J, budget=budget, seed=seed, **(solver_kw or {}))
    return res["x"].astype(bool), res["energy"], res["flips"]


@dataclass
class Record:
    pair: str
    oblique: float
    photometric: float
    method: str
    n_cand: int
    inlier_frac: float
    corner_err: float
    n_used: int
    energy: float
    flips: int
    seconds: float
    ncc: float
    iso: float = float("nan")
    padded: bool = False
    meta: dict = field(default_factory=dict)


def evaluate_pair(
    name: str,
    img1,
    img2,
    h_gt,
    *,
    w: QuboWeights = QuboWeights(),
    budget: int = 300_000,
    seed: int = 0,
    n_max: int = 160,
    oblique: float = 0.0,
    photometric: float = 0.0,
    solver_kw: dict | None = None,
    methods=ALL_METHODS,
    theta=None,
    learned_theta=None,
    theta_label: str = "*",
    target_inlier: float | None = None,
    cand_total: int = 240,
    decode: str = "dlt",
) -> list[Record]:
    fit = estimate_irls if decode == "irls" else estimate
    cand = match_candidates(img1, img2, n_max=n_max)
    if target_inlier is not None:
        cand = inject_outliers(cand, h_gt, target_inlier,
                              np.random.default_rng(abs(hash(name)) % (2 ** 32)),
                              total=cand_total)
    inl = gt_inlier_mask(cand, h_gt)
    shape = img1.shape
    out: list[Record] = []

    variants = [("", theta)]
    if learned_theta is not None:
        variants.append((theta_label, np.asarray(learned_theta, float)))

    plan: list[tuple[str, str, str, object]] = []
    for m in methods:
        if m in ("ransac", "oracle"):
            plan.append((m, m, "", None))
    for suffix, th in variants:
        for m in methods:
            if m in QUBO_METHODS:
                plan.append((m + suffix, m, suffix, th))

    for label, base, suffix, th in plan:
        t0 = time.perf_counter()
        energy, flips, n_used, padded = float("nan"), 0, 0, False
        if base == "ransac":
            H = ransac_transform(cand)
            pts = cand.p1
            if H is not None:
                n_used = int(len(cand))
        elif base == "oracle":
            H, n_used, _, padded = fit(cand, inl)
            pts = cand.p1[inl]
        else:
            sel, energy, flips = qubo_selected(cand, w, base, budget, seed, solver_kw, theta=th)
            H, n_used, _, padded = fit(cand, sel) if sel is not None else (None, 0, 0.0, False)
            pts = cand.p1[sel] if sel is not None else cand.p1[:0]
        dt = time.perf_counter() - t0
        out.append(Record(
            pair=name, oblique=oblique, photometric=photometric, method=label,
            n_cand=len(cand), inlier_frac=float(inl.mean()) if len(cand) else 0.0,
            corner_err=corner_error(H, h_gt, shape), n_used=n_used,
            energy=energy, flips=flips, seconds=dt,
            ncc=overlap_ncc(img1, img2, H), iso=spatial_isotropy(pts), padded=padded,
            meta={"base": base, "suffix": suffix},
        ))
    return out


def warmup():
    """Compile numba kernels on a tiny problem so timings are meaningful."""
    h = np.random.default_rng(0).normal(size=8)
    J = np.zeros((8, 8))
    for m in QUBO_METHODS:
        solve(m, h, J, budget=800, seed=0)


def _rank(a: np.ndarray) -> np.ndarray:
    order = a.argsort()
    r = np.empty(len(a), float)
    r[order] = np.arange(len(a), dtype=float)
    return r


def energy_quality_rank_corr(records: list[Record], suffix: str | None = None) -> float:
    """Spearman rho between solver energy and corner error, over QUBO records.

    A well-specified energy should be positively correlated with error.  Weak or
    negative rho means the energy is not a proxy for registration quality.

    `suffix` restricts the comparison to one theta ("" = hand-tuned, "*" = learned);
    energies from different theta are not comparable even after normalisation.
    """
    rs = [r for r in records if r.meta.get("base") in QUBO_METHODS
          and np.isfinite(r.energy)
          and (suffix is None or r.meta.get("suffix", "") == suffix)]
    if len(rs) < 4:
        return float("nan")
    e = _rank(np.array([r.energy for r in rs]))
    q = _rank(np.array([r.corner_err for r in rs]))
    e -= e.mean()
    q -= q.mean()
    den = np.sqrt((e * e).sum() * (q * q).sum())
    return float((e * q).sum() / den) if den > 0 else float("nan")


def solver_gap(records: list[Record], base_a: str = "sa", base_b: str = "sqa",
               suffix: str = "") -> float:
    """Paired median of err(A) - err(B) in px. Positive => B (sqa) is better."""
    st = solver_gap_stats(records, base_a, base_b, suffix)
    return float("nan") if st is None else st["median"]


def solver_gap_stats(records: list[Record], base_a: str = "sa", base_b: str = "sqa",
                     suffix: str = "") -> dict | None:
    """Paired gap statistics. The median alone is misleading: most pairs are exact ties
    and the spread lives in a few catastrophic divergences, so report mean/p90 and the
    win counts too."""
    per_pair: dict[str, dict[str, float]] = {}
    for r in records:
        if r.meta.get("base") in (base_a, base_b) and r.meta.get("suffix", "") == suffix:
            per_pair.setdefault(r.pair, {})[r.meta["base"]] = r.corner_err
    d = [v[base_a] - v[base_b] for v in per_pair.values() if len(v) == 2]
    if not d:
        return None
    d = np.clip(np.array(d, float), -1e4, 1e4)
    return {
        "n": len(d),
        "median": float(np.median(d)),
        "mean": float(d.mean()),
        "p90": float(np.percentile(d, 90)),
        "b_better": int((d > 1e-9).sum()),
        "a_better": int((d < -1e-9).sum()),
        "ties": int((np.abs(d) <= 1e-9).sum()),
    }


def summarise(records: list[Record], success_px: float = 5.0) -> str:
    methods = list(dict.fromkeys(r.method for r in records))
    lines = []
    hdr = f"{'method':>8} {'medErr':>8} {'p90Err':>8} {'succ@' + str(int(success_px)):>9} " \
          f"{'keep':>5} {'iso':>5} {'meanE':>8} {'flips':>9} {'ms':>7} {'ncc':>6}"
    lines.append(hdr)
    lines.append("-" * len(hdr))
    for m in methods:
        rs = [r for r in records if r.method == m]
        err = np.array([r.corner_err for r in rs])
        ok = np.isfinite(err)
        lines.append(
            f"{m:>8} {np.median(err[ok]) if ok.any() else float('nan'):>8.2f} "
            f"{np.percentile(err[ok], 90) if ok.any() else float('nan'):>8.2f} "
            f"{np.mean(err < success_px) if ok.any() else 0.0:>9.2f} "
            f"{np.mean([r.n_used / max(r.n_cand, 1) for r in rs]):>5.2f} "
            f"{np.nanmedian([r.iso for r in rs]):>5.2f} "
            f"{np.mean([r.energy for r in rs]) if np.isfinite([r.energy for r in rs]).any() else float('nan'):>8.1f} "
            f"{np.mean([r.flips for r in rs]):>9.0f} "
            f"{1000 * np.mean([r.seconds for r in rs]):>7.1f} "
            f"{np.nanmean([r.ncc for r in rs]):>6.3f}"
        )
    for sfx, tag in (("", "hand"), ("*", "learned")):
        rho = energy_quality_rank_corr(records, sfx)
        if np.isfinite(rho):
            lines.append(f"energy<->quality rho ({tag:>7}) = {rho:+.3f}   (want clearly > 0)")
    for sfx, tag in (("", "hand"), ("*", "learned")):
        st = solver_gap_stats(records, "sa", "sqa", sfx)
        if st:
            lines.append(
                f"gap SA-SQA ({tag:>7}) n={st['n']} med={st['median']:+.2f} "
                f"mean={st['mean']:+.2f} p90={st['p90']:+.2f} | "
                f"SQA better {st['b_better']}, SA better {st['a_better']}, ties {st['ties']}"
                f"   (+ => SQA better)")
    return "\n".join(lines)
