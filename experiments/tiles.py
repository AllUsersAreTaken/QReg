"""Per-tile registration under terrain relief, with the QUBO doing the selection.

    python experiments/tiles.py --relief 0.2 --baseline 0.4 --tiles 4

A global homography cannot represent a relief surface, so this is where the estimator
actually has to change - not just get faster. Each tile gets its own local homography,
fitted from a slightly expanded window so it has enough support.

Methods compared on identical candidates:
  global-h    one RANSAC homography for the whole frame (the plane model)
  moving-dlt  APAP-style: per tile, Gaussian spatial weights over ALL matches + Tukey
  ransac-tile per tile, RANSAC homography on the window
  qubo-tile   per tile, QUBO selection on the window, then robust refit
  oracle-tile per tile, robust fit on ground-truth inliers (ceiling)

Error is measured densely against the exact per-pixel GT map, not on corners.
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from qreg.decode import estimate_irls, residuals, weighted_dlt  # noqa: E402
from qreg.evaluate import qubo_selected, ransac_transform, warmup  # noqa: E402
from qreg.features import Candidates, match_candidates  # noqa: E402
from qreg.qubo import QuboWeights  # noqa: E402
from qreg.solvers import solve  # noqa: E402
from qreg.terrain import ReliefConfig, make_relief_pair  # noqa: E402
from qreg.train import THETA_NAMES, load_theta  # noqa: E402

METHODS = ("global-h", "moving-dlt", "ransac-tile", "qubo-tile", "oracle-tile")


def dense_inliers(cand: Candidates, pair: dict, thresh: float = 3.0) -> np.ndarray:
    m = pair["map"]
    h, w = m.shape[:2]
    x = np.clip(cand.p1[:, 0].round().astype(int), 0, w - 1)
    y = np.clip(cand.p1[:, 1].round().astype(int), 0, h - 1)
    return np.linalg.norm(m[y, x] - cand.p2, axis=1) < thresh


def window_idx(cand, x0, y0, x1, y1):
    p = cand.p1
    return np.flatnonzero((p[:, 0] >= x0) & (p[:, 0] < x1) & (p[:, 1] >= y0) & (p[:, 1] < y1))


def moving_dlt(sub: Candidates, all_cand: Candidates, center, sigma, iters=3):
    """APAP: spatial weights over ALL matches, Tukey reweighting, local RANSAC init.

    Needs a local initialisation. Starting from an unweighted least-squares fit (or from
    a global-plane RANSAC mask) is badly biased when the plane model is wrong, because
    under relief that mask rejects genuinely correct matches.
    """
    H, _ = cv2.findHomography(sub.p1.astype(np.float64), sub.p2.astype(np.float64),
                              cv2.RANSAC, 3.0, maxIters=4000, confidence=0.995)
    if H is None:
        return None
    src = all_cand.p1.astype(np.float64)
    dst = all_cand.p2.astype(np.float64)
    d = np.linalg.norm(src - np.asarray(center, float), axis=1)
    wsp = np.exp(-0.5 * (d / sigma) ** 2)
    for _ in range(iters):
        r = residuals(src, dst, H)
        u = np.clip(r / 3.0, 0, 1)
        w = wsp * (1 - u ** 2) ** 2
        if (w > 1e-9).sum() < 6:
            break
        H2 = weighted_dlt(src, dst, w)
        if H2 is None:
            break
        H = H2
    return H


def fit_tiles(pair, cand, n_tiles, expand, method, w, budget, robust=None, theta=None):
    s = pair["cfg"].size
    ts = s // n_tiles
    gt = dense_inliers(cand, pair)
    out = {}
    for ty in range(n_tiles):
        for tx in range(n_tiles):
            x0, y0 = tx * ts, ty * ts
            x1, y1 = x0 + ts, y0 + ts
            ex = int(expand * ts)
            wx0, wy0 = max(0, x0 - ex), max(0, y0 - ex)
            wx1, wy1 = min(s, x1 + ex), min(s, y1 + ex)
            idx = window_idx(cand, wx0, wy0, wx1, wy1)
            if len(idx) < 6:
                out[(ty, tx)] = None
                continue
            sub = cand.subset(idx)
            center = (0.5 * (x0 + x1), 0.5 * (y0 + y1))

            if method == "moving-dlt":
                out[(ty, tx)] = moving_dlt(sub, cand, center, sigma=0.75 * ts)
            elif method == "ransac-tile":
                H, _ = cv2.findHomography(sub.p1.astype(np.float64),
                                          sub.p2.astype(np.float64), cv2.RANSAC, 3.0,
                                          maxIters=4000, confidence=0.995)
                out[(ty, tx)] = H
            elif method == "oracle-tile":
                sub_gt = sub.subset(np.flatnonzero(gt[idx]))
                out[(ty, tx)] = estimate_irls(sub_gt, np.ones(len(sub_gt), bool))[0] \
                    if len(sub_gt) >= 6 else None
            elif method == "qubo-tile":
                sel, _, _ = qubo_selected(sub, w, "sa", budget, 0, {}, theta=theta)
                out[(ty, tx)] = estimate_irls(sub, sel)[0] if sel is not None else None
            else:
                raise ValueError(method)
    return out


def score(pair, tile_H, n_tiles, step=10):
    s = pair["cfg"].size
    ts = s // n_tiles
    m, v = pair["map"], pair["valid"]
    errs = []
    for ty in range(n_tiles):
        for tx in range(n_tiles):
            H = tile_H.get((ty, tx))
            ys, xs = np.mgrid[ty * ts:(ty + 1) * ts:step, tx * ts:(tx + 1) * ts:step]
            ok = v[ys, xs]
            if not ok.any():
                continue
            src = np.stack([xs[ok], ys[ok]], 1).astype(np.float64)
            gt = m[ys[ok], xs[ok]].astype(np.float64)
            if H is None:
                errs.append(np.full(len(src), 1e4))
                continue
            pred = cv2.perspectiveTransform(src.reshape(-1, 1, 2), H).reshape(-1, 2)
            errs.append(np.linalg.norm(pred - gt, axis=1))
    e = np.concatenate(errs) if errs else np.zeros(0)
    e = np.clip(e, 0, 1e4)
    return {"med": float(np.median(e)), "p90": float(np.percentile(e, 90)),
            "succ": float(np.mean(e < 3.0)), "n": len(e)}


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--relief", type=float, default=0.20)
    p.add_argument("--baseline", type=float, default=0.40)
    p.add_argument("--tilt", type=float, default=3.0)
    p.add_argument("--tiles", type=int, default=4)
    p.add_argument("--expand", type=float, default=0.55)
    p.add_argument("--size", type=int, default=512)
    p.add_argument("--n-max", type=int, default=300)
    p.add_argument("--budget", type=int, default=300_000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--seeds", type=int, default=3, help="number of seeds to aggregate")
    p.add_argument("--learned", default=None, help="theta JSON; adds qubo-tile*")
    a = p.parse_args(argv)

    learned = load_theta(a.learned) if a.learned else None
    methods = list(METHODS) + (["qubo-tile*"] if learned is not None else [])
    if learned is not None:
        print("learned theta: " + " ".join(f"{n}={v:+.2f}" for n, v in zip(THETA_NAMES, learned)))

    per_seed = []
    for seed in range(a.seed, a.seed + a.seeds):
        cfg = ReliefConfig(size=a.size, relief=a.relief, baseline=a.baseline,
                           tilt_deg=a.tilt, seed=seed)
        pair = make_relief_pair(cfg)
        cand = match_candidates(pair["img0"], pair["img1"], n_max=a.n_max)
        gt = dense_inliers(cand, pair)
        print(f"\nseed {seed}: relief {a.relief:.2f} baseline {a.baseline:.2f} "
              f"tilt {a.tilt}deg tiles {a.tiles}x{a.tiles} "
              f"candidates {len(cand)} GT inliers {gt.mean():.2f}")

        Hg, mask = cv2.findHomography(cand.p1.astype(np.float64), cand.p2.astype(np.float64),
                                      cv2.RANSAC, 3.0, maxIters=5000, confidence=0.995)
        tiles_all = {(ty, tx): Hg for ty in range(a.tiles) for tx in range(a.tiles)}
        res = {"global-h": score(pair, tiles_all, a.tiles)}
        w = QuboWeights()
        for m in METHODS:
            if m == "global-h":
                continue
            th = fit_tiles(pair, cand, a.tiles, a.expand, m, w, a.budget)
            res[m] = score(pair, th, a.tiles)
        if learned is not None:
            th = fit_tiles(pair, cand, a.tiles, a.expand, "qubo-tile", w, a.budget, theta=learned)
            res["qubo-tile*"] = score(pair, th, a.tiles)

        print(f"{'method':>12} {'medErr':>8} {'p90Err':>8} {'succ@3':>7}")
        print("-" * 40)
        for m in methods:
            r = res[m]
            print(f"{m:>12} {r['med']:>8.2f} {r['p90']:>8.2f} {r['succ']:>7.2f}")
        per_seed.append(res)

    if a.seeds > 1:
        print(f"\n=== median over {a.seeds} seeds ===")
        print(f"{'method':>12} {'medErr':>8} {'succ@3':>7}")
        print("-" * 30)
        for m in methods:
            meds = [r[m]["med"] for r in per_seed]
            succ = [r[m]["succ"] for r in per_seed]
            print(f"{m:>12} {np.median(meds):>8.2f} {np.mean(succ):>7.2f}")

        print("\n=== paired differences (per seed, same pair) ===")
        print("negative = first method better")
        for ma, mb in (("qubo-tile", "ransac-tile"), ("qubo-tile", "moving-dlt"),
                       ("qubo-tile", "oracle-tile"), ("ransac-tile", "moving-dlt")):
            if ma not in methods or mb not in methods:
                continue
            d = np.array([r[ma]["med"] - r[mb]["med"] for r in per_seed])
            wins = int((d < 0).sum())
            n = len(d)
            # two-sided sign test
            k = min(wins, n - wins)
            p = min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n)
            print(f"{ma:>11} - {mb:<11} n={n:>2} mean={d.mean():+7.2f} "
                  f"median={np.median(d):+7.2f} wins={wins}/{n} sign-test p={p:.3f}")

    print(f"\nplane-model floor (global-h, median over seeds): "
          f"{np.median([r['global-h']['med'] for r in per_seed]):.2f} px")
    return 0


if __name__ == "__main__":
    warmup()
    raise SystemExit(main())
