"""Train theta on the per-tile relief regime.

Why here: on flat or mildly oblique pairs the pairwise QUBO never had a real job. Under
relief a single homography is the wrong model, so each tile needs its own local
transform and the *selection* becomes the weak link. The measured failure is a tail, not
a median: most tiles land on the GT-inlier oracle, a few pick a locally-consistent but
wrong set and blow up. That is exactly what a learned energy is supposed to fix.

Loss = mean log1p(dense local registration error) over tiles, i.e. registration quality,
never energy.
"""

from __future__ import annotations

import argparse

import cv2
import numpy as np

from .decode import estimate_irls
from .features import Candidates, match_candidates
from .qubo import FeatureConfig, assemble, build_features, to_ising
from .solvers import solve
from .terrain import ReliefConfig, make_relief_pair
from .train import LOWER, THETA_NAMES, UPPER, init_theta, save_theta, train_es


class TileSample:
    __slots__ = ("feats", "cand", "src", "dst")

    def __init__(self, feats, cand: Candidates, src, dst):
        self.feats, self.cand, self.src, self.dst = feats, cand, src, dst


def _window_idx(cand, x0, y0, x1, y1):
    p = cand.p1
    return np.flatnonzero((p[:, 0] >= x0) & (p[:, 0] < x1) &
                          (p[:, 1] >= y0) & (p[:, 1] < y1))


def prepare_relief_tiles(cfgs, n_tiles=4, expand=0.35, n_max=800,
                         feat_cfg: FeatureConfig = FeatureConfig(), step=10,
                         min_cand=8, min_pts=20):
    samples: list[TileSample] = []
    for cfg in cfgs:
        pair = make_relief_pair(cfg)
        cand = match_candidates(pair["img0"], pair["img1"], n_max=n_max)
        s, ts = cfg.size, cfg.size // n_tiles
        m, v = pair["map"], pair["valid"]
        ex = int(expand * ts)
        for ty in range(n_tiles):
            for tx in range(n_tiles):
                x0, y0 = tx * ts, ty * ts
                x1, y1 = x0 + ts, y0 + ts
                idx = _window_idx(cand, max(0, x0 - ex), max(0, y0 - ex),
                                  min(s, x1 + ex), min(s, y1 + ex))
                if len(idx) < min_cand:
                    continue
                ys, xs = np.mgrid[y0:y1:step, x0:x1:step]
                ok = v[ys, xs]
                if ok.sum() < min_pts:
                    continue
                src = np.stack([xs[ok], ys[ok]], 1).astype(np.float64)
                dst = m[ys[ok], xs[ok]].astype(np.float64)
                sub = cand.subset(idx)
                samples.append(TileSample(build_features(sub, feat_cfg), sub, src, dst))
    return samples


def tile_error(theta, sample: TileSample, *, method="sa", budget=300_000, seed=0) -> float:
    u, J = assemble(theta[:2], theta[2:], sample.feats)
    h, Js = to_ising(u, J)
    res = solve(method, h, Js, budget=budget, seed=seed)
    H, _, _, _ = estimate_irls(sample.cand, res["x"].astype(bool))
    if H is None or not np.all(np.isfinite(H)):
        return 1e4
    pred = cv2.perspectiveTransform(sample.src.reshape(-1, 1, 2), H).reshape(-1, 2)
    return float(np.median(np.linalg.norm(pred - sample.dst, axis=1)))


def objective_tiles(theta, samples, *, method="sa", budget=300_000, seed=0) -> float:
    errs = np.array([tile_error(theta, s, method=method, budget=budget, seed=seed)
                     for s in samples])
    return float(np.mean(np.log1p(np.clip(errs, 0, 1e4))))


def tile_stats(theta, samples, budget=300_000):
    e = np.array([tile_error(theta, s, budget=budget) for s in samples])
    return {"med": float(np.median(e)), "p90": float(np.percentile(e, 90)),
            "mean": float(np.mean(e)), "n": len(e)}


def _cfgs(args, offset: int):
    return [ReliefConfig(size=args.size, relief=args.relief, baseline=args.baseline,
                         tilt_deg=args.tilt, seed=offset + i)
            for i in range(args.pairs)]


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Train QUBO theta per-tile under relief.")
    p.add_argument("--pairs", type=int, default=5)
    p.add_argument("--val-pairs", type=int, default=3)
    p.add_argument("--relief", type=float, default=0.08)
    p.add_argument("--baseline", type=float, default=0.40)
    p.add_argument("--tilt", type=float, default=3.0)
    p.add_argument("--size", type=int, default=512)
    p.add_argument("--tiles", type=int, default=3)
    p.add_argument("--expand", type=float, default=0.35)
    p.add_argument("--n-max", type=int, default=800)
    p.add_argument("--iters", type=int, default=10)
    p.add_argument("--popsize", type=int, default=6)
    p.add_argument("--sigma", type=float, default=0.35)
    p.add_argument("--lr", type=float, default=0.3)
    p.add_argument("--budget", type=int, default=200_000)
    p.add_argument("--method", default="sa", choices=["greedy", "sa", "pt", "sqa"])
    p.add_argument("--out", default=None)
    a = p.parse_args(argv)

    train = prepare_relief_tiles(_cfgs(a, 0), a.tiles, a.expand, a.n_max)
    val = prepare_relief_tiles(_cfgs(a, 1000), a.tiles, a.expand, a.n_max)
    print(f"train tiles {len(train)}, val tiles {len(val)} "
          f"(relief {a.relief}, {a.tiles}x{a.tiles}, solver {a.method})")

    theta0 = init_theta()
    s0 = tile_stats(theta0, train, a.budget)
    v0 = tile_stats(theta0, val, a.budget)
    print(f"hand-tuned  train med={s0['med']:.2f} p90={s0['p90']:.2f} | "
          f"val med={v0['med']:.2f} p90={v0['p90']:.2f}")

    fn = lambda th, d: objective_tiles(th, d, method=a.method, budget=a.budget)  # noqa: E731
    theta = train_es(theta0, train, val, iters=a.iters, popsize=a.popsize,
                     sigma=a.sigma, lr=a.lr, seed=0, objective_fn=fn)

    s1 = tile_stats(theta, train, a.budget)
    v1 = tile_stats(theta, val, a.budget)
    print(f"learned     train med={s1['med']:.2f} p90={s1['p90']:.2f} | "
          f"val med={v1['med']:.2f} p90={v1['p90']:.2f}")
    print("theta: " + " ".join(f"{n}={v:+.2f}" for n, v in zip(THETA_NAMES, theta)))
    if a.out:
        save_theta(a.out, theta)
        print(f"wrote {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
