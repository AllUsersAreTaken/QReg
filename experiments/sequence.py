"""Sequence test: what a drone actually produces, and where drift comes from.

    python experiments/sequence.py --frames 12 --overlap 0.75 --decode irls

Two strategies are compared for every method:

  chained  : register consecutive frames (t-1 -> t) and compose. Consecutive pairs are
             near-identity and high-inlier, so they are easy; the cost is drift, which
             is what makes mosaics ghost.
  direct   : register frame t straight to frame 0. Drift-free but the baseline grows,
             so beyond a few frames there is no overlap at all.

That trade-off is invisible in pairwise benchmarks and is the actual orthomosaic
problem. `drift` is corner error of the composed transform against ground truth.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import cv2

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from qreg.decode import corner_error, corners, estimate, estimate_irls  # noqa: E402
from qreg.evaluate import QUBO_METHODS, qubo_selected, ransac_transform, warmup  # noqa: E402
from qreg.features import gt_inlier_mask, match_candidates  # noqa: E402
from qreg.qubo import QuboWeights  # noqa: E402
from qreg.strip import StripConfig, gt_between, make_strip, overlap_fraction  # noqa: E402

METHODS = ("ransac",) + QUBO_METHODS


def register(frames, i, j, method, w, budget, solver_kw, decode, cache):
    key = (i, j)
    if key not in cache:
        cache[key] = match_candidates(frames[i], frames[j], n_max=160)
    cand = cache[key]
    fit = estimate_irls if decode == "irls" else estimate
    if method == "ransac":
        return ransac_transform(cand), cand
    sel, _, _ = qubo_selected(cand, w, method, budget, 0, solver_kw)
    if sel is None:
        return None, cand
    return fit(cand, sel)[0], cand


def _measure_edges(frames, n, max_gap, method, w, budget, decode, cache):
    """All pairwise transforms whose overlap is usable, as {(i, j): H_i->j}."""
    edges = {}
    for i in range(n):
        for j in range(i + 1, min(n, i + max_gap + 1)):
            H, _ = register(frames, i, j, method, w, budget, {}, decode, cache)
            if H is not None and np.all(np.isfinite(H)):
                edges[(i, j)] = H
    return edges


def _edge(edges, k, t):
    if (k, t) in edges:
        return edges[(k, t)]
    if (t, k) in edges:
        return np.linalg.inv(edges[(t, k)])
    return None


def _global_align(n, edges, shape, rounds=8):
    """Distribute error over the whole strip instead of chaining.

    G[t] maps frame t -> frame 0. Every edge gives a prediction G[t] = G[k] @ inv(M_k->t);
    predictions are combined by a robust median of the four mapped corners and the
    homography is refitted. This is the cheap version of "combine the rapid frames":
    it uses all pairs, not just the previous one, so independent errors cancel instead
    of accumulating.
    """
    src = corners(shape).reshape(-1, 1, 2).astype(np.float32)
    G = [np.eye(3) for _ in range(n)]
    for t in range(1, n):
        M = _edge(edges, t - 1, t)
        G[t] = G[t - 1] @ np.linalg.inv(M) if M is not None else G[t - 1].copy()

    for _ in range(rounds):
        for t in range(1, n):
            preds = [cv2.perspectiveTransform(src, G[t]).reshape(-1, 2)]
            for k in range(n):
                if k == t:
                    continue
                M = _edge(edges, k, t)
                if M is None:
                    continue
                preds.append(cv2.perspectiveTransform(src, G[k] @ np.linalg.inv(M)).reshape(-1, 2))
            if len(preds) < 2:
                continue
            med = np.median(np.stack(preds), axis=0).astype(np.float32)
            H, _ = cv2.findHomography(src, med, 0)
            if H is not None and np.all(np.isfinite(H)):
                G[t] = H
    return G


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--frames", type=int, default=12)
    p.add_argument("--size", type=int, default=384)
    p.add_argument("--scene", type=int, default=2048)
    p.add_argument("--overlap", type=float, default=0.75)
    p.add_argument("--tilt", type=float, default=3.0)
    p.add_argument("--photometric", type=float, default=0.35)
    p.add_argument("--blur", type=float, default=1.0)
    p.add_argument("--budget", type=int, default=300_000)
    p.add_argument("--decode", default="irls", choices=["dlt", "irls"])
    p.add_argument("--max-gap", type=int, default=2, help="widest pair used by global alignment")
    p.add_argument("--rounds", type=int, default=8)
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args(argv)

    cfg = StripConfig(size=a.size, scene=a.scene, n_frames=a.frames, overlap=a.overlap,
                      tilt_deg=a.tilt, photometric_drift=a.photometric, blur=a.blur,
                      seed=a.seed)
    strip = make_strip(cfg)
    frames, shape = strip["frames"], (a.size, a.size)
    w, cache = QuboWeights(), {}

    print(f"{a.frames} frames, size {a.size}, overlap {a.overlap:.2f}, "
          f"tilt {a.tilt}deg, photometric {a.photometric}, decode {a.decode}")
    print(f"{'step':>4} {'cand':>5} {'inl':>5} | "
          + " ".join(f"{m:>9}" for m in METHODS) + "   (chained drift vs frame 0, px)")

    chained = {m: np.eye(3) for m in METHODS}
    rows = []
    for t in range(1, a.frames):
        line = []
        cand0 = cache.get((t - 1, t))
        for m in METHODS:
            Hstep, cand = register(frames, t - 1, t, m, w, a.budget, {}, a.decode, cache)
            cand0 = cand
            chained[m] = Hstep @ chained[m] if Hstep is not None else chained[m]
            line.append(corner_error(chained[m], gt_between(strip, 0, t), shape))
        if cand0 is not None and len(cand0):
            inl = gt_inlier_mask(cand0, gt_between(strip, t - 1, t)).mean()
        else:
            inl = 0.0
        print(f"{t:>4} {len(cand0) if cand0 is not None else 0:>5} {inl:>5.2f} | "
              + " ".join(f"{v:>9.2f}" for v in line))
        rows.append((t, None, list(line)))

    print("\nsame methods, DIRECT t-1 -> t (per-step accuracy, no accumulated drift):")
    print(f"{'gap':>3} | " + " ".join(f"{m:>9}" for m in METHODS))
    for t in range(1, min(a.frames, 4)):
        line = []
        for m in METHODS:
            H, _ = register(frames, t - 1, t, m, w, a.budget, {}, a.decode, cache)
            line.append(corner_error(H, gt_between(strip, t - 1, t), shape))
        print(f"{t:>3} | " + " ".join(f"{v:>9.2f}" for v in line))

    print("\nDIRECT frame 0 -> t (wide baseline; stops when overlap runs out):")
    print(f"{'gap':>3} {'ovlp':>5} | " + " ".join(f"{m:>9}" for m in METHODS))
    for t in range(1, min(a.frames, 6)):
        ov = overlap_fraction(cfg, t)
        if ov < 0.10:
            print(f"{t:>3} {ov:>5.2f} |  no overlap")
            continue
        line = []
        for m in METHODS:
            H, _ = register(frames, 0, t, m, w, a.budget, {}, a.decode, cache)
            line.append(corner_error(H, gt_between(strip, 0, t), shape))
        print(f"{t:>3} {ov:>5.2f} | " + " ".join(f"{v:>9.2f}" for v in line))

    final = rows[-1][2]
    print("\ndrift at final frame (chained):")
    for m, v in zip(METHODS, final):
        print(f"  {m:>7} {v:>9.2f} px")

    # ---- global alignment: use every usable pair, not just the previous frame ----
    print(f"\nGLOBAL alignment over all pairs with gap <= {a.max_gap} "
          f"(robust corner averaging, {a.rounds} relaxation rounds):")
    print(f"{'step':>4} | " + " ".join(f"{m:>9}" for m in METHODS)
          + "   (drift vs frame 0, px)")
    edges = {m: _measure_edges(frames, a.frames, a.max_gap, m, w, a.budget, a.decode, cache)
             for m in METHODS}
    G = {m: _global_align(a.frames, edges[m], shape, a.rounds) for m in METHODS}
    for t in range(2, a.frames):
        line = [corner_error(np.linalg.inv(G[m][t]), gt_between(strip, 0, t), shape) for m in METHODS]
        print(f"{t:>4} | " + " ".join(f"{v:>9.2f}" for v in line))
    print(f"\ndrift at final frame (global):")
    for m in METHODS:
        print(f"  {m:>7} {corner_error(np.linalg.inv(G[m][a.frames-1]), gt_between(strip, 0, a.frames-1), shape):>9.2f} px")
    return 0


if __name__ == "__main__":
    warmup()
    raise SystemExit(main())
