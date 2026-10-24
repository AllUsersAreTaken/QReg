"""Does registration enable fusion that actually improves detectability?

Real survey texture (Demining Research Community ortho), synthetic partly-buried targets,
simulated drift/lighting between frames. The question is narrow and answerable without a
trained detector: after fusing N frames, how much better can a fixed bandpass detector
separate targets from background?

Variants:
  single        one frame, no fusion                       (lower bound)
  naive-mean    average frames without registering         (drift blurs the target)
  aligned-mean  register with SIFT+RANSAC, then mean
  aligned-med   register, then median                      (kills moving shadows)

Metric: recall at a fixed false-alarm rate and AUC, using the same bandpass response for
every variant, so only the input differs.

Frames are rendered on a padded canvas and centre-cropped, so no pixel is ever filled by
border reflection - reflected content creates phantom correspondences and silently
corrupts both the registration and the fusion.

    python experiments/fusion_demo.py --frames 6 --targets 60
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from qreg.evaluate import ransac_transform  # noqa: E402
from qreg.features import match_candidates  # noqa: E402
from qreg.preprocess import bandpass, illum_norm  # noqa: E402

BORDER = 140


def composite_targets(bg, n, rng, contrast=12.0, rmin=5.5, rmax=8.5, margin=40):
    """Small, low-contrast partly-buried objects: dark body + offset shadow + bright rim."""
    img = bg.astype(np.float32).copy()
    h, w = img.shape[:2]
    lo = BORDER + margin
    centers, tries = [], 0
    while len(centers) < n and tries < 20000:
        tries += 1
        cx, cy = rng.uniform(lo, w - lo), rng.uniform(lo, h - lo)
        if any((cx - x) ** 2 + (cy - y) ** 2 < 55 ** 2 for x, y in centers):
            continue
        r = rng.uniform(rmin, rmax)
        ang = rng.uniform(0, 180)
        asp = rng.uniform(0.55, 1.0)
        mean = float(img[..., 1].mean())
        dx, dy = (rng.uniform(4, 8) * np.cos(np.deg2rad(ang)),
                  rng.uniform(4, 8) * np.sin(np.deg2rad(ang)))
        cv2.line(img, (int(cx + dx), int(cy + dy)), (int(cx + dx + 2 * r), int(cy + dy)),
                 (mean - 1.3 * contrast,) * 3, 3, cv2.LINE_AA)
        cv2.ellipse(img, (int(cx), int(cy)), (int(r), int(r * asp)), ang, 0, 360,
                    (mean - contrast,) * 3, -1, cv2.LINE_AA)
        cv2.ellipse(img, (int(cx - 1), int(cy - 1)), (int(r), int(r * asp)), ang,
                    200, 340, (mean + 0.8 * contrast,) * 3, 2, cv2.LINE_AA)
        centers.append((cx, cy))
    img = cv2.GaussianBlur(img, (0, 0), 0.7)
    return np.clip(img, 0, 255).astype(np.uint8), np.array(centers, np.float32)


def perturb(rng, size, deg=2.0, shift=8.0, scale=0.02):
    """Similarity about the image centre, so content stays inside the padded canvas."""
    a = np.deg2rad(rng.uniform(-deg, deg))
    s = 1.0 + rng.uniform(-scale, scale)
    R = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]]) * s
    c = np.array([size / 2.0, size / 2.0])
    H = np.eye(3)
    H[:2, :2] = R
    H[:2, 2] = c - R @ c + rng.uniform(-shift, shift, 2)
    return H


def make_frame(base, H, rng, size, gain=0.22, noise=9.0, blur=0.5):
    full = cv2.warpPerspective(base, H, (base.shape[1], base.shape[0]),
                               flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    f = full[BORDER:BORDER + size, BORDER:BORDER + size].astype(np.float32)
    f *= 1.0 + rng.uniform(-gain, gain)
    f = 255.0 * np.clip(f / 255.0, 0, 1) ** (1.0 + rng.uniform(-0.15, 0.15))
    yy, xx = np.mgrid[0:size, 0:size].astype(np.float32)
    cx, cy = rng.uniform(0, size), rng.uniform(0, size)
    rad = rng.uniform(0.35, 0.6) * size
    shadow = np.exp(-(((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * rad ** 2)))
    f *= (1.0 - 0.18 * shadow)[..., None]
    # per-view sun-angle shading: this is what decorrelates static terrain clutter
    # between frames, without it fusion can only average sensor noise.
    ang = rng.uniform(0, 2 * np.pi)
    ramp = ((xx - size / 2) * np.cos(ang) + (yy - size / 2) * np.sin(ang)) / size
    f *= (1.0 + 0.22 * ramp)[..., None]
    f = np.clip(f + rng.normal(0, noise, f.shape), 0, 255).astype(np.uint8)
    if blur > 0:
        k = int(2 * round(blur) + 1)
        cv2.GaussianBlur(f, (k, k), 0.6 * blur, dst=f)
    return f


def grid_err(H, Hgt, shape, n=7):
    h, w = shape[:2]
    yy, xx = np.mgrid[0:h:n * 1j, 0:w:n * 1j]
    p = np.stack([xx.ravel(), yy.ravel()], 1).reshape(-1, 1, 2)
    a = cv2.perspectiveTransform(p, H).reshape(-1, 2)
    b = cv2.perspectiveTransform(p, Hgt).reshape(-1, 2)
    return float(np.mean(np.linalg.norm(a - b, axis=1)))


def measure(img, centers, target_px, rng, n_neg=120000, fa=1e-2, valid=None):
    r = bandpass(img, target_px).astype(np.float32)
    h, w = r.shape
    if valid is None:
        valid = np.ones((h, w), bool)
    pos = []
    for cx, cy in centers:
        x0, y0 = int(cx) - 3, int(cy) - 3
        if not valid[min(h - 1, max(0, y0 + 3)), min(w - 1, max(0, x0 + 3))]:
            continue
        pos.append(float(r[max(0, y0):min(h, y0 + 7), max(0, x0):min(w, x0 + 7)].max()))
    pos = np.array(pos)
    ys = rng.integers(0, h, n_neg)
    xs = rng.integers(0, w, n_neg)
    keep = valid[ys, xs]
    for cx, cy in centers:
        keep &= (np.abs(xs - cx) > 25) | (np.abs(ys - cy) > 25)
    neg = r[ys[keep], xs[keep]]
    thr = np.quantile(neg, 1 - fa)
    recall = float((pos > thr).mean())
    allv = np.concatenate([pos, neg])
    lab = np.concatenate([np.ones(len(pos)), np.zeros(len(neg))])
    order = np.argsort(allv)
    ranks = np.empty(len(allv))
    ranks[order] = np.arange(1, len(allv) + 1)
    auc = (ranks[lab == 1].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg))
    snr = float((pos.mean() - neg.mean()) / (neg.std() + 1e-9))
    return recall, float(auc), snr


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--bg", default="data/zenodo/F1A.png")
    p.add_argument("--size", type=int, default=1200)
    p.add_argument("--targets", type=int, default=60)
    p.add_argument("--frames", type=int, default=6)
    p.add_argument("--target-px", type=float, default=13.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--noise", default="4,9,16,28",
                   help="comma list of per-frame sensor noise levels (grey levels)")
    a = p.parse_args(argv)

    raw = cv2.imread(a.bg)
    if raw is None:
        raise SystemExit(f"missing {a.bg}")
    h, w = raw.shape[:2]
    y0, x0 = (h - a.size) // 2 - BORDER, (w - a.size) // 2 - BORDER
    region0 = raw[y0:y0 + a.size + 2 * BORDER, x0:x0 + a.size + 2 * BORDER].copy()

    print(f"{a.targets} targets ~{a.target_px:.0f} px, {a.frames} frames, "
          f"real texture {Path(a.bg).name}")
    noises = [float(x) for x in a.noise.split(",")]
    print(f"{'noise':>6} {'reg err':>8} {'single':>7} {'naive':>7} {'ali-mean':>9} {'ali-med':>8}"
          f"   {'(AUC; higher is better)':<24}")
    print("-" * 72)
    for noise in noises:
        rng = np.random.default_rng(a.seed + int(noise))
        region, centers = composite_targets(region0.copy(), a.targets, rng)
        centers = centers - np.array([BORDER, BORDER], np.float32)
        Hs = [np.eye(3)] + [perturb(rng, region.shape[1]) for _ in range(a.frames - 1)]
        frames = [make_frame(region, H, rng, a.size, noise=noise) for H in Hs]
        Tp = np.array([[1, 0, BORDER], [0, 1, BORDER], [0, 0, 1]], float)
        Tm = np.array([[1, 0, -BORDER], [0, 1, -BORDER], [0, 0, 1]], float)
        gt = [Tm @ np.linalg.inv(H) @ Tp for H in Hs]

        est, gerr = [], []
        for j in range(1, a.frames):
            c = match_candidates(frames[j], frames[0], n_max=2000)
            H = ransac_transform(c)
            est.append(H)
            gerr.append(grid_err(H, gt[j], frames[0].shape) if H is not None else np.nan)

        cube = [cv2.warpPerspective(frames[j + 1], est[j], (a.size, a.size),
                                    flags=cv2.INTER_CUBIC) for j in range(len(est))]
        aligned = [frames[0]] + cube
        ones = np.ones((a.size, a.size), np.uint8)
        valid = np.ones((a.size, a.size), bool)
        for j in range(len(est)):
            valid &= cv2.warpPerspective(ones, est[j], (a.size, a.size),
                                         flags=cv2.INTER_NEAREST).astype(bool)

        variants = {
            "single": frames[0],
            "naive": np.clip(np.mean([f.astype(np.float32) for f in frames], 0), 0, 255).astype(np.uint8),
            "ali-mean": np.clip(np.mean([f.astype(np.float32) for f in aligned], 0), 0, 255).astype(np.uint8),
            "ali-med": np.median(np.stack([f.astype(np.float32) for f in aligned], 0), 0).astype(np.uint8),
        }
        aucs = {k: measure(v, centers, a.target_px, np.random.default_rng(1), valid=valid)[1]
                for k, v in variants.items()}
        print(f"{noise:>6.0f} {np.nanmean(gerr):>8.2f} {aucs['single']:>7.3f} {aucs['naive']:>7.3f} "
              f"{aucs['ali-mean']:>9.3f} {aucs['ali-med']:>8.3f}")
        if noise == noises[0]:
            print(f"\nbandpass scale sweep on the single frame (targets are {a.target_px:.0f} px):")
            for tp in (4, 7, 10, 13, 18, 26, 36):
                auc = measure(variants["single"], centers, tp,
                              np.random.default_rng(1), valid=valid)[1]
                print(f"  sigma-fit {tp:>2} px  AUC {auc:.3f}")
            cv2.imwrite("runs/fusion_single.png", variants["single"])
            cv2.imwrite("runs/fusion_aligned_med.png", variants["ali-med"])
    print("\n(registration err is mean transform error over a grid, in px)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
