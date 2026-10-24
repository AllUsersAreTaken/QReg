"""Synthetic oblique-view + photometric-change pair generator.

Motivated by the target use case: aerial RGB frames from a drone, where the failure
modes are (a) out-of-plane-ish oblique viewing angles between overlapping frames and
(b) large illumination/exposure/shadow changes.  Both are dialled independently here
so that the RANSAC-vs-annealer crossover can be located instead of guessed at.

Geometry: a plane at z=1 viewed by a virtual camera with intrinsics K and a rotation
R, giving the exact homography  H = K (R + t n^T / d) K^-1.  Content is recentred so
overlap stays high at large tilt, isolating projective distortion from pure crop.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class PairConfig:
    size: int = 512
    oblique_deg: float = 30.0     # out-of-plane tilt between the two views
    yaw_deg: float = 10.0
    roll_deg: float = 0.0
    shift: float = 0.03           # translation / depth ratio
    f_scale: float = 0.9          # focal length as a fraction of image width
    photometric: float = 0.5      # 0 = none, 1 = strong
    blur: float = 0.5
    noise: float = 3.0
    repetitive: bool = False      # tiled self-similar content (RANSAC trap)
    seed: int = 0


def _lowfreq(rng, size, cells):
    small = rng.random((cells, cells, 3)).astype(np.float32)
    return cv2.resize(small, (size, size), interpolation=cv2.INTER_CUBIC)


def scene_texture(size: int = 512, seed: int = 0, repetitive: bool = False) -> np.ndarray:
    rng = np.random.default_rng(seed)
    img = 90.0 + 40.0 * _lowfreq(rng, size, 6)

    if repetitive:
        # Identical repeated tiles: globally ambiguous, locally distinctive.
        tile = np.zeros((64, 64, 3), np.float32)
        tile[:] = 120
        cv2.rectangle(tile, (8, 8), (56, 56), (200, 200, 200), -1)
        cv2.circle(tile, (32, 32), 12, (60, 60, 60), -1)
        cv2.line(tile, (0, 0), (63, 63), (240, 240, 240), 2)
        for y in range(0, size, 64):
            for x in range(0, size, 64):
                img[y:y + 64, x:x + 64] = tile[:min(64, size - y), :min(64, size - x)]
    else:
        # Aerial-plausible content: dark roads, bright fields, mid-tone blocks.
        for _ in range(max(6, int(6 * (size / 512) ** 2))):
            p1 = tuple(rng.integers(0, size, 2))
            p2 = tuple(rng.integers(0, size, 2))
            cv2.line(img, p1, p2, (70, 70, 70), int(rng.integers(3, 9)))
        for _ in range(int(120 * (size / 512) ** 2)):
            x, y = rng.integers(0, size, 2)
            w, h = rng.integers(6, 46, 2)
            col = tuple(float(c) for c in rng.integers(40, 230, 3))
            if rng.random() < 0.5:
                cv2.rectangle(img, (x, y), (x + w, y + h), col, -1)
                cv2.rectangle(img, (x, y), (x + w, y + h), (30, 30, 30), 1)
            else:
                cv2.ellipse(img, (int(x), int(y)), (int(w), int(h)),
                            float(rng.integers(0, 180)), 0, 360, col, -1)
        for _ in range(int(40 * (size / 512) ** 2)):
            x, y = rng.integers(0, size, 2)
            cv2.circle(img, (int(x), int(y)), int(rng.integers(2, 6)),
                       tuple(float(c) for c in rng.integers(20, 240, 3)), -1)

    img += rng.normal(0, 4.0, img.shape)
    img = cv2.GaussianBlur(img, (3, 3), 0.6)
    return np.clip(img, 0, 255).astype(np.uint8)


def _rot(pitch, yaw, roll):
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw), np.sin(yaw)
    cr, sr = np.cos(roll), np.sin(roll)
    rx = np.array([[1, 0, 0], [0, cp, -sp], [0, sp, cp]])
    ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    rz = np.array([[cr, -sr, 0], [sr, cr, 0], [0, 0, 1]])
    return rz @ ry @ rx


def oblique_homography(cfg: PairConfig) -> np.ndarray:
    s = cfg.size
    f = cfg.f_scale * s
    K = np.array([[f, 0, s / 2], [0, f, s / 2], [0, 0, 1]], float)
    R = _rot(np.deg2rad(cfg.oblique_deg), np.deg2rad(cfg.yaw_deg), np.deg2rad(cfg.roll_deg))
    t = np.array([cfg.shift, 0.5 * cfg.shift, 0.0])
    n = np.array([0.0, 0.0, 1.0])
    H = K @ (R + np.outer(t, n)) @ np.linalg.inv(K)

    # Normalise projected area back to the frame, then recentre. This keeps the
    # projected *scale* constant across tilt so that a sweep over oblique_deg
    # isolates projective distortion from a trivial crop / zoom change.
    c = np.array([[0, 0], [s, 0], [s, s], [0, s]], float).reshape(-1, 1, 2)
    quad = cv2.perspectiveTransform(c, H).reshape(-1, 2)
    area = abs(cv2.contourArea(quad.astype(np.float32)))
    k = s / np.sqrt(max(area, 1e-6))
    ctr = quad.mean(0)
    S = np.array([[k, 0, 0], [0, k, 0], [0, 0, 1]], float)
    tgt = s / 2.0
    T = np.array([[1, 0, tgt - k * ctr[0]], [0, 1, tgt - k * ctr[1]], [0, 0, 1]], float)
    return T @ S @ H


def _photometric(img, rng, strength: float):
    if strength <= 0:
        return img
    out = img.astype(np.float32)
    out *= rng.uniform(1.0 - 0.35 * strength, 1.0 + 0.35 * strength)
    out += rng.uniform(-25, 25) * strength
    out = 255.0 * np.clip(out / 255.0, 0, 1) ** rng.uniform(1 - 0.5 * strength, 1 + 0.5 * strength)
    out *= rng.uniform(1 - 0.25 * strength, 1 + 0.25 * strength, 3)
    # Soft shadow / vignette gradient (cloud shadow).
    h, w = img.shape[:2]
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    cx, cy = rng.uniform(0, w), rng.uniform(0, h)
    rad = rng.uniform(0.4, 0.9) * max(h, w)
    m = np.exp(-(((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * rad ** 2)))
    out *= (1.0 - 0.45 * strength * m)[..., None]
    return np.clip(out, 0, 255).astype(np.uint8)


def make_pair(cfg: PairConfig):
    """Return (img1, img2, H_gt, texture) with H_gt mapping image-1 coords to image-2."""
    rng = np.random.default_rng(cfg.seed)
    tex = scene_texture(cfg.size, cfg.seed, cfg.repetitive)
    H = oblique_homography(cfg)

    img1 = tex.copy()
    img2 = cv2.warpPerspective(tex, H, (cfg.size, cfg.size),
                               flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)

    img1 = _photometric(img1, rng, 0.25 * cfg.photometric)
    img2 = _photometric(img2, rng, cfg.photometric)
    for im in (img1, img2):
        if cfg.blur > 0:
            k = int(2 * round(cfg.blur) + 1)
            cv2.GaussianBlur(im, (k, k), 0.6 * cfg.blur, dst=im)
        if cfg.noise > 0:
            im += rng.normal(0, cfg.noise, im.shape).astype(np.uint8)
    return img1, img2, H, tex


def make_pair_from_images(img1, img2, h_gt):
    """Adapter so real datasets (Oxford / HPatches / drone pairs) share the same path."""
    return img1, img2, np.asarray(h_gt, float), None
