"""Drone flight-strip generator: a *sequence*, not a pair.

The pairwise benchmarks used everywhere in registration research are not the drone
problem. A drone captures a strip at a few Hz, so consecutive frames have ~70-80%
overlap and an almost-identity transform, and the mosaic is built by chaining
many such estimates. What matters then is not per-pair corner error but the *drift*
accumulated along the strip, and the fact that any pair far enough apart to be
informative barely overlaps.

Geometry is exact: every frame is a homography of one ground plane,

    H_t = K @ R_wc @ [[1,0,-X_t],[0,1,-Y_t],[0,0,h_t]]

so the ground-truth transform between any two frames is H_j @ inv(H_i). Tilt, yaw
drift, altitude jitter and photometric drift are independent knobs.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from .synth import scene_texture


@dataclass
class StripConfig:
    size: int = 384              # frame size
    scene: int = 2048            # ground-plane texture size
    n_frames: int = 12
    overlap: float = 0.75        # along-track overlap between consecutive frames
    tilt_deg: float = 3.0        # peak out-of-nadir tilt
    yaw_drift_deg: float = 2.0   # camera pans slowly along the strip
    lateral_amp: float = 0.25    # sinusoidal cross-track wander, in steps
    altitude_jitter: float = 0.03
    photometric_drift: float = 0.35
    blur: float = 1.0
    noise: float = 3.0
    seed: int = 0


def _rot(yaw, pitch, roll):
    cy, sy = np.cos(yaw), np.sin(yaw)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cr, sr = np.cos(roll), np.sin(roll)
    rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
    rx = np.array([[1, 0, 0], [0, cp, -sp], [0, sp, cp]])
    ry = np.array([[cr, 0, sr], [0, 1, 0], [-sr, 0, cr]])
    return rz @ rx @ ry


def _frame_homographies(cfg: StripConfig, rng: np.random.Generator) -> np.ndarray:
    s = cfg.size
    f = 0.9 * s
    K = np.array([[f, 0, s / 2], [0, f, s / 2], [0, 0, 1]], float)
    h0 = float(s)
    footprint = s / (f / h0)                     # scene px covered by one frame
    step = footprint * (1.0 - cfg.overlap)

    Hs = np.empty((cfg.n_frames, 3, 3))
    # Keep every frame window strictly inside the texture. Without this, the part of a
    # frame outside the scene is filled by border reflection, which injects mirrored
    # content and creates phantom "consensus" matches (a reflection reads as a valid
    # homography, so RANSAC happily locks onto it).
    pad = 0.65 * footprint
    x0 = pad
    y0 = cfg.scene / 2.0
    if x0 + (cfg.n_frames - 1) * step + pad > cfg.scene:
        raise ValueError(f"scene too small: need ~{x0 + (cfg.n_frames - 1) * step + pad:.0f}px")

    for t in range(cfg.n_frames):
        X = x0 + t * step
        Y = y0 + cfg.lateral_amp * step * np.sin(0.9 * t) + rng.normal(0, 0.02 * step)
        h = h0 * (1.0 + cfg.altitude_jitter * np.sin(0.7 * t + 0.3))
        yaw = np.deg2rad(cfg.yaw_drift_deg) * np.sin(0.45 * t)
        pitch = np.deg2rad(cfg.tilt_deg) * np.sin(0.55 * t + 0.4)
        roll = np.deg2rad(0.5 * cfg.tilt_deg) * np.sin(0.8 * t)

        R = _rot(yaw, pitch, roll)
        M = np.array([[1, 0, -X], [0, 1, -Y], [0, 0, h]], float)
        Hs[t] = K @ R @ M
    return Hs


def _photometric(img, rng, t, n, strength):
    if strength <= 0:
        return img
    a = strength * (0.5 + 0.5 * np.sin(np.pi * t / max(n - 1, 1)))
    out = img.astype(np.float32)
    out *= 1.0 + a * rng.uniform(-0.6, 0.6) + a * 0.5 * np.sin(0.5 * t)
    out *= 1.0 + a * rng.uniform(-0.15, 0.15, 3)          # slow colour-temperature drift
    out = 255.0 * np.clip(out / 255.0, 0, 1) ** (1.0 + a * rng.uniform(-0.25, 0.25))
    return np.clip(out, 0, 255).astype(np.uint8)


def _motion_blur(img, length: float):
    if length <= 0:
        return img
    k = int(round(length))
    if k < 2:
        return img
    kern = np.zeros((k, k), np.float32)
    kern[k // 2, :] = 1.0 / k
    return cv2.filter2D(img, -1, kern)


def make_strip(cfg: StripConfig = StripConfig()):
    """Return dict with frames, scene->frame homographies, and pairwise GT."""
    rng = np.random.default_rng(cfg.seed)
    tex = scene_texture(cfg.scene, cfg.seed, repetitive=False)
    Hs = _frame_homographies(cfg, rng)

    frames = []
    for t in range(cfg.n_frames):
        img = cv2.warpPerspective(tex, Hs[t], (cfg.size, cfg.size), flags=cv2.INTER_LINEAR,
                                  borderMode=cv2.BORDER_REPLICATE)
        img = _photometric(img, rng, t, cfg.n_frames, cfg.photometric_drift)
        img = _motion_blur(img, cfg.blur * (1.0 + 0.4 * np.sin(0.6 * t)))
        if cfg.noise > 0:
            img = np.clip(img + rng.normal(0, cfg.noise, img.shape), 0, 255).astype(np.uint8)
        frames.append(img)
    return {"frames": frames, "H": Hs, "cfg": cfg}


def gt_between(strip: dict, i: int, j: int) -> np.ndarray:
    """Ground-truth homography mapping frame i coordinates -> frame j coordinates."""
    H = strip["H"]
    return H[j] @ np.linalg.inv(H[i])


def overlap_fraction(cfg: StripConfig, gap: int) -> float:
    """Approximate along-track overlap between frames `gap` apart."""
    return max(0.0, 1.0 - gap * (1.0 - cfg.overlap))
