"""Relief-aware pair generator: drone frame vs a reference orthophoto.

A flat-plane homography is the wrong model as soon as the ground has relief, and
orthomosaic registration is exactly that case. Frame 0 here is the reference orthophoto
(pixel = world XY). Frame 1 is rendered from a real camera at altitude h with tilt, so
its pixels come from ray-marching each ray onto the height field. That makes the true
image-to-image mapping non-homographic, while still giving exact per-pixel ground truth.

Because relief displacement is (Z/h) * baseline, both a decent relief amplitude and a
decent baseline are needed before the plane model visibly breaks; `PairReliefConfig`
exposes both.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from .synth import scene_texture


@dataclass
class ReliefConfig:
    size: int = 512
    relief: float = 0.18        # height amplitude as a fraction of altitude
    relief_scale: int = 6       # blob size, larger = smoother terrain
    baseline: float = 0.35      # camera translation as a fraction of altitude
    tilt_deg: float = 3.0
    yaw_deg: float = 0.0
    photometric: float = 0.5
    blur: float = 0.4
    noise: float = 2.0
    seed: int = 0


def height_field(size: int, amplitude: float, scale: int = 6,
                 seed: int = 0) -> np.ndarray:
    """Smooth random terrain plus a few dome 'buildings', in world units (pixels)."""
    rng = np.random.default_rng(seed)
    n = max(3, size // max(scale, 1))
    base = rng.normal(size=(n + 2, n + 2)).astype(np.float32)
    up = cv2.resize(base, (size, size), interpolation=cv2.INTER_CUBIC)
    up = cv2.GaussianBlur(up, (0, 0), size / 40.0)
    up /= np.abs(up).max() + 1e-9

    yy, xx = np.mgrid[0:size, 0:size].astype(np.float32)
    for _ in range(rng.integers(3, 7)):
        cx, cy = rng.uniform(0.1, 0.9, 2) * size
        r = rng.uniform(0.05, 0.16) * size
        h = rng.uniform(0.5, 1.0) * (1 if rng.random() < 0.7 else -0.7)
        up += h * np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * r ** 2))

    up /= np.abs(up).max() + 1e-9
    return (up * amplitude).astype(np.float32)


class Camera:
    """Pinhole, looking down at a height field defined over ortho pixels."""

    def __init__(self, size: int, center, altitude: float, tilt_deg=0.0,
                 yaw_deg=0.0, f_scale: float = 0.9):
        self.size = size
        self.C = np.array([center[0], center[1], altitude], float)
        self.f = f_scale * size
        self.cx = size / 2.0
        self.cy = size / 2.0
        nadir = np.array([[1.0, 0, 0], [0, 1.0, 0], [0, 0, -1.0]])
        a, y = np.deg2rad(tilt_deg), np.deg2rad(yaw_deg)
        rt = np.array([[1, 0, 0],
                       [0, np.cos(a), -np.sin(a)],
                       [0, np.sin(a), np.cos(a)]])
        rz = np.array([[np.cos(y), -np.sin(y), 0],
                       [np.sin(y), np.cos(y), 0],
                       [0, 0, 1]])
        self.R = rt @ rz @ nadir          # world -> camera

    def project(self, P: np.ndarray):
        """World points (N,3) -> (u, v, depth)."""
        pc = (P - self.C) @ self.R.T
        depth = pc[:, 2]
        return self.f * pc[:, 0] / depth + self.cx, self.f * pc[:, 1] / depth + self.cy, depth

    def _ray_dirs(self):
        s = self.size
        yy, xx = np.mgrid[0:s, 0:s].astype(np.float32)
        d_cam = np.stack([(xx - self.cx) / self.f, (yy - self.cy) / self.f,
                          np.ones_like(xx)], -1)
        return d_cam @ self.R            # world directions, s > 0, pointing down


def _sample(img: np.ndarray, u: np.ndarray, v: np.ndarray) -> np.ndarray:
    return cv2.remap(img, u.astype(np.float32), v.astype(np.float32),
                     interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)


def _surface_points(cam: Camera, Z: np.ndarray, iters: int = 7):
    """Ray-march every pixel onto the height surface. Returns (P, valid)."""
    size = cam.size
    d = cam._ray_dirs()
    s = np.full((size, size), cam.C[2], np.float32)
    Zf = Z.astype(np.float32)
    for _ in range(iters):
        P = cam.C + s[..., None] * d
        Zi = _sample(Zf, P[..., 0], P[..., 1])
        s = (Zi - cam.C[2]) / d[..., 2]
    P = cam.C + s[..., None] * d
    valid = ((P[..., 0] >= 0) & (P[..., 0] < size - 1) &
             (P[..., 1] >= 0) & (P[..., 1] < size - 1) & (s > 0))
    return P, valid


def make_relief_pair(cfg: ReliefConfig = ReliefConfig()):
    """Orthophoto (img0), rendered drone frame (img1), and dense GT correspondence.

    Returns dict with img0, img1, map (u0,v0)->(u1,v1), valid mask, Z, cameras.
    """
    rng = np.random.default_rng(cfg.seed)
    s = cfg.size
    tex = scene_texture(s, cfg.seed)
    Z = height_field(s, cfg.relief * s, cfg.relief_scale, cfg.seed)

    cam0_center = (s / 2.0, s / 2.0)
    cam = Camera(s, (s / 2.0 + cfg.baseline * s, s / 2.0 + 0.4 * cfg.baseline * s),
                 altitude=s, tilt_deg=cfg.tilt_deg, yaw_deg=cfg.yaw_deg)

    # Render the drone frame: every pixel ray-marched onto the surface, then textured.
    P, valid = _surface_points(cam, Z)
    img1 = _sample(tex, P[..., 0], P[..., 1])
    img1[~valid] = 0

    # Reference orthophoto: frame-0 pixel (u, v) is world (u, v) with height Z(u, v).
    yy, xx = np.mgrid[0:s, 0:s].astype(np.float32)
    P0 = np.stack([xx, yy, Z], -1)
    u1, v1, _ = cam.project(P0.reshape(-1, 3))
    map1 = np.stack([u1.reshape(s, s), v1.reshape(s, s)], -1).astype(np.float32)
    gt_valid = valid & (v1.reshape(s, s) >= 0) & (v1.reshape(s, s) < s) & \
        (u1.reshape(s, s) >= 0) & (u1.reshape(s, s) < s)

    img0 = tex.copy()

    # Photometric difference + mild degradation, independent per image.
    def photo(im, strength):
        if strength <= 0:
            return im
        out = im.astype(np.float32)
        out *= 1.0 + strength * rng.uniform(-0.25, 0.25)
        out += strength * rng.uniform(-18, 18)
        out = 255.0 * np.clip(out / 255.0, 0, 1) ** (1.0 + strength * rng.uniform(-0.2, 0.2))
        return np.clip(out, 0, 255).astype(np.uint8)

    img0 = photo(img0, 0.3 * cfg.photometric)
    img1 = photo(img1, cfg.photometric)
    for im in (img0, img1):
        if cfg.blur > 0:
            k = int(2 * round(cfg.blur) + 1)
            cv2.GaussianBlur(im, (k, k), 0.6 * cfg.blur, dst=im)
        if cfg.noise > 0:
            im += rng.normal(0, cfg.noise, im.shape).astype(np.uint8)

    return {"img0": img0, "img1": img1, "map": map1, "valid": gt_valid,
            "Z": Z, "cam": cam, "cfg": cfg}


def local_affine_gt(pair: dict, x0, y0, x1, y1) -> np.ndarray:
    """Best affine fit to the dense GT mapping inside a rectangle (2x3)."""
    m, v = pair["map"], pair["valid"]
    ys, xs = np.mgrid[y0:y1, x0:x1]
    ok = v[y0:y1, x0:x1]
    if ok.sum() < 12:
        return None
    src = np.stack([xs[ok], ys[ok]], 1).astype(np.float64)
    dst = m[y0:y1, x0:x1][ok].astype(np.float64)
    A, _ = cv2.estimateAffine2D(src, dst)
    return A
