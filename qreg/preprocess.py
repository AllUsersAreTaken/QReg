"""Preprocessing operators for small-surface-object detection.

These operate on the input image and change nothing about the detector, which is the
point: they can be evaluated as an ablation against a fixed set of weights.

Design notes that matter for IED/surface-ordinate work:

- Bandpass at the target scale is the highest-value operator. Small partly-buried
  objects are anomalies at one spatial scale; terrain, ruts and vegetation live at
  others. It is also a relative-contrast filter, so it survives exposure change.
- Illumination normalisation must be edge-preserving. Shadows are *signal* here (a
  protruding edge reads as a shadow line), so CLAHE-style shadow suppression can
  delete the best cue. We divide by a heavily blurred copy instead.
- GSD canonicalisation resamples so the target occupies a fixed pixel footprint at any
  altitude. YOLO boxes are normalised, so resizing does not invalidate labels.
"""

from __future__ import annotations

import cv2
import numpy as np


def to_gray(img: np.ndarray) -> np.ndarray:
    if img.ndim == 2:
        return img
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def illum_norm(img: np.ndarray, sigma_frac: float = 0.06) -> np.ndarray:
    """Divide out low-frequency illumination, keep local structure (incl. shadow edges)."""
    g = to_gray(img).astype(np.float32) + 1.0
    s = max(3.0, sigma_frac * max(g.shape))
    bg = cv2.GaussianBlur(g, (0, 0), s)
    out = g / (bg + 1e-3)
    out -= out.min()
    out *= 255.0 / (out.max() + 1e-6)
    return out.astype(np.uint8)


def bandpass(img: np.ndarray, target_px: float, lo: float = 0.5, hi: float = 2.5) -> np.ndarray:
    """Difference of Gaussians tuned to an object of `target_px` across."""
    g = to_gray(img).astype(np.float32)
    s1 = max(0.6, lo * target_px / 2.0)
    s2 = max(s1 * 1.6, hi * target_px / 2.0)
    resp = cv2.GaussianBlur(g, (0, 0), s1) - cv2.GaussianBlur(g, (0, 0), s2)
    resp = np.abs(resp)
    resp -= resp.min()
    resp *= 255.0 / (resp.max() + 1e-6)
    return resp.astype(np.uint8)


def canonical_gsd(img: np.ndarray, scale: float, target_px: float, cur_px: float) -> tuple:
    """Resample so an object currently `cur_px` wide becomes `target_px` wide."""
    s = target_px / max(cur_px, 1e-6)
    out = cv2.resize(img, None, fx=s, fy=s,
                     interpolation=cv2.INTER_AREA if s < 1 else cv2.INTER_CUBIC)
    return out, s


def fuse(imgs: list[np.ndarray], method: str = "median") -> np.ndarray:
    st = np.stack([im.astype(np.float32) for im in imgs], 0)
    out = np.median(st, axis=0) if method == "median" else st.mean(axis=0)
    return np.clip(out, 0, 255).astype(np.uint8)


def apply_variant(img: np.ndarray, variant: str, target_px: float, scale: float = 1.0) -> np.ndarray:
    """Composable preprocessing used by the ablation exporter."""
    out = img
    if variant in ("illum", "illum+bandpass"):
        out = illum_norm(out)
    if variant in ("bandpass", "illum+bandpass"):
        out = bandpass(out, target_px)
    if scale != 1.0:
        out = cv2.resize(out, None, fx=scale, fy=scale,
                         interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_CUBIC)
    return out
