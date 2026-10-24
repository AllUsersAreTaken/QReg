"""Turn a selected subset of correspondences into a transform, and score it."""

from __future__ import annotations

import cv2
import numpy as np

from .features import Candidates


def estimate(cand: Candidates, sel, *, pad_to: int = 4, model: str = "homography"):
    """Least-squares (no RANSAC) estimate from the selected matches.

    Returns (H 3x3, n_used, rms, padded). H is None if estimation fails.
    """
    sel = np.asarray(sel)
    if sel.dtype == bool:
        idx = np.flatnonzero(sel)
    else:
        idx = sel.astype(np.int64)

    padded = False
    if idx.size < pad_to:
        order = np.argsort(cand.dist)
        extra = [i for i in order if i not in set(idx.tolist())]
        idx = np.concatenate([idx, np.array(extra[: pad_to - idx.size], np.int64)])
        padded = True
    if idx.size < pad_to:
        return None, int(idx.size), float("inf"), padded

    src = cand.p1[idx].astype(np.float64).reshape(-1, 1, 2)
    dst = cand.p2[idx].astype(np.float64).reshape(-1, 1, 2)

    H = None
    if model == "homography" and idx.size >= 4:
        H, _ = cv2.findHomography(src, dst, 0)
    if H is None:
        M, _ = cv2.estimateAffinePartial2D(src, dst)
        if M is not None:
            H = np.eye(3)
            H[:2] = M
    if H is None or not np.all(np.isfinite(H)):
        return None, int(idx.size), float("inf"), padded

    proj = cv2.perspectiveTransform(src, H).reshape(-1, 2)
    rms = float(np.sqrt(np.mean(np.sum((proj - dst.reshape(-1, 2)) ** 2, axis=1))))
    return H, int(idx.size), rms, padded


def _normalise(pts: np.ndarray):
    c = pts.mean(0)
    d = np.sqrt(((pts - c) ** 2).sum(1)).mean()
    s = np.sqrt(2.0) / max(d, 1e-9)
    T = np.array([[s, 0, -s * c[0]], [0, s, -s * c[1]], [0, 0, 1]])
    return T, (pts - c) * s


def weighted_dlt(src: np.ndarray, dst: np.ndarray, w: np.ndarray) -> np.ndarray:
    """Weighted DLT with Hartley normalisation (cv2.findHomography has no weighted API)."""
    T1, s1 = _normalise(src)
    T2, s2 = _normalise(dst)
    n = len(src)
    x, y = s1[:, 0], s1[:, 1]
    u, v = s2[:, 0], s2[:, 1]
    A = np.zeros((2 * n, 9))
    A[0::2, 0], A[0::2, 1], A[0::2, 2] = -x, -y, -1.0
    A[0::2, 6], A[0::2, 7], A[0::2, 8] = u * x, u * y, u
    A[1::2, 3], A[1::2, 4], A[1::2, 5] = -x, -y, -1.0
    A[1::2, 6], A[1::2, 7], A[1::2, 8] = v * x, v * y, v
    sw = np.sqrt(np.maximum(w, 0.0)).repeat(2)[:, None]
    _, _, Vt = np.linalg.svd(A * sw, full_matrices=False)
    Hn = Vt[-1].reshape(3, 3)
    H = np.linalg.inv(T2) @ Hn @ T1
    if abs(H[2, 2]) < 1e-12:
        return None
    return H / H[2, 2]


def residuals(src: np.ndarray, dst: np.ndarray, H) -> np.ndarray:
    p = cv2.perspectiveTransform(src.reshape(-1, 1, 2), H).reshape(-1, 2)
    return np.linalg.norm(p - dst, axis=1)


def estimate_irls(cand: Candidates, sel, *, iters: int = 15, pad_to: int = 4,
                  thresh: float = 3.0):
    """Robust refit over ALL candidates, seeded by the selection.

    The selected subset only *initialises* the fit; the final homography is fitted on
    the whole candidate set with Tukey reweighting at a fixed consensus threshold
    (the same 3 px RANSAC is given, so the comparison is fair). A subset-only decode
    is limited by the spatial conditioning of that subset — the reason subset-only
    QUBO lost to RANSAC regardless of energy quality.
    """
    sel = np.asarray(sel)
    idx = np.flatnonzero(sel) if sel.dtype == bool else sel.astype(np.int64)
    mask = np.zeros(len(cand), bool)
    mask[idx[idx < len(cand)]] = True
    if mask.sum() < pad_to:
        for i in np.argsort(cand.dist):
            mask[i] = True
            if mask.sum() >= pad_to:
                break
    if mask.sum() < pad_to:
        return None, int(mask.sum()), float("inf"), True

    src = cand.p1.astype(np.float64)
    dst = cand.p2.astype(np.float64)

    H = weighted_dlt(src[mask], dst[mask], np.ones(int(mask.sum())))
    if H is None or not np.all(np.isfinite(H)):
        H = weighted_dlt(src, dst, np.ones(len(cand)))
    if H is None or not np.all(np.isfinite(H)):
        return None, int(mask.sum()), float("inf"), False

    c = float(thresh)
    w = mask.astype(np.float64)
    for _ in range(iters):
        r = residuals(src, dst, H)
        u = np.clip(r / c, 0.0, 1.0)
        w = (1.0 - u ** 2) ** 2
        guard = 0
        while (w > 1e-9).sum() < max(pad_to + 2, 6) and guard < 8:
            c *= 1.5
            u = np.clip(r / c, 0.0, 1.0)
            w = (1.0 - u ** 2) ** 2
            guard += 1
        H_new = weighted_dlt(src, dst, w)
        if H_new is None or not np.all(np.isfinite(H_new)):
            break
        H = H_new

    r = residuals(src, dst, H)
    n_used = int((r < c).sum())
    rms = float(np.sqrt(np.mean(r ** 2)))
    return H, n_used, rms, False


def corners(shape) -> np.ndarray:
    h, w = shape[:2]
    return np.array([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]], np.float64)


def corner_error(H, h_gt, shape) -> float:
    """Mean L2 displacement of the 4 image corners against the ground-truth transform."""
    if H is None:
        return float("inf")
    c = corners(shape).reshape(-1, 1, 2)
    a = cv2.perspectiveTransform(c, H).reshape(-1, 2)
    b = cv2.perspectiveTransform(c, h_gt).reshape(-1, 2)
    return float(np.mean(np.linalg.norm(a - b, axis=1)))


def overlap_ncc(img1: np.ndarray, img2: np.ndarray, H) -> float:
    """GT-free registration score: NCC between img1 and img2 warped back by H."""
    if H is None:
        return float("nan")
    try:
        Hinv = np.linalg.inv(H)
    except np.linalg.LinAlgError:
        return float("nan")
    h, w = img1.shape[:2]
    warped = cv2.warpPerspective(img2, Hinv, (w, h), flags=cv2.INTER_LINEAR)
    valid = cv2.warpPerspective(np.ones(img2.shape[:2], np.uint8) * 255, Hinv, (w, h),
                                flags=cv2.INTER_NEAREST) > 0
    if valid.sum() < 100:
        return float("nan")
    a = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY).astype(np.float64)[valid]
    b = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY).astype(np.float64)[valid]
    a -= a.mean()
    b -= b.mean()
    den = np.sqrt((a * a).sum() * (b * b).sum())
    return float((a * b).sum() / den) if den > 0 else float("nan")
