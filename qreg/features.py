"""SIFT front-end: detection, RootSIFT descriptors, mutual-NN + ratio-test candidates.

The front-end is deliberately kept identical across all compared methods so that the
only variable is the *correspondence selection* stage.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class Candidates:
    """Candidate correspondences between two images.

    Attributes
    ----------
    p1, p2 : (N, 2) float32
        Keypoint coordinates in image 1 and image 2.
    dist : (N,) float32
        Descriptor distance (cosine after RootSIFT), lower is better.
    kp1, kp2 : (N,) int32
        Index of the source keypoint in each image (for one-to-one constraints).
    """

    p1: np.ndarray
    p2: np.ndarray
    dist: np.ndarray
    kp1: np.ndarray
    kp2: np.ndarray

    def __len__(self) -> int:
        return int(self.p1.shape[0])

    def subset(self, idx: np.ndarray) -> "Candidates":
        idx = np.asarray(idx)
        return Candidates(self.p1[idx], self.p2[idx], self.dist[idx], self.kp1[idx], self.kp2[idx])


def root_sift(desc: np.ndarray) -> np.ndarray:
    """RootSIFT: L1-normalise then element-wise sqrt. Cheap illumination robustness."""
    d = np.asarray(desc, dtype=np.float32)
    d /= d.sum(axis=1, keepdims=True) + 1e-7
    np.sqrt(d, out=d)
    return d


def detect(img: np.ndarray, nfeatures: int = 2000) -> tuple[np.ndarray, np.ndarray]:
    gray = img if img.ndim == 2 else cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    sift = cv2.SIFT_create(nfeatures=nfeatures, contrastThreshold=0.005, edgeThreshold=12)
    kps, desc = sift.detectAndCompute(gray, None)
    if desc is None or len(kps) == 0:
        return np.zeros((0, 2), np.float32), np.zeros((0, 128), np.float32)
    pts = np.array([k.pt for k in kps], dtype=np.float32)
    return pts, root_sift(desc)


def match_candidates(
    img1: np.ndarray,
    img2: np.ndarray,
    *,
    ratio: float = 0.85,
    n_max: int = 160,
    nfeatures: int = 2000,
    mutual: bool = True,
) -> Candidates:
    """Build candidate correspondences via (mutual) nearest neighbour + Lowe ratio test."""
    p1, d1 = detect(img1, nfeatures)
    p2, d2 = detect(img2, nfeatures)
    if len(p1) < 8 or len(p2) < 8:
        e = np.zeros((0, 2), np.float32)
        return Candidates(e, e, np.zeros(0, np.float32), np.zeros(0, np.int32), np.zeros(0, np.int32))

    # RootSIFT vectors are unit-ish; cosine similarity == dot product.
    sim = d1 @ d2.T
    nn12 = sim.argmax(axis=1)
    nn21 = sim.argmax(axis=0)

    # Lowe ratio on the top-2 in image 2.
    top2 = np.partition(sim, -2, axis=1)[:, -2:]
    best = top2[:, 1]
    second = top2[:, 0]
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio_ok = (1.0 - best) < ratio * (1.0 - second)

    idx1 = np.arange(len(p1))
    keep = ratio_ok.copy()
    if mutual:
        keep &= nn21[nn12] == idx1
    idx1 = idx1[keep]
    if idx1.size == 0:
        e = np.zeros((0, 2), np.float32)
        return Candidates(e, e, np.zeros(0, np.float32), np.zeros(0, np.int32), np.zeros(0, np.int32))

    idx2 = nn12[idx1]
    dist = (1.0 - sim[idx1, idx2]).astype(np.float32)

    # Keep the strongest n_max candidates.
    order = np.argsort(dist)[:n_max]
    return Candidates(p1[idx1[order]], p2[idx2[order]], dist[order],
                      idx1[order].astype(np.int32), idx2[order].astype(np.int32))


def gt_inlier_mask(cand: Candidates, h_gt: np.ndarray, thresh: float = 3.0) -> np.ndarray:
    """Label candidates as geometric inliers w.r.t. a ground-truth homography (img1 -> img2)."""
    if len(cand) == 0:
        return np.zeros(0, bool)
    ph = cv2.perspectiveTransform(cand.p1.reshape(-1, 1, 2), h_gt).reshape(-1, 2)
    return np.linalg.norm(ph - cand.p2, axis=1) < thresh


def spatial_isotropy(pts: np.ndarray) -> float:
    """Axis ratio sqrt(l_min/l_max) of the point cloud. 1 = isotropic, 0 = collinear.

    A DLT from a collinear or clustered set is ill-conditioned no matter how
    geometrically consistent every pair is, so this is the quantity the energy has
    no term for (it is a *global* property of the selected set).
    """
    pts = np.asarray(pts, float)
    if len(pts) < 3:
        return 0.0
    c = pts - pts.mean(0)
    ev = np.linalg.eigvalsh(c.T @ c / len(pts))
    return float(np.sqrt(max(ev[0], 0.0) / max(ev[1], 1e-9)))


def inject_outliers(cand: Candidates, h_gt: np.ndarray, target_frac: float,
                    rng: np.random.Generator, total: int = 240,
                    thresh: float = 3.0) -> Candidates:
    """Reshape the candidate set to a target inlier fraction at fixed size.

    Keeps the best true inliers, fills the remaining budget with the worst real
    outliers and then with random correspondences. Holding `total` fixed across a
    sweep over `target_frac` isolates selector robustness from descriptor quality and
    from QUBO size, which is exactly the axis this project claims an advantage on.
    """
    if len(cand) == 0 or not (0.0 < target_frac < 1.0):
        return cand
    inl = gt_inlier_mask(cand, h_gt, thresh)
    idx_in = np.flatnonzero(inl)
    idx_out = np.flatnonzero(~inl)
    n_in = min(int(round(target_frac * total)), len(idx_in))
    n_out = total - n_in

    keep_in = idx_in[np.argsort(cand.dist[idx_in])[:n_in]] if n_in else np.zeros(0, np.int64)
    keep_out = idx_out[np.argsort(cand.dist[idx_out])[:n_out]] if n_out else np.zeros(0, np.int64)
    base = np.sort(np.concatenate([keep_in, keep_out]).astype(np.int64))

    need = n_out - len(keep_out)
    if need <= 0:
        return cand.subset(base)

    lo1, hi1 = cand.p1.min(0), cand.p1.max(0)
    lo2, hi2 = cand.p2.min(0), cand.p2.max(0)
    next_id = int(max(cand.kp1.max(), cand.kp2.max())) + 1
    fake = Candidates(
        p1=rng.uniform(lo1, hi1, (need, 2)).astype(np.float32),
        p2=rng.uniform(lo2, hi2, (need, 2)).astype(np.float32),
        dist=np.full(need, float(cand.dist.max()) + 1.0, np.float32),
        kp1=np.arange(next_id, next_id + need, dtype=np.int32),
        kp2=np.arange(next_id, next_id + need, dtype=np.int32),
    )
    merged = cand.subset(base)
    return Candidates(
        p1=np.concatenate([merged.p1, fake.p1]),
        p2=np.concatenate([merged.p2, fake.p2]),
        dist=np.concatenate([merged.dist, fake.dist]),
        kp1=np.concatenate([merged.kp1, fake.kp1]),
        kp2=np.concatenate([merged.kp2, fake.kp2]),
    )
