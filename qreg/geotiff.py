"""GeoTIFF orthomosaic handling: pixel<->UTM affine from embedded tags.

The Demining Research Community release has drone RGB orthomosaics of the same seeded
field taken months/years apart, but they ship in different CRSs (one geographic, one
projected) and different footprints. Their geotags give an exact alignment, which is
what makes them usable as *real registration ground truth* with genuine appearance
change between acquisitions.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
import tifffile

UTM = "EPSG:32614"  # Oklahoma test site


@dataclass
class Ortho:
    rgb: np.ndarray          # HxWx3, RGB
    valid: np.ndarray        # HxW bool
    pix_to_utm: np.ndarray   # 2x3 affine, pixel(col,row) -> (X, Y)
    from_epsg: int | None

    @property
    def shape(self):
        return self.rgb.shape[:2]

    def footprint(self) -> tuple[float, float, float, float]:
        h, w = self.shape
        pts = np.array([[0, 0], [w, 0], [0, h], [w, h]], float)
        g = pts @ self.pix_to_utm[:, :2].T + self.pix_to_utm[:, 2]
        return g[:, 0].min(), g[:, 1].min(), g[:, 0].max(), g[:, 1].max()


def read_ortho(path: str) -> Ortho:
    with tifffile.TiffFile(path) as t:
        page = t.pages[0]
        scale = np.array(page.tags[33550].value, float)
        tie = np.array(page.tags[33922].value, float)
        geokeys = {}
        if 34735 in page.tags:
            kv = list(page.tags[34735].value)
            for k in range(4, len(kv), 4):
                geokeys[kv[k]] = (kv[k + 1], kv[k + 2], kv[k + 3])
        im = None

    # cv2 decodes the JPEG-compressed TIFF without needing imagecodecs.
    raw = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if raw is None:
        raise IOError(f"cannot decode {path}")
    rgb = np.ascontiguousarray(raw[:, :, 2::-1])      # BGR(A) -> RGB
    valid = raw[:, :, 3] > 0 if raw.shape[2] > 3 else np.ones(raw.shape[:2], bool)

    i0, j0 = tie[0], tie[1]
    x0, y0 = tie[3], tie[4]
    sx, sy = scale[0], scale[1]

    # ModelTypeGeoKey 1024 == 2 means a geographic CRS. Some exporters emit a
    # user-defined GeographicTypeGeoKey (32767) with only a citation, so keying on
    # 4326 alone silently misses them.
    is_geo = geokeys.get(1024, (None, None, None))[2] == 2
    if is_geo:
        # Latitude/longitude degrees -> UTM, linearised over the tile (valid to
        # well under a pixel across a 50 m field).
        from pyproj import Transformer
        tr = Transformer.from_crs("EPSG:4326", UTM, always_xy=True)
        cols = np.array([0.0, 1000.0, 0.0])
        rows = np.array([0.0, 0.0, 1000.0])
        lon = x0 + (cols - i0) * sx
        lat = y0 - (rows - j0) * sy
        X, Y = tr.transform(lon, lat)
        src = np.stack([cols, rows], 1)
        dst = np.stack([np.asarray(X), np.asarray(Y)], 1)
        M = np.hstack([src, np.ones((3, 1))])
        A = np.linalg.lstsq(M, dst, rcond=None)[0].T      # 2x3
        return Ortho(rgb, valid, A, 4326)

    A = np.array([[sx, 0.0, x0 - i0 * sx],
                  [0.0, -sy, y0 + j0 * sy]], float)
    return Ortho(rgb, valid, A, None)


def resample_to(ortho: Ortho, bbox, gsd, size) -> tuple[np.ndarray, np.ndarray]:
    """Resample onto a north-up UTM grid covering bbox=(xmin,ymin,xmax,ymax)."""
    w, h = size
    xmin, _, _, ymax = bbox
    Aout = np.array([[gsd, 0.0, xmin], [0.0, -gsd, ymax]], float)   # out pixel -> UTM
    # out pixel -> source pixel
    Minv = np.linalg.inv(np.vstack([ortho.pix_to_utm, [0, 0, 1]]))
    M = (Minv @ np.vstack([Aout, [0, 0, 1]]))[:2]
    img = cv2.warpAffine(ortho.rgb, M, (w, h), flags=cv2.INTER_AREA | cv2.WARP_INVERSE_MAP,
                         borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))
    msk = cv2.warpAffine(ortho.valid.astype(np.uint8), M, (w, h),
                         flags=cv2.INTER_NEAREST | cv2.WARP_INVERSE_MAP,
                         borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    return img, msk > 0


def common_grid(orthos, gsd: float):
    boxes = [o.footprint() for o in orthos]
    xmin = max(b[0] for b in boxes); ymin = max(b[1] for b in boxes)
    xmax = min(b[2] for b in boxes); ymax = min(b[3] for b in boxes)
    if xmax <= xmin or ymax <= ymin:
        raise ValueError("no overlap between orthos")
    w = int(round((xmax - xmin) / gsd)); h = int(round((ymax - ymin) / gsd))
    return (xmin, ymin, xmax, ymax), (w, h)
