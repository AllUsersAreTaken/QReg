"""Export a YOLO dataset with a preprocessing variant applied, so the *same* detector
weights can be trained/evaluated on each variant.

YOLO labels are normalised, so geometric-only changes (resampling for GSD
canonicalisation) leave labels valid. Nothing here is detector-specific: point
ultralytics at the output directory and the comparison is a controlled ablation.

    python experiments/ablation.py --src data/mine --out runs/abl_bandpass \
        --variant bandpass --target-px 7
    python experiments/ablation.py --src data/mine --out runs/abl_illum_bp \
        --variant illum+bandpass --target-px 7

Variants: none | bandpass | illum | illum+bandpass
Layout expected: SRC/images/{train,val}/*.jpg, SRC/labels/{train,val}/*.txt
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from qreg.preprocess import apply_variant  # noqa: E402

IMG_EXT = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"}


def export(src: Path, out: Path, variant: str, target_px: float, scale: float) -> int:
    n = 0
    for img_path in sorted(src.rglob("*")):
        if img_path.suffix.lower() not in IMG_EXT:
            continue
        rel = img_path.relative_to(src)
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        img = apply_variant(img, variant, target_px, scale)
        dst = out / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(dst.with_suffix(".jpg")), img, [cv2.IMWRITE_JPEG_QUALITY, 95])

        lab = src / "labels" / rel.with_suffix(".txt").relative_to("images")
        if lab.exists():
            ldst = out / "labels" / rel.with_suffix(".txt").relative_to("images")
            ldst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(lab, ldst)
        n += 1
    return n


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--src", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--variant", default="bandpass",
                   choices=["none", "bandpass", "illum", "illum+bandpass"])
    p.add_argument("--target-px", type=float, default=7.0,
                   help="scale of the bandpass; tune to the object *edge*, not its full width")
    p.add_argument("--scale", type=float, default=1.0,
                   help="resample factor for GSD canonicalisation (labels stay valid)")
    a = p.parse_args(argv)

    src, out = Path(a.src), Path(a.out)
    if not src.exists():
        raise SystemExit(f"missing {src}")
    out.mkdir(parents=True, exist_ok=True)
    n = export(src, out, a.variant, a.target_px, a.scale)

    for y in src.rglob("*.yaml"):
        shutil.copyfile(y, out / y.name)
        print(f"copied {y.name}")
    print(f"{a.variant} @ target_px={a.target_px} scale={a.scale}: {n} images -> {out}")
    print(f"  yolo detect train data={out} model=yolov8n.pt epochs=50 imgsz=640")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
