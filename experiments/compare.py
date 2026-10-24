"""Compare correspondence selectors under matched compute.

    python experiments/compare.py --pairs 4 --oblique 10,25,40,55 --budget 300000
    python experiments/compare.py --csv runs/sweep.csv --repetitive

`oracle` is the least-squares fit on ground-truth inliers: it bounds what any
selector can achieve with this SIFT front-end, i.e. how much headroom is left.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from qreg.evaluate import ALL_METHODS, Record, evaluate_pair, summarise, warmup  # noqa: E402
from qreg.qubo import QuboWeights  # noqa: E402
from qreg.synth import PairConfig, make_pair  # noqa: E402
from qreg.train import load_theta, THETA_NAMES  # noqa: E402


def weights_from_args(a) -> QuboWeights:
    return QuboWeights(
        w_app=a.w_app, lam_oo=a.lam_oo, lam_geo=a.lam_geo, lam_attr=a.lam_attr,
        lam_spread=a.lam_spread, slack=a.slack, lam_card=a.lam_card,
        k_target=a.k_target, k_frac=a.k_frac, tau_s=a.tau_s, tau_th=a.tau_th, local_k=a.local_k,
    )


def run_synthetic(args) -> list[Record]:
    recs: list[Record] = []
    w = weights_from_args(args)
    learned = load_theta(args.learned) if args.learned else None
    obliques = [float(x) for x in args.oblique.split(",")]
    fracs = [None] if not args.inlier_frac else [float(x) for x in args.inlier_frac.split(",")]

    for oblique in obliques:
        for frac in fracs:
            part: list[Record] = []
            for seed in range(args.pairs):
                cfg = PairConfig(size=args.size, oblique_deg=oblique, yaw_deg=0.35 * oblique,
                                 photometric=args.photometric, repetitive=args.repetitive,
                                 seed=1000 * int(oblique) + seed)
                img1, img2, h_gt, _ = make_pair(cfg)
                name = f"obl{oblique:.0f}" + (f"_in{frac:g}" if frac else "") + f"_s{seed}"
                part += evaluate_pair(
                    name, img1, img2, h_gt,
                    budget=args.budget, seed=seed, n_max=args.n_max, w=w,
                    oblique=oblique, photometric=args.photometric,
                    solver_kw={"n_trotter": args.trotter, "T": args.sqa_temp, "T1": args.sqa_temp1},
                    methods=ALL_METHODS + ("oracle",),
                    learned_theta=learned,
                    target_inlier=frac, cand_total=args.cand_total, decode=args.decode,
                )
            recs += part
            inl = np.mean([r.inlier_frac for r in part if r.method == "ransac"])
            tag = f"oblique {oblique:.0f} deg" + (f", target inliers {frac:g}" if frac else "")
            print(f"\n=== {tag}   (candidates/frac-inlier {part[0].n_cand}/{inl:.2f}) ===")
            print(summarise(part))
    return recs


def run_pairs_dir(args) -> list[Record]:
    recs: list[Record] = []
    root = Path(args.pairs_dir)
    learned = load_theta(args.learned) if args.learned else None
    for d in sorted(root.iterdir()):
        if not d.is_dir():
            continue
        a, b, gt = d / "1.png", d / "2.png", d / "H.npy"
        if not (a.exists() and b.exists() and gt.exists()):
            print(f"skip {d.name}: need 1.png, 2.png, H.npy")
            continue
        recs += evaluate_pair(
            d.name, cv2.imread(str(a)), cv2.imread(str(b)), np.load(gt),
            budget=args.budget, n_max=args.n_max, w=weights_from_args(args),
            solver_kw={"n_trotter": args.trotter, "T": args.sqa_temp, "T1": args.sqa_temp1},
            methods=ALL_METHODS + ("oracle",),
            learned_theta=learned,
        )
    return recs


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--pairs", type=int, default=4, help="seeds per condition")
    p.add_argument("--oblique", default="10,25,40,55", help="oblique angles in degrees")
    p.add_argument("--photometric", type=float, default=0.6)
    p.add_argument("--size", type=int, default=512)
    p.add_argument("--n-max", type=int, default=160, help="cap on candidate matches")
    p.add_argument("--budget", type=int, default=300_000, help="spin flips per solve")
    p.add_argument("--trotter", type=int, default=8)
    p.add_argument("--sqa-temp", type=float, default=1.0)
    p.add_argument("--sqa-temp1", type=float, default=0.02)
    p.add_argument("--repetitive", action="store_true")
    p.add_argument("--pairs-dir", default=None)
    p.add_argument("--csv", default=None)
    p.add_argument("--inlier-frac", default=None,
                   help="comma list of target inlier fractions; reshapes candidates to fixed size")
    p.add_argument("--cand-total", type=int, default=240)
    p.add_argument("--decode", default="dlt", choices=["dlt", "irls"],
                   help="subset fit: 'dlt' on the selection, or 'irls' refit over all candidates")
    p.add_argument("--learned", default=None,
                   help="theta JSON from `python -m qreg.train`; adds '*' variants")
    # QUBO weights (theta). Hand-tuned here; the training stage replaces them.
    p.add_argument("--w-app", type=float, default=1.0)
    p.add_argument("--lam-oo", type=float, default=6.0)
    p.add_argument("--lam-geo", type=float, default=3.0)
    p.add_argument("--lam-attr", type=float, default=0.0)
    p.add_argument("--lam-spread", type=float, default=2.0)
    p.add_argument("--slack", type=float, default=1.0)
    p.add_argument("--lam-card", type=float, default=2.0)
    p.add_argument("--k-target", type=int, default=None)
    p.add_argument("--k-frac", type=float, default=0.30)
    p.add_argument("--tau-s", type=float, default=0.25)
    p.add_argument("--tau-th", type=float, default=0.20)
    p.add_argument("--local-k", type=int, default=8)
    args = p.parse_args(argv)

    warmup()
    if args.learned:
        th = load_theta(args.learned)
        print(f"QUBO theta: learned  ({args.learned})")
        print("  " + " ".join(f"{n}={v:+.2f}" for n, v in zip(THETA_NAMES, th)))
    recs = run_pairs_dir(args) if args.pairs_dir else run_synthetic(args)

    print("\n=== overall ===")
    print(summarise(recs))

    if args.csv:
        out = Path(args.csv)
        out.parent.mkdir(parents=True, exist_ok=True)
        keys = [k for k in Record.__dataclass_fields__ if k != "meta"]
        with out.open("w", newline="") as f:
            wr = csv.DictWriter(f, fieldnames=keys)
            wr.writeheader()
            for r in recs:
                wr.writerow({k: getattr(r, k) for k in keys})
        print(f"\nwrote {out} ({len(recs)} rows)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
