# QReg - learned-QUBO correspondence selection for registration

SIFT front-end, then a QUBO decides *which* candidate matches to keep; a simulated
quantum annealer (path-integral Monte Carlo) solves it. Compared against SIFT+RANSAC at
matched compute.

The research question is not "can an annealer minimise a QUBO" - it can. The question is
whether an energy learned against registration quality beats a hand-tuned energy and
RANSAC. `evaluate.energy_quality_rank_corr` is the diagnostic that says why that is the
only version worth building, and the sequence test says the framing may be wrong anyway.

## Layout

```
qreg/features.py   SIFT + RootSIFT, mutual-NN + ratio test -> candidate matches
qreg/qubo.py       E(x) = u.x + sum_{i<j} J_ij x_i x_j, and the linear-in-theta feature stack
qreg/solvers.py    Ising solvers: greedy, SA, parallel tempering, path-integral SQA
qreg/decode.py     selected matches -> homography; subset DLT and robust IRLS refit
qreg/synth.py      oblique-view + photometric-change pair generator (plane homography)
qreg/preprocess.py  bandpass / illumination / GSD / fusion operators
qreg/strip.py      drone flight-strip generator (sequence, exact GT between any frames)
qreg/terrain.py    relief-aware orthophoto vs drone-frame pair (ray-marched, dense GT)
qreg/evaluate.py   per-method metrics, energy<->quality rank correlation, solver gap stats
qreg/train.py      ES over theta (works now) + blackbox-differentiation hook
qreg/selfcheck.py  invariants (Ising conversion exactness, end-to-end wiring)
experiments/compare.py    matched-compute comparison / sweep
experiments/sequence.py   drone-strip drift test (chained vs direct vs global)
experiments/tiles.py      per-tile registration under relief (QUBO does the selection)
experiments/fusion_demo.py multi-frame fusion vs single frame, on real texture
experiments/ablation.py  export a preprocessed YOLO dataset for detector ablation
```

## Quickstart

```
pip install -r requirements.txt
python -m qreg.selfcheck
python experiments/compare.py --pairs 4 --oblique 10,25,40 --budget 300000 --csv runs/sweep.csv
python experiments/compare.py --pairs 8 --oblique 25,35,45 --inlier-frac 0.2 --decode irls
python -m qreg.train --train-pairs 10 --val-pairs 4 --oblique 25,35,45 --inlier-frac 0.18 --out runs/theta.json
python experiments/compare.py --pairs 8 --oblique 25,35,45 --inlier-frac 0.2 --decode irls --learned runs/theta.json
python experiments/sequence.py --frames 10 --overlap 0.65 --tilt 8 --photometric 1.0 --blur 2.5
python experiments/tiles.py --relief 0.08 --baseline 0.4 --tiles 4 --n-max 1200 --seeds 4
```

`--learned PATH` adds `greedy*/sa*/pt*/sqa*` rows next to the hand-tuned ones, so one
table answers both questions: does learning the energy help, and does the annealer
advantage survive learning. `--inlier-frac` reshapes the candidate set to a target
inlier fraction at fixed size (`--cand-total`). `--decode irls` refits over all
candidates from the selection instead of fitting the selection alone.

Real pairs: `--pairs-dir DIR` reads `DIR/<name>/{1.png,2.png,H.npy}` (H maps image 1 to
image 2). Oxford/HPatches data just needs converting into that layout.

## Protocol (do not break these)

1. **Fixed front-end.** All methods consume the same candidates from
   `features.match_candidates`. Never let the selector touch descriptor extraction.
2. **Matched compute.** Solvers share a spin-flip budget (`--budget`), not an iteration
   count. SQA costs `P` flips per spin update, so it gets `1/P` the sweeps.
3. **Report the oracle.** `oracle` = robust fit on ground-truth inliers. It is the
   ceiling for this front-end.
4. **Report per condition, not pooled.** The per-angle spread is 1 px to 15 px, so a
   pooled median across angles is an artifact in either direction.
5. **Report gap statistics, not just the median.** Most pairs are exact ties; the spread
   lives in a few catastrophic divergences.


## Findings

Six findings, written up in [`docs/FINDINGS.md`](docs/FINDINGS.md), including the
ones that argue against the idea:

- **The decoder beats the selector.** A Tukey IRLS refit over *all* candidates
  (same 3 px threshold RANSAC gets) takes 20%-inlier, 35-degree pairs from
  40-450 px to 1-2 px. Most of what looked like a selection problem was fitting.
- **The annealer is not contributing.** At a matched spin-flip budget, SQA beats
  SA in 9/24 pairs and gets 1/8 the sweeps, so at coin-flip quality it is a strict
  compute loss. Greedy is within noise of both.
- **Learning the energy does work.** Rank correlation goes from **-0.65 to +0.83**;
  per-tile selection reaches the ground-truth-inlier oracle (1.70 vs 1.70) and
  beats per-tile RANSAC in 4 of 5 seeds.
- **Real orthomosaic data cannot answer the question** - the front-end finds no
  true correspondences across a season of appearance change. **Untested, not
  negative.**
- **Three bugs that were manufacturing results**, including border reflection
  creating a phantom 180-degree consensus that made RANSAC look like it failed at
  41% inliers.

The honest summary: the quantum-inspired solver was the part that did not work;
per-tile selection plus a robust decoder was the part that did.
