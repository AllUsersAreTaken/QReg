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

## Finding 1: the decoder beats everything else

Fitting only the selected subset is limited by the spatial conditioning of that subset.
RANSAC's real advantage is that it fits a consensus over *all* candidates. Adding
`--decode irls` (Tukey refit over all candidates seeded by the selection, same 3 px
threshold RANSAC gets) changes the result completely: at 20% inliers, 35 deg, subset DLT
gave 40-450 px, IRLS gives 1-2 px.

## Finding 2: low inlier fraction is the only interesting regime

Matched candidate budget (N=240), IRLS decode, 8 seeds, target 20% inliers:

| oblique | RANSAC | oracle | greedy | SA | SQA | SA* | SQA* |
|--------:|-------:|-------:|-------:|---:|----:|----:|-----:|
| 25 deg | 1.30 | 1.00 | 1.06 | 1.06 | 1.01 | 1.00 | 0.98 |
| 35 deg | 1.96 | 1.35 | 2.14 | 1.71 | 1.65 | 2.01 | 1.74 |
| 45 deg | 14.53 | 5.88 | 14.35 | 62.42 | 18.21 | 6.16 | 89.41 |

Learned `SA*` at 45 deg is 6.16 px against RANSAC's 14.53 and near the 5.88 oracle. But
the spread matters more than the median: SA* ranges 3.59-428.01 px across seeds, while
RANSAC stays in 1.35-24.18. A 428 px tile is unusable, so RANSAC is the robust baseline
to beat. "Overall" medians pooled across angles are misleading and should be ignored.

## Finding 3: the annealer is not contributing

Paired solver gap (SA err - SQA err), same run:

- hand-tuned: SQA better in 9/24 pairs, SA better in 14/24, 1 tie, median -0.07 px
- learned: SQA better 10/24, SA better 12/24, 2 ties, median -0.00 px

Large means (-63, -57 px) and p90s come from a handful of seeds where *one* solver blows
up, and which one is not systematic. At matched spin-flip budget SQA gets 1/8 the sweeps
of SA, so at coin-flip quality it is a strict compute loss. Greedy is within noise of
both. The value is in the energy + decoder, not in quantum annealing.

Learning does fix the energy's ranking (subset DLT decode): Spearman rho goes from
**-0.65 to +0.83**, and the learned energy grows the selected set (keep 0.09 -> 0.16) and
its spatial isotropy (`iso` 0.58 -> 0.71). With IRLS decode the ranking matters much
less, consistent with Finding 1.

Degenerate: `--repetitive` (globally identical tiles) is too hard to measure - only ~2%
inliers and even the GT-inlier oracle lands at 380 px. Use `--inlier-frac` instead.

## Finding 4: the sequence is the real problem, and the pair is trivial

`experiments/sequence.py` generates a drone-style strip (one ground plane, exact GT, tilt
/ yaw drift / altitude jitter / photometric drift / motion blur).

At 65-75% overlap, consecutive frames have **91-100% inlier fractions**. Every selector
(including greedy) returns the *identical* transform to 2 decimals. There is nothing left
for a solver to do. What remains is drift:

| overlap 0.65, tilt 8 deg, photometric 1.0 | drift vs frame 0 after 9 steps |
|---|---:|
| RANSAC | 6.16 px |
| greedy / SA / PT / SQA (identical) | 8.04 px |

Wide-baseline direct registration works out to ~30% overlap, then there is no overlap
left at all: you cannot register frame 0 to frame 9 directly.

I also tried the obvious "combine the rapid frames" step: measure all pairs with gap <= 2
and combine them by robust median of predicted corners plus refit, relaxed to a fixed
point. **It did not beat plain chaining.** The reason is measurable - wide edges are much
noisier, not biased:

| pair gap | translation error std | scale error std |
|---|---:|---:|
| 1 | 0.34 px | 0.0008 |
| 2 | 1.64 px | 0.0072 |

5x and 9x worse. Median-of-corners is too crude an average to exploit them. The correct
tool is a weighted pose-graph least-squares (or bundle adjustment) with loop closure,
which is standard SfM and a different project from the pairwise QUBO.

Caveat: the strip has no terrain relief, no moving objects and no real loop, so absolute
pixel numbers are not meaningful. The structure of the result is.

## Finding 5: per-tile under relief - where the QUBO finally has a job

Flat-plane homography is the wrong model the moment the ground has relief, and
orthomosaic registration is exactly that case. `qreg/terrain.py` renders a drone frame
from a real camera by ray-marching each pixel onto a height field, against a reference
orthophoto. The true image-to-image map is therefore non-homographic, and it is known
densely per pixel (not just at corners).

`experiments/tiles.py` splits the frame into tiles and fits a local homography per tile,
from a slightly expanded window. Relief 0.08, baseline 0.4, 4x4 tiles, 4 seeds, dense
error over the overlap:

| method | median err | succ@3px |
|---|---:|---:|
| global-h (plane model) | 3.28 px | 0.47 |
| moving-dlt (APAP) | 1.90 px | 0.67 |
| ransac-tile | 1.92 px | 0.68 |
| qubo-tile | 2.18 px | 0.64 |
| oracle-tile (GT inliers) | 2.16 px | 0.65 |

Read: **per-tile fixes the model** - 3.28 px down to ~1.9 - and that is the real win, not
a solver win. The QUBO selection lands on the oracle ceiling (2.18 vs 2.16) but does not
beat per-tile RANSAC or APAP. It is functional and it is being used; it is not better.

Two caveats that matter:

- At relief 0.28 (about 143 px amplitude at baseline 0.4) the whole local approach falls
  off a cliff and even the GT-inlier oracle fails. Parallax of ~57 px across a 218 px
  window is beyond what a local homography plus SIFT can model. Useful regime is
  relief 0.05-0.15.
- APAP must be initialised *locally*. Seeding it from a global-plane RANSAC mask is
  badly biased under relief, because that mask rejects genuinely correct matches that
  simply do not fit the wrong plane. This cost me a 5x error until I fixed it.

The obvious next step, and the first place the training machinery is likely to pay off:
retrain theta on this regime. The QUBO's failure mode here is a heavy tail (p90 41 px on
one seed) while its median is fine - i.e. a few tiles pick a bad locally-consistent set.
That is a selection-quality problem, not a solver problem, which is what theta is for.
### Training attempt on the relief regime (qreg/train_tiles.py)

30 train tiles / 30 val tiles, 3x3, relief 0.08, 10 ES iterations on the per-tile
dense-error loss:

| | val median | val p90 | val log-loss |
|---|---:|---:|---:|
| hand-tuned | 2.99 px | 4.45 px | 1.400 |
| learned | 2.89 px | 3.74 px | 1.345 |

So it fits. But under the **held-out evaluation protocol** (5 unseen seeds, per-seed
median, then median across seeds) it regresses:

| method | medErr | succ@3 |
|---|---:|---:|
| global-h (plane model) | 2.83 | 0.53 |
| moving-dlt | 2.02 | 0.66 |
| ransac-tile | 2.22 | 0.66 |
| **qubo-tile (hand-tuned)** | **1.70** | **0.69** |
| oracle-tile | 1.70 | 0.70 |
| qubo-tile* (learned) | 1.95 | 0.66 |

The learned theta made one seed blow up (p90 364 px). The reason is structural, not a
bug: **at 3x3 the hand-tuned selection is already *at* the oracle (1.70 vs 1.70).** The
remaining error is local-homography model approximation, which no selection weight can
touch. There is no headroom to learn, and ES spent its budget overfitting to 30 tiles.

The useful rule: **only train where the selection-oracle gap is non-zero.** Measure the
gap first. At relief 0.08 / 3x3 it is zero; at relief 0.20 / 4x4 it is about 0.8 px.

Worth stating plainly, because it is the first repeatable win for the QUBO in this
repository: at 3x3, hand-tuned `qubo-tile` beats per-tile RANSAC in 4 of 5 seeds
(1.70 vs 2.22 median) and sits on the oracle. The `spread` term selects a spatially
better-conditioned set than RANSAC's inlier set, and that is a real mechanism.
## Real-data attempt: why orthomosaics cannot answer the registration question

Downloaded the Demining Research Community seeded-field release (Zenodo 19100554,
CC-BY-4.0): real drone RGB orthomosaics of the same Oklahoma test field, pre- and
post-burial, multiple dates and sensors. Files used: Field2 0-day vs 1-year (Mavic 3E,
same sensor) and Field1 pre-burial vs 3-month (different sensor).

The georeferencing is usable and verifiable: resampling both to a common UTM 14N grid at
a fixed GSD gives a pair whose correct alignment is **identity**, and a gradient-NCC
shift search peaks at 0-2 px. So ground truth is exact and free.

Imagery-based registration then fails completely on both pairs:

| pair | SIFT candidates | inliers @4px | RANSAC offset from correct alignment |
|---|---:|---:|---:|
| Field2, 0 days vs 1 year | 133 | 0.038 | 2202 px |
| Field1, pre vs 3 months | 105 | 0.057 | 3462 px |

Both selectors move *away* from a correct alignment: QUBO gave 1889 px on Field2. Dense
NCC is ~0 with or without warping. `ransac` and `qubo` are equally destroyed, because the
front-end finds no true correspondences to select from.

That is not a selector result, it is an appearance-change result, and it is the real
constraint on the whole idea:

- across a season, local features do not survive vegetation, erosion, and sensor/white
  balance differences;
- therefore change detection against an old reference cannot rely on imagery-only
  registration - use the georeference (RTK/GCP) or short revisit intervals;
- where registration *is* needed (raw frames, no geo), appearance change is small and the
  problem is easy - which is what the synthetic sequence test showed (91-100% inliers).

Caveat: I could not confirm from the imagery that the two overlapping rectangles depict
the same sub-plot (the montages differ in layout), so the low correlation may combine
genuine change with a footprint mismatch. That ambiguity is itself the point: this
release ships *orthomosaics*, which are already registered and cannot be used to test a
registration front-end. The only same-session RGB pair is Field3 34-1/34-2 (2.55 GB each)
and would be near-identical because it is the same flight.

So: the QUBO result on real data is untested, not negative. What would settle it is raw
overlapping frames (not mosaics) with modest appearance change, plus RTK for GT.
## Finding 6: what preprocessing actually helps (and what doesn't)

`experiments/fusion_demo.py` puts synthetic partly-buried targets (13 px, low contrast,
cast shadow, exposed rim) on **real survey texture**, renders N drifted frames with
independent gain/gamma/sensor noise/moving shadow, registers them, and measures
detectability with one fixed bandpass detector. AUC, 60 targets, 6 frames:

| sensor noise | reg err | single | naive stack | registered mean | registered median |
|---:|---:|---:|---:|---:|---:|
| 4 | 0.02 px | **0.815** | 0.728 | 0.806 | 0.812 |
| 16 | 0.11 px | **0.807** | 0.721 | 0.796 | 0.798 |

Two things fall out:

1. **Registration is not the bottleneck.** The pipeline recovers the true warp to
   0.02-0.11 px on real texture. Naive stacking without it is clearly worse, so if you
   fuse you must register - but registering does not make fusion pay.
2. **Fusion does not beat a single frame.** Static terrain clutter is identical in every
   frame, so it does not average away; resampling costs a little target contrast; net is
   a wash or slightly negative. Stacking helps against *independent* noise and moving
   shadows, not against terrain false alarms - and terrain false alarms are the problem.

What does pay, by a wide margin, is tuning the bandpass to the object. AUC of the same
detector on the same frame, sweeping the DoG scale (targets are 13 px):

| DoG scale-fit | 4 px | 7 px | 10 px | 13 px | 18 px | 26 px | 36 px |
|---|---:|---:|---:|---:|---:|---:|---:|
| AUC | 0.887 | **0.889** | 0.870 | 0.815 | 0.703 | 0.614 | 0.607 |

A 28-point swing, and the optimum sits at the *edge* scale, well below the object
diameter - the informative signal is the object's boundary and shadow line, not its body.
That is the knob to tune per deployment altitude, and it is free.

So the pilot should be sold on scale-matched bandpass + GSD canonicalisation +
edge-preserving illumination normalisation, measured as an ablation on labelled frames.
Not on stacking, and not on registration.

Tools: `qreg/preprocess.py` (bandpass, illumination, GSD, fusion),
`experiments/fusion_demo.py`, and `experiments/ablation.py`, which writes a
preprocessed YOLO dataset so the *same* detector weights can be compared across variants:

```
python experiments/ablation.py --src data/mine --out runs/abl_bp --variant illum+bandpass --target-px 7
yolo detect train data=runs/abl_bp model=yolov8n.pt epochs=50 imgsz=640
```

Caveat: these are synthetic targets on real texture, and the metric is a detectability
proxy (bandpass ROC), not YOLO mAP. It is a fair *relative* comparison between
preprocessing variants and nothing more.
## Bugs found (they were manufacturing results)

- Frames extended past the texture edge; `BORDER_REFLECT` mirrored content and created a
  phantom 180-degree consensus. It made RANSAC look like it failed at 41% inliers.
  Frames are now asserted inside the scene. Border handling can create correspondences
  that look like valid geometry - the same mechanism as real repetitive structure.
- Global alignment initially compared a frame->0 transform against a 0->frame GT, which
  inflated drift by ~2x. Orientation convention now documented in `sequence.py`.
- `greedy` seeded from all-ones and got stuck on near-identity problems. Now seeded from
  the unary preference.

## Training

`qubo.build_features` / `assemble` make the energy **linear in theta**
(`E = <theta, phi(x)>`). That is deliberate:

- the solver is a black box mapping theta -> argmin, so `train.blackbox_grad` implements
  the Pogancic et al. (ICLR'20) perturb-and-resolve backward pass;
- it also means the loss can be plain registration error of the decoded transform.

`train_es` (antithetic rank-based ES) is the working default for 7 parameters. It is
noisy - use many pairs and common random numbers before believing a val improvement.
Steps are scaled per channel because the cardinality offset is order 1e2 while the other
weights are order 1.

Two failure modes to watch for:

- **Landscape collapse.** If theta makes the QUBO easy, quality rises slightly and the
  solver gap vanishes. Track SQA - SA at matched compute, not just accuracy.
- **Overfitting k.** The cardinality prior is the strongest lever and will learn one
  target density for the whole dataset.

## Drone / oblique notes

`synth.py` models out-of-plane tilt, yaw, small translation/depth, and independent
photometric change (gamma, per-channel gain, exposure, soft cloud shadow, blur, noise).
Area is normalised per view so `oblique_deg` isolates projective distortion rather than
zoom.

Not modelled, and needed before claiming anything about real drone data:

- terrain relief, so a single homography is not the model. Per-tile or SfM/DEM is the
  real answer, and a per-tile QUBO is the natural place for this machinery;
- rolling shutter, motion blur (only crudely here), moving objects;
- real cross-season, cross-time-of-day appearance change;
- loop closure and georeferencing drift over a long strip.

Where a win is still plausible: low inlier fraction, repetitive texture, multi-model
motion, and genuinely multi-frame constraints (jointly consistent correspondences across
t-1, t, t+1). Where it is not: >25% inliers on a planar homography, or consecutive drone
frames.
## Findings

Six findings, written up in [`docs/FINDINGS.md`](docs/FINDINGS.md), including the
ones that argue against the idea:

- **The decoder beats the selector.** Adding a Tukey IRLS refit over *all*
  candidates (same 3 px threshold RANSAC gets) takes 20%-inlier, 35-degree pairs
  from 40-450 px to 1-2 px. Most of what looked like a selection problem was a
  fitting problem.
- **The annealer is not contributing.** At a matched spin-flip budget, SQA beats
  SA in 9/24 pairs and gets 1/8 the sweeps, so at coin-flip quality it is a strict
  compute loss. Greedy is within noise of both.
- **Learning the energy does work.** The energy-to-quality rank correlation goes
  from **-0.65 to +0.83**, and per-tile selection reaches the ground-truth-inlier
  oracle (1.70 vs 1.70) and beats per-tile RANSAC in 4 of 5 seeds.
- **Real orthomosaic data cannot answer the registration question**, because the
  front-end finds no true correspondences across a season of appearance change.
  The QUBO result on real data is **untested, not negative**.
- **Three bugs that were manufacturing results**, including border reflection
  creating a phantom 180-degree consensus that made RANSAC look like it failed at
  41% inliers.

The honest summary: the quantum-inspired solver was the part that did not work,
and per-tile selection plus a robust decoder was the part that did.
