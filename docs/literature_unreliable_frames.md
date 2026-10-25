# Literature and tools for the two sub-problems: finding and repairing unreliable frames

Read 2026-10-25 for the next phase (`next_phase_plan_2026-10-24.md`). Every card is from the primary text (paper
full text, official docs, or the tool's source code); where a journal version is paywalled the open preprint was used
and is named. Quotes are at most 15 words.

## 0. Our setting and the fixed evaluation

Single-view 60 fps infrared video, 8 keypoints on the eye (4 pupil ellipse endpoints, 2 eyelid points, 2 corners),
one DeepLabCut model per video at its plateau (20–80 labels). Evaluation (step 8 of the report): 1330 evenly spaced
human-labeled frames never trained on; truth = frame RMSE > 20 px (155 frames) or > 50 px (32 frames); report flagged
share, recall, precision, AUC. Current best label-free detector: DLC's `fitting` at 20 px (flags 2%, recall 72% /
16% at 50 / 20 px, precision 85% / 89%). Known weakness of all our signals: a model that stays wrong over consecutive
frames ("stuck" on fur or a glint) produces neither a jump nor a SARIMAX deviation. Assets already on disk: for every
video the whole-video predictions of every training step (20, 40, 60 … labels), so multi-model signals cost nothing.

- P1 = flag the unreliable frames without labels, better than the table above.
- P2 = repair them at the smallest human cost.

## 1. Source cards

### 1.1 SLEAP (Pereira et al. 2022, Nat Methods) and LEAP (Pereira et al. 2019)

Read: paper full text (PMC9007740), docs (initial labeling, assisted labeling, proofreading) and, because the docs do
not define the methods, the source files `sleap/gui/suggestions.py`, `config/suggestions.yaml`, `info/summary.py`,
`info/feature_suggestions.py`; LEAP full text (PMC6899221).

- **What it does.** Multi-animal tracker with a GUI loop: label → train → predict on "suggested" frames → correct →
  retrain. The paper only says predictions can be sorted by score to initialise labels; all definitions below are
  from code.
- **Suggestion methods (defaults).** `sample`: 20 per video, stride or random. `image features`: 200 frames per
  video → raw pixels / BRISK bag-of-words / HOG → PCA (5) → k-means (5) → 5 random frames per cluster; label-free,
  no model. `prediction score`: per-instance score ≤ 3.0 (0–100 scale). `velocity`: one chosen node, displacement
  between consecutive frames, flag if (disp − min) > 0.1 × (max − min) of the video (relative threshold).
  `max point displacement`: mean displacement over nodes > 10 px. Corrections: a predicted instance becomes a user
  instance on double-click; no propagation to neighbouring frames; corrected frames join the training set.
- **Evidence.** Learning curves only (≈ 90% of peak accuracy below 200 labels for flies and mice). No evaluation of
  any suggestion method; no precision/recall of score or velocity flags. LEAP: PCA (50 components) + k-means (10)
  cluster sampling for the first frames; labeling time fell from ≈ 2 min to ≈ 6 s per frame once predictions were
  used as initialisation.
- **Needs.** Nothing beyond predictions; single camera fine.
- **P1.** Nothing new: `velocity` / `max point displacement` are our jump (not normalised by eye width),
  `prediction score` is our confidence rule. Both are single-frame-difference signals and miss a stuck run except at
  its entry and exit. The image-feature clustering is orthogonal: it finds visually unusual frames with no notion of
  error; could be used to pick which stuck frames a human inspects.
- **P2.** Manual only, one correction per flagged frame, no propagation. DLC's refine GUI already does the same.
- **Cost / verdict.** Zero cost, **not applicable as a new method**; cite as evidence that jump + confidence are the
  standard practice and that no tool documents their precision.

### 1.2 Lightning Pose (Biderman, Whiteway et al. 2024, Nat Methods) and EKS

Read: preprint full text (PMC10168383; the journal version is paywalled), EKS repository README and source
(`eks/singlecam_smoother.py`, `eks/core.py`, `eks/ibl_pupil_smoother.py`).

- **What it does.** DLC-style training (ResNet-50) plus label-free losses on unlabeled clips, a 5-frame temporal
  context network, and a post-hoc Ensemble Kalman Smoother (EKS) that fuses m independently trained networks using
  their disagreement as the observation noise. Has a dedicated IBL mouse-pupil variant with 4 pupil edge points, the
  same pupil layout as ours.
- **Signals.** Temporal-difference loss: max(0, ‖y_t − y_{t−1}‖ − ε), applied when both frames are confident; ε per
  dataset (20 px at 250 Hz for the mouse). **Pose-PCA loss**: PCA fitted on the labeled poses (components explaining
  99% of variance; 3 of 8 dimensions for the IBL pupil), per-frame reconstruction error of the predicted pose,
  loss = max(0, error − ε) with ε = largest reconstruction error on the labels. Multiview PCA: needs ≥ 2 cameras,
  not applicable. **EKS (single camera)**: per keypoint a random-walk latent, observation = ensemble median with
  noise = ensemble variance across the m models (default confidence-weighted: variance / mean confidence); process
  noise s fitted by maximum likelihood; m = 5 used, "m = 2 already gives substantial gains". Pupil variant: latent
  (diameter, centre x, centre y) with a fixed circle model (vertical = horizontal diameter by construction).
- **Evidence.** Correlation of signals with pixel error on out-of-distribution labels (mirror-mouse): confidence
  −0.54, temporal loss 0.26, **Pose-PCA 0.91**, multiview PCA 0.88. The 100 frames with the largest Pose-PCA loss
  contained true errors in 85 / 100 (mouse), 87 / 100 (fish), 100 / 100 (CRIM13). The paper states that during a
  paw-switch error confidence stays above 0.9 and the temporal loss flags only the jump back, not the frames where
  the network lingers; the PCA losses catch them. Ensemble variance: pixel error rises with ensemble SD (figure only,
  no precision/recall). IBL pupil: vertical–horizontal diameter correlation DLC 0.36 → LP 0.88; trial-to-trial
  consistency 0.35 → 0.62 → 0.74 with EKS; no pixel-error numbers for the pupil.
- **Needs.** Pose-PCA: the labels already used for training + predictions. EKS: ≥ 2 models' prediction CSVs, CPU
  (JAX), no labels, no retraining. Temporal loss / context network: retraining in LP (can import DLC projects).
- **P1.** Two new label-free signals: (a) ensemble variance across our existing 20/40/60-label models per video,
  per keypoint, per frame; models trained on different label sets are unlikely to stick on the same fur for the
  same frames, so the variance should stay high through a stuck run (must be measured: if all share the failure it
  is silent); (b) Pose-PCA reconstruction error, memoryless, flags every frame of a stuck run equally; caveat: fitted
  on 20–80 labels including blinks, the subspace may be poorly estimated (their 0.91 used 631 labels).
- **P2.** EKS is an automatic zero-label repair: the posterior mean replaces the prediction, frames with high ensemble
  variance are interpolated from the dynamics and confident neighbours; fixes glitches and short runs, not a long
  run where all members agree. The pupil variant forces a circle; our pupil is an ellipse, so use the generic
  single-camera smoother or write the ellipse observation model ourselves.
- **Cost / verdict.** Low: ensemble variance is a few lines on existing predictions; Pose-PCA ≈ 30 lines; EKS runs on
  our CSVs after renaming columns (minutes per video on CPU). **Try now** (both signals, then EKS as repair,
  all on the 1330-frame evaluation). LP retraining later.

### 1.3 DeepGraphPose (Wu, Buchanan et al. 2020, NeurIPS)

Read: main PDF and supplement from the NeurIPS proceedings, repository README.

- **What it does.** A graphical model on top of a DLC network: image potential (the CNN), a quadratic temporal
  potential between consecutive frames weighted by optical flow (more motion → weaker smoothness), a spatial potential
  that penalises keypoint pairs further apart than their mean labeled distance; trained semi-supervised on labeled +
  unlabeled frames. There is no inference-only step: the repair is the retrained network's output.
- **Evidence.** Mouse-wheel: 55 train / 945 test frames, DGP below DLC at every training size (figure only, no pixel
  numbers). Authors report they could not find Kalman-smoother or TV-denoiser settings that removed DLC glitches
  without over-smoothing (results not shown). Active-learning scores for choosing unlabeled frames (motion energy,
  ensemble disagreement, heatmap-peak uncertainty) help qualitatively.
- **Needs.** The repository's TensorFlow-1 DLC fork (Python ≈ 3.6), GPU, retraining per video; unmaintained.
- **P1 / P2.** No outlier score; the spatial constraint (e.g. pupil top–bottom distance vs eyelid distance) can be
  re-implemented as a rule in minutes; the temporal prior is our jump family. Repair only through retraining.
- **Verdict.** **Not applicable as software**; reuse the spatial-constraint idea as an extra P1 rule; note its finding
  that single-model post-hoc smoothing did not remove glitches, which argues against single-model smoothing as P2.

### 1.4 Uncertainty and error prediction for keypoints (general computer vision)

**Deep ensembles** (Lakshminarayanan et al. 2017, NeurIPS; arXiv 1612.01474). M independently initialised networks,
spread of their predictions = uncertainty; M = 5 default, M = 2 gives most of the gain. Evidence is classification /
regression benchmarks (ImageNet top-1 error 22.2 → 18.7% with 10 members; entropy rises on unseen classes while
MC-dropout stays overconfident). Stated caveat: members differ only by initialisation and data order, so a shortcut
all members learn (our fur / glint) leaves the disagreement low. *P1:* zero cost for us (per-frame, per-keypoint
standard deviation across the existing per-video models), expected to miss shared stuck errors. *Verdict: try now,
measure.*

**Learning Loss for Active Learning** (Yoo & Kweon 2019, CVPR). A small head predicts the network's own loss;
correlation predicted-vs-true loss 0.68 on MPII with thousands of labels; gain over entropy sampling 1–2 PCKh points.
*Needs retraining with a new head; with 20–80 labels per video there are almost no wrong examples to learn from.*
*Verdict: later / probably not.*

**Multiple Peak Entropy** (Liu & Ferrari 2017, ICCV, "Active learning for human pose estimation"). Per joint, the
entropy over the local maxima of the heatmap: high when several weak peaks compete (e.g. pupil edge vs glint),
silent when the network is confidently on one wrong place. Their Voronoi click interface (annotator picks among the
heatmap peaks instead of placing a point) saved ≈ 33% annotation time. *Needs heatmaps (DLC 3 exposes them).*
*P1: a new single-frame signal, complementary to confidence. P2: "pick among candidates" is a cheaper correction
than placing a point. Verdict: try later, after the free signals.*

**Temporal pose continuity** (Mori et al., Nagoya/RIKEN): displacement from t−1 and t+1 above a size-scaled threshold
ranks frames for labelling; same family as our SARIMAX deviation, same blind spot. *Not needed.*

**Keypoint self-consistency failure detector** (Chan 2026, arXiv 2608.03516, object pose). Hand-made consistency
features (pairwise distances, residuals, 1 − confidence) → logistic regression trained on a small calibration set;
AUROC 83 vs 66 for max confidence. The 3D part does not apply; *the recipe does*: ellipse residual of the 4 pupil
points, pupil–eyelid geometry, ensemble SD, SARIMAX deviation → logistic regression, cross-validated on the 1330
frames. *Verdict: try once the features exist; cheapest way to combine signals.*

**Test-time augmentation agreement.** No primary paper found that uses flip/scale disagreement as a per-keypoint
error estimator; untested, one extra inference pass if wanted, no published number to expect.

### 1.5 Industry practice: video annotation tools and correction propagation

**CVAT docs** (track mode, skeletons, AI tools). A track has keyframes; frames between keyframes are interpolated;
editing any frame makes it a keyframe and re-interpolates to its neighbours; skeletons (our 8 points) are one track
object. AI trackers propagate bounding boxes only; no keypoint tracker; no accuracy numbers anywhere. *P2: linear
interpolation between two human-corrected endpoints of a flagged run is the baseline repair, cost = 2 frames per run;
adequate for slow pupil motion inside short runs, not for saccades. Use the idea (5 lines of numpy), not the tool.*

**Label Studio video keypoints.** Keyframes + unnamed interpolation; paid tier; nothing beyond CVAT. *Not applicable.*

**Personalising video pose estimation** (Charles, Pfister, Magee, Hogg, Zisserman 2016, CVPR; arXiv 1511.06676).
Start from confident frames, propagate joints along dense optical flow up to 30 frames forward and backward, keep a
propagated label only if overlapping propagations agree within 20 px and come from ≥ 3 source frames, then fine-tune
the network on the kept pseudo-labels for that video. PCK@20 px 71 → 91 on YouTube Pose, 85 → 96 on BBC Pose, no new
human labels; ≈ 15 s per frame on one CPU core for the initial stage. *P1: the agreement rule is itself a label-free
flag with a failure mode (flow drift) different from a stuck network. P2: direct template — propagate a corrected
frame ± 30 frames by flow, accept where propagations from both run endpoints agree; one correction can repair a run
of ≈ 60 frames. Verdict: try now for P2 (compare with linear interpolation on the 1330 frames); fine-tuning later.*

**IMPose** (arXiv 2606.04480, abstract only, code not yet released). Correct one frame, a keypoint tracker propagates
the correction over the video; reported cost ≈ 27 clicks per 1050 frames. *Later; its metric "clicks per 1000 frames"
is the right way to report our P2 cost.*

### 1.6 Pupil and eye specific tools

**MEYE** (Mazziotti et al. 2021, eNeuro; PMC8489024). A tiny U-Net (0.03 M parameters) on 128 × 128 crops outputs a
pupil probability map plus two scalars, P(eye present) and P(blink). Threshold 0.15 for mouse, largest connected
component; blink epochs removed, linearly interpolated, then a 0.5 s median filter. Trained on 11 897 images (7612
mouse); test Dice 84%; pupil size vs EyeLink Spearman 0.88 (human); no blink-detection accuracy, no comparison with
DLC in the paper. Pretrained weights public, CPU is enough. *P1: an independent single-frame estimator; |MEYE area −
ellipse area| is a label-free cross-check that does not depend on temporal jumps; its blink output is a second blink
detector. Risk: our labelling rule counts the bright outer band as pupil; MEYE may segment the dark core only, so it
must be calibrated on the 1330 frames. P2: gives an area directly where the 4-point ellipse fails; cannot repair
eyelid points. Verdict: try now as cross-check; adopt as fallback only if its area error on the test frames is below
the ellipse's.*

**DLC pupillometry toolbox** (Privitera et al. 2020, Nat Protoc). Paywalled; only second-hand facts (11 markers,
diameter from marker-pair distances with medians). *Not usable as a primary source.*

**EyeLoop** (Arvin, Rasmussen, Yonehara 2021, Front Cell Neurosci; PMC8696164). Classical: binarise, 32 radial walks
from the previous centre, drop contour points > 1 SD from the mean radius, fit an ellipse; if the centre jumps too
far fall back to a Hough search, else mark the frame as blink and re-seed when an ellipse reappears. Authors note it
fails with whiskers and eyelid overlap, which is exactly our failure mode. *Not a repair route; the 1-SD radial
outlier rule and the "re-seed after blink" logic are cheap heuristics.*

**Pupil-DLC** (bioRxiv 2026.01.18.700183). 12 equidistant contour keypoints + 4 reference points; ellipse by least
squares on points with confidence ≥ 0.9, ≥ 6 points required, otherwise NaN; error 1.2–1.3 px; no temporal filter, no
blink detector. *Shows the redundancy route: with 12 points a frame survives 6 bad points and has a within-frame
residual; with our 4 endpoints one bad point ruins the ellipse and there is no residual. Later; only if the label
standard changes (not planned).*

**SOLOv2 pupil segmentation** (Sensors 2021, PMC8588114). Mask model on 737 labelled frames, IoU 0.93, diameter error
1.9%; no outlier handling; fails on whiskers, blur, reflections. *Not now (needs mask labels; MEYE gives the same
cross-check for free).*

### 1.7 DeepLabCut outlier extraction and refinement (Nath et al. 2019, Nat Protoc; DLC 3.x docs and source)

Read: bioRxiv preprint v1 of the protocol (journal paywalled), DLC 3 user guide ("Filter pose data", "Refinement"),
and the source files `refine_training_dataset/outlier_frames.py`, `post_processing/filtering.py`.

- **Definitions (from the code).** `uncertain`: any body part with likelihood < p_bound (0.01). `jump`: any body
  part displaced > epsilon (20 px) from the previous frame, per-part maximum, absolute pixels. `fitting`: per body
  part and coordinate a SARIMAX(ARdegree 3, 0, MAdegree 1) fit on the whole video with samples below p_bound set to
  missing; deviation of a part = distance between the one-step prediction and the observation; **frame score = mean
  over the compared body parts**, flagged if > epsilon (20 px). A second quantity, the number of parts outside the
  alpha confidence interval, is computed but not used ("Only the distance is used", docstring). Candidates are then
  subsampled to `numframes2pick` by k-means on downsampled images. `filterpredictions`: median (window 5), ARIMA
  (same SARIMAX, coordinates replaced by the predicted mean) or spline (gaps ≤ window). Refinement = napari GUI,
  merge, retrain from fresh weights.
- **Evidence.** None: the protocol gives one illustrative ARIMA trajectory and no precision/recall or before/after
  error for any outlier algorithm; the docs give no numbers either.
- **P1.** Our jump / uncertain / fitting are DLC's own, with jump normalised by eye width. Two variants follow from
  the code and are untried: the per-keypoint **maximum** of the SARIMAX deviation instead of the mean (one pupil
  point on a glint is diluted by seven good points), and the unused **count of points outside the confidence
  interval** (scale-aware, the interval adapts to each trajectory's noise). Why fitting misses a stuck run: the AR(3)
  one-step predictor re-anchors on the last observations within ≈ 3 frames, so after the first frames of the run the
  wrong trajectory is self-consistent.
- **P2.** Manual correction + retraining only; `filterpredictions` smooths a stuck run into itself.
- **Verdict.** *Try now*: max-over-keypoints and CI-count variants (a few lines on the existing fitting script).

### 1.8 refineDLC (Klecel, Rahael, Brooks 2025, Biology Methods and Protocols; PMC12744387)

Five CLI scripts on DLC CSVs: likelihood mask (absolute or per-video percentile, threshold chosen where trajectory
"roughness" plateaus), frame-to-frame displacement mask (fixed 30 px, or MAD / IQR / skew-adjusted IQR × 1.5),
interpolation of gaps ≤ 5 frames (cubic), no smoothing. Validated on 17 cattle and 524 horse videos **without
ground truth**: only amounts removed (2–11% of frames per body part at likelihood 0.5; 28–135 frames per video by the
jump rule) and visual checks. The authors state that mislabels with high likelihood exist and that "smooth but wrong"
errors (small displacement) escape the displacement rule, the same failure mode as ours. *Nothing beyond our
confidence and jump rules; the one borrowable idea is per-video, per-keypoint adaptive thresholds (percentile / IQR)
instead of one global number. Verdict: not applicable.*

### 1.9 Anipose 2D filters (Karashchuk et al. 2021, Cell Reports; PMC8498918; `anipose/filter_pose.py`)

- **Median filter**: per joint and axis, deviation from a 13-frame median (L1 over x and y) ≥ offset threshold
  (25 px) or score < 0.8 → missing, then cubic-spline or linear interpolation. The paper states it "cannot correct
  errors spanning multiple frames".
- **Viterbi filter**: needs the top-n candidate detections per joint per frame (≈ 20 heatmap peaks with scores);
  candidates below the score threshold dropped; particles from the last n_back = 3 frames with decayed scores;
  transition log-probability from the Gaussian of the inter-frame distance with σ = 30 px; a "missing" particle with
  probability 0.001; emission = candidate score; Viterbi max-path picks one candidate per frame. Designed, in the
  authors' words, "to correct for errors that persist over multiple frames".
- **Score autoencoder**: a one-hidden-layer MLP maps the frame's vector of joint scores to corrected scores, trained
  on synthetic score vectors built from the labeled set's visibility pattern (needs the convention "invisible =
  left unlabeled", which is ours); final score = min(predicted, original). A variant predicts each joint from the
  others and turns the residual into a score.
- **Evidence.** Paired t-tests on ≈ 1200 fly and 8608 human validation frames: median filter significant on human
  only; Viterbi significant on both (fly t = −4.4, human t = −10.9); autoencoder significant on fly; on the mouse
  dataset "filters did not further improve tracking accuracy". No detection rates, no pixel numbers in the text.
- **Needs.** Median filter and score autoencoder: existing predictions and labels. Viterbi: a DLC 3 inference pass
  that exports the k best heatmap peaks per keypoint (not a standard output), no retraining.
- **P1 / P2.** Median-deviation is a cousin of fitting with the same blind spot. Viterbi is the one genuinely new
  single-camera idea: where the top-1 peak sits on fur or a glint but the true edge is a secondary peak, the path
  re-assigns it, and disagreement between the Viterbi path and the raw prediction is a flag that does not depend on a
  jump; a run where the model is confidently stuck with no secondary peak is followed. Viterbi + interpolation is an
  automatic repair.
- **Verdict.** Median filter and score autoencoder *try now, low expectation*; Viterbi *later*, after the zero-cost
  signals, because it needs the top-k peak export.

## 2. Synthesis for P1: label-free signals, ranked by evidence × cost here

| rank | signal | source | evidence | catches a stuck run? | cost here | experiment |
|---|---|---|---|---|---|---|
| 1 | Pose-PCA reconstruction error of the 8-point pose (PCA on the video's labels, 99% variance) | Lightning Pose | r = 0.91 with pixel error; 85–100% of the 100 worst frames truly wrong | yes if the stuck pose is geometrically implausible; memoryless | 30 lines on existing labels and predictions | S1-7 |
| 2 | Ensemble disagreement across the existing 20/40/60-label models of the video (per keypoint SD, confidence-weighted) | Lightning Pose, deep ensembles | error rises with ensemble SD (figure); EKS gains grow with disagreement | only if the models do not share the failure; must be measured | a few lines on existing predictions | S1-8 |
| 3 | fitting variants: max over keypoints, pupil points only, count outside the confidence interval, forward + backward | DLC source | none published | partly (entry/exit frames) | refit SARIMAX (minutes per video) | S1-2, S1-3 |
| 4 | fitting threshold sweep and eye-width normalisation | DLC default | none published | no | trivial | S1-1, S1-4 |
| 5 | geometric constraints: ellipse axes crossing at their midpoints, pupil between the eyelids, eye width constant | DeepGraphPose spatial term, EyeLoop radial rule, our `axes_off` | `axes_off` AUC 0.92 for eye closure (our step 9) | yes, memoryless | minutes | S1-7b |
| 6 | MEYE segmentation area vs ellipse area | MEYE | Dice 84% on mouse pupils; no DLC comparison | yes, memoryless; needs calibration to our "band = pupil" rule | pip install + one pass over 1330 crops | S2-3 |
| 7 | per-video adaptive thresholds (percentile / IQR) for confidence and jump | refineDLC | removal counts only | no | trivial | S1-1 |
| 8 | multiple-peak entropy of the heatmap | Liu & Ferrari 2017 | 8–19% better active selection | partly (two peaks) ; silent when confidently wrong | heatmap export | S2-4 |
| 9 | Viterbi path vs raw prediction disagreement | Anipose | t-tests only | yes when the true location is a secondary peak | top-k peak export + 150 lines | S2-5 |
| 10 | combined score: logistic regression on all of the above, leave-one-video-out | Chan 2026 recipe, DGP active scores | AUROC 83 vs 66 for confidence (other domain) | inherits the features | minutes once features exist | S1-5 |
| — | Learning-loss head, LP retraining with temporal loss / context network | Yoo & Kweon; LP | 1–2 points; LP figures | — | retraining all 19 models | later |

What the literature says about runs of consecutive errors: every temporal rule (jump, SARIMAX, median deviation,
velocity, continuity) is acknowledged to miss them (DLC code structure, refineDLC and Anipose explicitly, Lightning
Pose's "lingering" example). The only signals that fire on every frame of a run are memoryless ones: geometry
(Pose-PCA, constraints), a second estimator (ensemble, segmentation) and secondary heatmap peaks (Viterbi, MPE). No
source reports precision/recall on runs; our evaluation is per frame, so a per-run count needs the clip labels
proposed in the plan.

## 3. Synthesis for P2: repair routes, ranked by human cost

| human cost | route | source | what it needs from P1 | evidence | experiment |
|---|---|---|---|---|---|
| 0 | EKS: ensemble median + Kalman smoothing with ensemble variance as noise (generic single-camera version; the pupil version forces a circle) | Lightning Pose / eks package | ≥ 2 models per video (we have 2–4) | beats median filter, ARIMA and ensemble mean on OOD error (figure); fixes glitches and short runs, not consensus failures | P2-1 |
| 0 | single-model smoothing (median, ARIMA, spline) | DLC `filterpredictions`, refineDLC, Anipose median | nothing | DGP: no setting removed glitches without over-smoothing; Anipose: no gain on mouse | P2-1 baseline |
| 0 | drop flagged frames from the area / blink analyses | — | a detector with precision ≥ 50% | — | P2-2 |
| 0 | segmentation area (MEYE) as fallback where the ellipse is rejected | MEYE | the flag | Dice 84%; must beat the 4-point ellipse on our test frames | P2-3 |
| 2 frames per run | correct the two endpoints of a flagged run, linear interpolation between them | CVAT practice | run boundaries | none published; fine for slow pupil motion | P2-4 |
| 1 frame per run | correct one frame, propagate ± 30 frames by optical flow, accept where propagations agree within 20 px | Charles et al. 2016 | run boundaries | PCK@20 px 71 → 91 without new labels (human video) | P2-4b |
| pick among peaks | show the 2–3 heatmap peaks, annotator clicks the right one | Liu & Ferrari 2017 | heatmaps | ≈ 33% time saved | P2-5 |
| 20 frames per batch | retrain with corrected frames (our active-learning loop) | DLC / SLEAP | the detector as selector | our step 5 | existing |
| — | Viterbi re-assignment + interpolation | Anipose | top-k peaks | t-tests only | S2-5 |

Report every route as "frames a human touched per 1000 frames" against "frames > 20 px / > 50 px remaining" and the
area error, as proposed by IMPose's click budget.

## 4. What this changes in `next_phase_plan_2026-10-24.md`

- Stage 1 of P1 gains two zero-cost items that the literature rates above all temporal rules: Pose-PCA reconstruction
  error (S1-7) and ensemble disagreement across the existing per-video models (S1-8); the fitting variants (max over
  keypoints, confidence-interval count) are now specified from DLC's own code.
- Stage 2 is reordered: geometric constraints and MEYE cross-check first (memoryless, cheap), then heatmap signals
  (MPE, Viterbi) which need an inference pass that exports peaks; LP retraining stays last.
- P2 gains EKS as the first automatic route and the "two endpoints + interpolation" and "one frame + optical-flow
  propagation" routes with the click-budget metric; single-model smoothing is kept only as a baseline because two
  sources report it does not remove glitches.
- Literature gap worth stating in the report: no tool reports precision / recall of its outlier rules against a
  per-frame pixel truth, and none measures runs of consecutive errors; our 1330-frame evaluation is itself new.

## 5. References (as read)

- Nath et al. 2019, Nat Protoc 14:2152 — bioRxiv 10.1101/476531; DLC 3 docs `standardDeepLabCut_UserGuide.md`; source
  `deeplabcut/refine_training_dataset/outlier_frames.py`, `deeplabcut/post_processing/filtering.py`.
- Pereira et al. 2022, Nat Methods 19:486 (PMC9007740); SLEAP docs and `sleap/gui/suggestions.py`; Pereira et al. 2019
  LEAP (PMC6899221).
- Biderman, Whiteway et al. 2024, Nat Methods 21:1316 — preprint PMC10168383; github.com/paninski-lab/eks.
- Karashchuk et al. 2021, Cell Reports 36:109730 (PMC8498918); anipose.readthedocs.io; `anipose/filter_pose.py`.
- Wu, Buchanan et al. 2020, NeurIPS — proceedings.neurips.cc paper 4379cf00…; github.com/paninski-lab/deepgraphpose.
- Klecel, Rahael, Brooks 2025, Biology Methods and Protocols 10:bpaf084 (PMC12744387); github.com/wer-kle/refineDLC.
- Lakshminarayanan, Pritzel, Blundell 2017, NeurIPS — arXiv 1612.01474.
- Yoo & Kweon 2019, CVPR — arXiv 1905.03677.
- Liu & Ferrari 2017, ICCV, "Active learning for human pose estimation".
- Mori et al., "Active learning for human pose estimation based on temporal pose continuity" (Nagoya University).
- Chan 2026, arXiv 2608.03516, "Detecting pose estimation failures via keypoint self-consistency".
- Charles, Pfister, Magee, Hogg, Zisserman 2016, CVPR — arXiv 1511.06676.
- IMPose, arXiv 2606.04480 (abstract only).
- CVAT docs (track mode, skeletons, AI tools); Label Studio docs (VideoVectorLabels).
- Mazziotti et al. 2021, eNeuro 8(5) (PMC8489024), MEYE.
- Arvin, Rasmussen, Yonehara 2021, Front Cell Neurosci 15:779628 (PMC8696164), EyeLoop.
- Pupil-DLC, bioRxiv 10.64898/2026.01.18.700183.
- Sensors 2021 (PMC8588114), SOLOv2 pupil segmentation.
- Privitera et al. 2020, Nat Protoc 15:2301 — paywalled, not read.
