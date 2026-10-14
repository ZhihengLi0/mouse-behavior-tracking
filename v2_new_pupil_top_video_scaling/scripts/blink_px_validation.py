"""Two checks of the per-mouse pixel threshold (section 10.3 of the page), added 2026-10-13 after Kaiwen chose the
threshold-scan method (figure 10-5) and one pixel threshold per mouse as the rule to use.

1. Where the judged frames fall on the eye-opening distribution.
   For each mouse: histogram of the eye opening O = y(eyelid bottom) - y(eyelid top) over every frame of all its videos
   (whole-video predictions of the plateau-point models, frames with plausible eyelid points), and underneath the frames the
   labeler judged blind (the 289 trusted frames of the two single-frame samples), red = judged not open, green = judged
   open, each at the O the model gives that frame, with the mouse threshold as a dashed line.
   -> opening_distribution_with_verdicts.png

2. How far the model's eyelid distance is from the labeler's own labels.
   The 50 frozen test frames of every video (labels/test50, never trained on): O from the labeler's two eyelid points against
   O from the plateau-point model on the same frame. Per mouse: median and 90th percentile of |model - label|, the bias, and
   how many test frames the two put on different sides of the mouse threshold (all of them, and among the frames within
   20 px of the line). This separates the model's share of the errors near the threshold from the judgment's share.
   -> eyelid_distance_label_vs_model.png, eyelid_distance_label_vs_model.csv (per frame),
      eyelid_distance_label_vs_model_summary.csv (per mouse and per opening band)

Inputs: results/labels_to_plateau.csv (which model is the plateau model), <unit>/training-data/predictions_stepNN/*.h5,
<unit>/training-data/labels/test50/CollectedData_*.csv, labels/frames/blink_notopen_frames_combined_all_videos.csv,
results/every_mouse/px/threshold_per_mouse.csv (the thresholds; falls back to 147.5 / 150.5).
"""
import glob, re, sys
from pathlib import Path
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent))
import pupil_trace as pt  # noqa: E402
from scale_step import HERE  # noqa: E402

DX_FRAC, CONF_MIN = 0.3, 0.3
ROOT = HERE / "results" / "blink_eyelid_distance"   # folder layout since 2026-10-12: labels/{segments,frames} = the labeler's verdicts; results/{every_video,one_threshold,every_mouse}
LAB_SEG, LAB_FR = ROOT / "labels" / "segments", ROOT / "labels" / "frames"
RES_EV, RES_ONE, RES_MOUSE = ROOT / "results" / "every_video", ROOT / "results" / "one_threshold", ROOT / "results" / "every_mouse"
OUT = RES_MOUSE / "px"; OUT.mkdir(parents=True, exist_ok=True)
PL = pd.read_csv(HERE / "results" / "labels_to_plateau.csv"); PL = PL[PL.status == "final"]
MICE = ["mouse A", "Pluto", "Terra"]
NEAR = 20.0   # px: "near the line"

THR = {"mouse A": None, "Pluto": 147.5, "Terra": 150.5}
tp = OUT / "threshold_per_mouse.csv"
if tp.exists():
    t = pd.read_csv(tp)
    for r in t.itertuples():
        if r.mouse in ("Pluto", "Terra") and np.isfinite(float(r.threshold_px)):
            THR[r.mouse] = float(r.threshold_px)

def frame_values(d):
    g = lambda b, c: d[b][c].to_numpy(float)
    O = g("eyelid_bottom", "y") - g("eyelid_top", "y")
    wf = np.hypot(g("eye_temporal_corner", "x") - g("eye_nasal_corner", "x"), g("eye_temporal_corner", "y") - g("eye_nasal_corner", "y"))
    W = float(np.nanmedian(wf))
    dx = np.abs(g("eyelid_top", "x") - g("eyelid_bottom", "x")) / W
    conf = np.minimum(g("eyelid_top", "likelihood"), g("eyelid_bottom", "likelihood"))
    trusted = (O >= 0) & (dx <= DX_FRAC) & (conf >= CONF_MIN) & np.isfinite(O)
    return O, trusted

# ---- load every video once: whole-video O and the test-frame comparison ----------------------------------------------
allO, rows = {m: [] for m in MICE}, []
for rr in PL.itertuples():
    step = max(1, int(rr.labels_at_plateau) // 20)
    h5 = sorted(glob.glob(str(HERE / rr.unit / "training-data" / f"predictions_step{step:02d}" / "*snapshot*120*.h5")))
    O, tr = frame_values(pt.load(h5[-1])); allO[rr.mouse].append(O[tr])
    L = pd.read_csv(glob.glob(str(HERE / rr.unit / "training-data" / "labels" / "test50" / "CollectedData_*.csv"))[0], header=[0, 1, 2], index_col=[0, 1, 2])
    L.columns = [f"{b}_{c}" for s, b, c in L.columns]
    for idx, r in L.iterrows():
        f = int(re.findall(r"\d+", idx[2])[0]); ol = r.eyelid_bottom_y - r.eyelid_top_y
        if np.isfinite(ol) and f < len(O):
            rows.append(dict(video=int(rr.video), mouse=rr.mouse, date=rr.date, frame=f, O_label_px=round(ol, 2), O_model_px=round(float(O[f]), 2), eyelid_plausible=bool(tr[f])))
D = pd.DataFrame(rows); D["diff_px"] = (D.O_model_px - D.O_label_px).round(2); D["absdiff_px"] = D.diff_px.abs()
D.to_csv(OUT / "eyelid_distance_label_vs_model.csv", index=False)

# ---- the judged frames ------------------------------------------------------------------------------------------------
Q = pd.read_csv(LAB_FR / "blink_notopen_frames_combined_all_videos.csv")
J = Q[Q.human_verdict.notna() & ~Q.human_verdict.astype(str).str.startswith("cannot") & Q.trusted.astype(bool)].copy()
J["notopen"] = J.human_verdict.str.startswith("not open")
J["mouse"] = J.video.map(lambda v: PL.set_index("video").mouse.get(int(v)))

# ---- figure 1: distribution with the verdicts --------------------------------------------------------------------------
fig, axes = plt.subplots(2, 3, figsize=(18, 8), gridspec_kw=dict(height_ratios=[3, 1.3]))
for k, m in enumerate(MICE):
    O = np.concatenate(allO[m]); g = J[J.mouse == m]; T = THR[m]
    a = axes[0, k]; a.hist(O, bins=np.arange(0, 400, 4), color="#bbbbbb", label=f"all frames of {m} ({len(O):,})")
    a.set_title(f"{m}: eye opening O over all frames of its videos"); a.set_xlim(0, 400); a.set_xlabel("eyelid distance O (px)"); a.set_ylabel("frames")
    if T: a.axvline(T, color="k", lw=1.5, ls="--", label=f"mouse threshold {T:g} px")
    a.legend(loc="upper right", fontsize=9)
    b = axes[1, k]; rng = np.random.default_rng(0)
    for lab, col, y in [("judged not open", "#c0392b", 1), ("judged open", "#2e8b57", 0)]:
        s = g[g.notopen == (y == 1)]; b.scatter(s.O_px, y + rng.uniform(-0.25, 0.25, len(s)), s=22, color=col, alpha=.75, label=f"{lab} ({len(s)})")
    if T:
        b.axvline(T, color="k", lw=1.5, ls="--"); y = g.notopen.values; p = (g.O_px < T).values
        b.text(0.01, 0.9, f"wrong side of the line: {int((p & ~y).sum())} judged open but O < T, {int((~p & y).sum())} judged not open but O >= T", transform=b.transAxes, fontsize=8)
    b.set_xlim(0, 400); b.set_yticks([0, 1]); b.set_yticklabels(["open", "not open"]); b.set_xlabel("O of the judged frame (px, model output)")
    b.set_title(f"the {len(g)} blind-judged frames of {m} (plausible eyelid points)", fontsize=10); b.legend(loc="center right", fontsize=8)
fig.suptitle("Where the blind-judged frames fall on the eye-opening distribution (gray = every frame of the mouse, dots = the labeler's verdicts)", fontsize=12)
fig.tight_layout(); fig.savefig(OUT / "opening_distribution_with_verdicts.png", dpi=110); plt.close(fig)

# ---- figure 2 + summary: label vs model on the test frames ------------------------------------------------------------
summ = []
fig, axes = plt.subplots(1, 3, figsize=(18, 6))
for k, m in enumerate(MICE):
    g = D[D.mouse == m]; T = THR[m]; a = axes[k]
    a.scatter(g.O_label_px, g.O_model_px, s=14, color="#2a6f97", alpha=.6)
    lim = [0, 400]; a.plot(lim, lim, color="k", lw=1, label="model = label")
    a.plot(lim, [lim[0] + 10, lim[1] + 10], color="#999", lw=.8, ls=":", label="±10 px"); a.plot(lim, [lim[0] - 10, lim[1] - 10], color="#999", lw=.8, ls=":")
    row = dict(mouse=m, test_frames=len(g), median_absdiff_px=round(g.absdiff_px.median(), 1), p90_absdiff_px=round(g.absdiff_px.quantile(.9), 1),
               max_absdiff_px=round(g.absdiff_px.max(), 1), median_bias_px=round(g.diff_px.median(), 1))
    if T:
        a.axvline(T, color="k", ls="--", lw=1.2); a.axhline(T, color="k", ls="--", lw=1.2)
        side = (g.O_label_px < T) != (g.O_model_px < T); near = (g.O_label_px - T).abs() <= NEAR
        row.update(threshold_px=T, different_side=int(side.sum()), different_side_pct=round(100 * side.mean(), 1), near_frames=int(near.sum()),
                   near_different_side=int(side[near].sum()), near_different_side_pct=round(100 * side[near].mean(), 1) if near.sum() else np.nan)
        a.text(0.02, 0.95, f"threshold {T:g} px (dashed)\nlabel and model on different sides: {int(side.sum())} of {len(g)} ({100 * side.mean():.1f}%)\n"
               f"within {NEAR:g} px of the line: {int(side[near].sum())} of {int(near.sum())}", transform=a.transAxes, va="top", fontsize=9, bbox=dict(fc="white", ec="none", alpha=.8))
    a.set_title(f"{m}: {len(g)} test frames\n|model - label| median {row['median_absdiff_px']} px, 90th percentile {row['p90_absdiff_px']} px", fontsize=11)
    a.set_xlim(lim); a.set_ylim(lim); a.set_xlabel("O from the labeler's eyelid points (px)"); a.set_ylabel("O from the model's eyelid points (px)"); a.legend(loc="lower right", fontsize=9)
    summ.append(row)
fig.suptitle("Eyelid distance: the labeler's labels against the plateau model on the 50 frozen test frames of every video (never trained on)", fontsize=12)
fig.tight_layout(); fig.savefig(OUT / "eyelid_distance_label_vs_model.png", dpi=110); plt.close(fig)

D["opening_band_px"] = pd.cut(D.O_label_px, [0, 100, 150, 200, 250, 400]).astype(str)
bands = D.groupby("opening_band_px").absdiff_px.agg(test_frames="size", median_absdiff_px="median", p90_absdiff_px=lambda x: x.quantile(.9)).round(1).reset_index()
S = pd.concat([pd.DataFrame(summ), bands], axis=0, ignore_index=True)
S.to_csv(OUT / "eyelid_distance_label_vs_model_summary.csv", index=False)
print(pd.DataFrame(summ).to_string(index=False)); print(bands.to_string(index=False))
print("judged frames per mouse:", J.groupby("mouse").size().to_dict())
