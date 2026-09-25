#!/usr/bin/env python3
"""Pupil-area variants under the new label standard (kaiwen's direction 3, 2026-09-23), read-only analysis.

    python pupil_area_variants.py

Part A - frozen test frames of all four videos (human labels = truth, truth area = 4-point ellipse from the human labels):
    current   3-point rule of pupil_trace.py (pupil_top unused): width = |R.x - L.x|, height = 2 (B.y - mean(L.y, R.y))
    4pt       width = |R.x - L.x|, height = |B.y - T.y|
    drop_B / drop_L / drop_R   the other three "3 of 4" rules (mirror the kept opposite point through the centre)
    prior     per video, the drop_* rule that leaves out the pupil point the LABELER corrected most (median move of the
              pre-label in the training batches) - the "labeler's annotations as prior" of kaiwen's message
    conf      per frame, leave out the pupil point with the lowest model confidence (kaiwen distrusts this; shown for comparison)
    oracle    per frame, the best of the four drop_* rules - a lower bound, not a method
  The last trained model of each video is used (final snapshot). Output: results/pupil_area_variants.csv/.png

Part B - the 5-minute video (video 0, mouse A), where the OLD-era model (Aug 17 project, 100 labels, snapshot best-100)
  and the NEW-standard model (video 0 step 5, 100 labels) both predicted the same frames: 3-point traces of both models
  overlaid (the 3-point rule ignores pupil_top, so the label-standard change does not enter), the new 4-point trace,
  the blink trigger of pupil_trace.py under each model, and a check of the trigger on the labeled frames whose pupil
  the labeler left EMPTY (eye closed). Output: 0_first5minvedio/results/pupil_old_vs_new_model.csv/.png
Nothing else is modified; pupil_trace.py stays as it is."""
import glob
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import pupil_trace as pt  # noqa: E402
from scale_step import HERE, flat, human_table  # noqa: E402

ROOT = HERE.parent
UNITS = [("0_first5minvedio", "x100_step05_final", "video 0 (5 min, mouse A)"),
         ("1_20251031_Pluto_spont_1", "x140_step07_final", "video 1 (Pluto 1, mouse B)"),
         ("2_pluton2", "x080_step04_final", "video 2 (Pluto 2, mouse B)"),
         ("3_pluto3", "x060_step03_final", "video 3 (Pluto 3, mouse B)")]
PUPIL = ["pupil_top", "pupil_bottom", "pupil_left", "pupil_right"]
DROP_OF = {"pupil_top": "current", "pupil_bottom": "drop_B", "pupil_left": "drop_L", "pupil_right": "drop_R"}
OLD_H5 = ROOT / "old_pupil_top/active-learning-jump-selection/training-data/face_first4minDLC_Resnet50_EyePupilBlinkAug17shuffle60_snapshot_best-100.h5"
FPS = 60.0


def areas(d):
    """All area rules on a flat frame table (bodypart, coord). Ellipse area = pi/4 * width * height."""
    T, B, L, R = (d[b] for b in PUPIL)
    cx, cy = (L["x"] + R["x"]) / 2, (L["y"] + R["y"]) / 2            # centre from the horizontal pair
    cx2, cy2 = (T["x"] + B["x"]) / 2, (T["y"] + B["y"]) / 2          # centre from the vertical pair
    w_lr, h_tb = (R["x"] - L["x"]).abs(), (B["y"] - T["y"]).abs()
    a = pd.DataFrame(index=d.index)
    a["current"] = np.pi / 4 * w_lr * 2 * (B["y"] - cy)              # drop_T = the rule in production
    a["drop_B"] = np.pi / 4 * w_lr * 2 * (cy - T["y"])
    a["drop_L"] = np.pi / 4 * 2 * (R["x"] - cx2) * h_tb
    a["drop_R"] = np.pi / 4 * 2 * (cx2 - L["x"]) * h_tb
    a["4pt"] = np.pi / 4 * w_lr * h_tb
    return a


def prelabel_moves(unit):
    """Median |human - pre-label| per pupil point over the training batches of a video (px), plus how often it was emptied."""
    lab = HERE / unit / "training-data" / "labels"
    moves, emptied, n = {b: [] for b in PUPIL}, {b: 0 for b in PUPIL}, 0
    for bdir in sorted(lab.glob("batch*")):
        ml = glob.glob(str(bdir / "machinelabels.h5")) or glob.glob(str(bdir / "_old_prelabels" / "machinelabels.h5"))
        hm = glob.glob(str(bdir / "CollectedData_*.h5"))
        if not ml or not hm:
            continue
        pre, hum = flat(pd.read_hdf(ml[0])), flat(pd.read_hdf(hm[0]))
        common = hum.index.intersection(pre.index)
        n += len(common)
        for b in PUPIL:
            dx = hum.loc[common, (b, "x")] - pre.loc[common, (b, "x")]
            dy = hum.loc[common, (b, "y")] - pre.loc[common, (b, "y")]
            dist = np.hypot(dx, dy)
            moves[b].extend(dist.dropna().tolist())
            emptied[b] += int((hum.loc[common, (b, "x")].isna() & pre.loc[common, (b, "x")].notna()).sum())
    # most pre-labels are accepted unchanged, so the median move is 0 for every point; the prior therefore uses
    # "how often the labeler CORRECTED the point": moved by more than 2 px or emptied (judged not visible)
    frac = {b: ((np.sum(np.asarray(moves[b]) > 2.0) + emptied[b]) / n if n else np.nan) for b in PUPIL}
    mean_move = {b: (float(np.mean(moves[b])) if moves[b] else np.nan) for b in PUPIL}
    return frac, mean_move, emptied, n


# ------------------------------------------------------------------ Part A
rows, prior_rows, per_video = [], [], {}
pooled = []
for unit, lab, title in UNITS:
    gt = flat(pd.read_hdf(HERE / unit / "training-data" / "labels" / "test_frozen" / "test50_labels.h5"))
    ph = glob.glob(str(HERE / unit / "training-data" / "eval" / lab / "image_predictions_*.h5"))[0]
    pr = flat(pd.read_hdf(ph)).loc[gt.index]
    ok = pd.concat([gt[(b, c)].notna() for b in PUPIL for c in ("x", "y")], axis=1).all(axis=1)
    gt, pr = gt[ok], pr[ok]
    G, P = areas(gt), areas(pr)
    truth = G["4pt"]
    frac, moves, emptied, n_pre = prelabel_moves(unit)
    worst = max(PUPIL, key=lambda b: (frac[b] if np.isfinite(frac[b]) else -1, moves[b] if np.isfinite(moves[b]) else -1))
    P["prior"] = P[DROP_OF[worst]]
    conf = pd.DataFrame({b: pr[(b, "likelihood")] for b in PUPIL})
    lowest = conf.idxmin(axis=1)
    P["conf"] = pd.Series([P.loc[i, DROP_OF[lowest[i]]] for i in P.index], index=P.index)
    drops = P[["current", "drop_B", "drop_L", "drop_R"]]
    err = (drops.sub(truth, axis=0)).abs()
    P["oracle"] = pd.Series([drops.loc[i, err.loc[i].idxmin()] for i in P.index], index=P.index)
    prior_rows.append({"video": unit, "n_prelabeled_frames": n_pre, **{f"frac_corrected_{b}": round(frac[b], 3) for b in PUPIL},
                       **{f"mean_move_px_{b}": round(moves[b], 2) for b in PUPIL},
                       **{f"emptied_{b}": emptied[b] for b in PUPIL}, "point_dropped_by_prior": worst, "prior_rule": DROP_OF[worst]})
    per_video[unit] = {}
    for v in ("current", "4pt", "drop_B", "drop_L", "drop_R", "prior", "conf", "oracle"):
        rel = (P[v] - truth) / truth * 100
        same = (P[v] - G[v]) / G[v] * 100 if v in G else pd.Series(np.nan, index=P.index)
        per_video[unit][v] = float(rel.abs().median())
        rows.append({"video": unit, "variant": v, "n": int(len(rel)), "median_abs_rel_err_pct": round(float(rel.abs().median()), 1),
                     "p90_abs_rel_err_pct": round(float(rel.abs().quantile(0.9)), 1), "median_signed_err_pct": round(float(rel.median()), 1),
                     "median_abs_rel_err_vs_same_rule_on_human_pct": round(float(same.abs().median()), 1) if same.notna().any() else np.nan,
                     "note": f"prior drops {worst}" if v == "prior" else ""})
        pooled.append(pd.DataFrame({"video": unit, "variant": v, "rel": rel}))
pooled = pd.concat(pooled)
for v, g in pooled.groupby("variant", sort=False):
    rows.append({"video": "all four videos pooled", "variant": v, "n": int(len(g)), "median_abs_rel_err_pct": round(float(g["rel"].abs().median()), 1),
                 "p90_abs_rel_err_pct": round(float(g["rel"].abs().quantile(0.9)), 1), "median_signed_err_pct": round(float(g["rel"].median()), 1),
                 "median_abs_rel_err_vs_same_rule_on_human_pct": np.nan, "note": ""})
outA = pd.DataFrame(rows)
outA.to_csv(HERE / "results" / "pupil_area_variants.csv", index=False)
pd.DataFrame(prior_rows).to_csv(HERE / "results" / "pupil_area_labeler_prior.csv", index=False)
pd.set_option("display.width", 250)
print(pd.DataFrame(prior_rows).to_string(index=False))
print(outA.to_string(index=False))

# ------------------------------------------------------------------ Part B (5-minute video, old vs new model)
old = pt.load(OLD_H5)
new_h5 = glob.glob(str(HERE / "0_first5minvedio" / "training-data" / "predictions_step05" / "*.h5"))[0]
new = pt.load(new_h5).iloc[: len(old)]                               # the old prediction covers the first 4 minutes (14,400 frames)
raw_o, fil_o, _, I_o = pt.pupil_trace(old, fps=FPS)
raw_n, fil_n, _, I_n = pt.pupil_trace(new, fps=FPS)
A_n = areas(new)
minconf_n = pd.DataFrame({b: new[(b, "likelihood")] for b in PUPIL}).min(axis=1).to_numpy()
bad_n4 = I_n["bad"] | (minconf_n < pt.PCUT)                          # old rule + lowest pupil confidence (any of the four points)
both = ~I_o["bad"] & ~I_n["bad"]
ratio = (raw_n["area"] / raw_o["area"])[both]
rowsB = [
    {"quantity": "frames compared", "value": int(len(old)), "note": "first 4 minutes of the 5-minute video, both models"},
    {"quantity": "corr(3-pt area old model, 3-pt area new model)", "value": round(float(np.corrcoef(raw_o["area"][both], raw_n["area"][both])[0, 1]), 3), "note": f"frames trusted by both rules, n={int(both.sum())}"},
    {"quantity": "median new/old 3-pt area ratio", "value": round(float(ratio.median()), 3), "note": "same rule, same frames; differs from 1 only through the models"},
    {"quantity": "median |new/old - 1| (%)", "value": round(float((ratio - 1).abs().median() * 100), 1), "note": "frame-wise disagreement of the two models"},
    {"quantity": "corr(3-pt area, 4-pt area) new model", "value": round(float(np.corrcoef(raw_n["area"][~bad_n4], A_n["4pt"][~bad_n4])[0, 1]), 3), "note": "trusted frames"},
    {"quantity": "median 3-pt / 4-pt - 1 (%) new model", "value": round(float(((raw_n["area"] / A_n["4pt"])[~bad_n4] - 1).median() * 100), 1), "note": "bias of the old rule on the new model's output"},
    {"quantity": "untrusted frames, old model, rule 2026-09-19", "value": int(I_o["bad"].sum()), "note": f"{I_o['bad'].mean()*100:.1f}% of frames, {I_o['n_runs']} runs"},
    {"quantity": "untrusted frames, new model, rule 2026-09-19", "value": int(I_n["bad"].sum()), "note": f"{I_n['bad'].mean()*100:.1f}% of frames, {I_n['n_runs']} runs"},
    {"quantity": "untrusted frames, new model, rule + any pupil conf < 0.6", "value": int(bad_n4.sum()), "note": f"{bad_n4.mean()*100:.1f}% of frames"},
    {"quantity": "frames flagged by both models", "value": int((I_o["bad"] & I_n["bad"]).sum()), "note": "overlap of the two untrusted sets"},
]
# labeled frames of video 0 with an EMPTY pupil (labeler: eye closed) -> does the trigger catch them?
closed, opened = [], []
for s in sorted((HERE / "0_first5minvedio" / "training-data" / "labels").glob("*")):
    if not s.is_dir() or s.name.startswith("_") or not glob.glob(str(s / "CollectedData_*.h5")):
        continue
    h = flat(human_table(s))
    for name, r in h.iterrows():
        fi = int(Path(name).stem[3:])
        if fi >= len(old):
            continue
        (closed if all(pd.isna(r[(b, "x")]) for b in PUPIL) else opened).append(fi)
closed, opened = np.array(sorted(set(closed)), dtype=int), np.array(sorted(set(opened)), dtype=int)
for name, bad in (("old model, rule 2026-09-19", I_o["bad"]), ("new model, rule 2026-09-19", I_n["bad"]), ("new model, rule + any pupil conf < 0.6", bad_n4)):
    rowsB.append({"quantity": f"labeled frames flagged untrusted: {name}",
                  "value": f"{int(bad[opened].sum())}/{len(opened)} with a labeled pupil" + (f", {int(bad[closed].sum())}/{len(closed)} with an empty pupil" if len(closed) else ""),
                  "note": "no labeled frame of video 0 has an empty pupil (mouse A never fully closed the eye in the labeled frames)" if not len(closed)
                          else "empty pupil = labeler judged the eye closed"})
outB = pd.DataFrame(rowsB)
R0 = HERE / "0_first5minvedio" / "results" / "blink_area_analysis"
outB.to_csv(R0 / "pupil_old_vs_new_model.csv", index=False)
print(outB.to_string(index=False))

# ------------------------------------------------------------------ figures
BLUE, RED, GRAY, GREEN, ORANGE = "#2F6B9A", "#D1495B", "0.55", "#2A9D8F", "#E39B3C"
variants = ["current", "4pt", "drop_B", "drop_L", "drop_R", "prior", "conf", "oracle"]
fig, axes = plt.subplots(1, 2, figsize=(15, 5.4), constrained_layout=True, gridspec_kw={"width_ratios": [1.6, 1]})
ax = axes[0]
w = 0.8 / (len(UNITS) + 1)
cols = [BLUE, RED, GREEN, ORANGE, "k"]
for j, (unit, _, title) in enumerate(UNITS + [("all four videos pooled", "", "pooled (200 frames)")]):
    vals = outA[(outA["video"] == unit)].set_index("variant").loc[variants, "median_abs_rel_err_pct"]
    ax.bar(np.arange(len(variants)) + (j - len(UNITS) / 2) * w, vals, width=w, color=cols[j], alpha=0.9 if j < 4 else 0.5, label=title)
ax.set_xticks(range(len(variants)))
ax.set_xticklabels(["current\n(3-pt, top unused)", "4-pt", "drop bottom", "drop left", "drop right", "labeler prior\n(3 of 4)", "lowest conf\n(3 of 4, per frame)", "oracle\n(lower bound)"], fontsize=8.5)
ax.set_ylabel("median |area error| vs human 4-pt area (%)"); ax.grid(axis="y", alpha=0.3); ax.legend(fontsize=8.5)
ax.set_title("Frozen test frames: which pupil points should the area use?", fontsize=11)
ax = axes[1]
pr_df = pd.DataFrame(prior_rows).set_index("video")
for j, (unit, _, title) in enumerate(UNITS):
    ax.bar(np.arange(4) + (j - 1.5) * 0.2, [100 * pr_df.loc[unit, f"frac_corrected_{b}"] for b in PUPIL], width=0.2, color=cols[j], label=title)
ax.set_xticks(range(4)); ax.set_xticklabels([b.replace("pupil_", "") for b in PUPIL])
ax.set_ylabel("pre-labels corrected by the labeler (moved > 2 px or emptied, %)"); ax.grid(axis="y", alpha=0.3); ax.legend(fontsize=8.5)
ax.set_title("Labeler prior: how often each pupil point had to be corrected", fontsize=11)
fig.suptitle("Pupil-area rules under the new label standard (truth = 4-point ellipse from the human labels)", fontsize=12)
fig.savefig(HERE / "results" / "pupil_area_variants.png", dpi=130)
print("saved", HERE / "results" / "pupil_area_variants.png")

t = np.arange(len(old)) / FPS
fig, axes = plt.subplots(3, 1, figsize=(16, 11), constrained_layout=True)
for ax, (t0, t1) in zip(axes, ((0, t[-1]), (60, 90), (150, 180))):
    sl = (t >= t0) & (t <= t1)
    ax.plot(t[sl], fil_o["area"][sl], lw=0.8, color=GRAY, label="OLD-era model (Aug 17, 100 labels): 3-pt area, blink-filled")
    ax.plot(t[sl], fil_n["area"][sl], lw=0.9, color=BLUE, label="NEW-standard model (video 0 step 5, 100 labels): 3-pt area, blink-filled")
    ax.plot(t[sl], pd.Series(A_n["4pt"].to_numpy()).where(~bad_n4).interpolate(limit_area="inside").to_numpy()[sl], lw=0.9, color=RED, label="NEW model: 4-pt area (rule + pupil conf)")
    ax.fill_between(t[sl], 0, 1, where=I_n["bad"][sl], transform=ax.get_xaxis_transform(), color=ORANGE, alpha=0.3, label="untrusted (new model, rule 2026-09-19)")
    cs = closed[(closed / FPS >= t0) & (closed / FPS <= t1)]
    if len(cs):
        ax.vlines(cs / FPS, 0, 1, transform=ax.get_xaxis_transform(), color="k", lw=0.8, ls=":", label="labeled frame with EMPTY pupil (eye closed)")
    ax.set_xlim(t0, t1); ax.set_ylabel("pupil area (px$^2$)"); ax.grid(alpha=0.2)
    ax.set_ylim(0, 1.2 * np.nanpercentile(fil_n["area"], 99.5))
axes[0].legend(fontsize=8.5, ncol=3, loc="upper right"); axes[-1].set_xlabel("time (s)")
axes[0].set_title(f"Video 0, first 4 minutes: corr(old, new 3-pt area) = {outB.iloc[1]['value']}, median new/old = {outB.iloc[2]['value']}; "
                  f"untrusted frames old {int(I_o['bad'].sum())}, new {int(I_n['bad'].sum())}", fontsize=11)
axes[1].set_title("zoom 60-90 s"); axes[2].set_title("zoom 150-180 s")
fig.suptitle("The 5-minute video under the old-era and the new-standard model (same 3-point rule, so only the models differ)", fontsize=12)
fig.savefig(R0 / "pupil_old_vs_new_model.png", dpi=130)
print("saved", R0 / "pupil_old_vs_new_model.png")
