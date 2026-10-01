#!/usr/bin/env python3
"""Build the project summary page.

    python report/build_report.py            # -> report/index.html (+ report/assets/*.jpg)
    python report/build_report.py --rebuild-assets   # also re-cut the era 1-2 images from the local videos

Every number on the page is read here from the result tables of the units, so the page can be
rebuilt whenever a table changes. Images cut from the raw videos need the local (untracked)
videos/predictions; when those are missing the previously built assets are kept.
"""
import glob
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
REP = ROOT / "report"
ASSETS = REP / "assets"
ASSETS.mkdir(exist_ok=True)
V1 = ROOT / "v1_old_pupil_top"                    # eras 1-2 (old label standards)
V2 = ROOT / "v2_new_pupil_top_video_scaling"      # era 3 (ellipse label standard, one folder per video)
sys.path.insert(0, str(V1 / "time-series-analysis" / "scripts"))

D = {}

# ---- 1 scaling curve (old label standard, overall keypoint RMSE) ----------------------------
s = pd.read_csv(V1 / "scaling-curve/results/scaling_summary.csv")
s = s[s["model"] == "ResNet-50"].sort_values("train_frames")
D["scaling"] = [{"n": int(r.train_frames), "rmse": round(r.overall_keypoint_rmse_px, 1)} for r in s.itertuples()]

# ---- 2 batch size ---------------------------------------------------------------------------
b = pd.read_csv(V1 / "batch-size-selection/results/batch_size_summary.csv", index_col=False)
D["batch"] = [{"batch": int(r.batch_size), "rmse": round(r.external_overall_keypoint_rmse_px, 2),
               "val_loss": round(r.minimum_internal_valid_loss, 5),
               "conf": int(r.external_points_likelihood_ge_0_6)} for r in b.itertuples()]

# ---- 3 backbone -----------------------------------------------------------------------------
m = pd.read_csv(V1 / "model-selection/results/model_selection_summary.csv")
D["model"] = [{"model": r.model, "rmse": round(r.external_overall_rmse_px, 1), "val_loss": round(r.min_internal_valid_loss, 5),
               "conf": int(r.external_points_ge_0_6), "hours": r.training_hours} for r in m.itertuples()]

# ---- 4 active learning (one metric, final snapshot) -----------------------------------------
a = pd.read_csv(V1 / "active-learning-jump-selection/results/convergence_final_snapshot.csv")
D["al"] = {br: [{"n": int(r.cumulative_training_frames), "rmse": r.median_frame_rmse_px}
                for r in g.sort_values("round").itertuples()] for br, g in a.groupby("branch")}
D["al_range"] = [float(a["median_frame_rmse_px"].min()), float(a["median_frame_rmse_px"].max())]

# ---- 5 new video ----------------------------------------------------------------------------
c = pd.read_csv(V1 / "new-video-generalization/results/scale_curve.csv")
seed2 = c["label"].str.contains("seed43")
fin = c[(c["snapshot_rule"].str.startswith("final") | c["snapshot_rule"].str.startswith("n/a")) & ~seed2].sort_values("pluto_training_frames")
D["scale"] = [{"n": int(r.pluto_training_frames), "rmse": r.median_frame_rmse_px, "p90": r.p90_frame_rmse_px,
               "fail": round(r.frac_frames_rmse_gt_50px * 100, 1), "conf": round(r._7 * 100) if False else None}
              for r in fin.itertuples()]
for row, (_, r) in zip(D["scale"], fin.iterrows()):
    row["conf"] = round(float(r["frac_points_conf_ge_0.6"]) * 100)
D["scale_seed2"] = [{"n": int(r.pluto_training_frames), "rmse": r.median_frame_rmse_px, "p90": r.p90_frame_rmse_px}
                    for r in c[c["snapshot_rule"].str.startswith("final") & seed2].itertuples()]
dflt = c[c["snapshot_rule"].str.startswith("best validation mAP") & ~c["snapshot_rule"].str.contains(">= 80")]
D["scale_default_rule"] = [{"n": int(r.pluto_training_frames), "rmse": r.median_frame_rmse_px, "rule": r.snapshot_rule}
                           for r in dflt.sort_values("pluto_training_frames").itertuples()]
j = pd.read_csv(V1 / "new-video-generalization/results/jump_flagged.csv")
D["flagged"] = [{"n": int(r.model_pluto_training_frames), "pct": round(r.jump_flagged_pool_frames / r.pool_frames * 100, 1),
                 "frames": int(r.jump_flagged_pool_frames)} for r in j.itertuples()]
D["n_test"] = int(c["n_test_frames"].iloc[0])

# ---- 6 pupil trace example + images cut from the videos (local data; optional) --------------
CACHE = REP / "assets" / "derived.json"
try:
    if "--rebuild-assets" not in sys.argv and CACHE.exists():      # eras 1-2 images are finished work: keep them as built
        raise RuntimeError("cached era 1-2 assets kept (pass --rebuild-assets to cut them again)")
    import cv2
    import pupil_trace as pt

    d = pt.load(pt.H5)
    raw, fil, hold, info = pt.pupil_trace(d)
    bad, op = info["bad"], info["rel_open"]
    lo, hi = int(231.9 * 60), int(234.9 * 60)
    D["blink_example"] = {"t": [round(i / 60, 3) for i in range(lo, hi)],
                          "raw": [None if not np.isfinite(v) else round(float(v)) for v in raw["area"][lo:hi]],
                          "filled": [None if not np.isfinite(v) else round(float(v)) for v in fil["area"][lo:hi]],
                          "open": [round(float(v), 3) for v in op[lo:hi]], "bad": [bool(v) for v in bad[lo:hi]]}
    D["blink_stats"] = {"frames": int(len(d)), "untrusted": int(bad.sum()), "runs": int(info["n_runs"]),
                        "longest_s": round(info["longest"] / 60, 2)}

    clip = V1 / "active-learning-jump-selection/training-data/face_first4min.mp4"
    cap = cv2.VideoCapture(str(clip))

    def frame(i):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
        ok, im = cap.read()
        return im

    def P(i, bp):
        return float(d[bp]["x"].iloc[i]), float(d[bp]["y"].iloc[i])

    def crop_around(im, i, half=(330, 230)):
        cx = np.mean([P(i, b)[0] for b in ("eye_nasal_corner", "eye_temporal_corner")])
        cy = np.mean([P(i, b)[1] for b in ("eyelid_top", "eyelid_bottom")])
        x0, y0 = int(max(cx - half[0], 0)), int(max(cy - half[1], 0))
        return im[y0:y0 + 2 * half[1], x0:x0 + 2 * half[0]].copy(), x0, y0

    NAMES = {"pupil_top": "pupil top", "pupil_bottom": "pupil bottom", "pupil_left": "pupil left", "pupil_right": "pupil right",
             "eyelid_top": "upper lid", "eyelid_bottom": "lower lid", "eye_nasal_corner": "nasal corner",
             "eye_temporal_corner": "temporal corner"}
    PUP, LID = (60, 60, 235), (220, 190, 40)      # BGR

    # a) the eight keypoints on a quiet open-eye frame
    i0 = int(np.nanargmax(np.where(bad, -np.inf, op * 0 + raw["area"].rolling(31, center=True, min_periods=1).median().to_numpy() * 0 + 1)) or 3000)
    i0 = 3000 if bad[3000] is np.False_ or not bad[3000] else int(np.where(~bad)[0][len(np.where(~bad)[0]) // 4])
    im, x0, y0 = crop_around(frame(i0), i0)
    for bp, nm in NAMES.items():
        x, y = P(i0, bp); x, y = int(x - x0), int(y - y0)
        col = PUP if bp.startswith("pupil") else LID
        cv2.circle(im, (x, y), 7, col, -1, cv2.LINE_AA); cv2.circle(im, (x, y), 7, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.imwrite(str(ASSETS / "keypoints.jpg"), im, [cv2.IMWRITE_JPEG_QUALITY, 88])
    D["keypoints_px"] = {bp: [round((P(i0, bp)[0] - x0) / im.shape[1] * 100, 1), round((P(i0, bp)[1] - y0) / im.shape[0] * 100, 1)]
                         for bp in NAMES}

    # b) pupil geometry: 4-point ellipse (visible pupil) vs 3-point endpoint ellipse (whole pupil) on the same frame
    im, x0, y0 = crop_around(frame(i0), i0)
    L, R, T, B = (np.array(P(i0, b)) - (x0, y0) for b in ("pupil_left", "pupil_right", "pupil_top", "pupil_bottom"))
    w = abs(R[0] - L[0])
    c4 = ((L[0] + R[0]) / 2, (T[1] + B[1]) / 2); h4 = abs(B[1] - T[1])
    ce = ((L[0] + R[0]) / 2, (L[1] + R[1]) / 2); he = 2 * (B[1] - ce[1])
    cv2.ellipse(im, (int(c4[0]), int(c4[1])), (int(w / 2), int(h4 / 2)), 0, 0, 360, (235, 160, 60), 2, cv2.LINE_AA)
    cv2.ellipse(im, (int(ce[0]), int(ce[1])), (int(w / 2), int(he / 2)), 0, 0, 360, (70, 70, 240), 3, cv2.LINE_AA)
    for p_, used in ((L, True), (R, True), (B, True), (T, False)):
        cv2.circle(im, (int(p_[0]), int(p_[1])), 7, (70, 70, 240) if used else (235, 160, 60), -1, cv2.LINE_AA)
        cv2.circle(im, (int(p_[0]), int(p_[1])), 7, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.drawMarker(im, (int(ce[0]), int(ce[1])), (70, 70, 240), cv2.MARKER_CROSS, 18, 2, cv2.LINE_AA)
    cv2.drawMarker(im, (int(c4[0]), int(c4[1])), (235, 160, 60), cv2.MARKER_TILTED_CROSS, 14, 2, cv2.LINE_AA)
    cv2.imwrite(str(ASSETS / "pupil_geometry.jpg"), im, [cv2.IMWRITE_JPEG_QUALITY, 88])

    # c) a blink in three frames (before / deepest / after), predicted points overlaid
    deepest = lo + int(np.nanargmin(op[lo:hi]))
    for k_, i in enumerate((deepest - 40, deepest, deepest + 45)):
        f_, fx, fy = crop_around(frame(i), deepest - 40, half=(330, 165))
        for bp in NAMES:
            x, y = P(i, bp)
            cv2.circle(f_, (int(x - fx), int(y - fy)), 6, PUP if bp.startswith("pupil") else LID, -1, cv2.LINE_AA)
        cv2.imwrite(str(ASSETS / f"blink_{k_}.jpg"), f_, [cv2.IMWRITE_JPEG_QUALITY, 85])
    D["blink_strip_t"] = [round((deepest - 40) / 60, 2), round(deepest / 60, 2), round((deepest + 45) / 60, 2)]
    cap.release()

    # d) existing result figures, downscaled
    for src, dst, width in [("v1_old_pupil_top/time-series-analysis/results/05_saccade_closeup.png", "saccade.jpg", 1500),
                            ("v1_old_pupil_top/new-video-generalization/results/02_batch01_selected_frames.png", "picks_batch01.jpg", 1600),
                            ("v1_old_pupil_top/new-video-generalization/results/01_batch01_kmeans_clusters.png", "clusters_batch01.jpg", 1600)]:
        im = cv2.imread(str(ROOT / src))
        sc = width / im.shape[1]
        cv2.imwrite(str(ASSETS / dst), cv2.resize(im, None, fx=sc, fy=sc, interpolation=cv2.INTER_AREA), [cv2.IMWRITE_JPEG_QUALITY, 84])
    CACHE.write_text(json.dumps({k: D[k] for k in ("blink_example", "blink_stats", "keypoints_px", "blink_strip_t")}))
except Exception as e:                                            # local videos/predictions not available
    print("local data not available, reusing cached derived data:", repr(e))
    D.update(json.loads(CACHE.read_text()))

# ---- 8-11 era 3: ellipse label standard, labels needed per video, back-test, area / blink -------
pl = pd.read_csv(V2 / "results/labels_to_plateau.csv")
CARRIED_0_3 = {0: 100, 1: 100, 2: 60, 3: 60}       # videos 0-3 carry the labels that existed when the next video started
vids = []
for r in pl.itertuples():
    sc = pd.read_csv(V2 / r.unit / "results/scale_curve.csv")
    reg = sc[sc["label"].str.match(r"^x\d{3}_step\d{2}_final$") & (sc["training_frames_this_video"] > 0)].sort_values("training_frames_this_video")
    sub = sc[sc["label"].str.contains(r"_final_sub\d+[ab]$")].sort_values("label")
    pt = lambda q: {"n": int(q["training_frames_this_video"]), "rmse": float(q["median_frame_rmse_px"]),
                    "p90": round(float(q["p90_frame_rmse_px"]), 1), "fail": round(float(q["frac_frames_rmse_gt_50px"]) * 100),
                    "conf": round(float(q["frac_points_conf_ge_0.6"]) * 100)}
    jf = V2 / r.unit / "results/jump_flagged.csv"
    jump = []
    if jf.exists():
        j3 = pd.read_csv(jf)
        if {"model_labels", "jump_flagged_pool_frames", "pool_frames"} <= set(j3.columns):
            jump = [{"n": int(q.model_labels), "pct": round(q.jump_flagged_pool_frames / q.pool_frames * 100, 1)} for q in j3.itertuples()]
    final = r.status == "final"
    vids.append({"video": int(r.video), "unit": r.unit, "mouse": r.mouse, "date": str(r.date), "final": bool(final),
                 "plateau_n": int(r.labels_at_plateau), "plateau_px": float(r.plateau_median_px),
                 "zero": None if pd.isna(r.zero_label_px) else float(r.zero_label_px),
                 "steps": [pt(q) for _, q in reg.iterrows()],
                 "subsets": [{**pt(q), "tag": q["label"][-3:]} for _, q in sub.iterrows()],
                 "jump": jump, "labeled": int(reg["training_frames_this_video"].max()),
                 "carried": CARRIED_0_3.get(int(r.video), int(r.labels_at_plateau) if final else None)})
D["videos"] = vids

ar = pd.read_csv(V2 / "results/blink_area_consistency_area.csv")
D["area"] = [{"video": str(r.video), "e3": float(r.err_3pt_median_abs_pct), "e4": float(r.err_4pt_median_abs_pct),
              "anchored": bool(r.anchored_prelabels)} for r in ar.itertuples()]
bk = pd.read_csv(V2 / "results/blink_area_consistency_blink.csv")
pooled = bk[bk["video"].str.startswith("all")].iloc[0]
D["blink3"] = {"closed": int(pooled["closed_frames"]), "open": int(pooled["open_frames"]),
               "auc": {k: float(pooled[f"AUC_{k}"]) for k in ("min_conf", "open_rel", "centre_off", "lid_rel", "aspect")},
               "hit": float(pooled["production_rule_hit_rate_closed"]), "false_alarm": float(pooled["production_rule_false_alarm_open"])}
cp = pd.read_csv(V2 / "results/blink_area_consistency_coupling.csv")
D["coupling"] = [round(float(cp["corr_area4_opening_trusted"].min()), 2), round(float(cp["corr_area4_opening_trusted"].max()), 2)]
D["built"] = pd.Timestamp.now().strftime("%Y-%m-%d")

try:                                                                # era-3 figures (result PNGs are tracked; label images need local data)
    import cv2

    for src, dst, width in [("results/cross_video_curves.png", "cross_video_curves.jpg", 2400),
                            ("results/cross_video_matrix.png", "cross_video_matrix.jpg", 2200)]:
        im = cv2.imread(str(V2 / src))
        sc_ = width / im.shape[1]
        cv2.imwrite(str(ASSETS / dst), cv2.resize(im, None, fx=sc_, fy=sc_, interpolation=cv2.INTER_AREA), [cv2.IMWRITE_JPEG_QUALITY, 82])
    sheet = ROOT / "local_data/label_review_2026-09-20/10_video1_batch01_relabeled.jpg"
    if sheet.exists():
        im = cv2.imread(str(sheet))
        sc_ = 2000 / im.shape[1]
        cv2.imwrite(str(ASSETS / "labels_batch01.jpg"), cv2.resize(im, None, fx=sc_, fy=sc_, interpolation=cv2.INTER_AREA), [cv2.IMWRITE_JPEG_QUALITY, 80])
    b01 = V2 / "0_first5minvedio/training-data/labels/batch01"
    h5 = sorted(glob.glob(str(b01 / "CollectedData_*.h5")))
    if h5:                                                          # one labeled frame with the ellipse through its four pupil points
        t = pd.read_hdf(h5[0])
        t.columns = t.columns.droplevel(0)
        row = t.iloc[0]
        name = t.index[0][-1] if isinstance(t.index[0], tuple) else Path(str(t.index[0])).name
        im = cv2.imread(str(b01 / name))
        lo_, hi_ = np.percentile(im, (2, 98))
        im = np.clip((im.astype(np.float32) - lo_) / (hi_ - lo_) * 255, 0, 255).astype(np.uint8)
        P3 = lambda bp: (float(row[bp]["x"]), float(row[bp]["y"]))
        (lx, ly), (rx, ry), (tx, ty), (bx, by) = (P3(k) for k in ("pupil_left", "pupil_right", "pupil_top", "pupil_bottom"))
        cv2.ellipse(im, (int((lx + rx) / 2), int((ty + by) / 2)), (int(abs(rx - lx) / 2), int(abs(by - ty) / 2)), 0, 0, 360, (70, 70, 240), 2, cv2.LINE_AA)
        bps3 = ["pupil_top", "pupil_bottom", "pupil_left", "pupil_right", "eyelid_top", "eyelid_bottom", "eye_nasal_corner", "eye_temporal_corner"]
        for bp in bps3:
            x, y = P3(bp)
            cv2.circle(im, (int(x), int(y)), 7, (60, 60, 235) if bp.startswith("pupil") else (220, 190, 40), -1, cv2.LINE_AA)
            cv2.circle(im, (int(x), int(y)), 7, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.imwrite(str(ASSETS / "label_standard.jpg"), im, [cv2.IMWRITE_JPEG_QUALITY, 88])
        D["label_px"] = {bp: [round(P3(bp)[0] / im.shape[1] * 100, 1), round(P3(bp)[1] / im.shape[0] * 100, 1)] for bp in bps3}
        (ASSETS / "derived_era3.json").write_text(json.dumps({"label_px": D["label_px"]}))
except Exception as e:
    print("era-3 images not rebuilt:", repr(e))
if "label_px" not in D and (ASSETS / "derived_era3.json").exists():
    D.update(json.loads((ASSETS / "derived_era3.json").read_text()))

tpl = (REP / "template.html").read_text(encoding="utf-8")
(REP / "index.html").write_text(tpl.replace("/*__DATA__*/", "const DATA = " + json.dumps(D, ensure_ascii=False) + ";"), encoding="utf-8")
print("built", REP / "index.html", "|", {k: (len(v) if hasattr(v, "__len__") else v) for k, v in D.items()})
