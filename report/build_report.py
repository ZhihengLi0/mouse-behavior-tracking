#!/usr/bin/env python3
"""Build the project summary page.

    python report/build_report.py            # -> report/index.html (+ report/assets/*.jpg)

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
sys.path.insert(0, str(ROOT / "time-series-analysis" / "scripts"))

D = {}

# ---- 1 scaling curve (old label standard, overall keypoint RMSE) ----------------------------
s = pd.read_csv(ROOT / "scaling-curve/results/scaling_summary.csv")
s = s[s["model"] == "ResNet-50"].sort_values("train_frames")
D["scaling"] = [{"n": int(r.train_frames), "rmse": round(r.overall_keypoint_rmse_px, 1)} for r in s.itertuples()]

# ---- 2 batch size ---------------------------------------------------------------------------
b = pd.read_csv(ROOT / "batch-size-selection/results/batch_size_summary.csv", index_col=False)
D["batch"] = [{"batch": int(r.batch_size), "rmse": round(r.external_overall_keypoint_rmse_px, 2),
               "val_loss": round(r.minimum_internal_valid_loss, 5),
               "conf": int(r.external_points_likelihood_ge_0_6)} for r in b.itertuples()]

# ---- 3 backbone -----------------------------------------------------------------------------
m = pd.read_csv(ROOT / "model-selection/results/model_selection_summary.csv")
D["model"] = [{"model": r.model, "rmse": round(r.external_overall_rmse_px, 1), "val_loss": round(r.min_internal_valid_loss, 5),
               "conf": int(r.external_points_ge_0_6), "hours": r.training_hours} for r in m.itertuples()]

# ---- 4 active learning (one metric, final snapshot) -----------------------------------------
a = pd.read_csv(ROOT / "active-learning/results/convergence_final_snapshot.csv")
D["al"] = {br: [{"n": int(r.cumulative_training_frames), "rmse": r.median_frame_rmse_px}
                for r in g.sort_values("round").itertuples()] for br, g in a.groupby("branch")}
D["al_range"] = [float(a["median_frame_rmse_px"].min()), float(a["median_frame_rmse_px"].max())]

# ---- 5 new video ----------------------------------------------------------------------------
c = pd.read_csv(ROOT / "new-video-generalization/results/scale_curve.csv")
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
j = pd.read_csv(ROOT / "new-video-generalization/results/jump_flagged.csv")
D["flagged"] = [{"n": int(r.model_pluto_training_frames), "pct": round(r.jump_flagged_pool_frames / r.pool_frames * 100, 1),
                 "frames": int(r.jump_flagged_pool_frames)} for r in j.itertuples()]
D["n_test"] = int(c["n_test_frames"].iloc[0])

# ---- 6 pupil trace example + images cut from the videos (local data; optional) --------------
CACHE = REP / "assets" / "derived.json"
try:
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

    clip = ROOT / "active-learning/training-data/face_first4min.mp4"
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
    for src, dst, width in [("time-series-analysis/results/05_saccade_closeup.png", "saccade.jpg", 1500),
                            ("new-video-generalization/results/02_batch01_selected_frames.png", "picks_batch01.jpg", 1600),
                            ("new-video-generalization/results/01_batch01_kmeans_clusters.png", "clusters_batch01.jpg", 1600)]:
        im = cv2.imread(str(ROOT / src))
        sc = width / im.shape[1]
        cv2.imwrite(str(ASSETS / dst), cv2.resize(im, None, fx=sc, fy=sc, interpolation=cv2.INTER_AREA), [cv2.IMWRITE_JPEG_QUALITY, 84])
    CACHE.write_text(json.dumps({k: D[k] for k in ("blink_example", "blink_stats", "keypoints_px", "blink_strip_t")}))
except Exception as e:                                            # local videos/predictions not available
    print("local data not available, reusing cached derived data:", repr(e))
    D.update(json.loads(CACHE.read_text()))

tpl = (REP / "template.html").read_text(encoding="utf-8")
(REP / "index.html").write_text(tpl.replace("/*__DATA__*/", "const DATA = " + json.dumps(D, ensure_ascii=False) + ";"), encoding="utf-8")
print("built", REP / "index.html", "|", {k: (len(v) if hasattr(v, "__len__") else v) for k, v in D.items()})
