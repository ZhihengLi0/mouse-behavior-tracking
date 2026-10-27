#!/usr/bin/env python3
"""P2 baseline: what dropping the flagged frames does to the whole-video area and blink series (zero human cost).

    python repair_baselines.py

Flag rule (from results/unreliable_frames_sweep and 8.2): fit_pupil = one-step SARIMAX(3,0,1) deviation averaged over
the 4 pupil points (cached per video in results/unreliable_frames_sweep/cache/videoNN_fwd301.npy, the plateau model's
whole-video prediction). The threshold is a share of the video's eye width (median corner-to-corner distance of the
prediction), so that one rule serves every video; the share is the quantile of fit_pupil / eye width on the 1330 truth
frames that flags 5 % (and, for comparison, 9.2 %) of them.

Per video (plateau model, every frame):
  flagged frames and share; runs of consecutive flagged frames (also with gaps <= 5 frames merged); run lengths;
  area series (four-point, step 9) before / after dropping the flagged frames: median, relative spread (1.4826 MAD /
  median); eye-closure rule of step 10 (opening O < mouse threshold on trusted frames, else mean pupil confidence
  < 0.527): closed share and closure events (runs >= 3 frames, gaps < 5 merged) before / after, events that contain a
  flagged frame, events whose first frame is flagged (= onset would move if the flagged frames were dropped from the
  blink series too; the recommended default drops them from the AREA series only, the flag is about the pupil).
  Interpolation inside short runs (<= 5 frames) as the comparison: frames whose value changes.
On the 1330 truth frames: error of the flagged / unflagged frames, residual > 20 / > 50 px / area error > 20 % among
the unflagged frames, and the area error of the flagged frames in short runs before / after linear interpolation
from the neighbouring unflagged predictions.

Outputs in results/repair_baselines/: thresholds.csv, drop_per_video.csv, truth_frames.csv, truth_summary.csv.
Nothing else is modified."""
import glob
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import pupil_trace as pt  # noqa: E402
from scale_step import HERE  # noqa: E402

FPS, GAP, MIN_LEN, PUPIL_THR, DX_FRAC, CONF_MIN, SHORT_RUN = 60.0, 5, 3, 0.527, 0.3, 0.3, 5
RES = HERE / "results"
OUT = RES / "repair_baselines"
CACHE = RES / "unreliable_frames_sweep" / "cache"
P4 = ["pupil_top", "pupil_bottom", "pupil_left", "pupil_right"]
PL = pd.read_csv(RES / "labels_to_plateau.csv")
PL = PL[PL.status == "final"]
TH = pd.read_csv(RES / "blink_eyelid_distance/results/every_mouse/px/threshold_per_mouse.csv").set_index("mouse").threshold_px
MOUSE_THR = {"Pluto": TH["Pluto"], "Terra": TH["Terra"], "mouse A": TH["all mice pooled"]}
SHARES = {"5pct": 0.05, "9pct": 0.092}


def runs(mask, gap=0):
    """(start, end) inclusive of runs of True; runs closer than `gap` frames are merged."""
    idx = np.flatnonzero(mask)
    if len(idx) == 0:
        return []
    brk = np.flatnonzero(np.diff(idx) > gap + 1)
    starts, ends = np.r_[idx[0], idx[brk + 1]], np.r_[idx[brk], idx[-1]]
    return list(zip(starts.tolist(), ends.tolist()))


def load_video(rr):
    step = max(1, int(rr.labels_at_plateau) // 20)
    h5 = sorted(glob.glob(str(HERE / rr.unit / "training-data" / f"predictions_step{step:02d}" / "*snapshot*120*.h5")))
    d = pt.load(h5[-1])
    fwd = np.load(CACHE / f"video{rr.video:02d}_fwd301.npy")
    assert len(fwd) == len(d), (rr.video, len(fwd), len(d))
    return d, np.nanmean(fwd[:, :4], axis=1)


def eye_width(d):
    return float(np.nanmedian(np.hypot(d["eye_nasal_corner"]["x"] - d["eye_temporal_corner"]["x"],
                                       d["eye_nasal_corner"]["y"] - d["eye_temporal_corner"]["y"])))


def area_series(d):
    return np.pi / 4 * np.abs(d["pupil_right"]["x"].to_numpy(float) - d["pupil_left"]["x"].to_numpy(float)) \
        * np.abs(d["pupil_bottom"]["y"].to_numpy(float) - d["pupil_top"]["y"].to_numpy(float))


def closed_series(d, mouse, W):
    O = (d["eyelid_bottom"]["y"] - d["eyelid_top"]["y"]).to_numpy(float)
    dx = np.abs(d["eyelid_top"]["x"] - d["eyelid_bottom"]["x"]).to_numpy(float)
    trusted = (O >= 0) & (dx <= DX_FRAC * W) & (d["eyelid_top"]["likelihood"].to_numpy(float) >= CONF_MIN) \
        & (d["eyelid_bottom"]["likelihood"].to_numpy(float) >= CONF_MIN)
    pupil_conf = np.mean([d[b]["likelihood"].to_numpy(float) for b in P4], axis=0)
    return np.where(trusted, O < MOUSE_THR[mouse], pupil_conf < PUPIL_THR)


def events(closed):
    return [(a, b) for a, b in runs(closed, GAP - 1) if b - a + 1 >= MIN_LEN]


def spread(x):
    x = x[np.isfinite(x)]
    return 1.4826 * np.median(np.abs(x - np.median(x))) / np.median(x) * 100


def main():
    OUT.mkdir(exist_ok=True)
    truth = pd.read_csv(RES / "error_detector" / "scores_lovo.csv")
    truth = truth[truth.plateau].drop_duplicates(["video", "frame"]).reset_index(drop=True)
    sig = pd.read_csv(RES / "unreliable_frames_sweep" / "signals_all_frames.csv")[["video", "frame", "fit_pupil"]]
    truth = truth.drop(columns=[c for c in truth.columns if c == "fit_pupil"]).merge(sig, on=["video", "frame"])
    assert len(truth) == 1330 and (truth.frame_rmse_px > 20).sum() == 155 and (truth.frame_rmse_px > 50).sum() == 32

    videos = {}
    for rr in PL.itertuples():
        d, fp = load_video(rr)
        videos[int(rr.video)] = (rr, d, fp, eye_width(d))
    truth["eye_width"] = truth.video.map({v: w for v, (_, _, _, w) in videos.items()})
    truth["fit_rel"] = truth.fit_pupil / truth.eye_width
    thr_rows = [{"rule": k, "target_share": s, "fit_rel_threshold": float(truth.fit_rel.quantile(1 - s))} for k, s in SHARES.items()]
    THR = {r["rule"]: r["fit_rel_threshold"] for r in thr_rows}
    pd.DataFrame(thr_rows).to_csv(OUT / "thresholds.csv", index=False)

    human_area = np.pi / 4 * np.abs(truth.human_pupil_right_x - truth.human_pupil_left_x) * np.abs(truth.human_pupil_bottom_y - truth.human_pupil_top_y)
    pred_area = np.pi / 4 * np.abs(truth.pred_pupil_right_x - truth.pred_pupil_left_x) * np.abs(truth.pred_pupil_bottom_y - truth.pred_pupil_top_y)
    truth["area_err_pct"] = (pred_area - human_area).abs() / human_area * 100

    rows, trows = [], []
    for v, (rr, d, fp, W) in videos.items():
        n = len(d)
        area = area_series(d)
        closed = closed_series(d, rr.mouse, W)
        ev = events(closed)
        for rule, frac in THR.items():
            flag = fp > frac * W
            r1, r5 = runs(flag), runs(flag, GAP)
            L = np.array([b - a + 1 for a, b in r1]) if r1 else np.array([0])
            keep = ~flag
            # interpolation inside short runs only
            interp = area.copy().astype(float)
            interp[flag] = np.nan
            changed = 0
            for a, b in r1:
                if b - a + 1 <= SHORT_RUN and a > 0 and b < n - 1 and np.isfinite(area[a - 1]) and np.isfinite(area[b + 1]):
                    interp[a:b + 1] = np.interp(np.arange(a, b + 1), [a - 1, b + 1], [area[a - 1], area[b + 1]])
                    changed += b - a + 1
            ev_flagged = sum(flag[a:b + 1].any() for a, b in ev)
            ev_onset = sum(bool(flag[a]) for a, b in ev)
            closed_drop = closed & keep
            ev_drop = events(closed_drop)
            rows.append({"video": v, "mouse": rr.mouse, "rule": rule, "threshold_px": round(frac * W, 2), "eye_width_px": round(W, 1), "frames": n,
                         "flagged": int(flag.sum()), "flagged_pct": round(100 * flag.mean(), 2), "runs": len(r1), "runs_gap5_merged": len(r5),
                         "run_len_median": int(np.median(L)), "run_len_p90": int(np.percentile(L, 90)), "run_len_max": int(L.max()),
                         "longest_run_ms": round(L.max() / FPS * 1000), "single_frame_runs": int((L == 1).sum()), "runs_ge10": int((L >= 10).sum()),
                         "area_median_before": round(float(np.nanmedian(area))), "area_median_after_drop": round(float(np.nanmedian(area[keep]))),
                         "area_spread_before_pct": round(spread(area), 1), "area_spread_after_drop_pct": round(spread(area[keep]), 1),
                         "interp_short_runs_frames_changed": changed, "interp_short_runs_pct": round(100 * changed / n, 2),
                         "closed_frames_pct": round(100 * closed.mean(), 2), "closure_events": len(ev),
                         "events_containing_flag": ev_flagged, "events_onset_flagged": ev_onset,
                         "closed_frames_pct_if_blink_dropped_too": round(100 * closed_drop.mean(), 2), "closure_events_if_blink_dropped_too": len(ev_drop),
                         "flagged_inside_closure_pct": round(100 * float(flag[closed].mean()), 1) if closed.any() else np.nan})
            # truth frames of this video: flagged?, run length, interpolated area
            t = truth[truth.video == v]
            for r in t.itertuples():
                f = int(r.frame)
                tr = {"video": v, "frame": f, "rule": rule, "flagged": bool(flag[f]), "run_len": 0, "area_err_interp_pct": np.nan}
                if flag[f]:
                    a, b = next((a, b) for a, b in r1 if a <= f <= b)
                    tr["run_len"] = b - a + 1
                    if np.isfinite(interp[f]):
                        tr["area_err_interp_pct"] = abs(interp[f] - human_area[r.Index]) / human_area[r.Index] * 100
                trows.append(tr)
    D = pd.DataFrame(rows)
    D.to_csv(OUT / "drop_per_video.csv", index=False)
    T = pd.DataFrame(trows).merge(truth[["video", "frame", "set", "frame_rmse_px", "area_err_pct", "fit_pupil"]], on=["video", "frame"])
    T.round(2).to_csv(OUT / "truth_frames.csv", index=False)

    srows = []
    for rule, g in T.groupby("rule"):
        fl, un = g[g.flagged], g[~g.flagged]
        sh = fl[fl.area_err_interp_pct.notna()]
        srows.append({"rule": rule, "frames": len(g), "flagged": len(fl), "flagged_pct": round(100 * len(fl) / len(g), 1),
                      "flagged_gt20px": int((fl.frame_rmse_px > 20).sum()), "flagged_gt50px": int((fl.frame_rmse_px > 50).sum()),
                      "flagged_area_gt20pct": int((fl.area_err_pct > 20).sum()),
                      "recall_gt20px": round(100 * (fl.frame_rmse_px > 20).sum() / (g.frame_rmse_px > 20).sum(), 1),
                      "recall_gt50px": round(100 * (fl.frame_rmse_px > 50).sum() / (g.frame_rmse_px > 50).sum(), 1),
                      "recall_area_gt20pct": round(100 * (fl.area_err_pct > 20).sum() / (g.area_err_pct > 20).sum(), 1),
                      "unflagged_gt20px": int((un.frame_rmse_px > 20).sum()), "unflagged_gt50px": int((un.frame_rmse_px > 50).sum()),
                      "unflagged_area_gt20pct": int((un.area_err_pct > 20).sum()),
                      "unflagged_gt20px_pct": round(100 * (un.frame_rmse_px > 20).mean(), 1), "unflagged_gt50px_pct": round(100 * (un.frame_rmse_px > 50).mean(), 1),
                      "unflagged_area_gt20pct_pct": round(100 * (un.area_err_pct > 20).mean(), 1),
                      "all_area_gt20pct_pct": round(100 * (g.area_err_pct > 20).mean(), 1),
                      "flagged_area_err_median_pct": round(fl.area_err_pct.median(), 1), "unflagged_area_err_median_pct": round(un.area_err_pct.median(), 1),
                      "flagged_in_short_runs": len(sh), "short_runs_area_err_before_median_pct": round(sh.area_err_pct.median(), 1) if len(sh) else np.nan,
                      "short_runs_area_err_interp_median_pct": round(sh.area_err_interp_pct.median(), 1) if len(sh) else np.nan,
                      "short_runs_interp_better": int((sh.area_err_interp_pct < sh.area_err_pct).sum()),
                      "short_runs_interp_worse": int((sh.area_err_interp_pct > sh.area_err_pct).sum())})
    S = pd.DataFrame(srows)
    S.to_csv(OUT / "truth_summary.csv", index=False)
    pd.set_option("display.width", 250)
    print(pd.DataFrame(thr_rows).to_string(index=False))
    print(D[D.rule == "5pct"].drop(columns=["rule", "mouse"]).to_string(index=False))
    print(D.groupby("rule")[["flagged", "runs", "runs_gap5_merged", "single_frame_runs", "events_containing_flag", "events_onset_flagged", "closure_events"]].sum().to_string())
    print(S.T.to_string())


if __name__ == "__main__":
    main()
