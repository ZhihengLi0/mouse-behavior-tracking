#!/usr/bin/env python3
"""Automatic per-frame repair of sporadic big errors by a geometry-checked second heatmap peak (P2 step 3, pilot).

    python second_peak_repair.py --video 8 [--k-mad 5] [--bands pooled|own] [--sheet] [--tag NAME]

Idea: a > 50 px frame is usually one pupil point thrown far (pupil_left in 21 of the 32 truth frames) while the
predictor's heatmap for that point often has a second, nearly as strong peak at the right place. Nothing temporal is
used: every frame is judged alone, a repair changes only the frames that violate the geometry, and the repair keeps the
model's own peak-2 coordinates (no interpolation, no smoothing).

Per frame, in the eye frame (origin = midpoint of the two corners, x along the corners, unit = median eye width):
  closed           eye opening (eyelid distance) below half the video's median opening: left alone, counted apart
  ratio check      scale-free per point: pupil_left = (midpoint of top/bottom - left) along, divided by the pupil
                   height; pupil_right likewise; pupil_top / pupil_bottom = across offset from the midpoint of left /
                   right divided by the pupil width (each ~ half the aspect ratio, stable across dilation); centre = the
                   video's own median (all open frames, label-free); scale = the median over the 19 videos of each
                   video's MAD (--bands pooled, default; a video whose error mode holds 10 % of its frames would widen
                   its own MAD) or the video's own MAD (--bands own); a point violates if |z| > k MADs (k = 5)
  inside check     a pupil point must lie between the corners (|along| <= 0.6) and between the eyelids (+- 0.1 W)
  ordering check   pupil_left is nasal of pupil_right and pupil_top above pupil_bottom (orientation from the medians)
A frame with no violation is left alone ("ok"). Otherwise each violating point is replaced by its own peak-2 position
(cell * stride + the per-point median offset between the regressed x, y and peak-1 cell * stride) and the checks are
repeated; the smallest replacement (one point, then two) that makes the frame pass is accepted ("repaired"); if none
does, the frame is "drop". Inputs: the plateau model's whole-video predictions, results/heatmap_signals/
videoNN_shuffleSSSS_peaks.csv (heatmap_signals.py --peaks) and the stats csv for the peak ratios, the cached
fitting deviations (orders (2,0,0) and (3,0,1)) only to report the overlap with the time-series flags.

Judged on the video's 70 truth frames (last 4 minutes, human labels): frame RMSE and the four pupil-point errors
before / after, > 50 / > 20 px counts before / after, false repairs (frames < 20 px that were changed and got worse),
false drops (frames < 20 px dropped). Outputs in results/second_peak_repair/videoNN/: README numbers come from
truth_frames.csv, summary.csv, whole_video.csv (geometry_bands.csv one level up: per-video medians and MADs); frames_all.csv.gz (every frame, local only) and repaired_sheet.jpg
(local only, red = model, blue = repaired, green ring = human). Nothing existing is modified."""
import argparse
import glob
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import pupil_trace as pt  # noqa: E402
from scale_step import HERE  # noqa: E402

RES = HERE / "results"
PL = pd.read_csv(RES / "labels_to_plateau.csv")
BPS = ["pupil_top", "pupil_bottom", "pupil_left", "pupil_right", "eyelid_top", "eyelid_bottom", "eye_nasal_corner", "eye_temporal_corner"]
P4 = BPS[:4]
ALONG_MAX, LID_MARGIN = 0.6, 0.1


def load(video):
    rr = PL[PL.video == video].iloc[0]
    step = max(1, int(rr.labels_at_plateau) // 20)
    h5 = sorted(glob.glob(str(HERE / rr.unit / "training-data" / f"predictions_step{step:02d}" / "*snapshot*120*.h5")))[-1]
    d = pt.load(h5)
    pk = sorted(glob.glob(str(RES / "heatmap_signals" / f"video{video:02d}_shuffle*_peaks.csv")))
    if not pk:
        raise SystemExit(f"no peaks file for video {video}: run heatmap_signals.py --peaks first")
    P = pd.read_csv(pk[-1]).set_index("frame").sort_index()
    H = pd.read_csv(pk[-1].replace("_peaks.csv", ".csv")).set_index("frame").sort_index()
    assert len(d) == len(P) == len(H), (len(d), len(P), len(H))
    truth = pd.read_csv(RES / "error_detector" / "scores_lovo.csv")
    truth = truth[truth.plateau & (truth.video == video)].drop_duplicates("frame").set_index("frame").sort_index()
    cache = RES / "unreliable_frames_sweep" / "cache"
    fwd = {k: np.load(cache / f"video{video:02d}_fwd{k}.npy") for k in ("200", "301")}
    sig = pd.read_csv(RES / "unreliable_frames_sweep" / "signals_all_frames.csv")
    return rr, d, P, H, truth, fwd, sig


def peak2_pixels(d, P):
    """Pixel position of peak 2 per point: cell * stride + the median offset (regressed - peak-1 cell * stride)."""
    out, off = {}, {}
    for b in BPS:
        s = P[f"{b}_stride"].to_numpy(float)
        x, y = d[b]["x"].to_numpy(float), d[b]["y"].to_numpy(float)
        assert np.allclose(x, P[f"{b}_x"].to_numpy(float), atol=0.01) and np.allclose(y, P[f"{b}_y"].to_numpy(float), atol=0.01)
        ox, oy = np.nanmedian(x - P[f"{b}_x1c"].to_numpy(float) * s), np.nanmedian(y - P[f"{b}_y1c"].to_numpy(float) * s)
        out[b] = np.c_[P[f"{b}_x2c"].to_numpy(float) * s + ox, P[f"{b}_y2c"].to_numpy(float) * s + oy]
        off[b] = (ox, oy)
    return out, off


class Geometry:
    """Eye-frame coordinates and the per-video bands (label-free, from all open frames of the video)."""

    def __init__(self, xy, scale=None):
        N, T = xy["eye_nasal_corner"], xy["eye_temporal_corner"]
        self.C = (N + T) / 2
        u = T - N
        W = np.linalg.norm(u, axis=1)
        self.Wm = float(np.nanmedian(W))
        self.u = u / W[:, None]
        self.v = np.c_[-self.u[:, 1], self.u[:, 0]]
        E = {b: self.to_eye(xy[b]) for b in BPS}
        self.lid_lo = np.minimum(E["eyelid_top"][:, 1], E["eyelid_bottom"][:, 1]) - LID_MARGIN
        self.lid_hi = np.maximum(E["eyelid_top"][:, 1], E["eyelid_bottom"][:, 1]) + LID_MARGIN
        opening = np.abs(E["eyelid_bottom"][:, 1] - E["eyelid_top"][:, 1])
        self.closed = opening < 0.5 * np.nanmedian(opening)
        self.sign_lr = np.sign(np.nanmedian(E["pupil_right"][:, 0] - E["pupil_left"][:, 0]))
        self.sign_tb = np.sign(np.nanmedian(E["pupil_bottom"][:, 1] - E["pupil_top"][:, 1]))
        self.med, self.mad, self.own_mad = {}, {}, {}
        for b in P4:
            r = self.ratio(E, b)[~self.closed]
            self.med[b] = float(np.nanmedian(r))
            self.own_mad[b] = float(1.4826 * np.nanmedian(np.abs(r - self.med[b])))
            self.mad[b] = self.own_mad[b] if scale is None else float(scale[b])

    def to_eye(self, p, idx=None):
        C, u, v = (self.C, self.u, self.v) if idx is None else (self.C[idx], self.u[idx], self.v[idx])
        r = p - C
        return np.c_[(r * u).sum(-1), (r * v).sum(-1)] / self.Wm if p.ndim == 2 else np.r_[(r * u).sum(), (r * v).sum()] / self.Wm

    def ratio(self, E, b):
        """Scale-free position of one pupil point relative to the other three (works on (T,2) arrays and (2,) rows)."""
        h = np.abs(E["pupil_bottom"][..., 1] - E["pupil_top"][..., 1])
        w = np.abs(E["pupil_right"][..., 0] - E["pupil_left"][..., 0])
        mid_tb = (E["pupil_top"][..., 0] + E["pupil_bottom"][..., 0]) / 2
        mid_lr = (E["pupil_left"][..., 1] + E["pupil_right"][..., 1]) / 2
        if b == "pupil_left":
            return self.sign_lr * (mid_tb - E[b][..., 0]) / np.maximum(h, 1e-6)
        if b == "pupil_right":
            return self.sign_lr * (E[b][..., 0] - mid_tb) / np.maximum(h, 1e-6)
        if b == "pupil_top":
            return self.sign_tb * (mid_lr - E[b][..., 1]) / np.maximum(w, 1e-6)
        return self.sign_tb * (E[b][..., 1] - mid_lr) / np.maximum(w, 1e-6)

    def violations(self, E, i, k):
        """Set of violating pupil points of frame i given eye-frame coordinates E (dict point -> (2,) arrays)."""
        bad = set()
        for b in P4:
            z = abs(self.ratio(E, b) - self.med[b]) / max(self.mad[b], 1e-6)
            if not np.isfinite(z) or z > k:
                bad.add(b)
            if abs(E[b][0]) > ALONG_MAX or not (self.lid_lo[i] <= E[b][1] <= self.lid_hi[i]):
                bad.add(b)
        if self.sign_lr * (E["pupil_right"][0] - E["pupil_left"][0]) <= 0:
            bad |= {"pupil_left", "pupil_right"}
        if self.sign_tb * (E["pupil_bottom"][1] - E["pupil_top"][1]) <= 0:
            bad |= {"pupil_top", "pupil_bottom"}
        return bad


def predictions(video):
    rr = PL[PL.video == video].iloc[0]
    step = max(1, int(rr.labels_at_plateau) // 20)
    h5 = sorted(glob.glob(str(HERE / rr.unit / "training-data" / f"predictions_step{step:02d}" / "*snapshot*120*.h5")))
    return rr, (pt.load(h5[-1]) if h5 else None)


def pooled_bands(out_dir):
    """Per-video median and MAD of the four ratios from every video's plateau predictions; scale = median of the MADs."""
    f = out_dir / "geometry_bands.csv"
    if f.exists():
        B = pd.read_csv(f)
    else:
        rows = []
        for v in PL.video:
            rr, d = predictions(int(v))
            if d is None:
                print(f"  bands: video {v} has no plateau prediction file, skipped")
                continue
            G = Geometry({b: np.c_[d[b]["x"].to_numpy(float), d[b]["y"].to_numpy(float)] for b in BPS})
            rows.append({"video": int(v), "frames": len(d), "closed": int(G.closed.sum()), **{f"{b}_median": G.med[b] for b in P4}, **{f"{b}_mad": G.own_mad[b] for b in P4}})
        B = pd.DataFrame(rows)
        B.round(4).to_csv(f, index=False)
    return B, {b: float(B[f"{b}_mad"].median()) for b in P4}


def repair_frame(G, xy, p2, i, k):
    if G.closed[i]:
        return "closed", (), xy_frame(xy, i)
    E = {b: G.to_eye(xy[b][i], i) for b in BPS}
    bad = G.violations(E, i, k)
    if not bad:
        return "ok", (), xy_frame(xy, i)
    cands = [(b,) for b in sorted(bad)]
    if len(bad) >= 2:
        cands += [(a, b) for a in sorted(bad) for b in sorted(bad) if a < b]
    for rep in cands:
        E2 = dict(E)
        for b in rep:
            E2[b] = G.to_eye(p2[b][i], i)
        if not G.violations(E2, i, k):
            new = xy_frame(xy, i)
            for b in rep:
                new[b] = p2[b][i]
            return "repaired", rep, new
    return "drop", tuple(sorted(bad)), xy_frame(xy, i)


def xy_frame(xy, i):
    return {b: xy[b][i].copy() for b in BPS}


def frame_rmse(new, hrow):
    e = []
    for b in BPS:
        hx, hy = hrow[f"human_{b}_x"], hrow[f"human_{b}_y"]
        if np.isfinite(hx) and np.isfinite(hy):
            e.append(np.hypot(new[b][0] - hx, new[b][1] - hy))
    return float(np.sqrt(np.mean(np.square(e)))) if e else np.nan


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", type=int, required=True)
    ap.add_argument("--k-mad", type=float, default=5.0)
    ap.add_argument("--bands", choices=["pooled", "own"], default="pooled")
    ap.add_argument("--sheet", action="store_true", help="also write the contact sheet of the repaired / dropped truth frames (local jpg)")
    ap.add_argument("--tag", default="", help="suffix of the output folder (videoNN_TAG) for parameter variants; default folder videoNN")
    a = ap.parse_args()
    out = RES / "second_peak_repair" / (f"video{a.video:02d}" + (f"_{a.tag}" if a.tag else ""))
    out.mkdir(parents=True, exist_ok=True)
    rr, d, P, H, truth, fwd, sig = load(a.video)
    xy = {b: np.c_[d[b]["x"].to_numpy(float), d[b]["y"].to_numpy(float)] for b in BPS}
    p2, off = peak2_pixels(d, P)
    scale = None
    if a.bands == "pooled":
        B, scale = pooled_bands(out.parent)
        print("pooled scale (median MAD over videos):", {b: round(v, 4) for b, v in scale.items()})
    G = Geometry(xy, scale)
    n = len(d)
    print(f"video {a.video}: {n} frames, eye width {G.Wm:.1f} px, peak-2 offsets " + ", ".join(f"{b} ({o[0]:.1f}, {o[1]:.1f})" for b, o in off.items()))
    for b in P4:
        print(f"  band {b}: median {G.med[b]:.3f} own MAD {G.own_mad[b]:.3f} used MAD {G.mad[b]:.3f} -> k = {a.k_mad:g}: [{G.med[b] - a.k_mad * G.mad[b]:.3f}, {G.med[b] + a.k_mad * G.mad[b]:.3f}]")
    print(f"  closed frames (opening < half the median): {int(G.closed.sum())}")

    status, reps, new_xy = [], [], {b: xy[b].copy() for b in BPS}
    for i in range(n):
        st, rep, new = repair_frame(G, xy, p2, i, a.k_mad)
        status.append(st); reps.append("+".join(rep))
        for b in rep:
            new_xy[b][i] = new[b]
    status = np.array(status)
    F = pd.DataFrame({"frame": np.arange(n), "status": status, "points": reps})
    for b in P4:
        F[f"{b}_ratio"] = H[f"{b}_ratio"].to_numpy(float)
        F[f"{b}_conf"] = d[b]["likelihood"].to_numpy(float)
        F[f"{b}_moved_px"] = np.hypot(*(new_xy[b] - xy[b]).T)

    # time-series flags, only to report the overlap (thresholds = the pooled 95 % quantiles of the 1330 truth frames)
    thr200 = float(np.nanquantile(sig.fit_o200_max, 0.95))
    thr301 = float(np.nanquantile(sig.fit_pupil, 0.95))
    F["fit_o200_max"], F["fit_pupil"] = np.nanmax(fwd["200"], axis=1), np.nanmean(fwd["301"][:, :4], axis=1)
    F["flag_o200"], F["flag_pupil"] = F.fit_o200_max > thr200, F.fit_pupil > thr301
    F.to_csv(out / "frames_all.csv.gz", index=False, float_format="%.3f")

    W = pd.DataFrame([{"video": a.video, "frames": n, "k_mad": a.k_mad, "bands": a.bands,
                       "ok": int((status == "ok").sum()), "closed": int((status == "closed").sum()), "repaired": int((status == "repaired").sum()), "drop": int((status == "drop").sum()),
                       "repaired_pct": round(100 * (status == "repaired").mean(), 2), "drop_pct": round(100 * (status == "drop").mean(), 2),
                       "flag_o200_pct": round(100 * F.flag_o200.mean(), 2), "flag_pupil_pct": round(100 * F.flag_pupil.mean(), 2),
                       "repaired_and_flag_o200": int(((status == "repaired") & F.flag_o200).sum()),
                       "drop_and_flag_o200": int(((status == "drop") & F.flag_o200).sum()),
                       "flag_o200_and_ok": int(((status == "ok") & F.flag_o200).sum()),
                       "thr_o200_px": round(thr200, 1), "thr_pupil_px": round(thr301, 1)}])
    for b in P4:
        W[f"repaired_{b}"] = int((F.points.str.contains(b).to_numpy() & (status == "repaired")).sum())
    W.to_csv(out / "whole_video.csv", index=False)
    print(W.T.to_string())

    # truth frames
    rows = []
    for f, hr in truth.iterrows():
        before = {b: xy[b][f] for b in BPS}
        after = {b: new_xy[b][f] for b in BPS}
        r = {"frame": f, "set": hr.set, "status": status[f], "points": reps[f], "rmse_before": frame_rmse(before, hr), "rmse_after": frame_rmse(after, hr),
             "rmse_file": hr.frame_rmse_px, "flag_o200": bool(F.flag_o200[f]), "flag_pupil": bool(F.flag_pupil[f])}
        for b in P4:
            hx, hy = hr[f"human_{b}_x"], hr[f"human_{b}_y"]
            r[f"{b}_before"] = np.hypot(before[b][0] - hx, before[b][1] - hy)
            r[f"{b}_after"] = np.hypot(after[b][0] - hx, after[b][1] - hy)
            r[f"{b}_peak2_err"] = np.hypot(p2[b][f][0] - hx, p2[b][f][1] - hy)
            r[f"{b}_ratio"] = F[f"{b}_ratio"][f]
            r[f"{b}_z"] = abs(G.ratio({c: G.to_eye(xy[c][f], f) for c in BPS}, b) - G.med[b]) / max(G.mad[b], 1e-6)
        rows.append(r)
    T = pd.DataFrame(rows)
    assert np.allclose(T.rmse_before, T.rmse_file, atol=0.05), "frame RMSE formula differs from scores_lovo.csv"
    T.round(2).to_csv(out / "truth_frames.csv", index=False)
    kept = T.status != "drop"
    S = {"video": a.video, "k_mad": a.k_mad, "bands": a.bands, "truth_frames": len(T), "gt50_before": int((T.rmse_before > 50).sum()), "gt20_before": int((T.rmse_before > 20).sum()),
         "ok": int((T.status == "ok").sum()), "closed": int((T.status == "closed").sum()), "repaired": int((T.status == "repaired").sum()), "drop": int((T.status == "drop").sum()),
         "gt50_repaired_frames": int(((T.rmse_before > 50) & (T.status == "repaired")).sum()),
         "gt50_repaired_to_lt50": int(((T.rmse_before > 50) & (T.status == "repaired") & (T.rmse_after <= 50)).sum()),
         "gt50_repaired_to_lt20": int(((T.rmse_before > 50) & (T.status == "repaired") & (T.rmse_after <= 20)).sum()),
         "gt50_dropped": int(((T.rmse_before > 50) & (T.status == "drop")).sum()),
         "gt50_left_untouched": int(((T.rmse_before > 50) & T.status.isin(["ok", "closed"])).sum()),
         "gt50_after_among_kept": int(((T.rmse_after > 50) & kept).sum()), "gt20_after_among_kept": int(((T.rmse_after > 20) & kept).sum()),
         "false_repairs_lt20_worse5px": int(((T.rmse_before <= 20) & (T.status == "repaired") & (T.rmse_after > T.rmse_before + 5)).sum()),
         "repairs_lt20": int(((T.rmse_before <= 20) & (T.status == "repaired")).sum()),
         "false_drops_lt20": int(((T.rmse_before <= 20) & (T.status == "drop")).sum()),
         "drops_20to50": int(((T.rmse_before > 20) & (T.rmse_before <= 50) & (T.status == "drop")).sum()),
         "median_rmse_before": round(float(T.rmse_before.median()), 2), "median_rmse_after_kept": round(float(T.rmse_after[kept].median()), 2),
         "repaired_median_before": round(float(T.rmse_before[T.status == "repaired"].median()), 2) if (T.status == "repaired").any() else np.nan,
         "repaired_median_after": round(float(T.rmse_after[T.status == "repaired"].median()), 2) if (T.status == "repaired").any() else np.nan}
    pd.DataFrame([S]).to_csv(out / "summary.csv", index=False)
    print(pd.Series(S).to_string())
    pd.set_option("display.width", 250)
    show = ["frame", "set", "status", "points", "rmse_before", "rmse_after", "flag_o200"] + [f"{b}_before" for b in P4] + [f"{b}_after" for b in P4]
    print(T[~T.status.isin(["ok", "closed"]) | (T.rmse_before > 50)].sort_values("rmse_before", ascending=False)[show].round(1).to_string(index=False))

    if a.sheet:
        sheet(rr, G, xy, new_xy, truth, T, out)


def sheet(rr, G, xy, new_xy, truth, T, out):
    import cv2
    from PIL import Image, ImageDraw, ImageFont
    TW, TH, COLS = 300, 225, 5
    try:
        FONT = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 13)
    except OSError:
        FONT = ImageFont.load_default()
    sel = T[~T.status.isin(["ok", "closed"]) | (T.rmse_before > 50)].sort_values("rmse_before", ascending=False)
    rows_n = int(np.ceil(len(sel) / COLS))
    S = Image.new("RGB", (COLS * TW, 40 + rows_n * TH), (255, 255, 255))
    ImageDraw.Draw(S).text((6, 10), f"video {rr.video}: truth frames changed by the second-peak repair or > 50 px. red = model, blue = repaired, green ring = human. "
                           f"caption: frame, status (points), RMSE before -> after", fill=(0, 0, 0), font=FONT)
    cap = cv2.VideoCapture(str(HERE / rr.unit / f"{rr.unit}.mp4"))
    W = np.linalg.norm(xy["eye_temporal_corner"] - xy["eye_nasal_corner"], axis=1)
    for j, (_, r) in enumerate(sel.iterrows()):
        f = int(r.frame)
        cap.set(cv2.CAP_PROP_POS_FRAMES, f)
        ok, img = cap.read()
        assert ok, f
        img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        cx, cy = G.C[f]
        w = max(1.7 * W[f], 240); h = w * TH / TW
        box = (int(cx - w / 2), int(cy - h / 2), int(cx + w / 2), int(cy + h / 2))
        crop = img.crop(box).resize((TW, TH)); sc = TW / (box[2] - box[0])
        dr = ImageDraw.Draw(crop)
        for b in BPS:
            px, py = (xy[b][f] - box[:2]) * sc
            dr.ellipse([px - 4, py - 4, px + 4, py + 4], fill=(255, 60, 60))
            nx, ny = (new_xy[b][f] - box[:2]) * sc
            if np.hypot(nx - px, ny - py) > 0.5:
                dr.line([px, py, nx, ny], fill=(80, 120, 255), width=2)
                dr.ellipse([nx - 4, ny - 4, nx + 4, ny + 4], fill=(80, 120, 255))
            hx, hy = (truth.loc[f, f"human_{b}_x"] - box[0]) * sc, (truth.loc[f, f"human_{b}_y"] - box[1]) * sc
            if np.isfinite(hx) and np.isfinite(hy):
                dr.ellipse([hx - 5, hy - 5, hx + 5, hy + 5], outline=(60, 255, 60), width=3)
        dr.rectangle([0, 0, TW, 18], fill=(0, 0, 0))
        pts = r.points.replace("pupil_", "") if r.points else ""
        dr.text((4, 1), f"{f} {r.status} {pts} {r.rmse_before:.0f} -> {r.rmse_after:.0f} px", fill=(255, 255, 255), font=FONT)
        S.paste(crop, ((j % COLS) * TW, 40 + (j // COLS) * TH))
    S.save(out / "repaired_sheet.jpg", quality=88)
    print("sheet", out / "repaired_sheet.jpg", len(sel), "tiles")


if __name__ == "__main__":
    main()
