"""One eye-opening threshold per mouse, on the eyelid distance in pixels (Kaiwen, Slack 2026-10-11 22:41: use the upper and
lower eyelid directly, not divided by the eye width; the mouse is placed at the same position every day; one threshold per
mouse). Companion of blink_per_mouse_threshold.py (which uses r = opening / eye width); same inputs, same frames, same
truth, only the quantity changes. Second look, 2026-10-11. Nothing existing is modified.

Quantity:
  eye opening  O = y(eyelid_bottom) - y(eyelid_top) in px, per frame, from the plateau model's whole-video prediction of
  each finished video (same prediction files as the first look)
  eyelid points plausible ("trusted") = O >= 0 and |x(eyelid_top) - x(eyelid_bottom)| <= 0.3 W and both confidences >= 0.3
  (W = median corner-to-corner distance of the video, used only for this plausibility test and for the crops)
  small opening at threshold T:  trusted frame -> O < T;  implausible frame -> mean confidence of the 4 pupil points < 0.527
Threshold per mouse: on the trusted blind single-frame verdicts of that mouse (blink_notopen_frames_all_videos.csv, videos
0-13), the T in px that maximizes recall - false-alarm rate (the criterion of the first round); middle of the best range;
leave-one-video-out range as a stability check. Candidates shown next to it: 120, 140, 160, 180 px.
Position check (is the mouse at the same place every day): median x, y of the eye centre (midpoint of the two corners) and
the eye width per video.

Outputs (results/blink_eyelid_distance/per_mouse_threshold_px/):
  eye_position_per_video.csv / .png     eye centre, eye width and opening (px) per video
  threshold_per_mouse.csv               chosen T (px) per mouse, accuracy at each candidate, leave-one-video-out range
  threshold_scan_per_mouse.png          accuracy and recall - false alarms against T, per mouse
  small_opening_share_per_video.csv     share of frames with a small opening per video at each candidate T
  opening_distribution_per_video.png    distribution of O per video with the candidate thresholds
  opening_timeseries_<mouse>.png        O over time (10-s median) per video: the continuous variable
  examples_<mouse>.png / .csv           per candidate T one row of frames whose O is closest to T (different videos)
"""
import glob, sys
from pathlib import Path
import cv2
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent))
import pupil_trace as pt  # noqa: E402
from scale_step import HERE  # noqa: E402

FPS, PUPIL_THR, DX_FRAC, CONF_MIN = 60.0, 0.527, 0.3, 0.3
CAND = [120, 140, 160, 180]                       # px
GRID = np.arange(60, 301, 1.0)                     # px
ROOT = HERE / "results" / "blink_eyelid_distance"   # folder layout since 2026-10-12: labels/{segments,frames} = the labeler's verdicts; results/{every_video,one_threshold,every_mouse}
LAB_SEG, LAB_FR = ROOT / "labels" / "segments", ROOT / "labels" / "frames"
RES_EV, RES_ONE, RES_MOUSE = ROOT / "results" / "every_video", ROOT / "results" / "one_threshold", ROOT / "results" / "every_mouse"
SRC = LAB_FR; OUT = RES_MOUSE / "px"; OUT.mkdir(parents=True, exist_ok=True)
PL = pd.read_csv(HERE / "results" / "labels_to_plateau.csv"); PL = PL[PL.status == "final"]
P4 = ["pupil_top", "pupil_bottom", "pupil_left", "pupil_right"]
KEY = {"mouse A": "mouseA", "Pluto": "Pluto", "Terra": "Terra"}
COLM = {"mouse A": "#6c757d", "Pluto": "#2a6f97", "Terra": "#c9622b"}

def frame_values(d):
    g = lambda b, c: d[b][c].to_numpy(float)
    O = g("eyelid_bottom", "y") - g("eyelid_top", "y")
    wf = np.hypot(g("eye_temporal_corner", "x") - g("eye_nasal_corner", "x"), g("eye_temporal_corner", "y") - g("eye_nasal_corner", "y"))
    W = float(np.nanmedian(wf))
    cx = (g("eye_temporal_corner", "x") + g("eye_nasal_corner", "x")) / 2; cy = (g("eye_temporal_corner", "y") + g("eye_nasal_corner", "y")) / 2
    dx = np.abs(g("eyelid_top", "x") - g("eyelid_bottom", "x")) / W
    conf = np.minimum(g("eyelid_top", "likelihood"), g("eyelid_bottom", "likelihood"))
    trusted = (O >= 0) & (dx <= DX_FRAC) & (conf >= CONF_MIN) & np.isfinite(O)
    pmean = np.mean([g(b, "likelihood") for b in P4], 0)
    return O, wf, W, cx, cy, trusted, pmean

def small(O, trusted, pmean, T):
    return np.where(trusted, np.nan_to_num(O, nan=np.inf) < T, pmean < PUPIL_THR)

# ---- per-video values ---------------------------------------------------------------------------------------------------
V = {}
for rr in PL.itertuples():
    v = int(rr.video); step = max(1, int(rr.labels_at_plateau) // 20)
    h5 = sorted(glob.glob(str(HERE / rr.unit / "training-data" / f"predictions_step{step:02d}" / "*snapshot*120*.h5")))
    d = pt.load(h5[-1]); O, wf, W, cx, cy, trusted, pmean = frame_values(d)
    V[v] = dict(unit=rr.unit, mouse=rr.mouse, date=rr.date, d=d, O=O, wf=wf, W=W, cx=cx, cy=cy, trusted=trusted, pmean=pmean)
MICE = [m for m in ["mouse A", "Pluto", "Terra"] if any(x["mouse"] == m for x in V.values())]
lab = lambda v: f"v{v}\n{V[v]['date'][5:] if V[v]['date'][:2] == '20' else ''}"

# ---- 1. position check --------------------------------------------------------------------------------------------------
rows = []
for v, x in V.items():
    q = lambda a, p: float(np.nanpercentile(a[x["trusted"]], p))
    rows.append(dict(video=v, mouse=x["mouse"], date=x["date"], frames=len(x["O"]),
                     eye_centre_x_median_px=round(float(np.nanmedian(x["cx"])), 1), eye_centre_y_median_px=round(float(np.nanmedian(x["cy"])), 1),
                     eye_centre_x_p05_px=round(float(np.nanpercentile(x["cx"], 5)), 1), eye_centre_x_p95_px=round(float(np.nanpercentile(x["cx"], 95)), 1),
                     eye_centre_y_p05_px=round(float(np.nanpercentile(x["cy"], 5)), 1), eye_centre_y_p95_px=round(float(np.nanpercentile(x["cy"], 95)), 1),
                     eye_width_median_px=round(x["W"], 1), opening_p05_px=round(q(x["O"], 5), 1), opening_p25_px=round(q(x["O"], 25), 1),
                     opening_median_px=round(q(x["O"], 50), 1), opening_p75_px=round(q(x["O"], 75), 1), opening_p95_px=round(q(x["O"], 95), 1),
                     implausible_frames_pct=round(100 * float((~x["trusted"]).mean()), 2)))
S = pd.DataFrame(rows); S.to_csv(OUT / "eye_position_per_video.csv", index=False)
for m in MICE:
    g = S[S.mouse == m]
    print(f"{m}: eye centre x {g.eye_centre_x_median_px.min():.0f}-{g.eye_centre_x_median_px.max():.0f}, y {g.eye_centre_y_median_px.min():.0f}-{g.eye_centre_y_median_px.max():.0f} px; "
          f"eye width {g.eye_width_median_px.min():.0f}-{g.eye_width_median_px.max():.0f} px; median opening {g.opening_median_px.min():.0f}-{g.opening_median_px.max():.0f} px")
fig, ax = plt.subplots(1, 3, figsize=(17, 4.6), constrained_layout=True); x_ = np.arange(len(S)); cols = [COLM[m] for m in S.mouse]
for m in MICE:
    g = S[S.mouse == m]
    ax[0].errorbar(g.eye_centre_x_median_px, g.eye_centre_y_median_px, xerr=[g.eye_centre_x_median_px - g.eye_centre_x_p05_px, g.eye_centre_x_p95_px - g.eye_centre_x_median_px],
                   yerr=[g.eye_centre_y_median_px - g.eye_centre_y_p05_px, g.eye_centre_y_p95_px - g.eye_centre_y_median_px], fmt="o", color=COLM[m], ms=5, lw=1, capsize=2, label=m)
    for r_ in g.itertuples(): ax[0].annotate(f"v{r_.video}", (r_.eye_centre_x_median_px, r_.eye_centre_y_median_px), fontsize=7, xytext=(3, 3), textcoords="offset points")
ax[0].invert_yaxis(); ax[0].set_xlabel("eye centre x (px, image)"); ax[0].set_ylabel("eye centre y (px, image)"); ax[0].legend(fontsize=8)
ax[0].set_title("A. Where the eye is in the image, per video (median; bars = 5th-95th percentile)", fontsize=10); ax[0].grid(alpha=0.3)
ax[1].bar(x_, S.eye_width_median_px, color=cols, alpha=0.85); ax[1].set_ylabel("eye width (px)"); ax[1].set_title("B. Median eye width per video (px)", fontsize=10)
ax[2].bar(x_, S.opening_median_px, color=cols, alpha=0.85)
ax[2].errorbar(x_, S.opening_median_px, yerr=[S.opening_median_px - S.opening_p05_px, S.opening_p95_px - S.opening_median_px], fmt="none", ecolor="k", lw=1, capsize=2)
ax[2].set_ylabel("eye opening (px)"); ax[2].set_title("C. Eye opening per video: median, bars = 5th-95th percentile", fontsize=10)
for a in ax[1:]: a.set_xticks(x_); a.set_xticklabels([f"v{v}" for v in S.video], fontsize=8); a.grid(axis="y", alpha=0.3)
fig.suptitle("Position check: is the eye at the same place and size in the image on every day?", fontsize=12)
fig.savefig(OUT / "eye_position_per_video.png", dpi=125); plt.close(fig)

# ---- 2. threshold per mouse on the blind verdicts ----------------------------------------------------------------------
Q = pd.read_csv(SRC / "blink_notopen_frames_all_videos.csv")
J = Q[Q.human_verdict.notna() & ~Q.human_verdict.astype(str).str.startswith("cannot")].copy()
J["notopen"] = J.human_verdict.str.startswith("not open"); J["mouse"] = J.video.map(lambda v: V[v]["mouse"])
J["O_now"] = [float(V[v]["O"][f]) for v, f in zip(J.video, J.frame)]; J["trusted_now"] = [bool(V[v]["trusted"][f]) for v, f in zip(J.video, J.frame)]
JT = J[J.trusted_now]

def scan(g):
    o, y = g.O_now.to_numpy(), g.notopen.to_numpy(); pos, neg = max(1, y.sum()), max(1, (~y).sum())
    pred = o[None, :] < GRID[:, None]
    rec = (pred & y).sum(1) / pos; fa = (pred & ~y).sum(1) / neg; acc = (pred == y).mean(1)
    return rec, fa, acc
def best(g):
    rec, fa, _ = scan(g); j = rec - fa; top = GRID[j >= j.max() - 1e-9]
    return float(np.round((top.min() + top.max()) / 2, 1)), float(top.min()), float(top.max())

trow = []; TM = {}
fig, ax = plt.subplots(1, len(MICE) + 1, figsize=(5.2 * (len(MICE) + 1), 4.4), constrained_layout=True)
for a, (name, g) in zip(ax, [("all mice pooled", JT)] + [(m, JT[JT.mouse == m]) for m in MICE]):
    rec, fa, acc = scan(g); T, lo, hi = best(g)
    one_class = g.notopen.nunique() < 2
    loo = [best(g[g.video != v])[0] for v in sorted(g.video.unique())] if g.video.nunique() > 1 and not one_class else []
    at = lambda t: float(acc[np.argmin(np.abs(GRID - t))])
    row = dict(mouse=name, videos_with_verdicts=g.video.nunique(), trusted_judged_frames=len(g), judged_not_open=int(g.notopen.sum()),
               threshold_px=np.nan if one_class else T, best_range_low_px=np.nan if one_class else lo, best_range_high_px=np.nan if one_class else hi,
               leave_one_video_out_min_px=min(loo) if loo else np.nan, leave_one_video_out_max_px=max(loo) if loo else np.nan,
               accuracy_at_threshold_pct=np.nan if one_class else round(100 * at(T), 1))
    for c in CAND: row[f"accuracy_at_{c}px_pct"] = round(100 * at(c), 1)
    trow.append(row)
    if name in MICE and not one_class: TM[name] = T
    a.plot(GRID, 100 * acc, color="#2a6f97", label="accuracy (%)"); a.plot(GRID, 100 * (rec - fa), color="#c9622b", label="recall - false alarms (%)")
    for t in CAND: a.axvline(t, color="0.6", ls=":", lw=1)
    if not one_class: a.axvspan(lo, hi, color="#c9622b", alpha=0.15); a.axvline(T, color="#c9622b", lw=1.5)
    a.set_title(f"{name}: {len(g)} frames ({int(g.notopen.sum())} not open)" + ("" if one_class else f"; T = {T:.0f} px"), fontsize=10)
    a.set_xlabel("threshold T on the eye opening (px)"); a.set_ylim(0, 100); a.grid(alpha=0.3)
ax[0].legend(fontsize=8, loc="lower center")
T_POOLED = [r for r in trow if r["mouse"] == "all mice pooled"][0]["threshold_px"]
for m in MICE: TM.setdefault(m, T_POOLED)          # a mouse without judged not-open frames gets the pooled value
fig.suptitle(f"Threshold scan on the blind single-frame verdicts (trusted frames); dotted = {' / '.join(str(c) for c in CAND)} px, band = best range", fontsize=11)
fig.savefig(OUT / "threshold_scan_per_mouse.png", dpi=125); plt.close(fig)
TT = pd.DataFrame(trow); TT.to_csv(OUT / "threshold_per_mouse.csv", index=False)
pd.set_option("display.width", 260); print(); print(TT.to_string(index=False))

# ---- 3. share of frames with a small opening per video ------------------------------------------------------------------
rows = []
for v, x in V.items():
    row = dict(video=v, mouse=x["mouse"], date=x["date"], opening_median_px=round(float(np.nanmedian(x["O"][x["trusted"]])), 1), mouse_threshold_px=TM[x["mouse"]])
    for t in CAND: row[f"share_lt_{t}px_pct"] = round(100 * float(small(x["O"], x["trusted"], x["pmean"], t).mean()), 1)
    row["share_lt_mouse_threshold_pct"] = round(100 * float(small(x["O"], x["trusted"], x["pmean"], TM[x["mouse"]]).mean()), 1)
    rows.append(row)
SH = pd.DataFrame(rows); SH.to_csv(OUT / "small_opening_share_per_video.csv", index=False); print(); print(SH.to_string(index=False))

# ---- 4. distribution of O per video -------------------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(15, 5), constrained_layout=True)
data = [x["O"][x["trusted"] & np.isfinite(x["O"])] for x in V.values()]
vp = ax.violinplot(data, positions=np.arange(len(V)), widths=0.85, showextrema=False)
for b, x in zip(vp["bodies"], V.values()): b.set_facecolor(COLM[x["mouse"]]); b.set_alpha(0.7)
ax.plot(np.arange(len(V)), [np.median(a) for a in data], "k_", ms=14, mew=2, label="median")
for t in CAND: ax.axhline(t, color="k", ls=":", lw=1); ax.text(len(V) - 0.45, t + 2, f"{t} px", fontsize=8)
for m in MICE:
    idx = [i for i, x in enumerate(V.values()) if x["mouse"] == m]
    ax.plot([idx[0] - 0.45, idx[-1] + 0.45], [TM[m]] * 2, color=COLM[m], lw=2.5, label=f"{m}: threshold {TM[m]:.0f} px")
ax.set_xticks(np.arange(len(V))); ax.set_xticklabels([lab(v) for v in V], fontsize=8)
ax.set_ylim(0, 350); ax.set_ylabel("eye opening (px)"); ax.grid(axis="y", alpha=0.3); ax.legend(fontsize=8, ncol=4, loc="upper right")
ax.set_title(f"Distribution of the eye opening over all trusted frames of each video (plateau model); dotted = {' / '.join(str(c) for c in CAND)} px", fontsize=11)
fig.savefig(OUT / "opening_distribution_per_video.png", dpi=125); plt.close(fig)

# ---- 5. O over time: the continuous variable ----------------------------------------------------------------------------
for m in MICE:
    vs = [v for v, x in V.items() if x["mouse"] == m]
    fig, ax = plt.subplots(len(vs), 1, figsize=(15, 1.55 * len(vs) + 0.9), sharey=True, constrained_layout=True, squeeze=False)
    for a, v in zip(ax[:, 0], vs):
        x = V[v]; o = np.where(x["trusted"], x["O"], np.nan); n = int(10 * FPS); k = len(o) // n
        med = np.nanmedian(o[:k * n].reshape(k, n), 1); p10 = np.nanpercentile(o[:k * n].reshape(k, n), 10, 1); t = (np.arange(k) + 0.5) * 10 / 60
        a.fill_between(t, p10, med, color=COLM[m], alpha=0.25); a.plot(t, med, color=COLM[m], lw=1.2)
        a.axhline(TM[m], color="k", lw=1)
        a.set_ylim(0, 320); a.set_ylabel(lab(v), fontsize=8, rotation=0, labelpad=22, va="center"); a.grid(alpha=0.3)
    ax[-1, 0].set_xlabel("time (min)")
    fig.suptitle(f"{m}: eye opening (px) over time (line = median of each 10 s, band down to the 10th percentile); black = mouse threshold {TM[m]:.0f} px", fontsize=10.5)
    fig.savefig(OUT / f"opening_timeseries_{KEY[m]}.png", dpi=115); plt.close(fig)

# ---- 6. example frames closest to each threshold -----------------------------------------------------------------------
NCOL, TILE_W, TOL = 6, 420, 3.0
def tile(v, f):
    x = V[v]; cap = cv2.VideoCapture(str(HERE / x["unit"] / f"{x['unit']}.mp4")); cap.set(cv2.CAP_PROP_POS_FRAMES, int(f)); ok, im = cap.read(); cap.release()
    if not ok: return None
    d = x["d"]; p = lambda b: (float(d[b]["x"].iloc[f]), float(d[b]["y"].iloc[f]))
    (nx, ny), (tx, ty) = p("eye_nasal_corner"), p("eye_temporal_corner"); cx, cy = (nx + tx) / 2, (ny + ty) / 2; W = x["W"]
    x0, x1 = int(max(0, cx - 0.72 * W)), int(min(im.shape[1], cx + 0.72 * W)); y0, y1 = int(max(0, cy - 0.5 * W)), int(min(im.shape[0], cy + 0.5 * W))
    crop = im[y0:y1, x0:x1].copy()
    for b, c in [("eyelid_top", (60, 220, 255)), ("eyelid_bottom", (60, 220, 255)), ("eye_nasal_corner", (255, 200, 60)), ("eye_temporal_corner", (255, 200, 60))]:
        px, py = p(b); cv2.circle(crop, (int(px - x0), int(py - y0)), 5, c, 1, cv2.LINE_AA)
    crop = cv2.resize(crop, (TILE_W, int(crop.shape[0] * TILE_W / crop.shape[1])), interpolation=cv2.INTER_AREA)
    band = np.full((30, TILE_W, 3), 255, np.uint8)
    cv2.putText(band, f"video {v}  frame {f}  opening = {x['O'][f]:.0f} px", (6, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1, cv2.LINE_AA)
    return np.vstack([crop, band])

erows = []
for m in MICE:
    vs = [v for v, x in V.items() if x["mouse"] == m]
    ts = sorted([float(t) for t in CAND if abs(t - TM[m]) >= 5] + [float(TM[m])])   # a candidate within 5 px of the mouse threshold would repeat its row
    strips = []
    for t in ts:
        cand = []   # per video: the trusted, confident frames whose O is closest to t
        for v in vs:
            x = V[v]; conf = np.minimum(x["d"]["eyelid_top"]["likelihood"].to_numpy(float), x["d"]["eyelid_bottom"]["likelihood"].to_numpy(float))
            ok = x["trusted"] & (conf >= 0.6) & np.isfinite(x["O"]); dist = np.where(ok, np.abs(x["O"] - t), np.inf)
            order = np.argsort(dist)[:4000]; picks = []
            for f in order:
                if not np.isfinite(dist[f]) or dist[f] > TOL: break
                if all(abs(int(f) - q) >= 600 for q in picks): picks.append(int(f))     # at least 10 s apart within a video
                if len(picks) >= NCOL: break
            cand.append((v, picks))
        chosen = []; k = 0
        while len(chosen) < NCOL and any(len(p) > k for _, p in cand):      # round-robin over the videos
            for v, p in cand:
                if len(p) > k and len(chosen) < NCOL: chosen.append((v, p[k]))
            k += 1
        chosen.sort()
        tiles = [tile(v, f) for v, f in chosen]; tiles = [x_ for x_ in tiles if x_ is not None]
        erows += [dict(mouse=m, threshold_px=t, video=v, frame=f, time_s=round(f / FPS, 1), opening_px=round(float(V[v]["O"][f]), 1)) for v, f in chosen]
        h = max([x_.shape[0] for x_ in tiles] + [60])
        tiles = [cv2.copyMakeBorder(x_, 0, h - x_.shape[0], 0, 0, cv2.BORDER_CONSTANT, value=(255, 255, 255)) for x_ in tiles] + [np.full((h, TILE_W, 3), 255, np.uint8)] * (NCOL - len(tiles))
        head = np.full((40, TILE_W * NCOL, 3), 235, np.uint8)
        tag = "  (threshold chosen for this mouse)" if abs(t - TM[m]) < 0.5 else ""
        cv2.putText(head, f"opening = {t:.0f} px{tag}: frames whose opening is closest to it ({len(chosen)} found within {TOL:.0f} px)", (10, 27), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 0), 2, cv2.LINE_AA)
        strips += [head, np.hstack(tiles)]
    top = np.full((54, TILE_W * NCOL, 3), 255, np.uint8)
    cv2.putText(top, f"{m}: what the eye looks like at each candidate threshold (eye opening in px; yellow = eyelid points, blue = eye corners)", (10, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2, cv2.LINE_AA)
    cv2.imwrite(str(OUT / f"examples_{KEY[m]}.png"), np.vstack([top] + strips))
    pd.DataFrame([e for e in erows if e["mouse"] == m]).to_csv(OUT / f"examples_{KEY[m]}.csv", index=False)
print("\nwritten to", OUT)
