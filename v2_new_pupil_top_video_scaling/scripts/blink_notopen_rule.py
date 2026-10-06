"""First-round "not open" rule (definition fixed by the user on 2026-10-06: not open = eye closed or eyelid covering part
of the pupil) applied to every finished video, plus its check on the 239 blind single-frame verdicts.

Rule (per frame, from the plateau model's whole-video prediction):
  eye opening  O = y(eyelid_bottom) - y(eyelid_top);  eye width W = median over the video of the corner-to-corner distance
  eyelid points plausible ("trusted") = O >= 0 and |x(eyelid_top) - x(eyelid_bottom)| <= 0.3 W and both confidences >= 0.3
  trusted frame:    not open  <=>  O / W < T_OPEN            (T_OPEN = 0.347, chosen on the 211 trusted judged frames,
                                                              maximizing recall - false-alarm rate; leave-one-video-out
                                                              gives the same decision on 88% of the frames)
  implausible frame: not open <=> mean confidence of the four pupil points < 0.527 (threshold of blink_and_area.py part B)
  events = runs of not-open frames of >= 3 frames, runs closer than 5 frames merged.
Unlike rule D of blink_final_rule.py the threshold is ABSOLUTE (not relative to the video's own median): the single-frame
check showed that in some recordings the eye is partly covered most of the time, which a relative threshold cannot see.

Outputs (results/blink_eyelid_distance/): blink_notopen_rule_check.csv (rule against the verdicts, overall and per stratum),
blink_notopen_rule_all_videos.csv (per video: not-open frames %, events, per minute, median duration, median O/W),
blink_notopen_rule_events_all_videos.csv, blink_notopen_rule_summary.png. Nothing else is modified.
"""
import glob, sys
from pathlib import Path
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent))
import pupil_trace as pt  # noqa: E402
from scale_step import HERE  # noqa: E402

FPS, T_OPEN, PUPIL_THR, DX_FRAC, CONF_MIN, GAP, MIN_LEN = 60.0, 0.347, 0.527, 0.3, 0.3, 5, 3
OUT = HERE / "results" / "blink_eyelid_distance"
PL = pd.read_csv(HERE / "results" / "labels_to_plateau.csv"); PL = PL[PL.status == "final"]
P4 = ["pupil_top", "pupil_bottom", "pupil_left", "pupil_right"]

def runs(mask, gap=0):
    i = np.flatnonzero(mask)
    if not len(i): return []
    cut = np.flatnonzero(np.diff(i) > gap + 1 if gap else np.diff(i) > 1)
    return list(zip(i[np.r_[0, cut + 1]], i[np.r_[cut, len(i) - 1]]))

def frame_states(d):
    g = lambda b, c: d[b][c].to_numpy(float)
    O = g("eyelid_bottom", "y") - g("eyelid_top", "y")
    W = float(np.nanmedian(np.hypot(g("eye_temporal_corner", "x") - g("eye_nasal_corner", "x"), g("eye_temporal_corner", "y") - g("eye_nasal_corner", "y"))))
    dx = np.abs(g("eyelid_top", "x") - g("eyelid_bottom", "x")) / W
    conf = np.minimum(g("eyelid_top", "likelihood"), g("eyelid_bottom", "likelihood"))
    trusted = (O >= 0) & (dx <= DX_FRAC) & (conf >= CONF_MIN) & np.isfinite(O)
    pmean = np.mean([g(b, "likelihood") for b in P4], 0)
    ow = O / W
    notopen = np.where(trusted, np.nan_to_num(ow, nan=np.inf) < T_OPEN, pmean < PUPIL_THR)
    return notopen, ow, trusted, pmean

Q = pd.read_csv(OUT / "blink_notopen_frames_all_videos.csv")
J = Q[Q.human_verdict.notna() & (Q.human_verdict != "") & ~Q.human_verdict.astype(str).str.startswith("cannot")].copy()
J["notopen"] = J.human_verdict.str.startswith("not open")
rows, ev_rows, pred = [], [], {}
for rr in PL.itertuples():
    v = int(rr.video); step = max(1, int(rr.labels_at_plateau) // 20)
    h5 = sorted(glob.glob(str(HERE / rr.unit / "training-data" / f"predictions_step{step:02d}" / "*snapshot*120*.h5")))
    d = pt.load(h5[-1]); no, ow, trusted, pmean = frame_states(d)
    for q in J[J.video == v].itertuples(): pred[q.item] = bool(no[q.frame])
    ev = [(a, b) for a, b in runs(no, GAP) if b - a + 1 >= MIN_LEN]
    dur = [(b - a + 1) / FPS * 1000 for a, b in ev]
    ev_rows += [dict(video=v, unit=rr.unit, start=int(a), end=int(b), start_s=round(a / FPS, 2), duration_ms=round((b - a + 1) / FPS * 1000), min_O_over_W=round(float(np.nanmin(ow[a:b + 1])), 3)) for a, b in ev]
    rows.append(dict(video=v, mouse=rr.mouse, date=rr.date, model_step=step, minutes=round(len(d) / FPS / 60, 2), median_O_over_W=round(float(np.nanmedian(ow)), 3),
                     notopen_frames_pct=round(100 * float(no.mean()), 1), implausible_frames_pct=round(100 * float((~trusted).mean()), 2),
                     events=len(ev), events_per_min=round(len(ev) / (len(d) / FPS / 60), 2), median_event_ms=round(float(np.median(dur))) if dur else np.nan,
                     longest_event_s=round(max(dur) / 1000, 1) if dur else np.nan))
S = pd.DataFrame(rows); S.to_csv(OUT / "blink_notopen_rule_all_videos.csv", index=False)
pd.DataFrame(ev_rows).to_csv(OUT / "blink_notopen_rule_events_all_videos.csv", index=False)
J["rule"] = J.item.map(pred)
def check(g, name):
    tp = int((g.rule & g.notopen).sum()); fl = int(g.rule.sum()); pos = int(g.notopen.sum())
    return dict(subset=name, frames=len(g), judged_not_open=pos, rule_not_open=fl, correct=int((g.rule == g.notopen).sum()), accuracy_pct=round(100 * (g.rule == g.notopen).mean(), 1),
                precision_pct=round(100 * tp / max(1, fl), 1), recall_pct=round(100 * tp / max(1, pos), 1))
C = pd.DataFrame([check(J, "all judged frames")] + [check(J[J.stratum == s], f"stratum {s}") for s in ["r<0.70", "0.70-0.85", "0.85-1.00", "r>=1.00", "eyelid points implausible"]]
                 + [check(J[J.video == v], f"video {v}") for v in sorted(J.video.unique())])
C.to_csv(OUT / "blink_notopen_rule_check.csv", index=False)
pd.set_option("display.width", 250); print(C.to_string(index=False)); print(); print(S.to_string(index=False))

# figure: per-video share of not-open frames (rule) with the judged share of the sample, and events per minute
fig, ax = plt.subplots(1, 2, figsize=(15, 5.2), constrained_layout=True)
x = np.arange(len(S)); share_j = J.groupby("video").notopen.mean().reindex(S.video).to_numpy() * 100
ax[0].bar(x, S.notopen_frames_pct, color="#D1495B", alpha=0.85, label="rule: share of frames not open (whole video)")
ax[0].plot(x, share_j, "o", color="k", ms=6, label="labeler: share of the 18 sampled frames judged not open (sample is stratified, not proportional)")
for xi, m in zip(x, S.median_O_over_W): ax[0].text(xi, S.notopen_frames_pct.iloc[xi] + 1.5, f"{m:.2f}", ha="center", fontsize=8, color="0.3")
ax[0].set_xticks(x); ax[0].set_xticklabels([f"v{v}\n{m[:5]}" for v, m in zip(S.video, S.mouse)], fontsize=8); ax[0].set_ylabel("% of frames"); ax[0].set_ylim(0, 105)
ax[0].set_title(f"A. Not-open frames per video; grey number = median eye opening / eye width (threshold {T_OPEN})", fontsize=10.5); ax[0].legend(fontsize=8, loc="upper left"); ax[0].grid(axis="y", alpha=0.3)
ax[1].bar(x, S.events_per_min, color="#2a9d8f", alpha=0.85)
for xi, (e, d_) in enumerate(zip(S.events, S.median_event_ms)): ax[1].text(xi, S.events_per_min.iloc[xi] + 0.1, f"{e}\n{d_:.0f} ms" if d_ == d_ else f"{e}", ha="center", fontsize=7.5)
ax[1].set_xticks(x); ax[1].set_xticklabels([f"v{v}" for v in S.video], fontsize=8); ax[1].set_ylabel("not-open events per minute"); ax[1].grid(axis="y", alpha=0.3)
ax[1].set_title("B. Not-open events (runs >= 3 frames, gaps < 5 frames merged): count and median duration", fontsize=10.5)
fig.suptitle("First-round 'not open' rule (eye opening / eye width < 0.347; implausible eyelid points -> pupil confidence < 0.527) on the 14 finished videos", fontsize=12)
fig.savefig(OUT / "blink_notopen_rule_summary.png", dpi=125)
