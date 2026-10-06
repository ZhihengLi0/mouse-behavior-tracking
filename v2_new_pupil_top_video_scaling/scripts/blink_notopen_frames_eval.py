"""Fit and check a frame-level "not open" rule on the blind single-frame verdicts (blink_notopen_frames_sample.py).

    python blink_notopen_frames_eval.py        # needs blink_notopen_frames_verdicts.csv in results/blink_eyelid_distance/

Definition (user, 2026-10-06): not open = eye closed or eyelid covering part of the pupil. Truth = the labeler's verdict on
each sampled frame ("cannot tell" excluded). Signals per frame from the plateau model's whole-video prediction, all
relative to the video's own median so that videos are comparable: r = eyelid distance / median; ph = pupil height /
median; pw = pupil width / median; asp = (pupil height / width) / median; area = pupil height x width / median;
pmean = mean confidence of the four pupil points; trusted = eyelid points plausible (as in blink_final_rule.py).
For each single signal: AUC and the threshold chosen leave-one-video-out (maximizing recall - false-alarm rate); for the
rules: r < t; ph < t; r < t1 OR ph < t2 (grid, leave one video out); every rule also counts implausible-eyelid frames as
not open. Output: blink_notopen_frames_result.csv (rules: precision, recall, flagged share, per stratum) and the per-frame
table blink_notopen_frames_all_videos.csv with the signals and verdicts filled in. Nothing else is modified.
"""
import glob, itertools, sys
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent))
import pupil_trace as pt  # noqa: E402
from scale_step import HERE  # noqa: E402

DX_FRAC, CONF_MIN = 0.3, 0.3
OUT = HERE / "results" / "blink_eyelid_distance"
Q = pd.read_csv(OUT / "blink_notopen_frames_all_videos.csv")
V = pd.read_csv(OUT / "blink_notopen_frames_verdicts.csv")
Q["human_verdict"] = Q.item.map(dict(zip(V.item, V.human_verdict))).fillna("")
PL = pd.read_csv(HERE / "results" / "labels_to_plateau.csv"); PL = PL[PL.status == "final"]
sig = {}
for rr in PL.itertuples():
    v = int(rr.video); step = max(1, int(rr.labels_at_plateau) // 20)
    h5 = sorted(glob.glob(str(HERE / rr.unit / "training-data" / f"predictions_step{step:02d}" / "*snapshot*120*.h5")))
    d = pt.load(h5[-1]); g = lambda b, c: d[b][c].to_numpy(float)
    O = g("eyelid_bottom", "y") - g("eyelid_top", "y"); r = O / np.nanmedian(O)
    w = float(np.nanmedian(np.hypot(g("eye_temporal_corner", "x") - g("eye_nasal_corner", "x"), g("eye_temporal_corner", "y") - g("eye_nasal_corner", "y"))))
    dx = np.abs(g("eyelid_top", "x") - g("eyelid_bottom", "x")) / w; conf = np.minimum(g("eyelid_top", "likelihood"), g("eyelid_bottom", "likelihood"))
    trusted = (O >= 0) & (dx <= DX_FRAC) & (conf >= CONF_MIN) & np.isfinite(r)
    ph = g("pupil_bottom", "y") - g("pupil_top", "y"); pw = np.abs(g("pupil_right", "x") - g("pupil_left", "x"))
    pmean = np.mean([g(b, "likelihood") for b in ("pupil_top", "pupil_bottom", "pupil_left", "pupil_right")], 0)
    wmed = float(np.nanmedian(np.hypot(g("eye_temporal_corner", "x") - g("eye_nasal_corner", "x"), g("eye_temporal_corner", "y") - g("eye_nasal_corner", "y"))))
    sig[v] = dict(r=r, ph=ph / np.nanmedian(ph), pw=pw / np.nanmedian(pw), asp=(ph / pw) / np.nanmedian(ph / pw), area=(ph * pw) / np.nanmedian(ph * pw), pmean=pmean, trusted=trusted,
                  O_w=O / wmed, O_px=O, ph_w=ph / wmed, asp_abs=ph / pw)     # absolute signals: not normalised by the video's median
for k in ("r", "ph", "pw", "asp", "area", "pmean", "trusted", "O_w", "O_px", "ph_w", "asp_abs"):
    Q[k] = [sig[v][k][f] for v, f in zip(Q.video, Q.frame)]
Q.to_csv(OUT / "blink_notopen_frames_all_videos.csv", index=False)
J = Q[Q.human_verdict.notna() & (Q.human_verdict != "") & ~Q.human_verdict.astype(str).str.startswith("cannot")].copy()
J["notopen"] = J.human_verdict.str.startswith("not open").astype(bool); J["trusted"] = J.trusted.astype(bool)
print(len(J), "judged frames,", int(J.notopen.sum()), "not open;", int((Q.human_verdict.str.startswith("cannot")).sum()), "cannot tell")
print(pd.crosstab(J.stratum, J.notopen))

def auc(pos, neg):
    pos, neg = np.asarray(pos, float), np.asarray(neg, float); pos, neg = pos[np.isfinite(pos)], neg[np.isfinite(neg)]
    return float((pos[:, None] < neg[None, :]).mean() + 0.5 * (pos[:, None] == neg[None, :]).mean())
T = J[J.trusted]
print("\nAUC (lower value = not open), trusted frames only:")
for k in ("r", "ph", "pw", "asp", "area", "pmean", "O_w", "O_px", "ph_w", "asp_abs"):
    print(f"  {k:7s} {auc(T[T.notopen][k], T[~T.notopen][k]):.3f}")
I = J[~J.trusted]
print(f"\nimplausible-eyelid frames: {len(I)} judged, {int(I.notopen.sum())} not open; AUC of mean pupil confidence (lower = not open) {auc(I[I.notopen].pmean, I[~I.notopen].pmean):.3f}")

def best_t(g, k):
    v = np.unique(g[k].dropna()); cand = (v[:-1] + v[1:]) / 2
    j = [(g[g.notopen][k] < t).mean() - (g[~g.notopen][k] < t).mean() for t in cand]
    return float(cand[int(np.argmax(j))])

def evaluate(name, flag):
    tp = int((flag & J.notopen).sum()); fl = int(flag.sum()); pos = int(J.notopen.sum())
    row = dict(rule=name, flagged=fl, flagged_pct=round(100 * fl / len(J), 1), precision_pct=round(100 * tp / max(1, fl), 1), recall_pct=round(100 * tp / max(1, pos), 1))
    for s in J.stratum.unique():
        m = J.stratum == s; row[f"recall_{s}"] = round(100 * (flag & J.notopen & m).sum() / max(1, (J.notopen & m).sum()), 0)
    return row
rows = []
for k in ("r", "ph", "area", "asp"):
    flag = pd.Series(False, index=J.index)
    for v in J.video.unique():
        t = best_t(T[T.video != v], k); m = J.video == v
        flag.loc[m] = ((J.loc[m, k] < t) | ~J.loc[m, "trusted"]).to_numpy()
    t_all = best_t(T, k); rows.append(evaluate(f"{k} < t (leave one video out; {t_all:.3f} on all) OR eyelid implausible", flag))
for k in ("O_w", "O_px", "ph_w"):                                        # absolute signals; implausible frames by pupil confidence
    for imp_name, imp in (("implausible -> not open", lambda m: ~J.loc[m, "trusted"]), ("implausible -> pupil conf < 0.527", lambda m: ~J.loc[m, "trusted"] & (J.loc[m, "pmean"] < 0.527))):
        flag = pd.Series(False, index=J.index, dtype=bool)
        for v in J.video.unique():
            t = best_t(T[T.video != v], k); m = J.video == v
            flag.loc[m] = (((J.loc[m, k] < t) & J.loc[m, "trusted"]) | imp(m)).to_numpy()
        t_all = best_t(T, k); rows.append(evaluate(f"{k} < t (leave one video out; {t_all:.3f} on all); {imp_name}", flag))
rows.append(evaluate("r < 0.70 OR eyelid implausible (rule D, relative)", (J.r < 0.70) | ~J.trusted))
# two-signal grid r OR ph, leave one video out
grid_r = np.arange(0.60, 1.01, 0.025); grid_p = np.arange(0.50, 1.01, 0.025)
def best_pair(g):
    best = (-9, None)
    for a, b in itertools.product(grid_r, grid_p):
        f = (g.r < a) | (g.ph < b); j = (f & g.notopen).sum() / max(1, g.notopen.sum()) - (f & ~g.notopen).sum() / max(1, (~g.notopen).sum())
        if j > best[0]: best = (j, (a, b))
    return best[1]
flag = pd.Series(False, index=J.index)
for v in J.video.unique():
    a, b = best_pair(T[T.video != v]); m = J.video == v
    flag.loc[m] = ((J.loc[m, "r"] < a) | (J.loc[m, "ph"] < b) | ~J.loc[m, "trusted"]).to_numpy()
a, b = best_pair(T); rows.append(evaluate(f"r < {a:.3f} OR ph < {b:.3f} (leave one video out) OR eyelid implausible", flag))
R = pd.DataFrame(rows)
R["accuracy_pct"] = np.nan
for i, name in enumerate(R.rule):            # overall accuracy is recomputed below for the rules we can rebuild cheaply
    pass
R.to_csv(OUT / "blink_notopen_frames_result.csv", index=False)
# per-video share of frames judged not open vs the video's median eye opening / eye width
pv = J.groupby("video").agg(frames=("notopen", "size"), notopen=("notopen", "sum"))
pv["notopen_share_pct"] = (100 * pv.notopen / pv.frames).round(0)
pv["median_O_w_video"] = [round(float(np.nanmedian(sig[v]["O_w"])), 3) for v in pv.index]
pv.to_csv(OUT / "blink_notopen_frames_per_video.csv")
print("\nper video: share of sampled frames judged not open vs the video's median eye opening / eye width"); print(pv.to_string())
pd.set_option("display.width", 250); pd.set_option("display.max_columns", 20)
print("\n", R.to_string(index=False))
