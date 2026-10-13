"""Second blind single-frame sample (videos 14-18, blink_notopen_frames_sample_r2.py): fill in the signals and the labeler's
verdicts, check the first-round rule of 2026-10-06 on these fresh frames (it was fitted on videos 0-13 only), and write the
combined table of both samples for the per-mouse threshold (blink_per_mouse_threshold_px_v2.py).

    python blink_notopen_frames_r2_eval.py     # needs blink_notopen_frames_r2_verdicts.csv in results/blink_eyelid_distance/

Signals exactly as in blink_notopen_frames_eval.py (same plateau-model predictions, same plausibility test). Rule checked:
trusted frame -> not open iff O / W < 0.347; implausible eyelid points -> not open iff mean pupil confidence < 0.527.
Outputs (results/blink_eyelid_distance/): blink_notopen_frames_r2_all_videos.csv (filled), blink_notopen_frames_r2_result.csv
(rule check on the new frames, per video and per stratum), blink_notopen_frames_combined_all_videos.csv (both samples).
Nothing existing is modified.
"""
import glob, sys
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent))
import pupil_trace as pt  # noqa: E402
from scale_step import HERE  # noqa: E402

DX_FRAC, CONF_MIN, OW_THR, PUPIL_THR = 0.3, 0.3, 0.347, 0.527
ROOT = HERE / "results" / "blink_eyelid_distance"   # folder layout since 2026-10-12: labels/{segments,frames} = the labeler's verdicts; results/{every_video,one_threshold,every_mouse}
LAB_SEG, LAB_FR = ROOT / "labels" / "segments", ROOT / "labels" / "frames"
RES_EV, RES_ONE, RES_MOUSE = ROOT / "results" / "every_video", ROOT / "results" / "one_threshold", ROOT / "results" / "every_mouse"
Q = pd.read_csv(LAB_FR / "blink_notopen_frames_r2_all_videos.csv")
V = pd.read_csv(LAB_FR / "blink_notopen_frames_r2_verdicts.csv")
Q["human_verdict"] = Q.item.map(dict(zip(V.item, V.human_verdict))).fillna("")
PL = pd.read_csv(HERE / "results" / "labels_to_plateau.csv"); PL = PL[(PL.status == "final") & PL.video.isin(Q.video.unique())]
sig = {}
for rr in PL.itertuples():
    v = int(rr.video); step = max(1, int(rr.labels_at_plateau) // 20)
    h5 = sorted(glob.glob(str(HERE / rr.unit / "training-data" / f"predictions_step{step:02d}" / "*snapshot*120*.h5")))
    d = pt.load(h5[-1]); g = lambda b, c: d[b][c].to_numpy(float)
    O = g("eyelid_bottom", "y") - g("eyelid_top", "y"); r = O / np.nanmedian(O)
    wmed = float(np.nanmedian(np.hypot(g("eye_temporal_corner", "x") - g("eye_nasal_corner", "x"), g("eye_temporal_corner", "y") - g("eye_nasal_corner", "y"))))
    dx = np.abs(g("eyelid_top", "x") - g("eyelid_bottom", "x")) / wmed; conf = np.minimum(g("eyelid_top", "likelihood"), g("eyelid_bottom", "likelihood"))
    trusted = (O >= 0) & (dx <= DX_FRAC) & (conf >= CONF_MIN) & np.isfinite(r)
    ph = g("pupil_bottom", "y") - g("pupil_top", "y"); pw = np.abs(g("pupil_right", "x") - g("pupil_left", "x"))
    pmean = np.mean([g(b, "likelihood") for b in ("pupil_top", "pupil_bottom", "pupil_left", "pupil_right")], 0)
    sig[v] = dict(r=r, ph=ph / np.nanmedian(ph), pw=pw / np.nanmedian(pw), asp=(ph / pw) / np.nanmedian(ph / pw), area=(ph * pw) / np.nanmedian(ph * pw), pmean=pmean, trusted=trusted,
                  O_w=O / wmed, O_px=O, ph_w=ph / wmed, asp_abs=ph / pw)
for k in ("r", "ph", "pw", "asp", "area", "pmean", "trusted", "O_w", "O_px", "ph_w", "asp_abs"):
    Q[k] = [sig[v][k][f] for v, f in zip(Q.video, Q.frame)]
Q["notopen"] = Q.human_verdict.astype(str).str.startswith("not open")
Q.to_csv(LAB_FR / "blink_notopen_frames_r2_all_videos.csv", index=False)
J = Q[(Q.human_verdict != "") & ~Q.human_verdict.astype(str).str.startswith("cannot")].copy(); J["trusted"] = J.trusted.astype(bool)
print(len(J), "judged of", len(Q), "frames;", int(J.notopen.sum()), "not open;", int(Q.human_verdict.astype(str).str.startswith("cannot").sum()), "cannot tell; unjudged:", list(Q[Q.human_verdict == ""].item))
J["rule"] = np.where(J.trusted, J.O_w < OW_THR, J.pmean < PUPIL_THR)

def ev(g, name):
    tp = int((g.rule & g.notopen).sum()); fl = int(g.rule.sum()); pos = int(g.notopen.sum())
    return dict(group=name, n=len(g), judged_not_open=pos, rule_not_open=fl, accuracy_pct=round(100 * float((g.rule == g.notopen).mean()), 1),
                precision_pct=round(100 * tp / max(1, fl), 1), recall_pct=round(100 * tp / max(1, pos), 1))
rows = [ev(J, "all new frames (videos 14-18)")]
rows += [ev(J[J.stratum == s], f"stratum {s}") for s in ["r<0.70", "0.70-0.85", "0.85-1.00", "r>=1.00", "eyelid points implausible"]]
rows += [ev(J[J.video == v], f"video {v}") for v in sorted(J.video.unique())]
R = pd.DataFrame(rows); R.to_csv(RES_ONE / "blink_notopen_frames_r2_result.csv", index=False)
pd.set_option("display.width", 200); print(R.to_string(index=False))
# combined table of both samples
Q1 = pd.read_csv(LAB_FR / "blink_notopen_frames_all_videos.csv"); Q1["sample"] = "2026-10-06 (videos 0-13)"; Q["sample"] = "2026-10-12 (videos 14-18)"
C = pd.concat([Q1, Q[[c for c in Q1.columns if c in Q.columns]]], ignore_index=True)
C.to_csv(LAB_FR / "blink_notopen_frames_combined_all_videos.csv", index=False)
JC = C[C.human_verdict.notna() & (C.human_verdict.astype(str) != "") & ~C.human_verdict.astype(str).str.startswith("cannot")].copy()
JC["notopen"] = JC.human_verdict.astype(str).str.startswith("not open"); JC["rule"] = np.where(JC.trusted.astype(bool), JC.O_w < OW_THR, JC.pmean < PUPIL_THR)
print("\ncombined:", len(JC), "judged frames,", int(JC.notopen.sum()), "not open; rule accuracy", round(100 * float((JC.rule == JC.notopen).mean()), 1), "%")
