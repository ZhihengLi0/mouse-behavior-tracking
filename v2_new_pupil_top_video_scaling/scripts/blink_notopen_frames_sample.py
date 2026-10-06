"""Frame-level blind sample for the first-round definition "not open = closed or partly closed" (user decision 2026-10-06).

Item-level verdicts (two rounds, 237 items) showed that above the eyelid-distance threshold no label-free signal separates
"partly closed" from "open" well (best AUC 0.74), so a frame-level truth is needed to fit and validate a rule. This script
draws single frames, stratified by the relative eye opening r (eyelid distance / video median) of the plateau model's
prediction, and writes a blind sheet: one frame per item (crop around the eye, 2x), no values, no predicted points.

Per video (seed 2): 4 frames with r < 0.70 (trusted), 4 with 0.70 <= r < 0.85, 4 with 0.85 <= r < 1.00, 4 with r >= 1.00,
2 frames with implausible eyelid points; frames at least 60 apart from each other and from the two earlier spot checks.
Verdict options: eye open / not open (closed or lid covers part of the pupil) / cannot tell.
Outputs (results/blink_eyelid_distance/): blink_notopen_frames_all_videos.csv, blink_notopen_frames.html, nf_*.jpg (git-excluded).
Evaluation after the verdicts: blink_notopen_frames_eval.py.
"""
import glob, sys
from pathlib import Path
import cv2, numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent))
import pupil_trace as pt  # noqa: E402
from scale_step import HERE  # noqa: E402

THR, DX_FRAC, CONF_MIN, SEED, N_PER, MIN_SEP = 0.70, 0.3, 0.3, 2, 4, 60
OUT = HERE / "results" / "blink_eyelid_distance"
PL = pd.read_csv(HERE / "results" / "labels_to_plateau.csv"); PL = PL[PL.status == "final"]
rng = np.random.default_rng(SEED)
prev = pd.concat([pd.read_csv(OUT / "blink_eyelid_spotcheck_all_videos.csv")[["video", "start", "end"]],
                  pd.read_csv(OUT / "blink_r2_spotcheck_all_videos.csv")[["video", "start", "end"]]])
rows = []
for rr in PL.itertuples():
    v = int(rr.video); step = max(1, int(rr.labels_at_plateau) // 20)
    h5 = sorted(glob.glob(str(HERE / rr.unit / "training-data" / f"predictions_step{step:02d}" / "*snapshot*120*.h5")))
    d = pt.load(h5[-1]); n = len(d)
    g = lambda b, c: d[b][c].to_numpy(float)
    O = g("eyelid_bottom", "y") - g("eyelid_top", "y"); r = O / np.nanmedian(O)
    w = float(np.nanmedian(np.hypot(g("eye_temporal_corner", "x") - g("eye_nasal_corner", "x"), g("eye_temporal_corner", "y") - g("eye_nasal_corner", "y"))))
    dx = np.abs(g("eyelid_top", "x") - g("eyelid_bottom", "x")) / w
    conf = np.minimum(g("eyelid_top", "likelihood"), g("eyelid_bottom", "likelihood"))
    trusted = (O >= 0) & (dx <= DX_FRAC) & (conf >= CONF_MIN) & np.isfinite(r)
    cx = np.nanmedian((g("eye_temporal_corner", "x") + g("eye_nasal_corner", "x")) / 2); cy = np.nanmedian((g("eyelid_top", "y") + g("eyelid_bottom", "y")) / 2)
    used = list(prev[prev.video == v].start) + list(prev[prev.video == v].end)
    def far(f): return all(abs(f - u) >= MIN_SEP for u in used)
    strata = [("r<0.70", trusted & (r < 0.70)), ("0.70-0.85", trusted & (r >= 0.70) & (r < 0.85)), ("0.85-1.00", trusted & (r >= 0.85) & (r < 1.00)),
              ("r>=1.00", trusted & (r >= 1.00)), ("eyelid points implausible", ~trusted)]
    pick = []
    for name, m in strata:
        cand = np.flatnonzero(m); rng.shuffle(cand); k = 2 if name.startswith("eyelid") else N_PER
        for f in cand:
            if len([p for p in pick if p[0] == name]) >= k: break
            if far(f): pick.append((name, int(f))); used.append(int(f))
    rng.shuffle(pick)
    cap = cv2.VideoCapture(str(HERE / rr.unit / f"{rr.unit}.mp4"))
    for j, (name, f) in enumerate(pick):
        cap.set(cv2.CAP_PROP_POS_FRAMES, f); ok, im = cap.read()
        if not ok: continue
        H, W = im.shape[:2]; half = int(1.1 * w)
        x0, x1 = int(np.clip(cx - half, 0, W)), int(np.clip(cx + half, 0, W)); y0, y1 = int(np.clip(cy - 0.75 * half, 0, H)), int(np.clip(cy + 0.75 * half, 0, H))
        crop = im[y0:y1, x0:x1]; crop = cv2.resize(crop, (crop.shape[1] * 2, crop.shape[0] * 2), interpolation=cv2.INTER_CUBIC)
        item = f"nf_v{v:02d}_{j + 1:02d}"
        cv2.putText(crop, f"{item}  video {v}  frame {f}  t = {f / 60:.1f} s", (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2, cv2.LINE_AA)
        cv2.imwrite(str(OUT / f"{item}.jpg"), crop, [cv2.IMWRITE_JPEG_QUALITY, 85])
        rows.append(dict(item=item, video=v, unit=rr.unit, frame=f, stratum=name, r=round(float(r[f]), 3), trusted=bool(trusted[f]), human_verdict=""))
    cap.release()
Q = pd.DataFrame(rows); Q.to_csv(OUT / "blink_notopen_frames_all_videos.csv", index=False)
OPTS = ["eye open", "not open (closed, or the lid covers part of the pupil)", "cannot tell"]
items = "\n".join(f'<div class="it" data-item="{q.item}"><img src="{q.item}.jpg" loading="lazy"><div class="b">' +
                  "".join(f'<label><input type="radio" name="{q.item}" value="{o}"> {o}</label>' for o in OPTS) + "</div></div>" for q in Q.itertuples())
(OUT / "blink_notopen_frames.html").write_text(f"""<!doctype html><meta charset="utf-8"><title>Not-open frames, blind</title>
<style>body{{font:15px -apple-system,Arial,sans-serif;margin:16px;background:#fafafa;color:#222}}.it{{display:inline-block;vertical-align:top;margin:0 14px 18px 0;padding:8px;background:#fff;border:1px solid #ddd;width:560px}}
img{{max-width:100%;display:block}}.b{{margin-top:6px;display:flex;gap:14px;flex-wrap:wrap}}.done{{border-color:#2a9d8f}}#bar{{position:sticky;top:0;background:#fafafa;padding:8px 0;border-bottom:1px solid #ddd;margin-bottom:12px}}button{{font:inherit;padding:4px 12px}}</style>
<div id="bar"><b>Single frames, blind.</b> Definition of the first round: a frame is <b>not open</b> when the eye is closed or the eyelid covers part of the pupil; otherwise <b>open</b>. Judge each frame on its own. <span id="n"></span> <button onclick="save()">Download verdicts (CSV)</button></div>
{items}
<script>
const K='blink_notopen_frames';let V={{}};try{{V=JSON.parse(localStorage.getItem(K)||'{{}}')}}catch(e){{}}
function upd(){{document.getElementById('n').textContent=Object.keys(V).length+' of {len(Q)} judged.';document.querySelectorAll('.it').forEach(d=>d.classList.toggle('done',!!V[d.dataset.item]))}}
document.querySelectorAll('input[type=radio]').forEach(r=>{{if(V[r.name]===r.value)r.checked=true;r.addEventListener('change',()=>{{V[r.name]=r.value;try{{localStorage.setItem(K,JSON.stringify(V))}}catch(e){{}}upd()}})}});upd();
function save(){{const rows=['item,human_verdict'].concat(Object.keys(V).sort().map(k=>k+',"'+V[k]+'"'));const a=document.createElement('a');a.href=URL.createObjectURL(new Blob([rows.join('\\n')],{{type:'text/csv'}}));a.download='blink_notopen_frames_verdicts.csv';a.click()}}
</script>""", encoding="utf-8")
print(len(Q), "frames:", Q.stratum.value_counts().to_dict())
