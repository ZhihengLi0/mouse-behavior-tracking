"""Second human spot check of the final closure rule (rule D of blink_final_rule.py), drawn after the rule was fixed.

Purpose: the 137 verdicts of the first round were used to choose between the rule variants, so the numbers measured on
them are slightly optimistic. This sample is new (items overlapping a first-round item are excluded), stratified by what
rule D says, and shown BLIND: the sheet does not say which stratum an item comes from, nor the opening value.

Per video (seed 1), at most:
  3  "closed"      : events of rule D (closed runs of >= 3 frames)
  2  "near 0.70-0.85": trusted troughs with r in [0.70, 0.85), rule D open, >= 3 frames, no closed frame within 5 frames
  1  "near 0.85-1.00": the same with r in [0.85, 1.00)
  2  "open"        : random 5-frame windows where every frame is trusted and r >= 1.0 and no closed frame within 30 frames
                     (to estimate closures the rule has no signal for at all)
Each item is shown as 5 frames: 100 ms before, start, middle (deepest for a trough), end, 100 ms after; cyan circles =
predicted eyelid points. The verdict options are the same four as in the first round.

Outputs (results/blink_eyelid_distance/): blink_r2_spotcheck_all_videos.csv (with the stratum, for the evaluation),
blink_r2_spotcheck.html (blind sheet), item_r2_*.jpg and blink_r2_spotcheck_videoNN.jpg (git-excluded).
Evaluation after the verdicts are in: blink_spotcheck_round2_eval.py.
"""
import glob
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import pupil_trace as pt  # noqa: E402
from scale_step import HERE  # noqa: E402
# the rule itself is copied from blink_final_rule.py (that script runs its analysis on import)
FPS, THR, GAP, MIN_LEN, DX_FRAC, CONF_MIN, PUPIL_THR, FULL_R = 60.0, 0.70, 5, 3, 0.3, 0.3, 0.527, 0.45
OUT = HERE / "results" / "blink_eyelid_distance"
PL = pd.read_csv(HERE / "results" / "labels_to_plateau.csv")
PL = PL[PL.status == "final"]
P4 = ["pupil_top", "pupil_bottom", "pupil_left", "pupil_right"]


def runs(mask, gap=0):
    i = np.flatnonzero(mask)
    if not len(i):
        return []
    cut = np.flatnonzero(np.diff(i) > gap + 1 if gap else np.diff(i) > 1)
    return list(zip(i[np.r_[0, cut + 1]], i[np.r_[cut, len(i) - 1]]))


def near(mask, gap):
    return np.convolve(mask.astype(float), np.ones(2 * gap + 1), mode="same") > 0


def states(d):
    O = (d["eyelid_bottom"]["y"] - d["eyelid_top"]["y"]).to_numpy(float)
    r = O / np.nanmedian(O)
    w = float(np.nanmedian(np.hypot(d["eye_temporal_corner"]["x"] - d["eye_nasal_corner"]["x"], d["eye_temporal_corner"]["y"] - d["eye_nasal_corner"]["y"])))
    dx = np.abs((d["eyelid_top"]["x"] - d["eyelid_bottom"]["x"]).to_numpy(float)) / w
    conf = np.minimum(d["eyelid_top"]["likelihood"].to_numpy(float), d["eyelid_bottom"]["likelihood"].to_numpy(float))
    trusted = (O >= 0) & (dx <= DX_FRAC) & (conf >= CONF_MIN) & np.isfinite(r)
    pupil = np.stack([d[b]["likelihood"].to_numpy(float) for b in P4], 1).mean(1)
    B = trusted & (np.nan_to_num(r, nan=np.inf) < THR)
    adj = ~trusted & near(B, GAP)
    D = B | adj
    for a, b in runs(~trusted & ~adj):
        if np.median(pupil[a:b + 1]) < PUPIL_THR:
            D[a:b + 1] = True
    return {"D": D}, r, trusted


def events(closed, r, trusted):
    out = []
    for a, b in runs(closed, GAP):
        if b - a + 1 < MIN_LEN:
            continue
        seg_r = r[a:b + 1]
        full = (np.nanmin(seg_r) < FULL_R) or (~trusted[a:b + 1]).any()
        out.append((int(a), int(b), round((b - a + 1) / FPS * 1000), round(float(np.nanmin(seg_r)), 3), "full" if full else "partial"))
    return out

PAD, SEED = 6, 1
N_CLOSED, N_NEAR_LO, N_NEAR_HI, N_OPEN = 3, 2, 1, 2
rng = np.random.default_rng(SEED)
R1 = pd.read_csv(OUT / "blink_eyelid_spotcheck_all_videos.csv")


def overlaps_round1(video, a, b):
    q = R1[R1.video == video]
    return bool(((q.start <= b + GAP) & (q.end >= a - GAP)).any())


sample, strips_all = [], []
for rr in PL.itertuples():
    v = int(rr.video)
    step = max(1, int(rr.labels_at_plateau) // 20)
    h5 = sorted(glob.glob(str(HERE / rr.unit / "training-data" / f"predictions_step{step:02d}" / "*snapshot*120*.h5")))
    d = pt.load(h5[-1])
    st, r, trusted = states(d)
    closed = st["D"]
    n = len(r)
    cand = {"closed": [], "near 0.70-0.85": [], "near 0.85-1.00": [], "open": []}
    for a, b, du, mr, dep in events(closed, r, trusted):
        if not overlaps_round1(v, a, b):
            k = a + int(np.nanargmin(np.where(trusted[a:b + 1], r[a:b + 1], np.inf)))
            cand["closed"].append((a, b, k, mr))
    guard = near(closed, GAP)
    for lo, hi, key in ((0.70, 0.85, "near 0.70-0.85"), (0.85, 1.00, "near 0.85-1.00")):
        for a, b in runs(trusted & (r >= lo) & (r < hi) & ~guard):
            if b - a + 1 >= MIN_LEN and a > 0 and b < n - 1 and not overlaps_round1(v, a, b):
                k = a + int(np.nanargmin(r[a:b + 1]))
                cand[key].append((a, b, k, round(float(r[k]), 3)))
    far = ~near(closed, 30) & trusted & (r >= 1.0)
    ok_start = np.flatnonzero(np.all(np.lib.stride_tricks.sliding_window_view(far, 5), 1))
    ok_start = [a for a in ok_start if a > PAD and a + 4 + PAD < n and not overlaps_round1(v, a, a + 4)]
    for a in ok_start:
        cand["open"].append((a, a + 4, a + 2, round(float(np.nanmin(r[a:a + 5])), 3)))
    pick = []
    for key, m in (("closed", N_CLOSED), ("near 0.70-0.85", N_NEAR_LO), ("near 0.85-1.00", N_NEAR_HI), ("open", N_OPEN)):
        c = cand[key]
        if c:
            pick += [(key, c[i]) for i in rng.choice(len(c), min(m, len(c)), replace=False)]
    rng.shuffle(pick)                                                        # blind: the order does not reveal the stratum
    cap = cv2.VideoCapture(str(HERE / rr.unit / f"{rr.unit}.mp4"))
    strips = []
    for j, (key, (a, b, k, mr)) in enumerate(pick):
        tiles = []
        for lab, f in (("-100 ms", a - PAD), ("start", a), ("middle", k), ("end", b), ("+100 ms", b + PAD)):
            f = int(np.clip(f, 0, n - 1))
            cap.set(cv2.CAP_PROP_POS_FRAMES, f); ok, im = cap.read()
            if not ok:
                im = np.zeros((736, 928, 3), np.uint8)
            for bp in ("eyelid_top", "eyelid_bottom"):
                x, y = d[bp]["x"].iat[f], d[bp]["y"].iat[f]
                if np.isfinite(x) and np.isfinite(y):
                    cv2.circle(im, (int(x), int(y)), 7, (255, 255, 0), 2)
            im = cv2.resize(im, (464, 368))
            cv2.putText(im, f"{lab}  frame {f}", (6, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2, cv2.LINE_AA)
            tiles.append(im)
        row = np.hstack(tiles)
        item = f"r2_v{v:02d}_{j + 1:02d}"
        head = np.full((34, row.shape[1], 3), 30, np.uint8)
        cv2.putText(head, f"{item}  video {v}  t = {a / FPS:.2f} s  frames {a}-{b} ({round((b - a + 1) / FPS * 1000)} ms)",
                    (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA)
        strip = np.vstack([head, row])
        cv2.imwrite(str(OUT / f"item_{item}.jpg"), strip, [cv2.IMWRITE_JPEG_QUALITY, 82])
        strips.append(strip)
        sample.append({"item": item, "video": v, "unit": rr.unit, "stratum": key, "start": int(a), "end": int(b), "middle": int(k), "start_s": round(a / FPS, 2),
                       "duration_ms": round((b - a + 1) / FPS * 1000), "min_rel_opening": mr, "rule_D_closed": key == "closed", "human_verdict": ""})
    cap.release()
    if strips:
        cv2.imwrite(str(OUT / f"blink_r2_spotcheck_video{v:02d}.jpg"), np.vstack(strips), [cv2.IMWRITE_JPEG_QUALITY, 80])

Q = pd.DataFrame(sample)
Q.to_csv(OUT / "blink_r2_spotcheck_all_videos.csv", index=False)
OPTS = ["eye closed (full closure)", "partly closed (lid covers part of the pupil)", "eye open", "cannot tell"]
items = "\n".join(
    f'<div class="it" data-item="{q.item}"><img src="item_{q.item}.jpg" loading="lazy"><div class="b">' +
    "".join(f'<label><input type="radio" name="{q.item}" value="{o}"> {o}</label>' for o in OPTS) + "</div></div>" for q in Q.itertuples())
(OUT / "blink_r2_spotcheck.html").write_text(f"""<!doctype html><meta charset="utf-8"><title>Blink spot check, round 2</title>
<style>body{{font:15px -apple-system,Arial,sans-serif;margin:16px;background:#fafafa;color:#222}}.it{{margin:0 0 22px;padding:8px;background:#fff;border:1px solid #ddd}}
img{{max-width:100%;display:block}}.b{{margin-top:6px;display:flex;gap:18px;flex-wrap:wrap}}.done{{border-color:#2a9d8f}}#bar{{position:sticky;top:0;background:#fafafa;padding:8px 0;border-bottom:1px solid #ddd;margin-bottom:12px}}
button{{font:inherit;padding:4px 12px}}</style>
<div id="bar"><b>Blink spot check, round 2 (blind)</b>: judge each item from the images only. Each row: 100 ms before, start, middle, end, 100 ms after; cyan circles = predicted eyelid points (they may be wrong). "Closed" = the eye is shut at some point in the item; "partly closed" = the lid covers part of the pupil; "open" = the eye stays open.
<span id="n"></span> <button onclick="save()">Download verdicts (CSV)</button></div>
{items}
<script>
const K='blink_spotcheck_r2';let V={{}};try{{V=JSON.parse(localStorage.getItem(K)||'{{}}')}}catch(e){{}}
function upd(){{document.getElementById('n').textContent=Object.keys(V).length+' of {len(Q)} judged.';document.querySelectorAll('.it').forEach(d=>d.classList.toggle('done',!!V[d.dataset.item]))}}
document.querySelectorAll('input[type=radio]').forEach(r=>{{if(V[r.name]===r.value)r.checked=true;r.addEventListener('change',()=>{{V[r.name]=r.value;try{{localStorage.setItem(K,JSON.stringify(V))}}catch(e){{}}upd()}})}});upd();
function save(){{const rows=['item,human_verdict'].concat(Object.keys(V).sort().map(k=>k+',"'+V[k]+'"'));const a=document.createElement('a');a.href=URL.createObjectURL(new Blob([rows.join('\\n')],{{type:'text/csv'}}));a.download='blink_r2_spotcheck_verdicts.csv';a.click()}}
</script>""", encoding="utf-8")
print(len(Q), "items:", Q.stratum.value_counts().to_dict()); print(Q.groupby("video").stratum.value_counts().unstack(fill_value=0).to_string())
