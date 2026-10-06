#!/usr/bin/env python3
"""Eye closure from the distance between the two eyelid points, with a sample for a human spot check
(first round agreed with Kaiwen 2026-10-04: "use the distance between upper and lower eyelid, test on a few videos,
spot-check part of the result by eye"), read-only analysis. Revised 2026-10-05 after the first sheets showed that about
half of the "events" were eyelid-point tracking errors with the eye open (user's observation).

    python blink_by_eyelid_distance.py

Per finished video (results/labels_to_plateau.csv), model at the plateau point (plateau at 0 labels: the 20-label model),
whole-video prediction (final snapshot, no confidence cut-off):
  eye opening      O_t = y(eyelid_bottom) - y(eyelid_top)            (px, image y points down)
  relative opening r_t = O_t / median of O over the whole video
  UNTRUSTED frame  the eyelid points are implausible, so r_t means nothing: O_t < 0 (upper point below the lower one),
                   or |x(eyelid_top) - x(eyelid_bottom)| > 0.3 x eye width (the two points are not above each other),
                   or the confidence of either eyelid point < 0.3. Not counted as closed or open.
  closure frame    trusted and r_t < THR  (THR = 0.70; chosen before the spot check, to be revised by it)
  closure event    closure frames less than GAP = 5 frames (83 ms) apart are merged; kept only if at least MIN_LEN = 3
                   frames (50 ms) long. start, end, duration, minimum r.
  untrusted run    untrusted frames merged the same way (>= 3 frames). Reported separately: a real closure can also make
                   the eyelid points scatter, so these runs are spot-checked too.
Outputs in results/blink_eyelid_distance/:
  blink_eyelid_events_all_videos.csv     every closure event and untrusted run of every video (column kind)
  blink_eyelid_summary_all_videos.csv    per video: closure events, per minute, median duration, share of closure frames,
                                         untrusted frames and runs, how many raw events the two filters removed
  blink_eyelid_spotcheck_all_videos.csv  the sampled items for the human check (verdict column empty)
  blink_eyelid_spotcheck_videoNN.jpg     one sheet per video; one row per sampled item = 5 frames
                                         (100 ms before the start, start, deepest frame, end, 100 ms after the end),
                                         cyan = predicted eyelid points
  blink_eyelid_spotcheck.html            the same items with buttons; verdicts are kept in the browser and exported as CSV
Sample per video (random, seed 0): up to 6 closure events, up to 3 untrusted runs, up to 2 "near misses" (trusted dips
with 0.70 <= minimum r < 0.85 that the rule does NOT flag). Nothing else is modified."""
import glob
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import pupil_trace as pt  # noqa: E402
from scale_step import HERE  # noqa: E402

FPS, THR, NEAR, GAP, PAD, MIN_LEN, DX_FRAC, CONF_MIN = 60.0, 0.70, 0.85, 5, 6, 3, 0.3, 0.3
rng = np.random.default_rng(0)
OUT = HERE / "results" / "blink_eyelid_distance"
OUT.mkdir(exist_ok=True)
PL = pd.read_csv(HERE / "results" / "labels_to_plateau.csv")
PL = PL[PL.status == "final"]


def runs(mask, gap):
    i = np.flatnonzero(mask)
    if not len(i):
        return []
    cut = np.flatnonzero(np.diff(i) > gap)
    return list(zip(i[np.r_[0, cut + 1]], i[np.r_[cut, len(i) - 1]]))


events, summary, sample = [], [], []
for r in PL.itertuples():
    step = max(1, int(r.labels_at_plateau) // 20)
    h5 = sorted(glob.glob(str(HERE / r.unit / "training-data" / f"predictions_step{step:02d}" / "*snapshot*120*.h5")))
    d = pt.load(h5[-1])
    O = (d["eyelid_bottom"]["y"] - d["eyelid_top"]["y"]).to_numpy(float)
    rel = O / np.nanmedian(O)
    width = float(np.nanmedian(np.hypot(d["eye_temporal_corner"]["x"] - d["eye_nasal_corner"]["x"], d["eye_temporal_corner"]["y"] - d["eye_nasal_corner"]["y"])))
    dx = np.abs((d["eyelid_top"]["x"] - d["eyelid_bottom"]["x"]).to_numpy(float)) / width
    conf = np.minimum(d["eyelid_top"]["likelihood"].to_numpy(float), d["eyelid_bottom"]["likelihood"].to_numpy(float))
    untrusted = (O < 0) | (dx > DX_FRAC) | (conf < CONF_MIN) | ~np.isfinite(rel)
    closed = ~untrusted & (rel < THR)
    raw = runs(rel < THR, GAP)                                                 # the rule before the revision, for the count
    ev, unt = [], []
    for a, b in runs(closed, GAP):
        if b - a + 1 < MIN_LEN:
            continue
        k = a + int(np.nanargmin(np.where(untrusted[a:b + 1], np.inf, rel[a:b + 1])))
        ev.append({"video": int(r.video), "unit": r.unit, "kind": "closure event", "start": int(a), "end": int(b), "deepest": int(k), "start_s": round(a / FPS, 2),
                   "duration_ms": round((b - a + 1) / FPS * 1000), "min_rel_opening": round(float(rel[k]), 3)})
    for a, b in runs(untrusted, GAP):
        if b - a + 1 < MIN_LEN:
            continue
        unt.append({"video": int(r.video), "unit": r.unit, "kind": "untrusted run (eyelid points implausible)", "start": int(a), "end": int(b), "deepest": int((a + b) // 2),
                    "start_s": round(a / FPS, 2), "duration_ms": round((b - a + 1) / FPS * 1000), "min_rel_opening": round(float(np.nanmin(rel[a:b + 1])), 3)})
    events += ev + unt
    dur = [e["duration_ms"] for e in ev]
    summary.append({"video": int(r.video), "unit": r.unit, "mouse": r.mouse, "model_step": step, "frames": len(d), "median_opening_px": round(float(np.nanmedian(O)), 1),
                    "closure_events": len(ev), "events_per_min": round(len(ev) / (len(d) / FPS / 60), 2), "median_duration_ms": float(np.median(dur)) if dur else np.nan,
                    "closure_frames_pct": round(100 * float(closed.mean()), 2), "untrusted_frames_pct": round(100 * float(untrusted.mean()), 2), "untrusted_runs": len(unt),
                    "raw_events_before_revision": len(raw), "removed_as_untrusted_or_short": len(raw) - len(ev)})
    # sample for the spot check (random)
    pick = []
    if ev:
        pick += [ev[i] for i in sorted(rng.choice(len(ev), min(6, len(ev)), replace=False))]
    if unt:
        pick += [unt[i] for i in sorted(rng.choice(len(unt), min(3, len(unt)), replace=False))]
    near = []
    for a, b in runs(~untrusted & (rel >= THR) & (rel < NEAR), GAP):
        if b - a + 1 >= MIN_LEN and a > 0 and b < len(rel) - 1 and not (closed[max(0, a - GAP):b + GAP + 1]).any():
            k = a + int(np.nanargmin(rel[a:b + 1]))
            near.append({"video": int(r.video), "unit": r.unit, "kind": "near miss (not flagged)", "start": int(a), "end": int(b), "deepest": int(k), "start_s": round(a / FPS, 2),
                         "duration_ms": round((b - a + 1) / FPS * 1000), "min_rel_opening": round(float(rel[k]), 3)})
    if near:
        pick += [near[i] for i in sorted(rng.choice(len(near), min(2, len(near)), replace=False))]
    if not pick:
        continue
    cap = cv2.VideoCapture(str(HERE / r.unit / f"{r.unit}.mp4"))
    rows_img = []
    for j, e in enumerate(pick):
        tiles = []
        for lab, f in (("-100 ms", e["start"] - PAD), ("start", e["start"]), ("deepest", e["deepest"]), ("end", e["end"]), ("+100 ms", e["end"] + PAD)):
            f = int(np.clip(f, 0, len(d) - 1))
            cap.set(cv2.CAP_PROP_POS_FRAMES, f); ok, im = cap.read()
            if not ok:
                im = np.zeros((736, 928, 3), np.uint8)
            for bp in ("eyelid_top", "eyelid_bottom"):
                cv2.circle(im, (int(d[bp]["x"].iat[f]), int(d[bp]["y"].iat[f])), 7, (255, 255, 0), 2)
            im = cv2.resize(im, (464, 368))
            cv2.putText(im, f"{lab}  frame {f}  opening {rel[f]:.2f}", (6, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2, cv2.LINE_AA)
            tiles.append(im)
        row = np.hstack(tiles)
        item = f"v{int(r.video):02d}_{j + 1:02d}"
        head = np.full((34, row.shape[1], 3), 30, np.uint8)
        cv2.putText(head, f"{item}  video {int(r.video)}  {e['kind']}  t = {e['start_s']} s  duration {e['duration_ms']} ms  minimum opening {e['min_rel_opening']:.2f} of the video median",
                    (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA)
        strip = np.vstack([head, row])
        cv2.imwrite(str(OUT / f"item_{item}.jpg"), strip, [cv2.IMWRITE_JPEG_QUALITY, 82])
        rows_img.append(strip)
        sample.append({"item": item, **e, "human_verdict": ""})
    cap.release()
    cv2.imwrite(str(OUT / f"blink_eyelid_spotcheck_video{int(r.video):02d}.jpg"), np.vstack(rows_img), [cv2.IMWRITE_JPEG_QUALITY, 80])

E_ = pd.DataFrame(events); S = pd.DataFrame(summary); Q = pd.DataFrame(sample)
E_.to_csv(OUT / "blink_eyelid_events_all_videos.csv", index=False)
S.to_csv(OUT / "blink_eyelid_summary_all_videos.csv", index=False)
Q.to_csv(OUT / "blink_eyelid_spotcheck_all_videos.csv", index=False)

OPTS = ["eye closed (full closure)", "partly closed (lid covers part of the pupil)", "eye open: the rule is wrong", "cannot tell"]
items = "\n".join(
    f'<div class="it" data-item="{q.item}"><img src="item_{q.item}.jpg" loading="lazy"><div class="b">' +
    "".join(f'<label><input type="radio" name="{q.item}" value="{o}"> {o}</label>' for o in OPTS) + "</div></div>" for q in Q.itertuples())
(OUT / "blink_eyelid_spotcheck.html").write_text(f"""<!doctype html><meta charset="utf-8"><title>Blink spot check</title>
<style>body{{font:15px -apple-system,Arial,sans-serif;margin:16px;background:#fafafa;color:#222}}.it{{margin:0 0 22px;padding:8px;background:#fff;border:1px solid #ddd}}
img{{max-width:100%;display:block}}.b{{margin-top:6px;display:flex;gap:18px;flex-wrap:wrap}}.done{{border-color:#2a9d8f}}#bar{{position:sticky;top:0;background:#fafafa;padding:8px 0;border-bottom:1px solid #ddd;margin-bottom:12px}}
button{{font:inherit;padding:4px 12px}}</style>
<div id="bar"><b>Spot check of the eyelid-distance rule</b> (closure when the eye opening is below {THR:.2f} of the video median, at least 3 frames, eyelid points plausible; "untrusted run" = the eyelid points were implausible). Each row: 100 ms before, start, deepest frame, end, 100 ms after; cyan circles = predicted eyelid points.
<span id="n"></span> <button onclick="save()">Download verdicts (CSV)</button></div>
{items}
<script>
const K='blink_spotcheck_v1';let V={{}};try{{V=JSON.parse(localStorage.getItem(K)||'{{}}')}}catch(e){{}}
function upd(){{document.getElementById('n').textContent=Object.keys(V).length+' of {len(Q)} judged.';document.querySelectorAll('.it').forEach(d=>d.classList.toggle('done',!!V[d.dataset.item]))}}
document.querySelectorAll('input[type=radio]').forEach(r=>{{if(V[r.name]===r.value)r.checked=true;r.addEventListener('change',()=>{{V[r.name]=r.value;try{{localStorage.setItem(K,JSON.stringify(V))}}catch(e){{}}upd()}})}});upd();
function save(){{const rows=['item,human_verdict'].concat(Object.keys(V).sort().map(k=>k+',"'+V[k]+'"'));const a=document.createElement('a');a.href=URL.createObjectURL(new Blob([rows.join('\\n')],{{type:'text/csv'}}));a.download='blink_eyelid_spotcheck_verdicts.csv';a.click()}}
</script>""", encoding="utf-8")
pd.set_option("display.width", 220)
print(S.drop(columns=["unit"]).to_string(index=False)); print(len(E_), "rows;", len(Q), "items sampled:", Q.kind.value_counts().to_dict())
