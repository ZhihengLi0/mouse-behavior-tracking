#!/usr/bin/env python3
"""Closed-eye frames of the repair: what they are, a stratified sample for the user to judge, and the calibration of
the closed-eye line on those judgements (audit 2026-11-07).

    python closed_frames_check.py                 # features of every frame with opening ratio < 0.7, sample, review page
    python closed_frames_check.py --html-only     # rebuild the page from the existing items
    python closed_frames_check.py --eval verdicts.csv

The repair (flag_and_repair.py) calls a frame closed when the eye opening (distance of the two eyelid points across the
corner line, in the frame's eye frame) is below 0.5 x the video's median opening; such frames are left alone. That line
was set by hand. The 1330 truth frames hold no truly closed eye (the 4 frames the rule calls closed all have human pupil
labels; humans labeled pupils down to an opening ratio of 0.36), so the line cannot be derived from the labels; it is
calibrated here on judged frames, the way the flagging threshold (0.14) and step 11's not-open line (0.347) were.

Per frame with opening ratio r < 0.7 (all 19 videos, plateau model's prediction): r; length of the run of frames with
r < 0.5 it belongs to (0 when r >= 0.5); fitting deviation of each eyelid point (eye widths) and whether an eyelid point
is over the flagging threshold; whether the finders flag the frame; which groups of the 12 pupil quantities pass the
label range (along / lidfrac / ratio); the opening ratio the frame would have with the eyelid points moved to their
heatmap second peaks (top, bottom, both) and whether the outline range accepts that.

Sample: 80 frames, strata = r bin (< 0.2, 0.2-0.35, 0.35-0.5, 0.5-0.7; the last bin is not closed under the current rule
and serves as control) x eyelid point flagged (yes / no), 10 per stratum, one frame per run, videos 11 and 15 at least
8 frames each (most closed frames with a visible pupil / most closed frames), seed 0. Review page: two panels (the frame,
the frame with the model's 8 points), one question with four answers: fully closed / half open, pupil can be labeled /
open, eyelid points wrong / cannot tell. Nothing but the images is shown.

--eval: per stratum the share of each answer; candidate closed-eye lines r = 0.10 ... 0.70 with the accuracy of
"r < line <=> fully closed" on the judged frames (the calibration); recall and precision of "eyelid point flagged" for
the answer "eyelid points wrong".

Outputs: results/flag_and_repair/closed_frames/{closed_frames_all.csv.gz, by_video.csv, sample.csv, verdict_summary.csv,
line_candidates.csv}, results/flag_and_repair/closed_review/{items/*.jpg (local only), closed_review.html, items.csv}.
Nothing existing is modified."""
import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).resolve().parent))
from acceptance_thresholds import QUANTS, eye_quantities  # noqa: E402
from outline_region import VideoRef, inside_outline, outline_quantities  # noqa: E402
from second_peak_repair import BPS, P4, PL, RES, Geometry, load, peak2_pixels, pooled_bands  # noqa: E402

FAR = RES / "flag_and_repair"
OUT = FAR / "closed_frames"
REV = FAR / "closed_review"
ITEMS = REV / "items"
LIDS = ("eyelid_top", "eyelid_bottom")
R_MAX, LINE_NOW = 0.7, 0.5
BINS = [(0.0, 0.2), (0.2, 0.35), (0.35, 0.5), (0.5, 0.7)]
N_PER, SEED, N_MIN_1115 = 10, 0, 8
ANSWERS = ["fully closed", "half open, pupil can be labeled", "open, eyelid points wrong", "cannot tell"]
PUPIL_ANSWERS = ["right", "wrong", "no pupil to see", "cannot tell"]
PW, PH = 440, 340
try:
    FONT = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 14)
except OSError:
    FONT = ImageFont.load_default()


def run_lengths(mask):
    """Length of the run of True values each frame belongs to (0 for False)."""
    rl = np.zeros(len(mask), int)
    idx = np.flatnonzero(np.diff(np.r_[0, mask.astype(int), 0]))
    for s, e in zip(idx[::2], idx[1::2]):
        rl[s:e] = e - s
    return rl


def opening_ratio(G, pts_top, pts_bottom, medop):
    """Opening ratio from pixel eyelid positions (arrays (n, 2)) in the frames' eye frames."""
    return np.abs(G.to_eye(pts_bottom)[:, 1] - G.to_eye(pts_top)[:, 1]) / medop


def features(v, thr, R, R_out):
    rr, d, P, H, truth, fwd, sig = load(v)
    xy = {b: np.c_[d[b]["x"].to_numpy(float), d[b]["y"].to_numpy(float)] for b in BPS}
    B, scale = pooled_bands(RES / "second_peak_repair")
    G = Geometry(xy, scale)
    ref = VideoRef(xy)
    p2, off = peak2_pixels(d, P)
    E = {b: G.to_eye(xy[b]) for b in LIDS}
    op = np.abs(E["eyelid_bottom"][:, 1] - E["eyelid_top"][:, 1])
    medop = float(np.nanmedian(op))
    r = op / medop
    assert np.array_equal(G.closed, r < LINE_NOW), "closed rule differs from Geometry.closed"
    rl = run_lengths(r < LINE_NOW)
    dev = fwd["200"] / G.Wm
    F = pd.read_csv(FAR / f"video{v:02d}_either_labelregion_all_smallest" / "frames_all.csv.gz")
    assert len(F) == len(d)
    sel = np.flatnonzero(r < R_MAX)
    r_top = opening_ratio(G, p2["eyelid_top"], xy["eyelid_bottom"], medop)
    r_bot = opening_ratio(G, xy["eyelid_top"], p2["eyelid_bottom"], medop)
    r_both = opening_ratio(G, p2["eyelid_top"], p2["eyelid_bottom"], medop)
    rows = []
    for i in sel:
        pts = {b: xy[b][i] for b in BPS}
        q = eye_quantities(pts, G.sign_lr, G.sign_tb)
        groups = {g: all(np.isfinite(q[k]) and R.loc[k, "min"] <= q[k] <= R.loc[k, "max"] for k in QUANTS if k.endswith(g)) for g in ("along", "lidfrac", "ratio")}
        acc = {}
        for name, rep in (("top", ("eyelid_top",)), ("bottom", ("eyelid_bottom",)), ("both", LIDS)):
            new = dict(pts)
            for b in rep:
                new[b] = p2[b][i]
            acc[name] = bool(all(np.isfinite(new[b]).all() for b in rep)) and inside_outline(outline_quantities(new, ref), R_out)
        rows.append({"video": v, "frame": int(i), "r": r[i], "closed_now": bool(r[i] < LINE_NOW), "run_len": int(rl[i]),
                     "dev_lid_top": dev[i, BPS.index("eyelid_top")], "dev_lid_bottom": dev[i, BPS.index("eyelid_bottom")],
                     "lid_flagged": bool(max(dev[i, BPS.index("eyelid_top")], dev[i, BPS.index("eyelid_bottom")]) > thr),
                     "corner_flagged": bool(max(dev[i, BPS.index("eye_nasal_corner")], dev[i, BPS.index("eye_temporal_corner")]) > thr),
                     "pupil_flagged": bool(np.nanmax(dev[i, [BPS.index(b) for b in P4]]) > thr),
                     "flag_fit": bool(F.flag_fit[i]), "flag_pca": bool(F.flag_pca[i]), "status_now": F.status[i],
                     "pupil_along_ok": groups["along"], "pupil_lidfrac_ok": groups["lidfrac"], "pupil_ratio_ok": groups["ratio"],
                     "pupil_lk_mean": float(np.nanmean([d[b]["likelihood"].iloc[i] for b in P4])),
                     "r_after_top": r_top[i], "r_after_bottom": r_bot[i], "r_after_both": r_both[i],
                     "outline_ok_top": acc["top"], "outline_ok_bottom": acc["bottom"], "outline_ok_both": acc["both"]})
    print(f"video {v}: {len(d)} frames, {int((r < LINE_NOW).sum())} closed now, {len(sel)} with r < {R_MAX}")
    return pd.DataFrame(rows), {"video": v, "frames": len(d), "median_opening_eye_widths": round(medop, 4)}


def draw_sample(A):
    rng = np.random.default_rng(SEED)
    A = A.copy()
    A["bin"] = pd.cut(A.r, [b[0] for b in BINS] + [R_MAX], right=False, labels=[f"{a:.2f}-{b:.2f}" for a, b in BINS])
    A["run_id"] = A.groupby("video").frame.transform(lambda f: np.cumsum(np.r_[True, np.diff(f.to_numpy()) > 1]))
    one_per_run = A.groupby(["video", "run_id"]).sample(1, random_state=SEED)   # one frame per run (or per isolated frame)
    chosen = []
    for v in (11, 15):
        g = one_per_run[one_per_run.video == v]
        k = min(N_MIN_1115, len(g))
        chosen.append(g.groupby(["bin", "lid_flagged"], observed=True).sample(frac=1, random_state=SEED).groupby(["bin", "lid_flagged"], observed=True).head(1).head(k) if k else g.head(0))
    C = pd.concat(chosen)
    for (b, lf), g in one_per_run.groupby(["bin", "lid_flagged"], observed=True):
        have = int(((C.bin == b) & (C.lid_flagged == lf)).sum())
        pool = g[~g.index.isin(C.index)]
        take = min(max(0, N_PER - have), len(pool))
        if take:
            C = pd.concat([C, pool.sample(take, random_state=int(rng.integers(1 << 30)))])
    C = C.sort_values(["video", "frame"]).reset_index(drop=True)
    C["item"] = [f"video{v:02d}_f{f:06d}" for v, f in zip(C.video, C.frame)]
    return C


def render(v, rows):
    rr, d, P, H, truth, fwd, sig = load(v)
    xy = {b: np.c_[d[b]["x"].to_numpy(float), d[b]["y"].to_numpy(float)] for b in BPS}
    G = Geometry(xy)
    Wd = np.linalg.norm(xy["eye_temporal_corner"] - xy["eye_nasal_corner"], axis=1)
    cap = cv2.VideoCapture(str(RES.parent / rr.unit / f"{rr.unit}.mp4"))
    for r in rows:
        f = int(r.frame)
        cap.set(cv2.CAP_PROP_POS_FRAMES, f)
        ok, img = cap.read()
        assert ok, (v, f)
        img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        cx, cy = np.nanmedian(G.C, axis=0)            # the video's median eye position: a thrown corner must not move the crop
        w = max(2.2 * np.nanmedian(Wd), 240)
        h = w * PH / PW
        box = (int(cx - w / 2), int(cy - h / 2), int(cx + w / 2), int(cy + h / 2))
        raw = img.crop(box).resize((PW, PH), Image.LANCZOS)
        sc = PW / (box[2] - box[0])
        im = raw.copy()
        dr = ImageDraw.Draw(im)
        for b in BPS:
            px, py = (xy[b][f] - box[:2]) * sc
            col = (255, 60, 60) if b in P4 else (255, 200, 0)
            if 0 <= px < PW and 0 <= py < PH:
                dr.ellipse([px - 5, py - 5, px + 5, py + 5], outline=col, width=2)
        S = Image.new("RGB", (2 * PW + 10, PH + 22), (255, 255, 255))
        d2 = ImageDraw.Draw(S)
        for k, (title, p) in enumerate((("frame as it is", raw), ("model's points: red = pupil, yellow = eyelids and corners", im))):
            S.paste(p, (k * (PW + 10), 22))
            d2.text((k * (PW + 10) + 4, 3), title, fill=(0, 0, 0), font=FONT)
        S.save(ITEMS / f"video{v:02d}_f{f:06d}.jpg", quality=90)
    cap.release()


def page(C):
    parts = []
    for r in C.sample(frac=1, random_state=SEED).itertuples():          # shuffled: no video or bin order visible
        parts.append(f'<div class="it" data-item="{r.item}"><div class="h">frame {r.item}</div><img src="items/{r.item}.jpg" loading="lazy">'
                     f'<div class="b"><span class="q">The eye in this frame is:</span>' +
                     "".join(f'<label><input type="radio" name="{r.item}" value="{o}"> {o}</label>' for o in ANSWERS) +
                     f'</div><div class="b"><span class="q">Model\'s pupil points (red):</span>' +
                     "".join(f'<label><input type="radio" name="{r.item}:pupil" value="{o}"> {o}</label>' for o in PUPIL_ANSWERS) + "</div></div>")
    n = len(C)
    (REV / "closed_review.html").write_text(f"""<!doctype html><meta charset="utf-8"><title>Closed-eye frames</title>
<style>body{{font:15px -apple-system,Arial,sans-serif;margin:16px;background:#fafafa;color:#222}}.it{{margin:0 0 22px;padding:8px;background:#fff;border:1px solid #ddd;max-width:{2 * PW + 40}px}}
img{{max-width:100%;display:block}}.h{{font-weight:600;margin-bottom:4px}}.b{{margin-top:8px;display:flex;gap:14px;flex-wrap:wrap;align-items:center}}.q{{font-weight:600;margin-right:6px}}
.done{{border-color:#2a9d8f;background:#f3fbf9}}#bar{{position:sticky;top:0;background:#fafafa;padding:8px 0;border-bottom:1px solid #ddd;margin-bottom:12px;z-index:2}}
button{{font:inherit;padding:4px 12px}}</style>
<div id="bar"><b>Closed-eye frames</b> ({n} frames). Left: the frame. Right: the model's points (red = pupil, yellow = eyelids and corners). Answer from the images only.<br>
每帧一问，眼睛是：<b>fully closed</b> = 全闭，看不到瞳孔；<b>half open, pupil can be labeled</b> = 半睁，瞳孔露出一部分、按标注规则能标；<b>open, eyelid points wrong</b> = 眼睛睁着（或半睁），但模型的黄色眼睑点明显不在眼睑上（例如两个眼睑点贴在一起、跑到瞳孔上）；<b>cannot tell</b> = 看不清。第二问：模型的红色瞳孔点对不对（<b>right</b> 四个点都在瞳孔边缘；<b>wrong</b> 有点不在；<b>no pupil to see</b> 全闭没有瞳孔）。
&nbsp; <span id="n"></span> <button onclick="save()">Download verdicts (CSV)</button></div>
{"".join(parts)}
<script>
const K='closed_review';let V={{}};try{{V=JSON.parse(localStorage.getItem(K)||'{{}}')}}catch(e){{}}
function upd(){{document.getElementById('n').textContent=new Set(Object.keys(V).map(k=>k.split(':')[0])).size+' of {n} judged.';document.querySelectorAll('.it').forEach(d=>d.classList.toggle('done',!!V[d.dataset.item]))}}
document.querySelectorAll('input[type=radio]').forEach(r=>{{if(V[r.name]===r.value)r.checked=true;r.addEventListener('change',()=>{{V[r.name]=r.value;try{{localStorage.setItem(K,JSON.stringify(V))}}catch(e){{}}upd()}})}});upd();
function save(){{const rows=['item,verdict'].concat(Object.keys(V).sort().map(k=>k+',"'+V[k]+'"'));const a=document.createElement('a');a.href=URL.createObjectURL(new Blob([rows.join('\\n')],{{type:'text/csv'}}));a.download='closed_review_verdicts.csv';a.click()}}
</script>""", encoding="utf-8")
    print(n, "items ->", REV / "closed_review.html")


def evaluate(verdicts):
    C = pd.read_csv(OUT / "sample.csv")
    Vd = pd.read_csv(verdicts)
    pupil = Vd[Vd.item.str.contains(":pupil")].assign(item=lambda x: x.item.str.replace(":pupil", "", regex=False)).rename(columns={"verdict": "pupil_verdict"})
    Vd = Vd[~Vd.item.str.contains(":")]
    J = C.merge(Vd, on="item", how="inner").merge(pupil, on="item", how="left")
    J = J[J.verdict != "cannot tell"].copy()
    J["fully_closed"] = J.verdict == "fully closed"
    J["lid_wrong"] = J.verdict == "open, eyelid points wrong"
    rows = []
    for (b, lf), g in J.groupby(["bin", "lid_flagged"]):
        rows.append({"bin": b, "lid_flagged": lf, "judged": len(g), **{a: int((g.verdict == a).sum()) for a in ANSWERS[:3]},
                     "pupil_right": int((g.pupil_verdict == "right").sum()), "pupil_wrong": int((g.pupil_verdict == "wrong").sum())})
    S = pd.DataFrame(rows)
    S.to_csv(OUT / "verdict_summary.csv", index=False)
    print(S.to_string(index=False))
    rows = []
    for line in np.round(np.arange(0.10, 0.71, 0.05), 2):
        pred = J.r < line
        rows.append({"line": line, "judged": len(J), "accuracy_pct": round(100 * float((pred == J.fully_closed).mean()), 1),
                     "closed_caught": int((pred & J.fully_closed).sum()), "closed_total": int(J.fully_closed.sum()),
                     "not_closed_called_closed": int((pred & ~J.fully_closed).sum()), "labelable_lost": int((pred & (J.verdict == ANSWERS[1])).sum())})
    Lc = pd.DataFrame(rows)
    Lc.to_csv(OUT / "line_candidates.csv", index=False)
    print(Lc.to_string(index=False))
    # which label-free features separate the three classes (one table), and the pupil-confidence line
    J["cls"] = J.verdict.map({"fully closed": "closed", "half open, pupil can be labeled": "open", "open, eyelid points wrong": "lid_wrong"})
    J["pupil_like"] = J.pupil_along_ok & J.pupil_ratio_ok
    feats = []
    for col in ("lid_flagged", "pupil_flagged", "flag_fit", "flag_pca", "pupil_like", "pupil_lidfrac_ok"):
        for val in (False, True):
            g = J[J[col] == val]
            feats.append({"feature": col, "value": val, "judged": len(g), **{c: int((g.cls == c).sum()) for c in ("closed", "lid_wrong", "open")}})
    for lo, hi in ((0, 0.3), (0.3, 0.5), (0.5, 0.7), (0.7, 1.01)):
        g = J[(J.pupil_lk_mean >= lo) & (J.pupil_lk_mean < hi)]
        feats.append({"feature": "pupil_lk_mean", "value": f"{lo:.1f}-{min(hi, 1):.1f}", "judged": len(g), **{c: int((g.cls == c).sum()) for c in ("closed", "lid_wrong", "open")}})
    Fe = pd.DataFrame(feats)
    Fe.to_csv(OUT / "feature_crosstabs.csv", index=False)
    print(Fe.to_string(index=False))
    # pupil-confidence line: accuracy on the judged frames and cost on the labeled truth frames
    rows = []
    lk_truth = []
    for v in sorted(PL[PL.status == "final"].video.astype(int)):
        rr, d, P, H, truth, fwd, sig = load(v)
        lk = np.nanmean(np.c_[[d[b]["likelihood"].to_numpy(float) for b in P4]].T, axis=1)
        hp = np.c_[[truth[f"human_{b}_x"].to_numpy(float) for b in P4]].T
        lab = np.isfinite(hp).all(1)
        xy = {b: np.c_[d[b]["x"].to_numpy(float), d[b]["y"].to_numpy(float)] for b in BPS}
        line = float(pd.read_csv(RES / "error_tier" / "tier.csv").iloc[0].line_eye_widths)
        lk_truth.append(pd.DataFrame({"video": v, "labeled": lab, "lk": lk[truth.index.to_numpy()], "wrong": (truth.frame_rmse_px.to_numpy() / Geometry(xy).Wm > line)}))
    Lt = pd.concat(lk_truth, ignore_index=True)
    Lt = Lt[Lt.labeled]
    for t in np.round(np.arange(0.20, 0.71, 0.05), 2):
        pred = J.pupil_lk_mean < t
        rows.append({"line": t, "judged": len(J), "accuracy_pct": round(100 * float((pred == J.fully_closed).mean()), 1),
                     "closed_caught": int((pred & J.fully_closed).sum()), "closed_total": int(J.fully_closed.sum()),
                     "not_closed_called_closed": int((pred & ~J.fully_closed).sum()),
                     "labeled_truth_frames_called_closed": int((Lt.lk < t).sum()), "labeled_truth_frames": len(Lt),
                     "of_which_wrong_frames": int(((Lt.lk < t) & Lt.wrong).sum())})
    Lc = pd.DataFrame(rows)
    Lc.to_csv(OUT / "confidence_line_candidates.csv", index=False)
    print(Lc.to_string(index=False))
    tp = int((J.lid_flagged & J.lid_wrong).sum())
    print(f"eyelid points wrong: {int(J.lid_wrong.sum())} judged; eyelid flagged catches {tp} (recall {100 * tp / max(1, int(J.lid_wrong.sum())):.0f} %), "
          f"flagged frames {int(J.lid_flagged.sum())} of which wrong {tp} (precision {100 * tp / max(1, int(J.lid_flagged.sum())):.0f} %)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval", default=None, help="verdict CSV downloaded from the review page")
    ap.add_argument("--html-only", action="store_true")
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    if a.eval:
        evaluate(a.eval)
        return
    if a.html_only:
        page(pd.read_csv(OUT / "sample.csv"))
        return
    thr = float(pd.read_csv(RES / "fit_threshold" / "chosen.csv").iloc[0].threshold_eye_widths)
    R = pd.read_csv(RES / "acceptance_thresholds" / "label_region_margin.csv").set_index("quantity")
    R_out = pd.read_csv(RES / "acceptance_thresholds" / "label_region_outline.csv").set_index("quantity")
    A, meta = [], []
    for v in sorted(PL[PL.status == "final"].video.astype(int)):
        f, m = features(v, thr, R, R_out)
        A.append(f)
        meta.append(m)
    A = pd.concat(A, ignore_index=True)
    A.round(4).to_csv(OUT / "closed_frames_all.csv.gz", index=False)
    c = A[A.closed_now]
    runs = c.groupby("video").apply(lambda g: int((np.diff(np.r_[-9, g.frame.to_numpy()]) > 1).sum()), include_groups=False)
    single = c[c.run_len == 1].groupby("video").size()
    by = pd.DataFrame(meta).set_index("video")
    by["closed"] = c.groupby("video").size()
    by["closed_pct"] = (100 * by.closed / by.frames).round(2)
    by["runs"] = runs
    by["single_frame_runs"] = single
    by["closed_lid_flagged"] = c.groupby("video").lid_flagged.sum()
    by["closed_pupil_quantities_pass"] = c.groupby("video").apply(lambda g: int((g.pupil_along_ok & g.pupil_lidfrac_ok & g.pupil_ratio_ok).sum()), include_groups=False)
    by["closed_reopen_possible"] = c.groupby("video").apply(lambda g: int((((g.r_after_top >= LINE_NOW) & g.outline_ok_top) | ((g.r_after_bottom >= LINE_NOW) & g.outline_ok_bottom) | ((g.r_after_both >= LINE_NOW) & g.outline_ok_both)).sum()), include_groups=False)
    by["r_0.5_0.7"] = A[~A.closed_now].groupby("video").size()
    by = by.fillna(0).astype({c_: int for c_ in ("closed", "runs", "single_frame_runs", "closed_lid_flagged", "closed_pupil_quantities_pass", "closed_reopen_possible", "r_0.5_0.7")})
    by.reset_index().to_csv(OUT / "by_video.csv", index=False)
    pd.set_option("display.width", 250)
    print(by.to_string())
    C = draw_sample(A)
    C.round(4).to_csv(OUT / "sample.csv", index=False)
    print(C.groupby(["bin", "lid_flagged"], observed=True).size().to_string())
    print("videos 11 / 15:", int((C.video == 11).sum()), int((C.video == 15).sum()))
    ITEMS.mkdir(parents=True, exist_ok=True)
    for v, g in C.groupby("video"):
        render(int(v), list(g.itertuples()))
    C[["item", "video", "frame", "r", "bin", "run_len", "lid_flagged"]].to_csv(REV / "items.csv", index=False)
    page(C)


if __name__ == "__main__":
    main()
