#!/usr/bin/env python3
"""Audit of the smallest-subset repair: the three subset orders side by side, with and without leave-one-video-out
ranges, and a by-eye sample of the frames where they disagree (2026-11-07).

    python subset_order_comparison.py            # tables + sample + review page
    python subset_order_comparison.py --eval verdicts.csv

Variants (flag_and_repair.py --finder either --accept labelregion --points all --replace smallest ...):
  A  type order (in use): outline points first, then pupil points, singles before pairs, first passing subset taken
  B  --order deviation: candidate points by decreasing deviation / threshold, the largest first, then the pairs with it
  C  --order deviation --tiebreak margin: as B, but among the passing subsets of the smallest size the one farthest
     inside the label ranges (range_margin)
and the same three with --lovo-range (ranges from the other 18 videos' labels: the video treated as new).

Tables (results/flag_and_repair/all_videos/): subset_order_truth.csv (1330 truth frames: wrong repaired / still
flagged / closed, good made worse, per variant), subset_order_whole.csv (whole videos: repaired, flagged not repaired,
repaired frames whose largest-deviation point is replaced, frames where more than one subset of the chosen size passes),
subset_order_by_video.csv, subset_order_disagreements.csv (frames where A, B, C differ in status or points).
Sample: 40 disagreeing frames, 2 per video where possible (seed 0), review page results/flag_and_repair/subset_review/
subset_review.html: panels = the frame, the model's points, A's result, B's result, C's result (only the variants that
differ are shown as separate panels); question: which panel has every point right? Second part of the page: 20 frames repaired by variant B without replacing
the point of largest deviation (its second peak was tried alone and in pairs and failed the ranges; another point's
replacement passed), one per episode, at most 2 per video; panels: the frame, the model's points with that point ringed
in magenta, B's result; questions: is the magenta point right as it is, is the blue replacement right.
Nothing existing is modified."""
import argparse
import sys
from itertools import combinations
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).resolve().parent))
from second_peak_repair import BPS, P4, PL, RES, Geometry, load, peak2_pixels, pooled_bands  # noqa: E402

FAR = RES / "flag_and_repair"
OUT = FAR / "all_videos"
REV = FAR / "subset_review"
ITEMS = REV / "items"
BASE = "either_labelregion_all_smallest"
VARIANTS = {"A": "", "B": "_devorder", "C": "_devorder_margin", "A_lovo": "_lovo", "B_lovo": "_devorder_lovo", "C_lovo": "_devorder_margin_lovo"}
N_SAMPLE, PER_VIDEO, SEED = 40, 2, 0
N_WORST, PER_VIDEO_WORST = 20, 2
PW, PH = 400, 310
try:
    FONT = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 14)
except OSError:
    FONT = ImageFont.load_default()


def folder(v, k):
    return FAR / f"video{v:02d}_{BASE}{VARIANTS[k]}"


def tables(videos):
    truth, whole, byv = [], [], []
    frames = {}
    for k in VARIANTS:
        S = pd.concat([pd.read_csv(folder(v, k) / "summary.csv") for v in videos], ignore_index=True)
        W = pd.concat([pd.read_csv(folder(v, k) / "whole_video.csv") for v in videos], ignore_index=True)
        F = {v: pd.read_csv(folder(v, k) / "frames_all.csv.gz", dtype={"points": str}) for v in videos}
        frames[k] = F
        rep = {v: f[f.status == "repaired"] for v, f in F.items()}
        worst_in = sum(int(sum(w in str(p).split("+") for w, p in zip(r.worst_point, r.points))) for r in rep.values())
        multi = sum(int((r.n_passing > 1).sum()) for r in rep.values() if "n_passing" in r) if k != "A" and k != "A_lovo" else np.nan
        truth.append({"variant": k, "wrong_frames": int(S.wrong_before.sum()), "wrong_repaired_to_good": int(S.wrong_repaired_to_good.sum()),
                      "wrong_still_flagged": int(S.wrong_flagged_not_repaired.sum()), "wrong_left_untouched": int(S.wrong_left_untouched.sum()),
                      "wrong_after_all": int(S.wrong_after_all.sum()), "good_made_worse": int(S.good_made_worse.sum()), "good_repaired": int(S.good_repaired.sum()),
                      "good_passes_check": int(S.good_flagged.sum() - S.good_repaired.sum() - S.good_flagged_not_repaired.sum()), "good_still_flagged": int(S.good_flagged_not_repaired.sum())})
        n_all = int(W.frames.sum())
        whole.append({"variant": k, "frames": n_all, "repaired": int(W.repaired.sum()), "repaired_pct": round(100 * W.repaired.sum() / n_all, 3),
                      "flagged_not_repaired": int(W.flagged_not_repaired.sum()), "flagged_not_repaired_pct": round(100 * W.flagged_not_repaired.sum() / n_all, 3),
                      "repaired_worst_point_replaced": worst_in, "repaired_worst_point_replaced_pct": round(100 * worst_in / max(1, W.repaired.sum()), 1),
                      "repaired_more_than_one_passing_subset": multi,
                      "replaced_points_outline": int(sum((r.points.fillna("").str.contains("eyelid|corner")).sum() for r in rep.values())),
                      "replaced_points_pupil_only": int(sum((~r.points.fillna("").str.contains("eyelid|corner")).sum() for r in rep.values()))})
        for v in videos:
            s, w = S[S.video == v].iloc[0], W[W.video == v].iloc[0]
            byv.append({"video": v, "variant": k, "repaired": int(w.repaired), "flagged_not_repaired": int(w.flagged_not_repaired), "wrong_before": int(s.wrong_before),
                        "wrong_repaired_to_good": int(s.wrong_repaired_to_good), "wrong_still_flagged": int(s.wrong_flagged_not_repaired), "good_made_worse": int(s.good_made_worse)})
    T, Wt, B = pd.DataFrame(truth), pd.DataFrame(whole), pd.DataFrame(byv)
    T.to_csv(OUT / "subset_order_truth.csv", index=False)
    Wt.to_csv(OUT / "subset_order_whole.csv", index=False)
    B.to_csv(OUT / "subset_order_by_video.csv", index=False)
    pd.set_option("display.width", 250)
    print(T.to_string(index=False))
    print(Wt.to_string(index=False))
    return frames


def disagreements(frames, videos):
    rows = []
    for v in videos:
        A, B, C = (frames[k][v] for k in ("A", "B", "C"))
        pa, pb, pc = (f.points.fillna("").to_numpy() for f in (A, B, C))
        sa, sb, sc = (f.status.to_numpy() for f in (A, B, C))
        diff = (sa != sb) | (sa != sc) | (pa != pb) | (pa != pc)
        for i in np.flatnonzero(diff):
            rows.append({"video": v, "frame": int(i), "status_A": sa[i], "points_A": pa[i], "status_B": sb[i], "points_B": pb[i], "status_C": sc[i], "points_C": pc[i],
                         "worst_point": A.worst_point[i], "fit_rel": A.fit_rel[i], "AB_differ": bool(sa[i] != sb[i] or pa[i] != pb[i]), "BC_differ": bool(sb[i] != sc[i] or pb[i] != pc[i])})
    D = pd.DataFrame(rows)
    D.to_csv(OUT / "subset_order_disagreements.csv", index=False)
    print(f"disagreeing frames: {len(D)} (A vs B: {int(D.AB_differ.sum()) if len(D) else 0}, B vs C: {int(D.BC_differ.sum()) if len(D) else 0})")
    return D


def draw_sample(D):
    rng = np.random.default_rng(SEED)
    D = D.copy()
    D["run_id"] = D.groupby("video").frame.transform(lambda f: np.cumsum(np.r_[True, np.diff(f.to_numpy()) > 5]))
    one = D.groupby(["video", "run_id"]).sample(1, random_state=SEED)
    C = one.groupby("video").sample(frac=1, random_state=SEED).groupby("video").head(PER_VIDEO)
    pool = one[~one.index.isin(C.index)]
    if len(C) < N_SAMPLE and len(pool):
        C = pd.concat([C, pool.sample(min(N_SAMPLE - len(C), len(pool)), random_state=int(rng.integers(1 << 30)))])
    C = C.head(N_SAMPLE).sort_values(["video", "frame"]).reset_index(drop=True)
    C["item"] = [f"video{v:02d}_f{f:06d}" for v, f in zip(C.video, C.frame)]
    return C


def draw_worst_left(frames, videos):
    rows = []
    for v in videos:
        B = frames["B"][v]
        R = B[(B.status == "repaired") & ~B.worst_in_points.astype(bool) & B.flag_fit]
        for r in R.itertuples():
            rows.append({"video": v, "frame": int(r.frame), "status_B": r.status, "points_B": r.points, "worst_point": r.worst_point, "fit_rel": r.fit_rel})
    W = pd.DataFrame(rows)
    W.to_csv(OUT / "subset_order_worst_left.csv", index=False)
    W["run_id"] = W.groupby("video").frame.transform(lambda f: np.cumsum(np.r_[True, np.diff(f.to_numpy()) > 5]))
    one = W.groupby(["video", "run_id"]).sample(1, random_state=SEED)
    C = one.groupby("video").sample(frac=1, random_state=SEED).groupby("video").head(PER_VIDEO_WORST)
    pool = one[~one.index.isin(C.index)]
    if len(C) < N_WORST and len(pool):
        C = pd.concat([C, pool.sample(min(N_WORST - len(C), len(pool)), random_state=SEED)])
    C = C.head(N_WORST).sort_values(["video", "frame"]).reset_index(drop=True)
    C["item"] = [f"video{v:02d}_f{f:06d}_w" for v, f in zip(C.video, C.frame)]
    print(f"frames repaired without the largest-deviation point (variant B, fitting-flagged): {len(W)}; sampled {len(C)}")
    return C


def render(v, rows):
    rr, d, Pk, H, truth, fwd, sig = load(v)
    xy = {b: np.c_[d[b]["x"].to_numpy(float), d[b]["y"].to_numpy(float)] for b in BPS}
    p2, off = peak2_pixels(d, Pk)
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

        def panel(points, ring=None):
            im = raw.copy()
            dr = ImageDraw.Draw(im)
            for b in BPS:
                rep = b in points
                src = p2[b][f] if rep else xy[b][f]
                px, py = (src - box[:2]) * sc
                col = (80, 120, 255) if rep else ((255, 60, 60) if b in P4 else (255, 200, 0))
                if 0 <= px < PW and 0 <= py < PH:
                    dr.ellipse([px - 5, py - 5, px + 5, py + 5], outline=col, width=2)
                    if b == ring:
                        dr.ellipse([px - 10, py - 10, px + 10, py + 10], outline=(255, 0, 255), width=2)
                else:
                    dr.text((6, PH - 18), f"{b}: outside the crop", fill=col, font=FONT)
            return im

        if getattr(r, "item", "").endswith("_w"):
            panels = [("frame", raw), (f"model's points; magenta ring = {r.worst_point} (largest deviation, left as is)", panel((), r.worst_point)),
                      (f"variant B: blue = replaced {str(r.points_B).replace('+', ', ')}", panel(set(str(r.points_B).split("+")), r.worst_point))]
            S = Image.new("RGB", (len(panels) * PW + 10 * (len(panels) - 1), PH + 22), (255, 255, 255))
            d2 = ImageDraw.Draw(S)
            for k, (title, im) in enumerate(panels):
                S.paste(im, (k * (PW + 10), 22))
                d2.text((k * (PW + 10) + 4, 3), title, fill=(0, 0, 0), font=FONT)
            S.save(ITEMS / f"{r.item}.jpg", quality=90)
            continue

        panels = [("frame", raw), ("model's points (red pupil, yellow outline)", panel(()))]
        seen = {}
        for k, pts, st in (("A", r.points_A, r.status_A), ("B", r.points_B, r.status_B), ("C", r.points_C, r.status_C)):
            key = (st, pts)
            if key in seen:
                seen[key][0] += "+" + k
                continue
            label = [k, f"{'not repaired' if st != 'repaired' else 'blue = replaced ' + str(pts).replace('+', ', ')}"]
            seen[key] = label
            panels.append((label, panel(set(str(pts).split("+")) if st == "repaired" else ())))
        S = Image.new("RGB", (len(panels) * PW + 10 * (len(panels) - 1), PH + 22), (255, 255, 255))
        d2 = ImageDraw.Draw(S)
        for k, (title, im) in enumerate(panels):
            S.paste(im, (k * (PW + 10), 22))
            t = title if isinstance(title, str) else f"variant {title[0]}: {title[1]}"
            d2.text((k * (PW + 10) + 4, 3), t, fill=(0, 0, 0), font=FONT)
        S.save(ITEMS / f"video{v:02d}_f{f:06d}.jpg", quality=90)
    cap.release()


def page(C, Wc=None):
    parts = []
    for r in C.sample(frac=1, random_state=SEED).itertuples():
        groups = {}
        for k, pts, st in (("A", r.points_A, r.status_A), ("B", r.points_B, r.status_B), ("C", r.points_C, r.status_C)):
            groups.setdefault((st, str(pts)), []).append(k)
        opts = ["+".join(g) for g in groups.values()] + ["none of them", "cannot tell"]
        parts.append(f'<div class="it" data-item="{r.item}"><div class="h">frame {r.item}</div><img src="items/{r.item}.jpg" loading="lazy">'
                     f'<div class="b"><span class="q">Which variant has every point in the right place?</span>' +
                     "".join(f'<label><input type="radio" name="{r.item}" value="{o}"> {o}</label>' for o in opts) + "</div></div>")
    if Wc is not None and len(Wc):
        parts.append('<h2>Part 2: the point of largest deviation was left as it is, another point was replaced</h2>')
        for r in Wc.sample(frac=1, random_state=SEED).itertuples():
            parts.append(f'<div class="it" data-item="{r.item}"><div class="h">frame {r.item}</div><img src="items/{r.item}.jpg" loading="lazy">'
                         f'<div class="b"><span class="q">Magenta point (left as is):</span>' +
                         "".join(f'<label><input type="radio" name="{r.item}" value="{o}"> {o}</label>' for o in ("right where it is", "wrong, should have moved", "cannot tell")) +
                         f'</div><div class="b"><span class="q">Blue replacement:</span>' +
                         "".join(f'<label><input type="radio" name="{r.item}:blue" value="{o}"> {o}</label>' for o in ("right", "wrong", "cannot tell")) + "</div></div>")
    n = len(C) + (len(Wc) if Wc is not None else 0)
    (REV / "subset_review.html").write_text(f"""<!doctype html><meta charset="utf-8"><title>Subset order check</title>
<style>body{{font:15px -apple-system,Arial,sans-serif;margin:16px;background:#fafafa;color:#222}}.it{{margin:0 0 22px;padding:8px;background:#fff;border:1px solid #ddd;max-width:{5 * PW + 60}px}}
img{{max-width:100%;display:block}}.h{{font-weight:600;margin-bottom:4px}}.b{{margin-top:8px;display:flex;gap:14px;flex-wrap:wrap;align-items:center}}.q{{font-weight:600;margin-right:6px}}
.done{{border-color:#2a9d8f;background:#f3fbf9}}#bar{{position:sticky;top:0;background:#fafafa;padding:8px 0;border-bottom:1px solid #ddd;margin-bottom:12px;z-index:2}}
button{{font:inherit;padding:4px 12px}}</style>
<div id="bar"><b>Subset order check</b> ({n} frames; part 1 = frames where the three variants disagree, part 2 = frames repaired without the largest-deviation point). Panels: the frame; the model's points (red = pupil, yellow = eyelids and corners); then one panel per distinct result, blue = the points that variant replaced (its second peaks). Answer from the images only.<br>
每帧问：哪个变体的 8 个点全都在正确位置？A = 现在的顺序（先轮廓后瞳孔），B = 偏差最大的点先试，C = B 加验收余量平手裁决。同样结果的变体合成一个面板。都不对选 none of them，看不清选 cannot tell。第二部分：品红圈 = 偏差最大、但修复没动的点，问它原地对不对；蓝点 = 修复换上的点，问换得对不对。
&nbsp; <span id="n"></span> <button onclick="save()">Download verdicts (CSV)</button></div>
{"".join(parts)}
<script>
const K='subset_review';let V={{}};try{{V=JSON.parse(localStorage.getItem(K)||'{{}}')}}catch(e){{}}
function upd(){{document.getElementById('n').textContent=new Set(Object.keys(V).map(k=>k.split(':')[0])).size+' of {n} judged.';document.querySelectorAll('.it').forEach(d=>d.classList.toggle('done',!!V[d.dataset.item]))}}
document.querySelectorAll('input[type=radio]').forEach(r=>{{if(V[r.name]===r.value)r.checked=true;r.addEventListener('change',()=>{{V[r.name]=r.value;try{{localStorage.setItem(K,JSON.stringify(V))}}catch(e){{}}upd()}})}});upd();
function save(){{const rows=['item,verdict'].concat(Object.keys(V).sort().map(k=>k+',"'+V[k]+'"'));const a=document.createElement('a');a.href=URL.createObjectURL(new Blob([rows.join('\\n')],{{type:'text/csv'}}));a.download='subset_review_verdicts.csv';a.click()}}
</script>""", encoding="utf-8")
    print(n, "items ->", REV / "subset_review.html")


def evaluate(verdicts):
    C = pd.read_csv(OUT / "subset_order_sample.csv")
    J = C.merge(pd.read_csv(verdicts), on="item")
    J = J[J.verdict != "cannot tell"]
    rows = []
    for k in ("A", "B", "C"):
        right = J.verdict.str.split("+").apply(lambda g, k=k: k in g)
        rows.append({"variant": k, "judged": len(J), "right": int(right.sum()), "right_pct": round(100 * float(right.mean()), 1)})
    S = pd.DataFrame(rows)
    S["none_right"] = int((J.verdict == "none of them").sum())
    S.to_csv(OUT / "subset_order_verdicts.csv", index=False)
    print(S.to_string(index=False))
    Vd = pd.read_csv(verdicts)
    W = pd.read_csv(OUT / "subset_order_worst_left_sample.csv")
    m = Vd[~Vd.item.str.contains(":")].rename(columns={"verdict": "magenta"})
    b = Vd[Vd.item.str.contains(":")].assign(item=lambda x: x.item.str.replace(":blue", "", regex=False)).rename(columns={"verdict": "blue"})
    Jw = W.merge(m, on="item").merge(b, on="item", how="left")
    if len(Jw):
        T = pd.DataFrame([{"judged": len(Jw), "magenta_right": int((Jw.magenta == "right where it is").sum()), "magenta_wrong": int((Jw.magenta == "wrong, should have moved").sum()),
                           "blue_right": int((Jw.blue == "right").sum()), "blue_wrong": int((Jw.blue == "wrong").sum())}])
        T.to_csv(OUT / "subset_order_worst_left_verdicts.csv", index=False)
        print(T.to_string(index=False))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval", default=None)
    ap.add_argument("--html-only", action="store_true")
    a = ap.parse_args()
    if a.eval:
        evaluate(a.eval)
        return
    videos = sorted(PL[PL.status == "final"].video.astype(int))
    videos = [v for v in videos if all((folder(v, k) / "summary.csv").exists() for k in VARIANTS)]
    print("videos with all variants:", videos)
    frames = tables(videos)
    D = disagreements(frames, videos)
    if not len(D):
        return
    C = draw_sample(D)
    C.to_csv(OUT / "subset_order_sample.csv", index=False)
    Wc = draw_worst_left(frames, videos)
    Wc.to_csv(OUT / "subset_order_worst_left_sample.csv", index=False)
    ITEMS.mkdir(parents=True, exist_ok=True)
    if not a.html_only:
        for v, g in pd.concat([C, Wc]).groupby("video"):
            render(int(v), list(g.itertuples()))
    page(C, Wc)


if __name__ == "__main__":
    main()
