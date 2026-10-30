#!/usr/bin/env python3
"""flag_and_repair.py across all nineteen videos, both acceptance tests side by side.

    python flag_and_repair_all_videos.py [--no-sheets]

Reads results/flag_and_repair/videoNN_{fit,geometry}/ and writes to results/flag_and_repair/all_videos/:
  whole_video_all.csv, truth_summary_all.csv   the 19 x 2 rows side by side (with mouse, date, plateau labels)
  totals.csv                                    sums per acceptance test over all videos
  comparison.csv                                the two acceptance tests in one table, the numbers that matter
  touched_structure.csv                         per video and test: episodes of touched frames (repaired or still
                                                flagged, at most 5 frames apart), lengths, which point, how far
  share_per_video.png                           flagged / repaired / still flagged share per video, both tests
  spotcheck_sample.csv, spotcheck/videoNN_<test>.jpg (jpg local only)   the same rule on every video: up to 10
                                                repaired and 10 still-flagged frames, one per episode, random episodes
                                                (seed 0); red = model, blue = accepted second peak; empty "verdict"
Nothing is deleted anywhere: "still flagged" = flagged_not_repaired. Nothing existing is modified."""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from second_peak_repair import BPS, P4, PL, RES, Geometry, load, peak2_pixels, pooled_bands  # noqa: E402

FAR = RES / "flag_and_repair"
OUT = FAR / "all_videos"
TESTS = ("geometry", "fit", "conf", "jump", "labelgeo", "labelregion0", "labelregion")
GAP, LONG, N_PER, SEED = 5, 30, 10, 0


def episodes(frames):
    f = np.asarray(frames)
    return np.cumsum(np.r_[True, np.diff(f) > GAP]) if len(f) else np.array([], int)


def structure(v, test, F):
    n = len(F)
    T = F[F.status.isin(["repaired", "flagged_not_repaired"])].copy()
    T["episode"] = episodes(T.frame.to_numpy())
    ep_len = T.groupby("episode").size() if len(T) else pd.Series(dtype=int)
    rep = T[T.status == "repaired"]
    mv = rep[[f"{b}_moved_px" for b in P4]].max(axis=1) if len(rep) else pd.Series(dtype=float)
    pts = rep.points.fillna("").str.split("+").explode()
    pts = pts[pts != ""]
    row = {"video": v, "accept": test, "frames": n, "touched": len(T), "touched_pct": round(100 * len(T) / n, 2),
           "episodes": int(ep_len.size), "episode_len_median": float(ep_len.median()) if len(ep_len) else np.nan,
           "episode_len_max": int(ep_len.max()) if len(ep_len) else 0, "singletons": int((ep_len == 1).sum()),
           "touched_in_long_episodes_pct": round(100 * float(T.episode.map(ep_len).ge(LONG).mean()), 1) if len(T) else np.nan,
           **{f"repaired_{b}": int((pts == b).sum()) for b in P4},
           "repair_move_px_median": round(float(mv.median()), 1) if len(mv) else np.nan,
           "repair_move_px_p90": round(float(mv.quantile(0.9)), 1) if len(mv) else np.nan}
    return row, T


def draw_sample(v, test, T, rng):
    rows = []
    for st in ("repaired", "flagged_not_repaired"):
        S = T[T.status == st]
        if S.empty:
            continue
        eps = S.episode.unique()
        rng.shuffle(eps)
        for e in eps[:N_PER]:
            E = S[S.episode == e]
            r = E.iloc[rng.integers(len(E))]
            rows.append({"video": v, "accept": test, "frame": int(r.frame), "status": st, "points": r.points if isinstance(r.points, str) else "",
                         "episode_len": int((T.episode == e).sum()), "fit_rel": round(float(r.fit_rel), 3), "worst_point": r.worst_point, "verdict": ""})
    return rows


def sheet(v, test, sample, ctx):
    import cv2
    from PIL import Image, ImageDraw, ImageFont
    TW, TH, COLS = 300, 225, 5
    try:
        FONT = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 13)
    except OSError:
        FONT = ImageFont.load_default()
    rr, xy, p2, G, W = ctx
    sel = sorted(sample, key=lambda r: (r["status"] != "repaired", r["frame"]))
    rows_n = int(np.ceil(len(sel) / COLS))
    S = Image.new("RGB", (COLS * TW, 40 + rows_n * TH), (255, 255, 255))
    ImageDraw.Draw(S).text((6, 10), f"video {v}, acceptance = {test}: repaired and still-flagged frames, one per episode, random. red = model, "
                           f"blue = accepted second peak. caption: frame, status, points, episode length, fitting deviation", fill=(0, 0, 0), font=FONT)
    cap = cv2.VideoCapture(str(RES.parent / rr.unit / f"{rr.unit}.mp4"))
    for j, r in enumerate(sel):
        f = r["frame"]
        cap.set(cv2.CAP_PROP_POS_FRAMES, f)
        ok, img = cap.read()
        assert ok, (v, f)
        img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        cx, cy = G.C[f]
        w = max(1.7 * W[f], 240)
        h = w * TH / TW
        box = (int(cx - w / 2), int(cy - h / 2), int(cx + w / 2), int(cy + h / 2))
        crop = img.crop(box).resize((TW, TH))
        sc = TW / (box[2] - box[0])
        dr = ImageDraw.Draw(crop)
        pts = [p for p in r["points"].split("+") if p]
        for b in BPS:
            px, py = (xy[b][f] - box[:2]) * sc
            dr.ellipse([px - 4, py - 4, px + 4, py + 4], fill=(255, 60, 60))
            if r["status"] == "repaired" and b in pts:
                nx, ny = (p2[b][f] - box[:2]) * sc
                dr.line([px, py, nx, ny], fill=(80, 120, 255), width=2)
                dr.ellipse([nx - 4, ny - 4, nx + 4, ny + 4], fill=(80, 120, 255))
        dr.rectangle([0, 0, TW, 18], fill=(0, 0, 0))
        st = "still flagged" if r["status"] == "flagged_not_repaired" else r["status"]
        dr.text((4, 1), f"{f} {st} {r['points'].replace('pupil_', '')} ep {r['episode_len']} fit {100 * r['fit_rel']:.0f}%", fill=(255, 255, 255), font=FONT)
        S.paste(crop, ((j % COLS) * TW, 40 + (j // COLS) * TH))
    cap.release()
    (OUT / "spotcheck").mkdir(exist_ok=True)
    S.save(OUT / "spotcheck" / f"video{v:02d}_{test}.jpg", quality=88)


def figure(Wv):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(len(TESTS), 1, figsize=(12, 3.2 * len(TESTS)), sharex=True, constrained_layout=True)
    cols = {"flagged": "#9bb7d4", "repaired": "#2f6db3", "still flagged": "#e0a458"}
    bw = 0.27
    for ax, test in zip(axes, TESTS):
        D = Wv[Wv.accept == test].sort_values("video")
        x = np.arange(len(D))
        for k, (name, lab) in enumerate([("flagged_pct", "flagged"), ("repaired_pct", "repaired"), ("flagged_not_repaired_pct", "still flagged")]):
            ax.bar(x + (k - 1) * bw, D[name], width=bw - 0.03, color=cols[lab], label=lab)
        ax.set_title(f"acceptance = {test}", loc="left", fontsize=10)
        ax.set_ylabel("share of all frames (%)")
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", color="0.9")
        ax.set_axisbelow(True)
        ax.set_xticks(x)
        ax.set_xticklabels([f"{int(r.video)}\n{r.mouse}" for r in D.itertuples()], fontsize=8)
    axes[0].legend(frameon=False, ncol=3)
    fig.suptitle(f"Flag by the fitting deviation (> {100 * Wv.thr_eye_widths.iloc[0]:.0f} % of eye width), repair by the second peak: share per video", fontsize=11)
    fig.savefig(OUT / "share_per_video.png", dpi=150)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-sheets", action="store_true")
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    meta = PL[PL.status == "final"].set_index("video")[["mouse", "date", "labels_at_plateau", "plateau_median_px"]]
    videos = sorted(set.intersection(*[{int(p.name[5:7]) for p in FAR.glob(f"video*_{t}")} for t in TESTS]))
    W, S, ST, sample = [], [], [], []
    rng = np.random.default_rng(SEED)
    for v in videos:
        for test in TESTS:
            d = FAR / f"video{v:02d}_{test}"
            W.append(pd.read_csv(d / "whole_video.csv"))
            S.append(pd.read_csv(d / "summary.csv"))
            F = pd.read_csv(d / "frames_all.csv.gz", dtype={"points": str})
            row, T = structure(v, test, F)
            ST.append(row)
            sample += draw_sample(v, test, T, rng)
    Wv = pd.concat(W).merge(meta, left_on="video", right_index=True, how="left")
    Wv["touched_pct"] = (Wv.repaired_pct + Wv.flagged_not_repaired_pct).round(2)
    Wv.to_csv(OUT / "whole_video_all.csv", index=False)
    Sv = pd.concat(S).merge(meta, left_on="video", right_index=True, how="left")
    Sv.to_csv(OUT / "truth_summary_all.csv", index=False)
    pd.DataFrame(ST).to_csv(OUT / "touched_structure.csv", index=False)
    P = pd.DataFrame(sample)
    P.to_csv(OUT / "spotcheck_sample.csv", index=False)

    rows = []
    cols = ["wrong_before", "wrong_flagged", "wrong_repaired_frames", "wrong_repaired_to_good", "wrong_flagged_not_repaired", "wrong_left_untouched", "wrong_unflagged",
            "wrong_after_all", "wrong_after_unflagged", "good_flagged", "good_repaired", "good_made_worse", "good_pushed_over_line", "good_flagged_not_repaired"]
    for test in TESTS:
        s, w = Sv[Sv.accept == test], Wv[Wv.accept == test]
        rows.append({"accept": test, "videos": len(s), "thr_eye_widths": float(s.thr_eye_widths.iloc[0]), "line_eye_widths": float(s.line_eye_widths.iloc[0]),
                     "truth_frames": int(s.truth_frames.sum()), **{c: int(s[c].sum()) for c in cols},
                     "frames_total": int(w.frames.sum()), "flagged_pct": round(100 * (w.frames * w.flagged_pct / 100).sum() / w.frames.sum(), 2),
                     "repaired_pct": round(100 * w.repaired.sum() / w.frames.sum(), 2),
                     "flagged_not_repaired_pct": round(100 * w.flagged_not_repaired.sum() / w.frames.sum(), 2),
                     "touched_pct_min": float(w.touched_pct.min()), "touched_pct_median": float(w.touched_pct.median()), "touched_pct_max": float(w.touched_pct.max())})
    tot = pd.DataFrame(rows)
    tot.to_csv(OUT / "totals.csv", index=False)
    print(tot.set_index("accept")[cols + ["flagged_pct", "repaired_pct", "flagged_not_repaired_pct"]].T.to_string())
    C = Sv.pivot(index="video", columns="accept", values=["wrong_before", "wrong_repaired_to_good", "wrong_flagged_not_repaired", "good_made_worse", "good_flagged_not_repaired"])
    C.columns = [f"{a}_{b}" for a, b in C.columns]
    C = C.merge(Wv.pivot(index="video", columns="accept", values=["flagged_pct", "repaired_pct", "flagged_not_repaired_pct"]).pipe(lambda D: D.set_axis([f"{a}_{b}" for a, b in D.columns], axis=1)), left_index=True, right_index=True)
    C.reset_index().to_csv(OUT / "comparison.csv", index=False)
    figure(Wv)
    if not a.no_sheets:
        for v in videos:
            rr, d, Pk, H, truth, fwd, sig = load(v)
            xy = {b: np.c_[d[b]["x"].to_numpy(float), d[b]["y"].to_numpy(float)] for b in BPS}
            p2, off = peak2_pixels(d, Pk)
            B, scale = pooled_bands(RES / "second_peak_repair")
            G = Geometry(xy, scale)
            Wd = np.linalg.norm(xy["eye_temporal_corner"] - xy["eye_nasal_corner"], axis=1)
            for test in TESTS:
                sheet(v, test, [r for r in sample if r["video"] == v and r["accept"] == test], (rr, xy, p2, G, Wd))
            print("sheets", v)


if __name__ == "__main__":
    main()
