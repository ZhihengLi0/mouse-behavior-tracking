#!/usr/bin/env python3
"""The chosen pipeline across all nineteen videos: one table, one figure, one spot-check sample per video.

    python triggered_repair_all_videos.py [--no-sheets]

Reads what triggered_repair.py wrote for every video (results/second_peak_repair/videoNN_trig/: summary.csv,
whole_video.csv, frames_all.csv.gz) and produces, in results/second_peak_repair/all_videos/:
  whole_video_all.csv     the nineteen whole-video rows side by side, with mouse, date and plateau labels
  truth_summary_all.csv   the nineteen truth-frame summaries side by side
  touched_structure.csv   per video, how the touched frames (repaired or dropped) sit in time: episodes (touched
                          frames at most 5 frames apart), their lengths, the share of touched frames inside episodes of
                          at least one second (30 frames), which point was replaced, how far it moved, the median
                          trigger value of the touched frames, and in which quarter of the video they fall
  totals.csv              the sums over all videos (truth frames and whole videos)
  touched_share_per_video.png   flagged / repaired / dropped share per video
  spotcheck_sample.csv    the frames drawn for a check by eye: the same rule on every video — up to 10 repaired and
                          10 dropped frames, one per episode, episodes drawn at random (seed 0), with an empty
                          "verdict" column to fill in (real error / false alarm / cannot tell)
  spotcheck/videoNN.jpg   (local only) the drawn frames: red = model, blue = the accepted second peak (repaired
                          frames), yellow ring = the points the geometry check objected to (dropped frames)
Nothing existing is modified; the per-video folders are only read."""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from second_peak_repair import BPS, P4, PL, RES, Geometry, load, peak2_pixels, pooled_bands  # noqa: E402

SPR = RES / "second_peak_repair"
OUT = SPR / "all_videos"
GAP, LONG, N_PER, SEED = 5, 30, 10, 0


def episodes(frames):
    """Episode id per touched frame: a new episode starts after a gap of more than GAP frames."""
    f = np.asarray(frames)
    return np.cumsum(np.r_[True, np.diff(f) > GAP]) if len(f) else np.array([], int)


def structure(v, F):
    n = len(F)
    T = F[F.status.isin(["repaired", "drop"])].copy()
    T["episode"] = episodes(T.frame.to_numpy())
    ep_len = T.groupby("episode").size() if len(T) else pd.Series(dtype=int)
    rep = T[T.status == "repaired"]
    mv = rep[[f"{b}_moved_px" for b in P4]].max(axis=1) if len(rep) else pd.Series(dtype=float)
    pts = rep.points.fillna("").str.split("+").explode()
    pts = pts[pts != ""]
    q = np.histogram(T.frame, bins=np.linspace(0, n, 5))[0] if len(T) else np.zeros(4, int)
    worst_pupil = T.worst_point.isin(P4).mean() if len(T) else np.nan
    return {"video": v, "frames": n, "touched": len(T), "touched_pct": round(100 * len(T) / n, 2),
            "episodes": int(ep_len.size), "episode_len_median": float(ep_len.median()) if len(ep_len) else np.nan,
            "episode_len_max": int(ep_len.max()) if len(ep_len) else 0, "singletons": int((ep_len == 1).sum()),
            "touched_in_long_episodes_pct": round(100 * float(T.episode.map(ep_len).ge(LONG).mean()), 1) if len(T) else np.nan,
            **{f"repaired_{b}": int((pts == b).sum()) for b in P4},
            "repair_move_px_median": round(float(mv.median()), 1) if len(mv) else np.nan,
            "repair_move_px_p90": round(float(mv.quantile(0.9)), 1) if len(mv) else np.nan,
            "trigger_px_median_touched": round(float(T.fit_o200_max.median()), 1) if len(T) else np.nan,
            "worst_point_is_pupil_pct": round(100 * float(worst_pupil), 1) if len(T) else np.nan,
            **{f"touched_q{i + 1}": int(q[i]) for i in range(4)}}, T


def draw_sample(v, T, rng):
    """Up to N_PER repaired and N_PER dropped frames, one per episode, episodes in random order."""
    rows = []
    for st in ("repaired", "drop"):
        S = T[T.status == st]
        if S.empty:
            continue
        eps = S.episode.unique()
        rng.shuffle(eps)
        for e in eps[:N_PER]:
            E = S[S.episode == e]
            r = E.iloc[rng.integers(len(E))]
            rows.append({"video": v, "frame": int(r.frame), "status": st, "points": r.points if isinstance(r.points, str) else "",
                         "episode_len": int((T.episode == e).sum()), "trigger_px": round(float(r.fit_o200_max), 1),
                         "worst_point": r.worst_point, "verdict": ""})
    return rows


def sheet(v, sample):
    import cv2
    from PIL import Image, ImageDraw, ImageFont
    TW, TH, COLS = 300, 225, 5
    try:
        FONT = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 13)
    except OSError:
        FONT = ImageFont.load_default()
    rr, d, P, H, truth, fwd, sig = load(v)
    xy = {b: np.c_[d[b]["x"].to_numpy(float), d[b]["y"].to_numpy(float)] for b in BPS}
    p2, off = peak2_pixels(d, P)
    B, scale = pooled_bands(SPR)
    G = Geometry(xy, scale)
    W = np.linalg.norm(xy["eye_temporal_corner"] - xy["eye_nasal_corner"], axis=1)
    sel = sorted(sample, key=lambda r: (r["status"] != "repaired", r["frame"]))
    rows_n = int(np.ceil(len(sel) / COLS))
    S = Image.new("RGB", (COLS * TW, 40 + rows_n * TH), (255, 255, 255))
    ImageDraw.Draw(S).text((6, 10), f"video {v}: frames touched by the pipeline, one per episode, drawn at random. red = model, "
                           f"blue = accepted second peak, yellow ring = points the geometry check objected to. caption: frame, status, points, episode length",
                           fill=(0, 0, 0), font=FONT)
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
            if r["status"] == "drop" and b in pts:
                dr.ellipse([px - 7, py - 7, px + 7, py + 7], outline=(255, 220, 0), width=2)
            dr.ellipse([px - 4, py - 4, px + 4, py + 4], fill=(255, 60, 60))
            if r["status"] == "repaired" and b in pts:
                nx, ny = (p2[b][f] - box[:2]) * sc
                dr.line([px, py, nx, ny], fill=(80, 120, 255), width=2)
                dr.ellipse([nx - 4, ny - 4, nx + 4, ny + 4], fill=(80, 120, 255))
        dr.rectangle([0, 0, TW, 18], fill=(0, 0, 0))
        dr.text((4, 1), f"{f} {r['status']} {r['points'].replace('pupil_', '')} ep {r['episode_len']}", fill=(255, 255, 255), font=FONT)
        S.paste(crop, ((j % COLS) * TW, 40 + (j // COLS) * TH))
    cap.release()
    (OUT / "spotcheck").mkdir(exist_ok=True)
    S.save(OUT / "spotcheck" / f"video{v:02d}.jpg", quality=88)


def figure(Wv):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(12, 4.2), constrained_layout=True)
    x = np.arange(len(Wv))
    cols = {"flagged": "#9bb7d4", "repaired": "#2f6db3", "dropped": "#d1495b"}
    bw = 0.27
    for k, (name, col) in enumerate([("flag_o200_pct", "flagged"), ("repaired_pct", "repaired"), ("drop_pct", "dropped")]):
        ax.bar(x + (k - 1) * bw, Wv[name], width=bw - 0.03, color=cols[col], label=col)
    for i, r in enumerate(Wv.itertuples()):
        if r.flag_o200_pct >= 9:
            ax.text(i - bw, r.flag_o200_pct + 0.2, f"{r.flag_o200_pct:.0f}", ha="center", va="bottom", fontsize=8, color="0.25")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{int(r.video)}\n{r.mouse}" for r in Wv.itertuples()], fontsize=8)
    ax.set_ylabel("share of all frames (%)")
    ax.set_title("The chosen pipeline on every video: frames flagged by the time-series rule, repaired, dropped")
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color="0.9")
    ax.set_axisbelow(True)
    ax.legend(frameon=False, ncol=3)
    fig.savefig(OUT / "touched_share_per_video.png", dpi=150)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-sheets", action="store_true")
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    meta = PL[PL.status == "final"].set_index("video")[["mouse", "date", "labels_at_plateau", "plateau_median_px"]]
    videos = sorted(int(p.name[5:7]) for p in SPR.glob("video*_trig"))
    W, S, ST, sample = [], [], [], []
    rng = np.random.default_rng(SEED)
    for v in videos:
        d = SPR / f"video{v:02d}_trig"
        W.append(pd.read_csv(d / "whole_video.csv"))
        S.append(pd.read_csv(d / "summary.csv"))
        F = pd.read_csv(d / "frames_all.csv.gz", dtype={"points": str})
        row, T = structure(v, F)
        ST.append(row)
        sample += draw_sample(v, T, rng)
        print(f"video {v}: touched {row['touched']} in {row['episodes']} episodes, {row['touched_in_long_episodes_pct']} % in episodes >= {LONG} frames")
    Wv = pd.concat(W).merge(meta, left_on="video", right_index=True, how="left")
    Wv["touched_pct"] = (Wv.repaired_pct + Wv.drop_pct).round(2)
    Wv.to_csv(OUT / "whole_video_all.csv", index=False)
    Sv = pd.concat(S).merge(meta, left_on="video", right_index=True, how="left")
    Sv.to_csv(OUT / "truth_summary_all.csv", index=False)
    pd.DataFrame(ST).to_csv(OUT / "touched_structure.csv", index=False)
    P = pd.DataFrame(sample)
    P.to_csv(OUT / "spotcheck_sample.csv", index=False)
    tot = {"videos": len(videos), "truth_frames": int(Sv.truth_frames.sum()), "gt50_before": int(Sv.gt50_before.sum()),
           "gt50_repaired_to_lt50": int(Sv.gt50_repaired_to_lt50.sum()), "gt50_repaired_to_lt20": int(Sv.gt50_repaired_to_lt20.sum()),
           "gt50_dropped": int(Sv.gt50_dropped.sum()), "gt50_left_untouched": int(Sv.gt50_left_untouched.sum()),
           "gt50_after_among_kept": int(Sv.gt50_after_among_kept.sum()),
           "frames_le20": int((Sv.truth_frames - Sv.gt20_before).sum()), "false_repairs_lt20_worse5px": int(Sv.false_repairs_lt20_worse5px.sum()),
           "false_drops_lt20": int(Sv.false_drops_lt20.sum()), "drops_20to50": int(Sv.drops_20to50.sum()),
           "frames_total": int(Wv.frames.sum()), "flagged_total": int(round((Wv.frames * Wv.flag_o200_pct / 100).sum())),
           "repaired_total": int(Wv.repaired.sum()), "dropped_total": int(Wv["drop"].sum()),
           "flagged_pct": round(100 * (Wv.frames * Wv.flag_o200_pct / 100).sum() / Wv.frames.sum(), 2),
           "repaired_pct": round(100 * Wv.repaired.sum() / Wv.frames.sum(), 2), "dropped_pct": round(100 * Wv["drop"].sum() / Wv.frames.sum(), 2),
           "touched_pct_min": float(Wv.touched_pct.min()), "touched_pct_median": float(Wv.touched_pct.median()), "touched_pct_max": float(Wv.touched_pct.max()),
           "spotcheck_frames": len(P)}
    pd.DataFrame([tot]).to_csv(OUT / "totals.csv", index=False)
    print(pd.Series(tot).to_string())
    figure(Wv)
    if not a.no_sheets:
        for v in videos:
            sheet(v, [r for r in sample if r["video"] == v])
            print("sheet", v)


if __name__ == "__main__":
    main()
