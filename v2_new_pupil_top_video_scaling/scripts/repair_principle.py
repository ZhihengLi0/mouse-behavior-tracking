#!/usr/bin/env python3
"""Why a second-peak repair falls short and why a flagged frame is sent back: the reasons, read off the data.

    python repair_principle.py reasons    # every still-flagged frame of the 19 videos, fitting and PCA route
    python repair_principle.py repaired   # the repaired frames of the by-eye check, by verdict, with their heatmaps
    python repair_principle.py locref     # the second peak with the network's own sub-pixel offset, on the truth frames

Direction 1 of the 2026-10-31 meeting: do not add rules, look at the mechanism.

reasons   For every frame with status flagged_not_repaired in results/flag_and_repair/videoNN_labelregion (fitting
          route) and videoNN_pca_labelregion (PCA route), the reason it was not repaired, in the order the code
          decides it: (1) no candidate, an eye-outline point is over the threshold / has the largest residual;
          (2) a candidate pupil point has no second peak; (3) the frame as it is fails the label range and so does the
          frame with the candidate replaced by its second peak. For (3): which of the twelve quantities fail after the
          replacement, whether they belong to the replaced point (its along / lidfrac), to another pupil point, or
          only to the shape ratios; and whether replacing a different single pupil point or a pair would pass.
          Outputs reasons.csv (per video and route), reasons_total.csv, reasons.png, reasons_by_verdict.csv (the 184
          still-flagged frames of the by-eye check, fitting route, joined to their verdicts).
repaired  The 174 repaired frames of the by-eye check (verdicts yes / better but still not right / no / cannot tell):
          heatmap statistics of the replaced point from the peaks run (peak scores, their ratio, distance, number of
          local maxima) by verdict (repaired_by_verdict.csv, heatmap_stats.png); and for the 19 + 4 frames judged not
          right plus 23 frames judged right, the full heatmap recomputed on the CPU: every local maximum near the eye
          with its score (heatmap_maxima.csv) and a gallery (closer_not_right.jpg, right_for_comparison.jpg, local).
locref    DeepLabCut reports a point as the highest heatmap cell plus the offset the location-refinement head outputs
          for that cell. The peaks file keeps only the cell of the second peak, so a repaired point sits at the cell
          plus the video's median offset. Here the second peak is recomputed with the network's own offset at its cell,
          on the truth frames the repair changed (both routes), and the error against the human label is compared
          (peak2_locref.csv).

Reads only existing outputs and the models; writes to results/repair_principle/. Nothing existing is modified."""
import argparse
import re
import sys
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from acceptance_thresholds import QUANTS, eye_quantities  # noqa: E402
from flag_and_repair import candidates, candidates_pca  # noqa: E402
from second_peak_repair import BPS, P4, PL, RES, Geometry, frame_rmse, load, peak2_pixels, pooled_bands, xy_frame  # noqa: E402

OUT = RES / "repair_principle"
OUT.mkdir(parents=True, exist_ok=True)
FR = RES / "flag_and_repair"
THR = 0.14
ROUTES = {"fit": "video{v:02d}_labelregion", "pca": "video{v:02d}_pca_labelregion"}
REASONS = ["outline point over threshold", "no second peak", "replaced frame still outside the range"]
SUB = ["replaced point still outside", "another pupil point outside", "only the shape ratios outside"]


def failing(q, R):
    return [k for k in QUANTS if not (np.isfinite(q[k]) and R.loc[k, "min"] <= q[k] <= R.loc[k, "max"])]


def video_context(v):
    rr, d, P, H, truth, fwd, sig = load(v)
    xy = {b: np.c_[d[b]["x"].to_numpy(float), d[b]["y"].to_numpy(float)] for b in BPS}
    p2, off = peak2_pixels(d, P)
    B, scale = pooled_bands(RES / "second_peak_repair")
    G = Geometry(xy, scale)
    dev = fwd["200"] / G.Wm
    z = np.load(RES / "finder_comparison" / "pca_cache" / f"video{v:02d}_pca.npz")
    R = pd.read_csv(RES / "acceptance_thresholds" / "label_region_margin.csv").set_index("quantity")
    return dict(rr=rr, d=d, P=P, H=H, truth=truth, xy=xy, p2=p2, off=off, G=G, dev=dev, res=z["residuals"], thr_pca=float(z["threshold"]), R=R)


def classify(c, i, route):
    """Reason a flagged frame was not repaired, replaying the decision of flag_and_repair.py (labelregion)."""
    G, xy, p2, R = c["G"], c["xy"], c["p2"], c["R"]
    cand = candidates(c["dev"], i, THR) if route == "fit" else candidates_pca(c["res"], i, c["thr_pca"])
    row = {"frame": i, "candidate": "+".join(cand) if cand else "", "alt_pass": ""}
    if not cand:
        row["reason"] = REASONS[0]
        return row
    if any(not np.isfinite(p2[b][i]).all() for b in cand):
        row["reason"] = REASONS[1]
        return row
    orig = xy_frame(xy, i)
    f0 = failing(eye_quantities(orig, G.sign_lr, G.sign_tb), R)
    assert f0, (route, i, "the frame as it is passes the range, it would have been left alone")
    new = xy_frame(xy, i)
    for b in cand:
        new[b] = p2[b][i]
    f1 = failing(eye_quantities(new, G.sign_lr, G.sign_tb), R)
    assert f1, (route, i, "the replaced frame passes, it would have been repaired")
    row["reason"] = REASONS[2]
    row["fails_before"] = " ".join(f0)
    row["fails_after"] = " ".join(f1)
    own = [k for k in f1 if k.rsplit("_", 1)[1] != "ratio" and any(k.startswith(b) for b in cand)]
    other = [k for k in f1 if k.rsplit("_", 1)[1] != "ratio" and not any(k.startswith(b) for b in cand)]
    row["sub"] = SUB[0] if own else (SUB[1] if other else SUB[2])
    # would another choice of points pass?
    alts = []
    for k in (1, 2):
        for combo in combinations(P4, k):
            if set(combo) == set(cand) or any(not np.isfinite(p2[b][i]).all() for b in combo):
                continue
            t = xy_frame(xy, i)
            for b in combo:
                t[b] = p2[b][i]
            if not failing(eye_quantities(t, G.sign_lr, G.sign_tb), R):
                alts.append("+".join(combo))
        if alts:
            break
    row["alt_pass"] = ";".join(alts)
    row["alt_kind"] = "" if not alts else ("another single point" if "+" not in alts[0] else "a pair of points")
    return row


def reasons():
    rows, per = [], []
    for v in sorted(PL[PL.status == "final"].video.astype(int)):
        c = video_context(v)
        for route, pat in ROUTES.items():
            F = pd.read_csv(FR / pat.format(v=v) / "frames_all.csv.gz")
            W = pd.read_csv(FR / pat.format(v=v) / "whole_video.csv").iloc[0]
            idx = F.index[F.status == "flagged_not_repaired"].to_numpy()
            assert len(idx) == int(W.flagged_not_repaired), (v, route, len(idx), W.flagged_not_repaired)
            rr = [dict(video=v, route=route, **classify(c, int(i), route)) for i in idx]
            rows += rr
            D = pd.DataFrame(rr)
            n = len(D)
            per.append({"video": v, "route": route, "frames": len(F), "flagged_not_repaired": n,
                        **{f"r{k + 1}": int((D.reason == r).sum()) if n else 0 for k, r in enumerate(REASONS)},
                        **{f"s{k + 1}": int((D.get("sub", pd.Series(dtype=str)) == s).sum()) if n else 0 for k, s in enumerate(SUB)},
                        "alt_single": int((D.get("alt_kind", pd.Series(dtype=str)) == "another single point").sum()) if n else 0,
                        "alt_pair": int((D.get("alt_kind", pd.Series(dtype=str)) == "a pair of points").sum()) if n else 0})
            print(f"video {v:2d} {route}: {n:5d} still flagged  " + "  ".join(f"{r}: {per[-1][f'r{k + 1}']}" for k, r in enumerate(REASONS)))
    A = pd.DataFrame(rows)
    A.to_csv(OUT / "reasons_frames.csv.gz", index=False)
    Pv = pd.DataFrame(per)
    Pv.to_csv(OUT / "reasons.csv", index=False)
    T = Pv.groupby("route")[["flagged_not_repaired", "r1", "r2", "r3", "s1", "s2", "s3", "alt_single", "alt_pair"]].sum()
    for k in ("r1", "r2", "r3"):
        T[f"{k}_pct"] = (100 * T[k] / T.flagged_not_repaired).round(1)
    T.to_csv(OUT / "reasons_total.csv")
    print(T.T.to_string())
    # the by-eye check (fitting route, labelregion): 184 still-flagged frames with verdicts
    I = pd.read_csv(FR / "spotcheck_review" / "items.csv").merge(pd.read_csv(FR / "spotcheck_review" / "spotcheck_review_verdicts.csv"), on="item", how="left")
    I = I[I.status == "flagged_not_repaired"].merge(A[A.route == "fit"], on=["video", "frame"], how="left")
    I["verdict"] = I.verdict.fillna("not judged")
    V = pd.crosstab(I.reason, I.verdict)
    V.to_csv(OUT / "reasons_by_verdict.csv")
    print(V.to_string())
    V2 = pd.crosstab(I["sub"].fillna(""), I.verdict)
    V2.to_csv(OUT / "reasons_sub_by_verdict.csv")
    print(V2.to_string())

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2), constrained_layout=True)
    cols = ["#9bb7d4", "#e0a458", "#d1495b"]
    for ax, route in zip(axes, ("fit", "pca")):
        Q = Pv[Pv.route == route].set_index("video")
        bottom = np.zeros(len(Q))
        for k, r in enumerate(REASONS):
            share = 100 * Q[f"r{k + 1}"] / Q.frames
            ax.bar(Q.index, share, bottom=bottom, color=cols[k], label=r)
            bottom += share.to_numpy()
        ax.set_title(f"{'fitting' if route == 'fit' else 'PCA'} finder: why a flagged frame stays flagged")
        ax.set_xlabel("video")
        ax.set_ylabel("% of the video's frames")
        ax.set_xticks(Q.index)
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", color="0.92")
        ax.set_axisbelow(True)
    axes[0].legend(frameon=False, fontsize=9)
    fig.savefig(OUT / "reasons.png", dpi=150)
    print("figure", OUT / "reasons.png")


# ---------------------------------------------------------------- heatmaps with the location-refinement map
def infer(video, frames):
    """Full heatmap and location-refinement map of the plateau model on the given frames (CPU)."""
    import cv2
    import torch
    from deeplabcut.pose_estimation_pytorch.apis.utils import get_pose_inference_runner
    from heatmap_signals import MODELS
    rr = PL[PL.video == video].iloc[0]
    pk = sorted((RES / "heatmap_signals").glob(f"video{video:02d}_shuffle*_peaks.csv"))[-1]
    shuffle = int(re.search(r"shuffle(\d+)", pk.name).group(1))
    folder = next(MODELS.glob(f"*shuffle{shuffle}"))
    snap = folder / "train" / "snapshot-120.pt"
    if not snap.exists():
        snap = next((folder / "train").glob("snapshot-best-120.pt"))
    runner = get_pose_inference_runner(model_config=folder / "train" / "pytorch_config.yaml", snapshot_path=snap, batch_size=1, device="cpu",
                                       inference_cfg={"multithreading": {"enabled": False}})
    pred = runner.model.heads["bodypart"].predictor
    orig, store = pred.forward, []

    def forward(stride, outputs):
        with torch.no_grad():
            Hm = outputs["heatmap"]
            if pred.apply_sigmoid:
                Hm = torch.sigmoid(Hm)
            Hm = Hm.float().clamp(min=0)[0].cpu().numpy()  # (K, h, w)
            L = outputs["locref"][0].float().cpu().numpy()  # (2K, h, w)
            K, h, w = Hm.shape
            L = L.transpose(1, 2, 0).reshape(h, w, K, 2) * float(pred.locref_std)  # (h, w, K, 2) in px
            store.append((Hm, L, float(stride)))
        return orig(stride, outputs)

    pred.forward = forward
    cap = cv2.VideoCapture(str(RES.parent / rr.unit / f"{rr.unit}.mp4"))
    imgs = []
    for f in frames:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(f))
        ok, img = cap.read()
        assert ok, (video, f)
        imgs.append(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    cap.release()
    out = runner.inference(images=iter(imgs))
    return imgs, store, out


def local_maxima(Hk):
    """Cells that are the maximum of their 3x3 neighbourhood (ties to the first), as (score, row, col) sorted down."""
    from scipy.ndimage import maximum_filter
    m = (Hk == maximum_filter(Hk, size=3, mode="nearest")) & (Hk > 0)
    r, c = np.nonzero(m)
    s = Hk[r, c]
    o = np.argsort(-s)
    return list(zip(s[o], r[o], c[o]))


def repaired():
    I = pd.read_csv(FR / "spotcheck_review" / "items.csv").merge(pd.read_csv(FR / "spotcheck_review" / "spotcheck_review_verdicts.csv"), on="item", how="left")
    I = I[I.status == "repaired"].copy()
    I["verdict"] = I.verdict.fillna("not judged")
    rows = []
    ctx = {}
    for v, g in I.groupby("video"):
        c = ctx[v] = video_context(int(v))
        H, G = c["H"], c["G"]
        for _, r in g.iterrows():
            for b in str(r.points).split("+"):
                h = H.loc[r.frame]
                rows.append({"item": r["item"], "video": v, "frame": r.frame, "verdict": r.verdict, "point": b, "episode_len": r.episode_len, "fit_rel": r.fit_rel,
                             "peak1": h[f"{b}_peak1"], "peak2": h[f"{b}_peak2"], "ratio": h[f"{b}_ratio"], "dist_eye_widths": h[f"{b}_dist_px"] / G.Wm,
                             "n_peaks": h[f"{b}_n_peaks"], "entropy": h[f"{b}_entropy"], "mass_near": h[f"{b}_mass_near"],
                             "moved_eye_widths": np.hypot(*(c["p2"][b][r.frame] - c["xy"][b][r.frame])) / G.Wm})
    D = pd.DataFrame(rows)
    D.round(4).to_csv(OUT / "repaired_points.csv", index=False)
    S = D.groupby("verdict")[["peak1", "peak2", "ratio", "dist_eye_widths", "n_peaks", "entropy", "mass_near", "moved_eye_widths", "episode_len", "fit_rel"]].median()
    S.insert(0, "points", D.groupby("verdict").size())
    S.round(3).to_csv(OUT / "repaired_by_verdict.csv")
    print(S.round(3).to_string())

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    order = [k for k in ["yes", "better, but still not right", "no", "cannot tell"] if k in set(D.verdict)]
    short = {"yes": "right", "better, but still not right": "closer, not right", "no": "not better", "cannot tell": "cannot tell"}
    stats = [("ratio", "peak 2 / peak 1"), ("dist_eye_widths", "distance peak 1 to peak 2 (eye widths)"), ("n_peaks", "local maxima above 0.3 x peak 1"), ("peak2", "score of peak 2")]
    fig, axes = plt.subplots(1, len(stats), figsize=(4 * len(stats), 3.8), constrained_layout=True)
    rng = np.random.default_rng(0)
    for ax, (k, lab) in zip(axes, stats):
        for j, vd in enumerate(order):
            y = D[D.verdict == vd][k].to_numpy(float)
            ax.scatter(j + rng.uniform(-0.18, 0.18, len(y)), y, s=12, alpha=0.6, color=["#2f6db3", "#e0a458", "#d1495b", "0.6"][j])
            ax.hlines(np.nanmedian(y), j - 0.3, j + 0.3, color="k", lw=1.5)
        ax.set_xticks(range(len(order)))
        ax.set_xticklabels([f"{short[o]}\n({(D.verdict == o).sum()})" for o in order], fontsize=8)
        ax.set_title(lab, fontsize=10)
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", color="0.92")
        ax.set_axisbelow(True)
    fig.suptitle("Heatmap of the replaced point, repaired frames of the by-eye check, by verdict (black = median)")
    fig.savefig(OUT / "heatmap_stats.png", dpi=150)
    print("figure", OUT / "heatmap_stats.png")

    # full heatmaps: the frames judged not right, and as many judged right (spread over the videos)
    bad = I[I.verdict.isin(["better, but still not right", "no"])]
    good = I[I.verdict == "yes"].iloc[:: max(1, (I.verdict == "yes").sum() // len(bad))].head(len(bad))
    mx = []
    for name, S_ in (("closer_not_right", bad), ("right_for_comparison", good)):
        panels = []
        for v, g in S_.groupby("video"):
            c = ctx[int(v)]
            frames = g.frame.tolist()
            imgs, store, out = infer(int(v), frames)
            for (_, r), img, (Hm, L, stride) in zip(g.iterrows(), imgs, store):
                for b in str(r.points).split("+"):
                    k = BPS.index(b)
                    P1 = c["P"].loc[r.frame]
                    assert Hm[k].argmax() == P1[f"{b}_y1c"] * Hm.shape[2] + P1[f"{b}_x1c"], (v, r.frame, b, "peak 1 cell differs from the peaks file")
                    final = c["p2"][b][r.frame]
                    C = c["G"].C[r.frame]
                    lm = [(s, rr_, cc) for s, rr_, cc in local_maxima(Hm[k]) if s >= 0.2 * Hm[k].max() and np.hypot(cc * stride + stride / 2 - C[0], rr_ * stride + stride / 2 - C[1]) < 0.8 * c["G"].Wm]
                    for j, (s, rr_, cc) in enumerate(lm[:6]):
                        mx.append({"group": name, "item": r["item"], "video": v, "frame": r.frame, "verdict": r.verdict, "point": b, "rank": j + 1, "score": s, "row": rr_, "col": cc,
                                   "to_final_eye_widths": np.hypot(cc * stride + stride / 2 - final[0], rr_ * stride + stride / 2 - final[1]) / c["G"].Wm,
                                   "to_model_eye_widths": np.hypot(cc * stride + stride / 2 - c["xy"][b][r.frame][0], rr_ * stride + stride / 2 - c["xy"][b][r.frame][1]) / c["G"].Wm})
                    panels.append((r, b, img, Hm[k], stride, lm, c))
        ncol = 4
        nrow = int(np.ceil(len(panels) / ncol))
        fig, axes = plt.subplots(nrow, ncol, figsize=(4.2 * ncol, 4.2 * nrow), constrained_layout=True)
        for ax in np.ravel(axes):
            ax.axis("off")
        for ax, (r, b, img, Hk, stride, lm, c) in zip(np.ravel(axes), panels):
            G = c["G"]
            C, W = G.C[r.frame], G.Wm
            x0, y0 = int(max(C[0] - 0.9 * W, 0)), int(max(C[1] - 0.9 * W, 0))
            x1, y1 = int(min(C[0] + 0.9 * W, img.shape[1])), int(min(C[1] + 0.9 * W, img.shape[0]))
            ax.imshow(img[y0:y1, x0:x1])
            hm = np.kron(Hk, np.ones((int(stride), int(stride))))[y0:y1, x0:x1]
            ax.imshow(hm, cmap="summer", alpha=0.45, vmin=0, vmax=max(Hk.max(), 1e-6))
            for j, (s, rr_, cc) in enumerate(lm[:6]):
                ax.plot(cc * stride + stride / 2 - x0, rr_ * stride + stride / 2 - y0, "o", mfc="none", mec=["red", "blue", "white", "white", "white", "white"][j], mew=1.5, ms=9 if j < 2 else 6)
                ax.text(cc * stride + stride / 2 - x0 + 4, rr_ * stride + stride / 2 - y0 - 4, f"{s:.2f}", color="white", fontsize=7)
            for q in P4:
                p = c["xy"][q][r.frame]
                ax.plot(p[0] - x0, p[1] - y0, "r.", ms=4)
            f = c["p2"][b][r.frame]
            ax.plot(f[0] - x0, f[1] - y0, "b+", ms=10, mew=1.5)
            ax.set_title(f"v{int(r.video)} f{int(r.frame)} {b.replace('pupil_', '')}: {short.get(r.verdict, r.verdict)}\nmaxima {len(lm)}, peak2/peak1 {lm[1][0] / lm[0][0] if len(lm) > 1 else float('nan'):.2f}", fontsize=8)
        fig.suptitle(f"{name.replace('_', ' ')}: heatmap of the replaced point (red ring = peak 1, blue ring = peak 2, white = other maxima above 0.2 x peak 1 near the eye; blue + = the point put in; red dots = the model's pupil points)", fontsize=9)
        fig.savefig(OUT / f"{name}.jpg", dpi=80)
        print("gallery", OUT / f"{name}.jpg", len(panels), "panels")
    M = pd.DataFrame(mx)
    M.round(4).to_csv(OUT / "heatmap_maxima.csv", index=False)
    n_max = M.groupby(["group", "item", "point"]).size().groupby("group").agg(["count", "mean", "median", "max"])
    print("number of local maxima above 0.2 x peak 1 within 0.8 eye width of the eye centre, per replaced point:")
    print(n_max.round(2).to_string())
    third = M[M["rank"] >= 3].groupby(["group", "item", "point"]).score.max().groupby("group").agg(["count", "median", "max"])
    print("score of the best third-or-later maximum:")
    print(third.round(3).to_string())
    d2 = M[M["rank"] == 2].groupby("group").to_model_eye_widths.median()
    print("distance from peak 2 to the model's point, median eye widths:")
    print(d2.round(3).to_string())
    pd.concat([n_max.add_prefix("maxima_"), third.add_prefix("third_"), d2.rename("peak2_to_model_median")], axis=1).round(3).to_csv(OUT / "heatmap_maxima_summary.csv")


def locref():
    rows = []
    for v in sorted(PL[PL.status == "final"].video.astype(int)):
        T = {}
        for route, pat in ROUTES.items():
            t = pd.read_csv(FR / pat.format(v=v) / "truth_frames.csv")
            t = t[t.status == "repaired"]
            for _, r in t.iterrows():
                T.setdefault(int(r.frame), {})[route] = r
        if not T:
            continue
        c = video_context(v)
        frames = sorted(T)
        imgs, store, out = infer(v, frames)
        for f, (Hm, L, stride) in zip(frames, store):
            hr = c["truth"].loc[f]
            got = np.asarray(out[frames.index(f)]["bodyparts"])[0, :, :2]
            saved = np.array([c["xy"][b][f] for b in BPS])
            assert np.abs(got - saved).max() < 1.5, (v, f, np.abs(got - saved).max())
            for route, r in T[f].items():
                before = xy_frame(c["xy"], f)
                cur, sub = xy_frame(c["xy"], f), xy_frame(c["xy"], f)
                for b in str(r.points).split("+"):
                    k = BPS.index(b)
                    P1 = c["P"].loc[f]
                    cx, cy = int(P1[f"{b}_x2c"]), int(P1[f"{b}_y2c"])
                    cur[b] = c["p2"][b][f]
                    sub[b] = np.array([cx * stride + stride / 2 + L[cy, cx, k, 0], cy * stride + stride / 2 + L[cy, cx, k, 1]])
                e0, e1, e2 = frame_rmse(before, hr) / c["G"].Wm, frame_rmse(cur, hr) / c["G"].Wm, frame_rmse(sub, hr) / c["G"].Wm
                assert abs(e1 - r.error_after) < 1e-3, (v, f, route, e1, r.error_after)
                rows.append({"video": v, "frame": f, "route": route, "points": r.points, "error_before": e0, "error_after_cell": e1, "error_after_locref": e2,
                             "locref_shift_px": float(np.mean([np.hypot(*(sub[b] - cur[b])) for b in str(r.points).split("+")]))})
                print(f"video {v} frame {f} {route}: before {100 * e0:.2f} %, after (cell + median offset) {100 * e1:.2f} %, after (cell + locref) {100 * e2:.2f} %, shift {rows[-1]['locref_shift_px']:.1f} px")
    D = pd.DataFrame(rows)
    D.round(4).to_csv(OUT / "peak2_locref.csv", index=False)
    print(D.groupby("route")[["error_before", "error_after_cell", "error_after_locref", "locref_shift_px"]].agg(["median", "mean"]).round(4).to_string())


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("part", choices=["reasons", "repaired", "locref"])
    a = ap.parse_args()
    {"reasons": reasons, "repaired": repaired, "locref": locref}[a.part]()
