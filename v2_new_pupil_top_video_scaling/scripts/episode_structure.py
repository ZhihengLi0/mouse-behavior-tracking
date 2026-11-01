#!/usr/bin/env python3
"""Flagged frames come in episodes. How long are they, where do the truth frames sit in them, and what can the good
frames on both sides of an episode give, without any smoothing of unflagged frames?

    python episode_structure.py

Direction 2 of the 2026-10-31 meeting. Reads the whole-video outputs of flag_and_repair.py (videoNN_pca_labelregion,
which carries both finders' flags), the truth-frame outputs of both routes, the plateau predictions, the human labels
of the truth frames and the cached one-step forecasts. Writes to results/episode_structure/:

  episodes_by_video.csv   per video and finder (fit, pca, either): flagged frames, episodes, median and max length,
                          share of flagged frames in episodes of >= 10 and >= 50 frames
  episode_lengths.png     pooled episode-length histogram per finder
  timeline.png            one strip per video: share of frames flagged in each of 600 time bins, fitting and PCA
  truth_in_episodes.csv   every flagged truth frame of both routes: wrong or right, episode length, position inside it
  temporal_fill_truth.csv on the same frames, the error against the human label when the flagged points are filled
                          from the frames around the episode (never touching an unflagged frame): linear interpolation
                          between the last unflagged frame before and the first after, the nearest of the two copied,
                          the one-step forecast the fitting finder already uses; next to the error before and after the
                          second-peak repair
  temporal_fill.png       the same as a figure, by episode length
  displacement_vs_gap.csv / .png  label-free bound: on unflagged stretches of every video, the error of interpolating
                          a pupil point across a gap of L frames (the two ends known, the middle predicted linearly)
Nothing existing is modified."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from second_peak_repair import BPS, P4, PL, RES, Geometry, frame_rmse, load, pooled_bands, xy_frame  # noqa: E402

OUT = RES / "episode_structure"
OUT.mkdir(parents=True, exist_ok=True)
FR = RES / "flag_and_repair"
LINE, THR = 0.14, 0.14
FINDERS = {"fit": "flag_fit", "pca": "flag_pca", "either": None}
GAPS = [1, 2, 3, 5, 10, 20, 30, 50, 100, 200]


def episodes(flag):
    """(start, end) inclusive of every maximal run of True."""
    f = np.asarray(flag, bool)
    if not f.any():
        return np.zeros((0, 2), int)
    d = np.diff(np.r_[0, f.astype(int), 0])
    return np.c_[np.flatnonzero(d == 1), np.flatnonzero(d == -1) - 1]


def episode_of(E, i):
    k = np.searchsorted(E[:, 0], i, side="right") - 1
    assert k >= 0 and E[k, 0] <= i <= E[k, 1]
    return E[k]


def fill_errors(xy, hr, i, s, e, over, pm, G, n):
    """Error against the human label when the points `over` of frame i are filled from the frames around episode
    [s, e]; returns dict of variant -> error in eye widths (NaN when there is no anchor on one side)."""
    a, b = s - 1, e + 1
    out = {}
    have = (a >= 0, b < n)
    for variant in ("interp", "nearest", "forecast"):
        new = xy_frame(xy, i)
        ok = True
        for p in over:
            if variant == "interp":
                if not all(have):
                    ok = False
                    break
                t = (i - a) / (b - a)
                new[p] = (1 - t) * xy[p][a] + t * xy[p][b]
            elif variant == "nearest":
                cands = [k for k, h in zip((a, b), have) if h]
                if not cands:
                    ok = False
                    break
                k = min(cands, key=lambda k: abs(k - i))
                new[p] = xy[p][k].copy()
            else:
                q = pm[i, BPS.index(p)]
                if not np.isfinite(q).all():
                    ok = False
                    break
                new[p] = q
        out[variant] = frame_rmse(new, hr) / G.Wm if ok else np.nan
    return out


def main():
    rows_ep, rows_truth, rows_gap = [], [], []
    strips = {}
    rng = np.random.default_rng(0)
    for v in sorted(PL[PL.status == "final"].video.astype(int)):
        rr, d, P, H, truth, fwd, sig = load(v)
        xy = {b: np.c_[d[b]["x"].to_numpy(float), d[b]["y"].to_numpy(float)] for b in BPS}
        B, scale = pooled_bands(RES / "second_peak_repair")
        G = Geometry(xy, scale)
        n = len(d)
        F = pd.read_csv(FR / f"video{v:02d}_pca_labelregion" / "frames_all.csv.gz")
        assert len(F) == n
        flags = {"fit": F.flag_fit.to_numpy(bool), "pca": F.flag_pca.to_numpy(bool)}
        flags["either"] = flags["fit"] | flags["pca"]
        dev = fwd["200"] / G.Wm                                   # (n, 8) fitting deviation of every point
        pm = np.load(FR / "cache" / f"video{v:02d}_pred200.npy")  # (n, 8, 2) one-step forecast
        res = np.load(RES / "finder_comparison" / "pca_cache" / f"video{v:02d}_pca.npz")["residuals"]
        E = {}
        for name, f in flags.items():
            E[name] = ep = episodes(f)
            L = ep[:, 1] - ep[:, 0] + 1 if len(ep) else np.zeros(0, int)
            rows_ep.append({"video": v, "finder": name, "frames": n, "flagged": int(f.sum()), "flagged_pct": round(100 * f.mean(), 2), "episodes": len(ep),
                            "len_median": float(np.median(L)) if len(L) else np.nan, "len_mean": round(float(L.mean()), 1) if len(L) else np.nan, "len_max": int(L.max()) if len(L) else 0,
                            "singletons": int((L == 1).sum()), "episodes_ge10": int((L >= 10).sum()), "episodes_ge50": int((L >= 50).sum()),
                            "flagged_in_ge10_pct": round(100 * L[L >= 10].sum() / max(f.sum(), 1), 1), "flagged_in_ge50_pct": round(100 * L[L >= 50].sum() / max(f.sum(), 1), 1)})
        strips[v] = {name: np.histogram(np.flatnonzero(f), bins=600, range=(0, n))[0] / np.histogram(np.arange(n), bins=600, range=(0, n))[0] for name, f in flags.items() if name != "either"}
        print(f"video {v:2d}: " + "  ".join(f"{r['finder']} {r['flagged']} flagged / {r['episodes']} episodes, median {r['len_median']:.0f}, max {r['len_max']}" for r in rows_ep[-3:]))

        # truth frames in episodes, and the temporal fill on them
        for route, pat in (("fit", "video{v:02d}_labelregion"), ("pca", "video{v:02d}_pca_labelregion")):
            T = pd.read_csv(FR / pat.format(v=v) / "truth_frames.csv")
            T = T[T.flag & (T.status != "closed")]
            ep = E[route]
            f = flags[route]
            for _, r in T.iterrows():
                i = int(r.frame)
                s, e = episode_of(ep, i)
                hr = truth.loc[i]
                if route == "fit":
                    over = [b for b in BPS if dev[i, BPS.index(b)] > THR]
                else:
                    over = [BPS[int(np.nanargmax(np.nan_to_num(res[i], nan=-1)))]]
                fe = fill_errors(xy, hr, i, s, e, over, pm, G, n)
                fe_all = fill_errors(xy, hr, i, s, e, list(BPS), pm, G, n)
                rows_truth.append({"video": v, "route": route, "frame": i, "wrong": bool(r.error_before > LINE), "status": r.status, "episode_start": int(s), "episode_len": int(e - s + 1),
                                   "position": int(i - s), "over": "+".join(over), "error_before": r.error_before, "error_after_peak2": r.error_after,
                                   "interp": fe["interp"], "nearest": fe["nearest"], "forecast": fe["forecast"],
                                   "interp_all8": fe_all["interp"], "nearest_all8": fe_all["nearest"]})
        # label-free bound: interpolation error across a gap of L frames on unflagged stretches (either finder)
        good = ~flags["either"]
        pts = np.stack([xy[b] for b in P4], axis=1)  # (n, 4, 2)
        for L in GAPS:
            a = np.arange(0, n - L - 1)
            ok = good[a] & good[a + L + 1]
            # the whole stretch unflagged, checked with a cumulative sum
            cs = np.r_[0, np.cumsum(~good)]
            ok &= (cs[a + L + 2] - cs[a]) == 0
            idx = np.flatnonzero(ok)
            if len(idx) > 4000:
                idx = rng.choice(idx, 4000, replace=False)
            if len(idx) == 0:
                continue
            mid = a[idx] + (L + 1) // 2
            t = (mid - a[idx]) / (L + 1)
            pred = (1 - t)[:, None, None] * pts[a[idx]] + t[:, None, None] * pts[a[idx] + L + 1]
            err = np.linalg.norm(pred - pts[mid], axis=2) / G.Wm  # (m, 4)
            fr = np.sqrt((err ** 2).mean(axis=1))
            rows_gap.append({"video": v, "gap": L, "pairs": len(idx), "point_err_median": float(np.nanmedian(err)), "point_err_p90": float(np.nanpercentile(err, 90)),
                             "frame_err_median": float(np.nanmedian(fr)), "frame_err_p90": float(np.nanpercentile(fr, 90)), "frame_err_over_line_pct": float(100 * (fr > LINE).mean())})

    Ep = pd.DataFrame(rows_ep)
    Ep.to_csv(OUT / "episodes_by_video.csv", index=False)
    tot = Ep.groupby("finder").agg(flagged=("flagged", "sum"), episodes=("episodes", "sum"), len_max=("len_max", "max"), singletons=("singletons", "sum"),
                                   episodes_ge10=("episodes_ge10", "sum"), episodes_ge50=("episodes_ge50", "sum"))
    print(tot.to_string())
    Tr = pd.DataFrame(rows_truth)
    Tr.round(4).to_csv(OUT / "truth_in_episodes.csv", index=False)
    Tr.round(4).to_csv(OUT / "temporal_fill_truth.csv", index=False)
    Gp = pd.DataFrame(rows_gap)
    Gp.round(4).to_csv(OUT / "displacement_vs_gap_by_video.csv", index=False)
    Gs = Gp.groupby("gap").apply(lambda g: pd.Series({"videos": len(g), "pairs": g.pairs.sum(), "point_err_median": np.average(g.point_err_median, weights=g.pairs),
                                                       "point_err_p90_median_video": g.point_err_p90.median(), "frame_err_median": np.average(g.frame_err_median, weights=g.pairs),
                                                       "frame_err_p90_worst_video": g.frame_err_p90.max(), "frame_err_over_line_pct": np.average(g.frame_err_over_line_pct, weights=g.pairs)}))
    Gs.round(4).to_csv(OUT / "displacement_vs_gap.csv")
    print(Gs.round(4).to_string())

    # summary of the fill test
    def summ(g):
        o = {"frames": len(g)}
        for k in ("error_before", "error_after_peak2", "interp", "nearest", "forecast", "interp_all8"):
            o[f"{k}_median"] = g[k].median()
            o[f"{k}_wrong_after"] = int((g[k] > LINE).sum())
            o[f"{k}_nan"] = int(g[k].isna().sum())
        return pd.Series(o)
    Tr["len_bin"] = pd.cut(Tr.episode_len, [0, 1, 3, 10, 30, 100, 10000], labels=["1", "2-3", "4-10", "11-30", "31-100", ">100"])
    S = Tr.groupby(["route", "wrong"]).apply(summ)
    S.round(4).to_csv(OUT / "temporal_fill_summary.csv")
    print(S.round(3).T.to_string())
    S2 = Tr[Tr.wrong].groupby(["route", "len_bin"], observed=True).apply(summ)
    S2.round(4).to_csv(OUT / "temporal_fill_by_length.csv")
    print(S2[["frames", "error_before_median", "error_after_peak2_median", "interp_median", "nearest_median", "forecast_median"]].round(3).to_string())

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    # episode lengths
    fig, ax = plt.subplots(figsize=(8, 4), constrained_layout=True)
    bins = np.logspace(0, np.log10(2000), 40)
    for name, col in (("fit", "#2f6db3"), ("pca", "#d1495b")):
        Ls = []
        for v in strips:
            F = pd.read_csv(FR / f"video{v:02d}_pca_labelregion" / "frames_all.csv.gz", usecols=[FINDERS[name]])
            ep = episodes(F[FINDERS[name]].to_numpy(bool))
            Ls.append(ep[:, 1] - ep[:, 0] + 1)
        Ls = np.concatenate(Ls)
        ax.hist(Ls, bins=bins, histtype="step", lw=1.8, color=col, label=f"{'fitting' if name == 'fit' else 'PCA'}: {len(Ls)} episodes, median {np.median(Ls):.0f} frames, longest {Ls.max()}")
    ax.set_xscale("log")
    ax.set_xlabel("episode length (consecutive flagged frames), 60 frames = 1 s")
    ax.set_ylabel("episodes, all 19 videos")
    ax.set_title("Flagged frames come in episodes")
    ax.legend(frameon=False, fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)
    fig.savefig(OUT / "episode_lengths.png", dpi=150)
    # timeline strips
    vids = sorted(strips)
    fig, axes = plt.subplots(len(vids), 1, figsize=(12, 0.55 * len(vids) + 1), constrained_layout=True, sharex=True)
    for ax, v in zip(axes, vids):
        M = np.vstack([strips[v]["fit"], strips[v]["pca"]])
        ax.imshow(M, aspect="auto", cmap="Greys", vmin=0, vmax=1, interpolation="nearest", extent=(0, 100, 2, 0))
        ax.set_yticks([0.5, 1.5])
        ax.set_yticklabels(["fitting", "PCA"], fontsize=7)
        ax.set_ylabel(f"video {v}", rotation=0, ha="right", va="center", fontsize=8)
        ax.tick_params(length=0)
        for s in ax.spines.values():
            s.set_visible(False)
    axes[-1].set_xlabel("position in the video (%); darkness = share of the bin's frames flagged (600 bins per video)")
    fig.suptitle("Where the flagged frames are in each video", fontsize=11)
    fig.savefig(OUT / "timeline.png", dpi=130)
    # temporal fill figure: per wrong truth frame, error under each treatment, by episode length
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.3), constrained_layout=True)
    for ax, route in zip(axes, ("fit", "pca")):
        g = Tr[(Tr.route == route) & Tr.wrong].sort_values("episode_len")
        x = np.arange(len(g))
        for k, lab, col, mk in (("error_before", "before", "0.5", "x"), ("error_after_peak2", "second-peak repair", "#2f6db3", "o"), ("interp", "interpolation across the episode", "#e0a458", "s"),
                                ("nearest", "nearest unflagged frame copied", "#7fb069", "^"), ("forecast", "one-step forecast", "#d1495b", "v")):
            ax.plot(x, 100 * g[k], mk, color=col, label=lab, ms=6, alpha=0.85, mfc="none" if k == "error_before" else col)
        ax.axhline(100 * LINE, color="k", ls="--", lw=1)
        ax.set_xticks(x)
        ax.set_xticklabels([f"v{int(r.video)}\n{int(r.episode_len)}" for _, r in g.iterrows()], fontsize=7)
        ax.set_xlabel("wrong truth frame (video, episode length in frames), sorted by episode length")
        ax.set_ylabel("frame error against the human label (% of eye width)")
        ax.set_title(f"{'fitting' if route == 'fit' else 'PCA'} finder: the {len(g)} wrong truth frames it flagged")
        ax.set_yscale("log")
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", color="0.92")
        ax.set_axisbelow(True)
    axes[0].legend(frameon=False, fontsize=8, loc="lower right")
    fig.suptitle("Filling a flagged frame from the frames around its episode, against the second-peak repair (dashed = the wrong-frame line)", fontsize=10)
    fig.savefig(OUT / "temporal_fill.png", dpi=150)
    # displacement vs gap
    fig, ax = plt.subplots(figsize=(7, 4), constrained_layout=True)
    ax.plot(Gs.index, 100 * Gs.frame_err_median, "o-", color="#2f6db3", label="median over all unflagged stretches")
    ax.plot(Gs.index, 100 * Gs.frame_err_p90_worst_video, "s--", color="#d1495b", label="90th percentile, worst video")
    ax.axhline(100 * LINE, color="k", ls="--", lw=1, label="wrong-frame line")
    ax.set_xscale("log")
    ax.set_xlabel("gap bridged by linear interpolation (frames), 60 frames = 1 s")
    ax.set_ylabel("error of the interpolated pupil points\nat the middle of the gap (% of eye width)")
    ax.set_title("How far the pupil moves inside a gap: the cost of interpolating across it")
    ax.legend(frameon=False, fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(color="0.92")
    ax.set_axisbelow(True)
    fig.savefig(OUT / "displacement_vs_gap.png", dpi=150)
    print("figures written to", OUT)


if __name__ == "__main__":
    main()
