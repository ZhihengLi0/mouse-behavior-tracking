#!/usr/bin/env python3
"""If the model's own predictions, after the finder and the repair, became training labels (self-training), how noisy
would they be? Measured on the 1330 truth frames, where the human label is known.

    python pseudo_label_noise.py

Direction 3 of the 2026-10-31 meeting. A self-training round would take, from every video, frames that no finder
flagged plus frames the repair accepted, and train on the model's coordinates there. On the truth frames those
coordinates have a known error against the human label. Reads results/flag_and_repair/videoNN_pca_labelregion/
truth_frames.csv (both finders' flags, status, error before and after the repair, per-point errors) and the
whole-video frames_all.csv.gz of the same folder (how many frames each class holds). Writes to
results/pseudo_label_noise/: noise.csv (per video: frames and error of each class), noise_total.csv, noise_points.csv
(per point), noise.png. Nothing existing is modified."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from second_peak_repair import BPS, P4, PL, RES  # noqa: E402

OUT = RES / "pseudo_label_noise"
OUT.mkdir(parents=True, exist_ok=True)
FR = RES / "flag_and_repair"
LINE = 0.14
CLASSES = {"unflagged": "no finder flagged it (pseudo label as predicted)", "repaired": "flagged, repaired, accepted (pseudo label = repaired)",
           "passes_check": "flagged, judged plausible as it is (pseudo label as predicted)", "still_flagged": "flagged, not repaired (excluded)", "closed": "eye closed (excluded)"}


def classes(T):
    c = pd.Series("still_flagged", index=T.index)
    c[~(T.flag_fit | T.flag_pca)] = "unflagged"
    c[T.status == "repaired"] = "repaired"
    c[T.status == "flagged_passes_check"] = "passes_check"
    c[T.status == "closed"] = "closed"
    return c


def main():
    rows, pts, allT = [], [], []
    for v in sorted(PL[PL.status == "final"].video.astype(int)):
        T = pd.read_csv(FR / f"video{v:02d}_pca_labelregion" / "truth_frames.csv")
        T["cls"] = classes(T)
        T["video"] = v
        allT.append(T)
        F = pd.read_csv(FR / f"video{v:02d}_pca_labelregion" / "frames_all.csv.gz", usecols=["status", "flag_fit", "flag_pca"])
        F["cls"] = classes(F.assign(flag_fit=F.flag_fit.astype(bool), flag_pca=F.flag_pca.astype(bool)))
        r = {"video": v, "truth_frames": len(T), "whole_frames": len(F)}
        for k in CLASSES:
            g = T[T.cls == k]
            r[f"{k}_truth"] = len(g)
            r[f"{k}_whole_pct"] = round(100 * (F.cls == k).mean(), 2)
            r[f"{k}_median"] = round(float(g.error_after.median()), 4) if len(g) else np.nan
            r[f"{k}_p90"] = round(float(g.error_after.quantile(0.9)), 4) if len(g) else np.nan
            r[f"{k}_over_line"] = int((g.error_after > LINE).sum())
        rows.append(r)
        print(f"video {v:2d}: unflagged {r['unflagged_truth']} truth frames, median {100 * r['unflagged_median']:.2f} %, p90 {100 * r['unflagged_p90']:.2f} %, over line {r['unflagged_over_line']}; "
              f"usable share of the video {r['unflagged_whole_pct'] + r['repaired_whole_pct'] + r['passes_check_whole_pct']:.1f} %")
    N = pd.DataFrame(rows)
    N.to_csv(OUT / "noise.csv", index=False)
    A = pd.concat(allT, ignore_index=True)
    tot = []
    for k, desc in CLASSES.items():
        g = A[A.cls == k]
        tot.append({"class": k, "meaning": desc, "truth_frames": len(g), "whole_video_pct": round(float(np.average(N[f"{k}_whole_pct"], weights=N.whole_frames)), 2),
                    "median_pct": round(100 * float(g.error_after.median()), 2), "p75_pct": round(100 * float(g.error_after.quantile(0.75)), 2), "p90_pct": round(100 * float(g.error_after.quantile(0.9)), 2),
                    "p99_pct": round(100 * float(g.error_after.quantile(0.99)), 2), "over_5pct": int((g.error_after > 0.05).sum()), "over_10pct": int((g.error_after > 0.10).sum()), "over_line": int((g.error_after > LINE).sum())})
    Tt = pd.DataFrame(tot)
    Tt.to_csv(OUT / "noise_total.csv", index=False)
    print(Tt.to_string())
    # per point, on the frames a pseudo label would use
    use = A[A.cls.isin(["unflagged", "repaired", "passes_check"])]
    for b in P4:
        e = use[f"{b}_after"]
        pts.append({"point": b, "frames": int(e.notna().sum()), "median_pct": round(100 * float(e.median()), 2), "p90_pct": round(100 * float(e.quantile(0.9)), 2), "over_line": int((e > LINE).sum())})
    Pp = pd.DataFrame(pts)
    Pp.to_csv(OUT / "noise_points.csv", index=False)
    print(Pp.to_string())

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2), constrained_layout=True)
    bins = np.logspace(np.log10(0.003), np.log10(1.5), 45)
    ax = axes[0]
    for k, col in (("unflagged", "#2f6db3"), ("repaired", "#7fb069"), ("still_flagged", "#d1495b")):
        g = A[A.cls == k].error_after
        ax.hist(g, bins=bins, histtype="step", lw=1.8, color=col, label=f"{k.replace('_', ' ')}: {len(g)} truth frames, median {100 * g.median():.1f} %")
    ax.axvline(LINE, color="k", ls="--", lw=1)
    ax.set_xscale("log")
    ax.set_xlabel("error of the would-be pseudo label against the human label (% of eye width)")
    ax.set_ylabel("truth frames")
    ticks = [0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0]
    ax.set_xticks(ticks)
    ax.set_xticklabels([f"{100 * t:g}" for t in ticks])
    ax.set_title("Noise a self-training round would start from")
    ax.legend(frameon=False, fontsize=8)
    ax.spines[["top", "right"]].set_visible(False)
    ax = axes[1]
    x = N.video
    ax.bar(x - 0.2, 100 * N.unflagged_median, width=0.4, color="#2f6db3", label="median, unflagged truth frames")
    ax.bar(x + 0.2, 100 * N.unflagged_p90, width=0.4, color="#9bb7d4", label="90th percentile")
    ax.set_xticks(x)
    ax.set_xlabel("video")
    ax.set_ylabel("error (% of eye width)")
    ax.set_title("Per video, the frames no finder flagged")
    ax.legend(frameon=False, fontsize=8)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color="0.92")
    ax.set_axisbelow(True)
    fig.savefig(OUT / "noise.png", dpi=150)
    print("figure", OUT / "noise.png")


if __name__ == "__main__":
    main()
