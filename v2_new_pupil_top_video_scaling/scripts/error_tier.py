#!/usr/bin/env python3
"""Where does a prediction error stop being noise and become a wrong frame? One line, in units of eye width.

    python error_tier.py

Decisions (user, 2026-10-30): errors are measured in units of the video's eye width (median distance between the two
eye corners in the plateau model's predictions, label-free), so that the same numbers hold on another camera or
another distance; the 20-px level is label noise and is not reported any more; the only line is the one that separates
the two groups visible in the distribution of the truth-frame errors: the bulk of frames whose error is a few per
cent of the eye width, and the frames whose point was thrown far. The line is read off the histogram by eye, the way
the blink threshold was: the main group ends at about 13 % of the eye width (no frame between 12.7 % and 15.4 %), the
sparse tail of thrown points starts after it, so LINE = 0.14 of the eye width (about 60 px on these videos).

Outputs in results/error_tier/: eye_width.csv (per video), truth_errors.csv (1330 frames, error in px and in eye
widths), tier.csv (the line, its pixel equivalent per video, the number of wrong frames), error_histogram.png.
Nothing existing is modified."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from second_peak_repair import PL, RES, predictions  # noqa: E402

OUT = RES / "error_tier"
OUT.mkdir(parents=True, exist_ok=True)
LINE = 0.14


def eye_widths():
    rows = []
    for v in sorted(PL[PL.status == "final"].video.astype(int)):
        rr, d = predictions(v)
        w = np.hypot(d["eye_temporal_corner"]["x"] - d["eye_nasal_corner"]["x"], d["eye_temporal_corner"]["y"] - d["eye_nasal_corner"]["y"])
        rows.append({"video": v, "unit": rr.unit, "mouse": rr.mouse, "eye_width_px": round(float(np.nanmedian(w)), 1), "frames": len(d)})
    return pd.DataFrame(rows)


def main():
    E = eye_widths()
    E.to_csv(OUT / "eye_width.csv", index=False)
    L = pd.read_csv(RES / "error_detector" / "scores_lovo.csv")
    L = L[L.plateau].drop_duplicates(["video", "frame"])[["video", "frame", "set", "frame_rmse_px"]]
    L = L.merge(E[["video", "eye_width_px"]], on="video")
    L["error_rel"] = L.frame_rmse_px / L.eye_width_px
    L.round(4).to_csv(OUT / "truth_errors.csv", index=False)
    line = LINE
    x = np.sort(L.error_rel.to_numpy())
    lo, hi = x[x <= line].max(), x[x > line].min()
    n_wrong = int((L.error_rel > line).sum())
    T = pd.DataFrame([{"line_eye_widths": line, "last_frame_below": round(float(lo), 4), "first_frame_above": round(float(hi), 4),
                       "line_px_min": round(line * E.eye_width_px.min(), 1), "line_px_median": round(line * E.eye_width_px.median(), 1), "line_px_max": round(line * E.eye_width_px.max(), 1),
                       "truth_frames": len(L), "wrong_frames": n_wrong, "wrong_frames_gt50px": int((L.frame_rmse_px > 50).sum()),
                       "wrong_per_video": " ".join(f"{int(v)}:{int(n)}" for v, n in L[L.error_rel > line].groupby("video").size().items())}])
    T.to_csv(OUT / "tier.csv", index=False)
    print(T.T.to_string())

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    bins = np.logspace(np.log10(0.002), np.log10(1.5), 50)
    fig, ax = plt.subplots(figsize=(9, 4.2), constrained_layout=True)
    ax.hist(L.error_rel[L.error_rel <= line], bins=bins, color="#9bb7d4", label=f"{len(L) - n_wrong} frames: the main group")
    ax.hist(L.error_rel[L.error_rel > line], bins=bins, color="#d1495b", label=f"{n_wrong} frames: a point thrown far (wrong frame)")
    ax.axvline(line, color="0.25", lw=1.2, ls="--")
    ax.text(line * 1.08, ax.get_ylim()[1] * 0.55, f"line: {100 * line:.0f} % of eye width\n≈ {line * E.eye_width_px.median():.0f} px on these videos", fontsize=9, color="0.25", va="top")
    ax.set_xscale("log")
    ax.set_xlabel("frame error of the plateau model against the human labels, in units of the eye width (log scale)")
    ax.set_ylabel("truth frames")
    ax.set_title("Prediction errors of the 1330 truth frames: a main group and a tail of thrown points")
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color="0.92")
    ax.set_axisbelow(True)
    ax.legend(frameon=False, fontsize=9, loc="upper left")
    ticks = [0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0]
    ax.set_xticks(ticks)
    ax.set_xticklabels([f"{100 * t:g} %" for t in ticks])
    fig.savefig(OUT / "error_histogram.png", dpi=150)
    print("figure", OUT / "error_histogram.png")


if __name__ == "__main__":
    main()
