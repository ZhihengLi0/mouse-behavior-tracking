#!/usr/bin/env python3
"""Figure 8-4 of the report page: recall against flagged share for the signals and detectors that were tried for
finding unreliable frames (section 8.2). Reads results/unreliable_frames_sweep/ and results/error_detector/ only;
writes results/error_detector/detectors_page.png."""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from scale_step import HERE  # noqa: E402
from unreliable_frames_sweep import auc, roc  # noqa: E402

SWEEP = HERE / "results" / "unreliable_frames_sweep"
DET = HERE / "results" / "error_detector"

S = pd.read_csv(SWEEP / "signals_all_frames.csv")
D1 = pd.read_csv(DET / "scores_lovo.csv")
D2 = pd.read_csv(DET / "scores_image.csv")
key = ["video", "frame"]
G = S.merge(D1[key + ["d1_pairs_all"]], on=key).merge(D2[key + ["d2_frame"]], on=key)
assert len(G) == len(S) == 1330

CURVES = [  # (label, column, sign, colour, line width)
    ("fitting, 4 pupil points (fit_pupil)", "fit_pupil", 1, "#7f3c1f", 2.4),
    ("fitting, 8 points (step 8)", "fitting", 1, "#d98b5f", 1.6),
    ("jump", "jump", 1, "#1f77b4", 1.6),
    ("uncertain (lowest confidence)", "min_conf", -1, "#2ca02c", 1.6),
    ("pose-PCA reconstruction error", "pca_video", 1, "#9467bd", 1.6),
    ("disagreement between the video's models", "ens_sdw_max", 1, "#8c564b", 1.6),
    ("D1: gradient boosting on 93 features", "d1_pairs_all", 1, "#e377c2", 1.6),
    ("D2: ResNet-18 on the eye crop", "d2_frame", 1, "#7f7f7f", 1.6),
]

fig, ax = plt.subplots(1, 2, figsize=(15, 6.2))
for j, (w, title) in enumerate((("wrong20", "A  frame RMSE > 20 px (155 of 1330 frames)"), ("wrong50", "B  frame RMSE > 50 px (32 of 1330 frames)"))):
    a = ax[j]
    for label, col, sg, c, lw in CURVES:
        s = sg * G[col]
        R = roc(s, G[w])
        a.plot(np.r_[0, R.flagged * 100], np.r_[0, R.recall * 100], color=c, lw=lw, label=f"{label}  (AUC {auc(s[G[w]], s[~G[w]]):.2f})")
    for x in (5, 10):
        a.axvline(x, color="0.55", ls=":", lw=1)
    a.text(5.3, 2, "5%", color="0.4", fontsize=9); a.text(10.3, 2, "10%", color="0.4", fontsize=9)
    a.set_xlim(0, 60); a.set_ylim(0, 101)
    a.set_xlabel("frames flagged (% of the 1330 frames)"); a.set_ylabel("recall (% of the wrong frames that are flagged)")
    a.set_title(title, fontsize=11.5, loc="left"); a.grid(alpha=0.3)
    a.legend(fontsize=8.5, loc="lower right", framealpha=0.95)
fig.suptitle("Finding unreliable frames: recall against the share of frames flagged, over every threshold of every signal\n"
             "1330 test + validation frames of 19 videos, plateau-point model of each video; the fitted detectors D1 and D2 are scored on videos they were not fitted on", fontsize=10.5)
fig.tight_layout()
fig.savefig(DET / "detectors_page.png", dpi=150)
print("written", DET / "detectors_page.png")
