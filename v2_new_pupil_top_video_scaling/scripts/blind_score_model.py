"""Blind-frame score of one student model that blind_test_set.py's score command does not list (experiment 4c:
round1_reopened, round2_reopened), without touching results/blind_test/: the 24 scored blind frames and the 10
supplement frames, teacher and the model, written to results/self_training/videoNN/<model>/blind_scores.csv in the
format of the existing round1_reopened/blind_scores.csv (set, model, frames, median_error_pct, wrong_frames).
Same labels, same error (frame RMSE in teacher eye widths) and same line as blind_test_set.py (its functions are used).

  python blind_score_model.py --video 1 --model round2_reopened [--out other.csv]
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

import blind_test_set as B


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", type=int, required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    v = a.video
    h5 = sorted((B.LOCAL / f"video{v:02d}" / a.model / "predictions").glob("*.h5"))[-1]
    models = {"teacher": B.teacher_h5(v), a.model: h5}
    rows = []
    for name, sup in (("main", False), ("supplement", True)):
        B.SUP = sup
        E, _ = B.errors(v, B.read_labels(v), models)
        for m, g in E.groupby("model", sort=False):          # unweighted median and count over the scored frames, as in the existing file
            g = g[np.isfinite(g.error)]
            rows.append({"set": name, "model": m, "frames": len(g), "median_error_pct": round(100 * float(g.error.median()), 2), "wrong_frames": int((g.error > B.LINE).sum())})
    out = Path(a.out) if a.out else B.RES / "self_training" / f"video{v:02d}" / a.model / "blind_scores.csv"
    pd.DataFrame(rows).to_csv(out, index=False)
    print(pd.DataFrame(rows).to_string(index=False))


if __name__ == "__main__":
    main()
