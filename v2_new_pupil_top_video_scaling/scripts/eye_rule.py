#!/usr/bin/env python3
"""A parallel fitting rule for the 4 eyelid / corner points next to fit_pupil (page section 8.2).

fit_pupil averages the fitting deviation of the 4 pupil points only, so an error on an eyelid or corner point enters
the score only through the pupil. This script evaluates the rule "flag if fit_pupil >= t_p OR fit_eye >= t_e" on the
1330 step-8 frames, with both thresholds chosen leave-one-video-out for a target flagged share, and compares it with
fit_pupil alone at the same total flagged share. Reads results/unreliable_frames_sweep/signals_all_frames.csv and
results/error_detector/truth_pairs.csv only; writes results/eye_rule/.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from scale_step import BPS, HERE  # noqa: E402

OUT = HERE / "results" / "eye_rule"
OUT.mkdir(parents=True, exist_ok=True)
EYE, PUPIL = BPS[4:], BPS[:4]


def threshold_for_share(values, share):
    """Smallest t such that mean(values >= t) <= share."""
    return float(np.quantile(values, 1 - share))


def lovo_flags(E, col, share):
    """Flag with a threshold chosen on the other 18 videos for the given flagged share."""
    f = np.zeros(len(E), bool)
    for v in E.video.unique():
        t = threshold_for_share(E.loc[E.video != v, col].values, share)
        m = (E.video == v).values
        f[m] = E.loc[m, col].values >= t
    return pd.Series(f, index=E.index)


def summary(name, f, E, eye_idx):
    w20, w50 = E.wrong20.astype(bool), E.wrong50.astype(bool)
    return {"rule": name, "flagged_pct": 100 * f.mean(), "recall20_pct": 100 * w20[f].sum() / w20.sum(),
            "precision20_pct": 100 * w20[f].mean(), "recall50_pct": 100 * w50[f].sum() / w50.sum(),
            "eye_wrong_caught": int(f[eye_idx].sum()), "eye_wrong_total": len(eye_idx),
            "pupil_wrong_caught": int(f[w20 & ~E.index.isin(eye_idx)].sum()), "pupil_wrong_total": int((w20 & ~E.index.isin(eye_idx)).sum())}


def main():
    S = pd.read_csv(HERE / "results" / "unreliable_frames_sweep" / "signals_all_frames.csv")
    T = pd.read_csv(HERE / "results" / "error_detector" / "truth_pairs.csv")
    T = T[T["eval"] == True] if "eval" in T else T  # noqa: E712
    errc = [f"err_{b}" for b in BPS]
    E = S.merge(T[["video", "frame"] + errc], on=["video", "frame"])
    assert len(E) == 1330, len(E)
    er = np.nan_to_num(E[errc].values, nan=-1)
    E["max_point"] = [BPS[i] for i in er.argmax(1)]
    w20 = E.wrong20.astype(bool)
    eye_idx = E.index[w20 & E.max_point.isin(EYE)]
    print(f"{len(E)} frames, {int(w20.sum())} wrong20; largest error on an eyelid/corner point in {len(eye_idx)}:",
          E.loc[eye_idx, "max_point"].value_counts().to_dict())
    rows = []
    for p_share in (0.05, 0.092):
        fp = lovo_flags(E, "fit_pupil", p_share)
        rows.append(summary(f"fit_pupil alone, {p_share:.1%}", fp, E, eye_idx))
        for e_share in (0.01, 0.015, 0.025, 0.05):
            fe = lovo_flags(E, "fit_eye", e_share)
            f = fp | fe
            rows.append(summary(f"fit_pupil {p_share:.1%} OR fit_eye {e_share:.1%}", f, E, eye_idx))
            # fit_pupil alone at the same total share, for a fair comparison
            rows.append(summary(f"  fit_pupil alone at {f.mean():.1%}", lovo_flags(E, "fit_pupil", f.mean()), E, eye_idx))
    R = pd.DataFrame(rows)
    R.to_csv(OUT / "eye_rule_table.csv", index=False, float_format="%.1f")
    pd.set_option("display.width", 200)
    print(R.round(1).to_string(index=False))
    # thresholds on all 19 videos, for the record
    for col, share in (("fit_pupil", 0.05), ("fit_pupil", 0.092), ("fit_eye", 0.01), ("fit_eye", 0.015), ("fit_eye", 0.025)):
        print(f"threshold on all videos: {col} for {share:.1%} flagged = {threshold_for_share(E[col].values, share):.1f} px")
    E.loc[eye_idx, ["video", "frame", "set", "frame_rmse_px", "max_point", "fit_pupil", "fit_eye", "min_conf", "jump"] + errc].round(1) \
        .to_csv(OUT / "eye_wrong_frames.csv", index=False)


if __name__ == "__main__":
    main()
