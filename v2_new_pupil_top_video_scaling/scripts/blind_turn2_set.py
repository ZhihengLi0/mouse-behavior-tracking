#!/usr/bin/env python3
"""A second blind set, drawn to judge turn 2 against turn 1 of the final loop (experiment 5b), 2026-11-14.

The 70 truth frames and the 34 blind frames were drawn to judge the starting model against turn 1; after turn 1 they
hold almost no wrong frame, so they cannot show whether turn 2 changes anything. This script draws new frames
stratified by what the two turns of the final loop say about them (per-frame finder state, fitting or PCA):

    T1  flagged by the turn-1 model only      turn 2 claims to have fixed them
    TB  flagged by both
    T2  flagged by the turn-2 model only      possible new errors of turn 2
    TN  flagged by neither

Where: only frames within 30 frames of an existing truth frame or blind frame (first draw of 50 and supplement of 10).
Experiments 2, 3, 4c and 5b keep 60 frames on each side of those frames out of every training set, so a frame drawn
here is at least 31 frames (half a second) away from every training frame of those experiments, at every turn, and no
model has to be retrained (one training frame of experiment 2 on video 15 lies inside that band; the frames within 30
of it are left out). Experiment 1 does not keep the blind band free; as before its numbers on blind frames are
not measured under the same condition. Excluded as in blind_test_set.py: 5 frames on each side of every frame that
already has a human label (truth, blind draws, training labels) and of every pseudo-label frame of any student; frames
where the starting model has a missing coordinate. Inside a stratum the frames are spread over the pose space
(k-means on the starting model's eight points / eye width, one frame per cluster). Weight = stratum share of the
eligible frames / frames drawn, so weighted numbers refer to the eligible part of the video.

    python blind_turn2_set.py select --video 1     -> results/blind_test/video01/turn2/selection_key.csv (do not open before labeling),
                                                      selection_summary.csv, labeling/blind/img201.png ...
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import blind_test_set as B  # noqa: E402
from second_peak_repair import load  # noqa: E402
from self_training import mkdir, teacher_eye_width, unit_of  # noqa: E402

STRATA = {"T1": "flagged by the turn-1 model only", "TB": "flagged by both", "T2": "flagged by the turn-2 model only", "TN": "flagged by neither"}
QUOTA = {"T1": 14, "TB": 12, "T2": 10, "TN": 4}
NEAR, MODELS, FIRST_NAME = 30, ("round1_formal", "round2_formal"), 201


def folder(v):
    return B.OUT / f"video{v:02d}" / "turn2"


def flags(v, m):
    F = pd.read_csv(B.RES / "self_training" / f"video{v:02d}" / m / "flags.csv.gz")
    return (F.fit_flag | F.pca_flag).to_numpy(bool)


def select(v, seed):
    rr, dt, P, H, truth, fwd, sig = load(v)
    n = len(dt)
    Wm = teacher_eye_width(dt)
    XY = B.U.pose_matrix(dt)
    f1, f2 = flags(v, MODELS[0]), flags(v, MODELS[1])
    assert len(f1) == len(f2) == n
    anchors = np.r_[truth.index.astype(int).to_numpy(),
                    pd.read_csv(B.OUT / f"video{v:02d}" / "selection_key.csv").frame.astype(int).to_numpy(),
                    pd.read_csv(B.OUT / f"video{v:02d}" / "supplement" / "selection_key.csv").frame.astype(int).to_numpy()]
    pseudo = B.pseudo_frames(v)
    excl = B.band(n, anchors, B.GUARD_LABEL) | B.band(n, B.label_frames(v), B.GUARD_LABEL) | B.band(n, pseudo, B.GUARD_PSEUDO) | ~np.isfinite(XY).all(axis=(1, 2))
    elig = B.band(n, anchors, NEAR) & ~excl
    # the frames drawn must be free of every training frame of experiments 2, 3, 4c, 5b: at least 31 frames away
    far = []
    for f in sorted((B.RES / "self_training" / f"video{v:02d}").glob("pseudo_*/pseudo_frames.csv")):
        if "every" in f.parent.name:
            continue                                           # experiment 1 does not keep the blind band free
        K = pd.read_csv(f)
        far += K[K.kept].frame.astype(int).tolist()
    close = B.band(n, far, 30) & elig                          # video 15: one experiment-2 turn-1 training frame (35908) lies 39 frames from a blind frame
    elig &= ~close                                             # so every frame drawn is at least 31 frames from every training frame of experiments 2 to 5
    stratum = np.where(f1 & ~f2, "T1", np.where(f1 & f2, "TB", np.where(~f1 & f2, "T2", "TN")))
    whole = {c: int((stratum == c).sum()) for c in STRATA}
    avail = {c: int(((stratum == c) & elig).sum()) for c in STRATA}
    tot = sum(avail.values())
    quota = {c: min(QUOTA[c], avail[c]) for c in STRATA}
    rng = np.random.default_rng(seed)
    rows = []
    for c in STRATA:
        pool = np.flatnonzero((stratum == c) & elig)
        chosen, lab = B.draw(pool, quota[c], XY, Wm, seed, rng)
        rows += [{"frame": int(f), "stratum": c, "cluster": int(k)} for f, k in zip(chosen, lab)]
    sel = pd.DataFrame(rows)
    assert elig[sel.frame].all() and sel.frame.is_unique
    for c in STRATA:
        cnt = int((sel.stratum == c).sum())
        sel.loc[sel.stratum == c, "stratum_frames"] = avail[c]
        sel.loc[sel.stratum == c, "stratum_share"] = avail[c] / tot
        sel.loc[sel.stratum == c, "weight"] = (avail[c] / tot) / cnt if cnt else np.nan
    sel["turn1_flag"], sel["turn2_flag"] = f1[sel.frame], f2[sel.frame]
    sel["nearest_anchor_distance"] = [int(np.abs(anchors - f).min()) for f in sel.frame]
    sel = sel.sample(frac=1, random_state=int(rng.integers(1 << 31))).reset_index(drop=True)
    sel.insert(0, "blind_name", [f"img{i + FIRST_NAME:03d}.png" for i in range(len(sel))])
    sel["video"] = v
    ld = folder(v) / "labeling" / "blind"
    if ld.exists():
        raise SystemExit(f"{ld} exists; remove it by hand for a fresh set")
    ld.mkdir(parents=True)
    out = mkdir(folder(v))
    sel.to_csv(out / "selection_key.csv", index=False)
    pd.DataFrame([{"video": v, "unit": unit_of(v), "frames": n, "models": " / ".join(MODELS), "within_frames_of_truth_or_blind": NEAR, "eligible": int(elig.sum()), "removed_near_a_training_frame": int(close.sum()),
                   **{f"whole_{c}": whole[c] for c in STRATA}, **{f"eligible_{c}": avail[c] for c in STRATA}, **{f"drawn_{c}": int((sel.stratum == c).sum()) for c in STRATA},
                   "eye_width_px": round(Wm, 1), "seed": seed}]).to_csv(out / "selection_summary.csv", index=False)
    B.write_pngs(v, dict(zip(sel.frame.astype(int), sel.blind_name)), ld)
    print(f"video {v}: {len(sel)} frames drawn; " + ", ".join(f"{c} {int((sel.stratum == c).sum())} of {avail[c]} eligible ({whole[c]} in the video)" for c in STRATA))
    print(f"PNGs in {ld}; key in {out / 'selection_key.csv'} (do not open it before labeling)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["select"])
    ap.add_argument("--video", type=int, required=True)
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()
    select(a.video, a.seed)


if __name__ == "__main__":
    main()
