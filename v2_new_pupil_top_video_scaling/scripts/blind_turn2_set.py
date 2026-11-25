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

--turn 3 (2026-11-24, user: "the turn-3 check is drawn on its own, exactly as every earlier turn"): the same draw for
turn 3 against turn 2 (models round2_formal / round3_formal, strata T2 / TB / T3 / TN with the same quotas 14 / 12 / 10 /
4), in results/blind_test/videoNN/turn3/, PNGs from img401. Anchors also include the reserve centres of the videos
without a blind set (final_check_set.py reserve; every turn keeps 60 frames on each side out of training, like the blind
frames), and 5 frames on each side of every frame already drawn for an earlier check (turn2/, final/) are excluded.
Then, as the user decided for turn 2, a random subset per stratum is kept for labeling (10 / 8 / 7 / 3; its random generator is seeded with --seed + 1, 43 by default, so that it is
independent of the draw's): the key
gets `subset` and `weight_subset` (stratum share / kept count), the other PNGs move to labeling/not_labeled/.
Without --turn (or --turn 2) the script does exactly what it did on 2026-11-14.
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
TURN = 2
SUBSET = (10, 8, 7, 3)                                         # kept for labeling per stratum, in the order of STRATA (user, 2026-11-14)


def set_turn(t):
    """Point the module at turn t against turn t-1 (t = 2 leaves everything as it was)."""
    global STRATA, QUOTA, MODELS, FIRST_NAME, TURN
    p, c = f"T{t - 1}", f"T{t}"
    STRATA = {p: f"flagged by the turn-{t - 1} model only", "TB": "flagged by both", c: f"flagged by the turn-{t} model only", "TN": "flagged by neither"}
    QUOTA = dict(zip(STRATA, (14, 12, 10, 4)))
    MODELS, FIRST_NAME, TURN = (f"round{t - 1}_formal", f"round{t}_formal"), 200 * (t - 1) + 1, t


def folder(v):
    return B.OUT / f"video{v:02d}" / f"turn{TURN}"


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
    if TURN == 2:
        anchors = np.r_[truth.index.astype(int).to_numpy(),
                        pd.read_csv(B.OUT / f"video{v:02d}" / "selection_key.csv").frame.astype(int).to_numpy(),
                        pd.read_csv(B.OUT / f"video{v:02d}" / "supplement" / "selection_key.csv").frame.astype(int).to_numpy()]
        drawn_before = []
    else:   # blind + supplement (videos 1, 6, 8, 15) or reserve centres (the other 15 videos): the frames every turn keeps 60 frames away from
        keys = [B.OUT / f"video{v:02d}" / s / "selection_key.csv" for s in ("", "supplement", "reserve")]
        anchors = np.r_[truth.index.astype(int).to_numpy(), *[pd.read_csv(k).frame.astype(int).to_numpy() for k in keys if k.exists()]]
        assert len(anchors) > len(truth), f"video {v}: no blind or reserve centres"
        drawn_before = [f for k in (B.OUT / f"video{v:02d}" / f"turn{t}" / "selection_key.csv" for t in range(2, TURN)) if k.exists() for f in pd.read_csv(k).frame.astype(int)]
        fk = B.OUT / f"video{v:02d}" / "final" / "selection_key.csv"
        drawn_before += pd.read_csv(fk).frame.astype(int).tolist() if fk.exists() else []
    pseudo = B.pseudo_frames(v)
    excl = B.band(n, anchors, B.GUARD_LABEL) | B.band(n, B.label_frames(v), B.GUARD_LABEL) | B.band(n, pseudo, B.GUARD_PSEUDO) | ~np.isfinite(XY).all(axis=(1, 2))
    if drawn_before:
        excl |= B.band(n, drawn_before, B.GUARD_LABEL)
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
    sp, _, sc, _ = STRATA                                      # T1 TB T2 TN for turn 2; T2 TB T3 TN for turn 3
    stratum = np.where(f1 & ~f2, sp, np.where(f1 & f2, "TB", np.where(~f1 & f2, sc, "TN")))
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
    sel[f"turn{TURN - 1}_flag"], sel[f"turn{TURN}_flag"] = f1[sel.frame], f2[sel.frame]
    if TURN > 2:   # 31 frames from every training frame of the final loop, at every turn (asserted, not only filtered)
        assert all(np.abs(np.asarray(far) - f).min() >= 31 for f in sel.frame), "a drawn frame is closer than 31 to a training frame"
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
    if TURN > 2:
        keep_subset(v, sel, out, ld, seed)
    print(f"video {v}: {len(sel)} frames drawn; " + ", ".join(f"{c} {int((sel.stratum == c).sum())} of {avail[c]} eligible ({whole[c]} in the video)" for c in STRATA))
    print(f"PNGs in {ld}; key in {out / 'selection_key.csv'} (do not open it before labeling)")


def keep_subset(v, sel, out, ld, seed):
    """The user's turn-2 decision applied to a later turn: a random subset per stratum (SUBSET, fewer when fewer were
    drawn) is labeled; weight_subset = stratum share / kept count; the other PNGs move to labeling/not_labeled/."""
    rng = np.random.default_rng(seed + 1)
    keep = []
    for c, k in zip(STRATA, SUBSET):
        g = sel[sel.stratum == c].blind_name.tolist()
        keep += list(rng.choice(g, size=min(k, len(g)), replace=False)) if g else []
    sel["subset"] = sel.blind_name.isin(keep)
    for c in STRATA:
        cnt = int((sel.subset & (sel.stratum == c)).sum())
        sel.loc[sel.stratum == c, "weight_subset"] = sel.loc[sel.stratum == c, "stratum_share"] / cnt if cnt else np.nan
    sel.loc[~sel.subset, "weight_subset"] = np.nan
    sel.to_csv(out / "selection_key.csv", index=False)
    nl = mkdir(ld.parent / "not_labeled")
    for nm in sel.blind_name[~sel.subset]:
        (ld / nm).rename(nl / nm)
    S = pd.read_csv(out / "selection_summary.csv")
    for c in STRATA:
        S[f"kept_{c}"] = int((sel.subset & (sel.stratum == c)).sum())
    S.to_csv(out / "selection_summary.csv", index=False)
    assert len(list(ld.glob("img*.png"))) == len(keep)
    print(f"video {v}: {len(keep)} frames kept for labeling (" + ", ".join(f"{c} {int((sel.subset & (sel.stratum == c)).sum())}" for c in STRATA) + f"), the others in {nl}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["select"])
    ap.add_argument("--video", type=int, required=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--turn", type=int, default=2, help="the turn checked against the one before (default 2, as drawn on 2026-11-14)")
    a = ap.parse_args()
    set_turn(a.turn)
    select(a.video, a.seed)


if __name__ == "__main__":
    main()
