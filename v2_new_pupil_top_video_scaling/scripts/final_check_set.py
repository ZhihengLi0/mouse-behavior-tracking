#!/usr/bin/env python3
"""Hand-labeled check of the final loop (experiment 5b) on the videos that have no blind set (2026-11-16).

Kaiwen 2026-11-16: run the loop on every video. The 34 blind frames and the 28 frames of the turn-2 check exist only
for videos 1, 8, 15, 6. For the other videos one draw per video, made after turn 2, serves both questions (user
2026-11-16): does turn 1 reduce the wrong frames of the starting model, and does turn 2 reduce those of turn 1.

Three steps:

    python final_check_set.py reserve --video V   BEFORE the loop runs on the video: centre frames whose surroundings (60
                                                  frames on each side) no turn may train on, the same condition as the 60
                                                  blind-set frames of the four videos. 60 centres per 71,748 frames (fewer
                                                  on a shorter video), half on frames the starting model flags (fitting or
                                                  PCA), half on frames it does not; inside each half spread over the pose
                                                  space (k-means, one frame per cluster); at least 121 frames from every
                                                  truth frame and from each other. No image, no label: a table of frame
                                                  numbers that self_training.blind_frames reads.
                                                  -> results/blind_test/videoNN/reserve/selection_key.csv, summary.csv

    python final_check_set.py select --video V    AFTER turn 2: the frames to label, in five strata by the per-frame
                                                  finder state (fitting or PCA) of the starting model, turn 1 and turn 2:
                                                      S   starting model only        turn 1 claims to have fixed them
                                                      T1  turn 1, not turn 2         turn 2 claims to have fixed them
                                                      TB  turn 1 and turn 2          never fixed
                                                      T2  turn 2, not turn 1         possible new errors of turn 2
                                                      N   none of the three          control
                                                  Only frames within 30 frames of a truth frame or a reserved centre, so
                                                  every frame is at least 31 frames from every training frame of the loop
                                                  (checked); 5 frames on each side of every human-labeled frame and frames
                                                  with a missing starting-model coordinate are left out. Inside a stratum
                                                  the frames are spread over the pose space. Weight = stratum share of the
                                                  eligible frames / frames drawn.
                                                  -> results/blind_test/videoNN/final/selection_key.csv (do not open before
                                                     labeling), selection_summary.csv, labeling/blind/img301.png ...

    python final_check_set.py score --video V     after labeling: starting model and every loop turn that has a prediction,
                                                  same error (RMSE over the labeled points / the starting model's median
                                                  eye width) and line as blind_test_set.py.
                                                  -> final/per_frame.csv, scores.csv
                                                  --folder turn2 --no-write: recompute the turn-2 check of the four videos
                                                  without writing (the check that this scorer equals blind_turn2_score.py)
"""
import argparse
import glob
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import blind_test_set as B  # noqa: E402
from second_peak_repair import load  # noqa: E402
from self_training import FINAL, GUARD, blind_frames, mkdir, teacher_eye_width, unit_of  # noqa: E402

FULL, CENTRES = 71748, 60                    # 60 centres on a video of the usual length, in proportion on a shorter one
SPACING = 2 * GUARD + 1                      # centres at least 121 frames from each other and from the truth frames
NEAR = 30                                    # frames are drawn within 30 frames of a truth frame or a centre
STRATA = {"S": "flagged by the starting model only", "T1": "flagged by turn 1, not by turn 2", "TB": "flagged by turn 1 and turn 2",
          "T2": "flagged by turn 2, not by turn 1", "N": "flagged by none of the three"}
QUOTA = {"S": 10, "T1": 8, "TB": 5, "T2": 7, "N": 4}
MODELS, FIRST_NAME = ("round1_formal", "round2_formal"), 301
ST = B.RES / "self_training"


def spaced(pool, k, XY, Wm, seed, rng, taken):
    """Up to k frames of a pool, one per pose cluster where possible, each at least SPACING frames from every frame in
    `taken` (which grows); clusters that have no such frame are made up from the rest of the pool."""
    from sklearn.cluster import KMeans
    out = []
    if not len(pool) or k <= 0:
        return out
    ok = lambda f: all(abs(int(f) - t) >= SPACING for t in taken)
    if len(pool) > k:
        lab = KMeans(n_clusters=k, random_state=seed, n_init=4).fit_predict((XY[pool] / Wm).reshape(len(pool), -1))
    else:
        lab = np.arange(len(pool))
    for c in rng.permutation(np.unique(lab)):
        for f in rng.permutation(pool[lab == c]):
            if ok(f):
                out.append((int(f), int(c)))
                taken.append(int(f))
                break
    if len(out) < k:
        for f in rng.permutation(pool):
            if len(out) >= k:
                break
            if ok(f):
                out.append((int(f), -1))
                taken.append(int(f))
    return out


def reserve(v, seed):
    out = B.OUT / f"video{v:02d}" / "reserve"
    if (B.OUT / f"video{v:02d}" / "selection_key.csv").exists():
        raise SystemExit(f"video {v} has a blind set; its bands are reserved already")
    if (out / "selection_key.csv").exists():
        raise SystemExit(f"{out / 'selection_key.csv'} exists; the reserve of a video is drawn once")
    if list((ST / f"video{v:02d}").glob("pseudo_formal*/pseudo_frames.csv")):
        raise SystemExit(f"video {v}: the final loop has a frame table already; the reserve must be drawn before")
    rr, dt, P, H, truth, fwd, sig = load(v)
    n = len(dt)
    Wm = teacher_eye_width(dt)
    XY = B.U.pose_matrix(dt)
    F = pd.read_csv(B.RES / "flag_and_repair" / FINAL.format(v=v) / "frames_all.csv.gz")
    assert len(F) == n
    flag = F.flag.to_numpy(bool)
    tf = truth.index.astype(int).to_numpy()
    k = max(2, round(CENTRES * n / FULL))
    kf = k // 2
    univ = ~B.band(n, tf, SPACING - 1) & ~B.band(n, B.label_frames(v), NEAR + B.GUARD_LABEL) & np.isfinite(XY).all(axis=(1, 2))
    univ[:GUARD] = False
    univ[n - GUARD:] = False
    rng = np.random.default_rng(seed)
    taken = []
    a = spaced(np.flatnonzero(univ & flag), kf, XY, Wm, seed, rng, taken)
    b = spaced(np.flatnonzero(univ & ~flag), k - len(a), XY, Wm, seed, rng, taken)   # a shortfall of flagged centres is made up here
    sel = pd.DataFrame([{"frame": f, "group": "flagged", "cluster": c} for f, c in a] + [{"frame": f, "group": "unflagged", "cluster": c} for f, c in b])
    sel = sel.sort_values("frame").reset_index(drop=True)
    sel["video"] = v
    fr = sel.frame.to_numpy()
    assert sel.frame.is_unique and (np.diff(fr) >= SPACING).all() and min(abs(int(f) - int(t)) for f in fr for t in tf) >= SPACING
    mkdir(out)
    sel.to_csv(out / "selection_key.csv", index=False)
    band_c, band_t = B.band(n, fr, GUARD), B.band(n, tf, GUARD)
    S = {"video": v, "unit": unit_of(v), "frames": n, "centres": len(sel), "centres_flagged": len(a), "centres_unflagged": len(b),
         "starting_model_flagged_pct": round(100 * float(flag.mean()), 2), "guard_frames": GUARD, "min_spacing": SPACING,
         "frames_in_centre_bands": int(band_c.sum()), "frames_in_truth_bands": int(band_t.sum()),
         "kept_out_of_training_pct": round(100 * float((band_c | band_t).mean()), 1), "flagged_frames_in_centre_bands": int((band_c & flag).sum()), "seed": seed}
    pd.DataFrame([S]).to_csv(out / "summary.csv", index=False)
    print(pd.Series(S).to_string())


def flags(v, m):
    F = pd.read_csv(ST / f"video{v:02d}" / m / "flags.csv.gz")
    return (F.fit_flag | F.pca_flag).to_numpy(bool)


def folder(v, name="final"):
    return B.OUT / f"video{v:02d}" / name


def select(v, seed):
    rr, dt, P, H, truth, fwd, sig = load(v)
    n = len(dt)
    Wm = teacher_eye_width(dt)
    XY = B.U.pose_matrix(dt)
    f0 = pd.read_csv(ST / f"video{v:02d}" / "repair_formal" / "frames_all.csv.gz").flag.to_numpy(bool)   # the starting model's flags as the loop saw them
    f1, f2 = flags(v, MODELS[0]), flags(v, MODELS[1])
    assert len(f0) == len(f1) == len(f2) == n
    tf = truth.index.astype(int).to_numpy()
    centres = np.array(sorted(blind_frames(v)), int)
    assert len(centres), f"video {v} has no reserved frames: run reserve before the loop"
    anchors = np.r_[tf, centres]
    labeled = list(tf) + B.label_frames(v)
    for f in glob.glob(str(B.OUT / f"video{v:02d}" / "**" / "selection_key.csv"), recursive=True):   # frames of an earlier labeled draw (the four videos); the reserve has no labels
        if "reserve" not in f:
            labeled += pd.read_csv(f).frame.astype(int).tolist()
    excl = B.band(n, labeled, B.GUARD_LABEL) | ~np.isfinite(XY).all(axis=(1, 2))
    elig = B.band(n, anchors, NEAR) & ~excl
    far = []
    for f in sorted((ST / f"video{v:02d}").glob("pseudo_formal*/pseudo_frames.csv")):
        K = pd.read_csv(f)
        far += K[K.kept].frame.astype(int).tolist()
    assert far, "no frame table of the final loop"
    close = B.band(n, far, NEAR) & elig
    elig &= ~close                                              # so every frame drawn is at least 31 frames from every training frame of the loop
    stratum = np.where(f1 & ~f2, "T1", np.where(f1 & f2, "TB", np.where(f2, "T2", np.where(f0, "S", "N"))))
    whole = {c: int((stratum == c).sum()) for c in STRATA}
    avail = {c: int(((stratum == c) & elig).sum()) for c in STRATA}
    tot = sum(avail.values())
    rng = np.random.default_rng(seed)
    rows = []
    for c in STRATA:
        pool = np.flatnonzero((stratum == c) & elig)
        chosen, lab = B.draw(pool, min(QUOTA[c], avail[c]), XY, Wm, seed, rng)
        rows += [{"frame": int(f), "stratum": c, "cluster": int(k)} for f, k in zip(chosen, lab)]
    sel = pd.DataFrame(rows)
    assert elig[sel.frame].all() and sel.frame.is_unique
    fa = np.array(sorted(far))
    sel["nearest_training_frame_distance"] = [int(np.abs(fa - f).min()) for f in sel.frame]
    assert sel.nearest_training_frame_distance.min() > NEAR
    for c in STRATA:
        cnt = int((sel.stratum == c).sum())
        sel.loc[sel.stratum == c, "stratum_frames"] = avail[c]
        sel.loc[sel.stratum == c, "stratum_share"] = avail[c] / tot
        sel.loc[sel.stratum == c, "weight"] = (avail[c] / tot) / cnt if cnt else np.nan
    sel["start_flag"], sel["turn1_flag"], sel["turn2_flag"] = f0[sel.frame], f1[sel.frame], f2[sel.frame]
    sel["nearest_anchor_distance"] = [int(np.abs(anchors - f).min()) for f in sel.frame]
    sel = sel.sample(frac=1, random_state=int(rng.integers(1 << 31))).reset_index(drop=True)
    sel.insert(0, "blind_name", [f"img{i + FIRST_NAME:03d}.png" for i in range(len(sel))])
    sel["video"] = v
    ld = folder(v) / "labeling" / "blind"
    if ld.exists():
        raise SystemExit(f"{ld} exists; remove it by hand for a fresh set")
    ld.mkdir(parents=True)
    out = folder(v)
    sel.to_csv(out / "selection_key.csv", index=False)
    pd.DataFrame([{"video": v, "unit": unit_of(v), "frames": n, "models": "starting model / " + " / ".join(MODELS), "within_frames_of_truth_or_centre": NEAR,
                   "anchors": len(anchors), "eligible": int(elig.sum()), "removed_near_a_training_frame": int(close.sum()),
                   "min_distance_to_a_training_frame": int(sel.nearest_training_frame_distance.min()),
                   **{f"whole_{c}": whole[c] for c in STRATA}, **{f"eligible_{c}": avail[c] for c in STRATA}, **{f"drawn_{c}": int((sel.stratum == c).sum()) for c in STRATA},
                   "eye_width_px": round(Wm, 1), "seed": seed}]).to_csv(out / "selection_summary.csv", index=False)
    B.write_pngs(v, dict(zip(sel.frame.astype(int), sel.blind_name)), ld)
    print(f"video {v}: {len(sel)} frames drawn; " + ", ".join(f"{c} {int((sel.stratum == c).sum())} of {avail[c]} eligible ({whole[c]} in the video)" for c in STRATA))
    print(f"PNGs in {ld}; key in {out / 'selection_key.csv'} (do not open it before labeling)")


def read_labels(v, name):
    d = folder(v, name)
    key = pd.read_csv(d / "selection_key.csv")
    key = key[key.subset] if "subset" in key.columns else key
    key = key.set_index("blind_name")
    csv = sorted(glob.glob(str(d / "labeling" / "blind" / "CollectedData_*.csv")))
    if not csv:
        raise SystemExit(f"no CollectedData csv in {d / 'labeling' / 'blind'}: label first")
    df = pd.read_csv(csv[-1], header=[0, 1, 2], index_col=[0, 1, 2])
    sc = df.columns.get_level_values(0)[0]
    rows = []
    for i, row in df.iterrows():
        nm = Path(i[-1] if isinstance(i, tuple) else i).name
        if nm not in key.index:
            continue                                           # the placeholder frame napari-deeplabcut needs
        r = {"blind_name": nm, "frame": int(key.loc[nm, "frame"]), "stratum": key.loc[nm, "stratum"],
             "weight": float(key.loc[nm, "weight_subset"] if "weight_subset" in key.columns else key.loc[nm, "weight"])}
        for b in B.BPS:
            r[f"human_{b}_x"], r[f"human_{b}_y"] = float(row[(sc, b, "x")]), float(row[(sc, b, "y")])
        rows.append(r)
    L = pd.DataFrame(rows).set_index("frame")
    L["labeled_points"] = sum(np.isfinite(L[f"human_{b}_x"]).astype(int) for b in B.BPS)
    return L[L.labeled_points > 0]


def score(v, name, write):
    L = read_labels(v, name)
    models = {"teacher": B.teacher_h5(v)}
    for d in sorted((B.LOCAL / f"video{v:02d}").glob("round*_formal*")):
        p = sorted((d / "predictions").glob("*.h5"))
        if p:
            models[d.name] = p[-1]
    E, _ = B.errors(v, L, models)
    E = E[np.isfinite(E.error)]
    E["wrong"] = E.error > B.LINE
    keep = ["model", "frame", "blind_name", "stratum", "weight", "labeled_points", "error", "error_px"]
    agg = dict(frames=("wrong", "size"), wrong=("wrong", "sum"), median_error_pct=("error", lambda x: round(100 * float(x.median()), 2)))
    Sx = E.groupby(["model", "stratum"], sort=False).agg(**agg).reset_index()
    T = E.groupby("model", sort=False).agg(**agg).reset_index()
    T.insert(1, "stratum", "all")
    if write:
        out = folder(v, name)
        E[keep].round(5).to_csv(out / "per_frame.csv", index=False)
        pd.concat([Sx, T]).to_csv(out / "scores.csv", index=False)
    order = [s for s in list(STRATA) + ["T1", "TB", "T2", "TN"] if s in set(L.stratum)]
    order = list(dict.fromkeys(order))
    Pv = E.pivot_table(index="stratum", columns="model", values="wrong", aggfunc="sum").reindex(order)
    Pv.loc["all"] = Pv.sum()
    Pv.insert(0, "frames", [int((L.stratum == s).sum()) for s in order] + [len(L)])
    pd.set_option("display.width", 250)
    print(f"video {v} ({name}): {len(L)} labeled frames; wrong frames (error over the {B.LINE:.2f} line) by stratum and model")
    print(Pv[["frames"] + [m for m in models if m in Pv.columns]].astype(int).to_string())
    return E


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["reserve", "select", "score"])
    ap.add_argument("--video", type=int, required=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--folder", default="final", help="score: the set's folder under results/blind_test/videoNN/ (turn2 = the turn-2 check of the four videos)")
    ap.add_argument("--no-write", action="store_true", help="score: print only")
    a = ap.parse_args()
    if a.cmd == "reserve":
        reserve(a.video, a.seed)
    elif a.cmd == "select":
        select(a.video, a.seed)
    else:
        score(a.video, a.folder, not a.no_write)


if __name__ == "__main__":
    main()
