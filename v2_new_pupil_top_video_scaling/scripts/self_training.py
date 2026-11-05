#!/usr/bin/env python3
"""Self-training pilot (direction 3 of the meeting of 2026-10-31): retrain the plateau model of a video on its human
labels plus the model's own coordinates on frames the pipeline passed, and see whether the truth-frame error and the
flagged share move. Pilot videos: 1 and 8 (hard), 7 (clean control).

    python self_training.py pseudo   --video 8 --every 30
    python self_training.py train    --video 8 --every 30 --round 1 [--epochs E]
    python self_training.py predict  --video 8 --shuffle 4081 --out DIR          (train hands over to it with os.execv)
    python self_training.py evaluate --video 8 --teacher
    python self_training.py evaluate --video 8 --every 30 --round 1
    python self_training.py compare

Teacher: the plateau model, i.e. the model whose whole-video predictions every step-9 number was computed on
(results/labels_to_plateau.csv; video 7's is shuffle 811 with 20 labels, although the table says 0 labels).

Pseudo labels (pseudo): the final pipeline (flag_and_repair.py --finder either --accept labelregion --points all
--replace smallest) re-run on the teacher's predictions into results/self_training/videoNN/repair/ with --save-coords
and checked against the committed summary. Frames: every --every-th frame whose status is "ok" (no finder flagged it)
or "repaired" (its repaired coordinates), excluding the 70 truth frames and 60 frames on either side of each. Per
point (a replaced point is kept as it is): a point is left out of the label when its likelihood is below 0.01, its
fitting deviation exceeds the flagging threshold or its PCA residual exceeds the video's PCA threshold; a frame is
kept when all four pupil points and at least six points remain. PNGs and the DeepLabCut label table go to
local_data/self_training/videoNN/pseudo_everyNN/ (git-ignored), the frame list and counts to
results/self_training/videoNN/pseudo_everyNN/.

Student (train): the teacher's training set (same human labels, same prior videos, read from the teacher shuffle's
training-set documentation pickle) plus the pseudo-labeled frames; same recipe as scale_step.py (ResNet-50, batch 2,
seed 42, DeepLabCut's augmentation, ImageNet backbone); validation = val20 as before. The schedule is matched in
ITERATIONS, not epochs: the teacher ran 120 epochs of ceil(N/2) iterations; the student runs round(that / ceil(N'/2))
epochs, learning-rate steps at 80 % and 95 % as before, so one student costs about one teacher. --every 100 is the
sparse variant (pseudo frames at about 0.3 of the density of --every 30): the stand-in for the 0.3 loss weight of the
design, which DeepLabCut's loss does not expose. Shuffle = 4000 + 10 * video + 2 * (round - 1) + {30: 1, 100: 2}.

Evaluation (evaluate): the same code on the whole-video predictions of the teacher and of every student:
(a) the 70 truth frames, error per frame = RMSE over the labeled points divided by the TEACHER's eye width, median,
90th percentile, count over the 0.14 line, per-point medians; (b) test50 median frame RMSE in px (the scale-curve
metric, from the same whole-video file; the teacher's value is checked against scale_curve.csv); (c) whole video:
share of frames flagged by the fitting finder (one-step SARIMAX (2,0,0) deviation > threshold, as in
flag_and_repair.py; the teacher's deviations are checked against the sweep's cache) and by the PCA finder (score >
multiple x the model's own median, the finder's rule; and > the teacher's absolute threshold), union share. Written to
results/self_training/videoNN/<model>/eval.csv and truth_errors.csv; compare gathers every eval.csv into
results/self_training/comparison.csv and comparison.png.

Round 2 (student as teacher) only if round 1 lowers the median truth error and the wrong-frame count; it needs the
student's heatmap peaks and is not implemented here. Nothing existing is modified."""
import argparse
import glob
import math
import os
import pickle
import re
import shutil
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
import pupil_trace as pt  # noqa: E402
import unreliable_frames_sweep as U  # noqa: E402
from flag_and_repair import FIT_PBOUND, one_step_prediction  # noqa: E402
from pca_threshold import residuals  # noqa: E402
from scale_step import BATCH, CONFIG, DATE, EPOCHS, HERE, PROJECT, ROOT, SCORER, TASK, ensure_fraction, ensure_project, log, stage  # noqa: E402
from second_peak_repair import BPS, P4, PL, RES, frame_rmse, load  # noqa: E402

OUT = RES / "self_training"
LOCAL = ROOT / "local_data" / "self_training"
GUARD = 60
MIN_POINTS = 6
PIPE = ["--finder", "either", "--accept", "labelregion", "--points", "all", "--replace", "smallest"]
FINAL = "video{v:02d}_either_labelregion_all_smallest"
VARIANT = {30: 1, 100: 2}
LINE = float(pd.read_csv(RES / "error_tier" / "tier.csv").iloc[0].line_eye_widths)


def unit_of(v):
    return PL[PL.video == v].iloc[0].unit


def step_of(v):
    return max(1, int(PL[PL.video == v].iloc[0].labels_at_plateau) // 20)


def teacher_h5(v):
    return sorted(glob.glob(str(HERE / unit_of(v) / "training-data" / f"predictions_step{step_of(v):02d}" / "*snapshot*120*.h5")))[-1]


def teacher_shuffle(v):
    return int(re.search(r"shuffle(\d+)_", Path(teacher_h5(v)).name).group(1))


def student_shuffle(v, every, rnd):
    return 4000 + 10 * v + 2 * (rnd - 1) + VARIANT[every]


def model_name(every=None, rnd=None, teacher=False):
    return "teacher" if teacher else f"round{rnd}_every{every}"


def pseudo_unit(v, every, rnd=1):
    return f"{unit_of(v)}_pseudo{every}" + (f"r{rnd}" if rnd > 1 else "")


def pseudo_dirname(every, rnd=1):
    return f"pseudo_every{every}" + (f"_round{rnd}" if rnd > 1 else "")


def teacher_eye_width(d):
    return float(np.nanmedian(U.eye_width(d)))


def mkdir(p):
    p.mkdir(parents=True, exist_ok=True)
    return p


# ------------------------------------------------------------------------------------------------------ pseudo labels
def student_source(v, every, rnd):
    """Coordinates, likelihoods and finder state of the round-rnd student (the source of round rnd+1 pseudo labels):
    the same fitting and PCA measures evaluate() computes, with the student's own PCA threshold (median x multiple),
    as the pipeline uses the teacher's own threshold in round 1. Returns a coords table in the layout of the repair's
    coords_after.csv.gz (status ok = passed by both finders, flagged otherwise; no repaired points)."""
    unit, step = unit_of(v), step_of(v)
    name = model_name(every, rnd)
    h5 = sorted((LOCAL / f"video{v:02d}" / name / "predictions").glob("*.h5"))[-1]
    d = pt.load(str(h5))
    rr, dt, P, H, truth, fwd, sig = load(v)
    Wm = teacher_eye_width(dt)
    thr = float(pd.read_csv(RES / "fit_threshold" / "chosen.csv").iloc[0].threshold_eye_widths)
    xy = {b: np.c_[d[b]["x"].to_numpy(float), d[b]["y"].to_numpy(float)] for b in BPS}
    cache = OUT / f"video{v:02d}" / name / "pred200.npy"
    pm = np.load(cache) if cache.exists() else one_step_prediction(d)
    if not cache.exists():
        np.save(cache, pm)
    XY = np.stack([xy[b] for b in BPS], axis=1)
    dev = np.hypot(*(XY - pm).transpose(2, 0, 1)) / Wm
    z = np.load(RES / "finder_comparison" / "pca_cache" / f"video{v:02d}_pca.npz")
    res_px, nc = residuals(U.labeled_poses(unit, step), U.pose_matrix(d))
    res = res_px / float(z["eye_width_px"])
    score = np.sqrt(np.mean(res ** 2, axis=1))
    thr_pca = float(z["multiple"]) * float(np.nanmedian(score))
    flagged = (np.nanmax(dev, axis=1) > thr) | (score > thr_pca)
    C = pd.DataFrame({"frame": np.arange(len(d)), "status": np.where(flagged, "flagged", "ok"), "points": ""})
    for b in BPS:
        C[f"{b}_x"], C[f"{b}_y"] = xy[b][:, 0], xy[b][:, 1]
    lik = {b: d[b]["likelihood"].to_numpy(float) for b in BPS}
    log(f"video {v}: round {rnd + 1} pseudo labels from {Path(h5).name}: {int(flagged.sum())} of {len(d)} frames flagged "
        f"(fitting thr {thr:.3f}, own PCA thr {thr_pca:.4f} eye widths)")
    return C, d, lik, dev, res, Wm, thr, thr_pca, truth, Path(h5).name


def pseudo(v, every, rnd=1):
    unit = unit_of(v)
    out = mkdir(OUT / f"video{v:02d}" / pseudo_dirname(every, rnd))
    loc = mkdir(LOCAL / f"video{v:02d}" / pseudo_dirname(every, rnd))
    if rnd == 1:
        rep = OUT / f"video{v:02d}" / "repair"
        if not (rep / "coords_after.csv.gz").exists():
            log(f"video {v}: re-running the final pipeline into {rep}")
            subprocess.run([sys.executable, str(HERE / "scripts" / "flag_and_repair.py"), "--video", str(v), *PIPE, "--out", str(rep), "--save-coords"],
                           check=True, cwd=str(HERE / "scripts"), stdout=subprocess.DEVNULL)
        for f in ("summary.csv", "whole_video.csv"):
            a, b = pd.read_csv(rep / f), pd.read_csv(RES / "flag_and_repair" / FINAL.format(v=v) / f)
            pd.testing.assert_frame_equal(a, b)
        log(f"video {v}: re-run summary and whole_video identical to the committed run")
        C = pd.read_csv(rep / "coords_after.csv.gz")
        W = pd.read_csv(rep / "whole_video.csv").iloc[0]
        Wm, thr = float(W.eye_width_px), float(W.thr_eye_widths)
        rr, d, P, H, truth, fwd, sig = load(v)
        n = len(C)
        assert n == len(d) == len(fwd["200"])
        lik = {b: d[b]["likelihood"].to_numpy(float) for b in BPS}
        dev = fwd["200"] / Wm
        z = np.load(RES / "finder_comparison" / "pca_cache" / f"video{v:02d}_pca.npz")
        res, thr_pca = z["residuals"], float(z["threshold"])
        source = Path(teacher_h5(v)).name
    else:
        C, d, lik, dev, res, Wm, thr, thr_pca, truth, source = student_source(v, every, rnd - 1)
        n = len(C)
    assert res.shape == (n, len(BPS))
    tf = truth.index.to_numpy()
    near = np.zeros(n, bool)
    for f in tf:
        near[max(0, f - GUARD):f + GUARD + 1] = True
    usable = C.status.isin(["ok", "repaired"]).to_numpy()
    cand = usable & (C.frame.to_numpy() % every == 0) & ~near
    rows, drop_count = [], Counter()
    for i in np.flatnonzero(cand):
        replaced = set(str(C.points[i]).split("+")) if C.status[i] == "repaired" and isinstance(C.points[i], str) else set()
        kept, reasons = {}, {}
        for k, b in enumerate(BPS):
            why = None
            if b not in replaced:
                if lik[b][i] < FIT_PBOUND:
                    why = "likelihood"
                elif dev[i, k] > thr:
                    why = "fitting"
                elif res[i, k] > thr_pca:
                    why = "pca"
            if why is None:
                kept[b] = (float(C[f"{b}_x"][i]), float(C[f"{b}_y"][i]))
            else:
                reasons[b] = why
                drop_count[why] += 1
        keep_frame = all(b in kept for b in P4) and len(kept) >= MIN_POINTS
        r = {"frame": int(C.frame[i]), "status": C.status[i], "replaced": "+".join(sorted(replaced)), "points_kept": len(kept),
             "points_dropped": "+".join(f"{b}:{w}" for b, w in reasons.items()), "kept": keep_frame}
        for b in BPS:
            r[f"{b}_x"], r[f"{b}_y"] = kept.get(b, (np.nan, np.nan))
        rows.append(r)
    F = pd.DataFrame(rows)
    F.round(3).to_csv(out / "pseudo_frames.csv", index=False)
    K = F[F.kept]
    S = {"video": v, "unit": unit, "every": every, "round": rnd, "source": source, "frames": n, "usable_frames": int(usable.sum()), "near_truth_excluded": int((usable & (C.frame.to_numpy() % every == 0) & near).sum()),
         "candidates": int(cand.sum()), "kept_frames": len(K), "dropped_frames": int((~F.kept).sum()),
         "kept_ok": int((K.status == "ok").sum()), "kept_repaired": int((K.status == "repaired").sum()),
         "points_total": int(cand.sum()) * len(BPS), **{f"points_dropped_{w}": drop_count.get(w, 0) for w in ("likelihood", "fitting", "pca")},
         "labeled_points_in_kept_frames": int(K[[f"{b}_x" for b in BPS]].notna().sum().sum()),
         "frames_with_8_points": int((K.points_kept == 8).sum()), "eye_width_px": Wm, "thr_eye_widths": thr, "thr_pca_eye_widths": thr_pca, "guard_frames": GUARD}
    pd.DataFrame([S]).to_csv(out / "pseudo_summary.csv", index=False)
    print(pd.Series(S).to_string())

    # the DeepLabCut label table, as CollectedData_<scorer>.h5 of a labeled-data folder
    names = [f"img{f:06d}.png" for f in K.frame]
    cols = pd.MultiIndex.from_product([[SCORER], BPS, ["x", "y"]], names=["scorer", "bodyparts", "coords"])
    T = pd.DataFrame(K[[f"{b}_{c}" for b in BPS for c in "xy"]].to_numpy(float), columns=cols,
                     index=pd.MultiIndex.from_tuples([("labeled-data", pseudo_unit(v, every, rnd), nm) for nm in names]))
    T.to_hdf(loc / f"CollectedData_{SCORER}.h5", key="df_with_missing", mode="w")
    T.to_csv(loc / f"CollectedData_{SCORER}.csv")
    T.round(3).to_csv(out / "pseudo_labels.csv")

    # the PNGs, one sequential pass over the video (cv2.imwrite of the frame as read, as select_frames.py does)
    import cv2
    if rnd > 1:   # the frames are the same video frames as round 1's: copy the PNGs that round already extracted
        prev = LOCAL / f"video{v:02d}" / pseudo_dirname(every, rnd - 1)
        for nm in names:
            if not (loc / nm).exists() and (prev / nm).exists():
                shutil.copy2(prev / nm, loc / nm)
    want = {int(f): nm for f, nm in zip(K.frame, names) if not (loc / nm).exists()}
    if want:
        cap = cv2.VideoCapture(str(HERE / unit / f"{unit}.mp4"))
        last = max(want)
        i, t0, written = 0, time.time(), 0
        while i <= last:
            ok = cap.grab()
            if not ok:
                break
            if i in want:
                ok, im = cap.retrieve()
                assert ok
                cv2.imwrite(str(loc / want[i]), im)
                written += 1
            i += 1
        cap.release()
        log(f"video {v}: {written} PNGs written to {loc} in {(time.time() - t0) / 60:.1f} min")
    missing = [nm for nm in names if not (loc / nm).exists()]
    assert not missing, f"{len(missing)} PNGs missing"
    log(f"video {v}: pseudo set every {every}: {len(K)} frames, {S['labeled_points_in_kept_frames']} labeled points")


# ------------------------------------------------------------------------------------------------------------ training
def teacher_training_set(v):
    """(prior units as [(unit, last_batch)], own labels, N train, N val) of the teacher shuffle, from its documentation."""
    sh = teacher_shuffle(v)
    f = glob.glob(str(PROJECT / "training-datasets" / "iteration-0" / "*" / f"Documentation_data-*shuffle{sh}.pickle"))[0]
    doc = pickle.load(open(f, "rb"))
    data, tr, te = doc[0], doc[1], doc[2]
    # the documentation's data list holds the training images only (in video 1's shuffle 211 the train indices run
    # past its length, so they are not positions in that list); count the units over the list itself
    assert len(data) == len(tr), (len(data), len(tr))
    cnt = Counter(d["image"][1] for d in data)
    units = list(PL.unit)
    prior, own = [], 0
    for name, c in cnt.items():
        u = [x for x in units if x == name or x.endswith("_" + name)]
        if not u and re.match(r"\d+_", name):
            # units renamed after the early videos were trained (video 2's '2_pluton2', video 4's '3_20251031_pluto3'):
            # the leading video number is the stable part
            u = [unit_of(int(name.split("_")[0]))]
        assert len(u) == 1, (name, u)
        if u[0] == unit_of(v):
            own = c
        else:
            assert c % 20 == 0
            prior.append((u[0], c // 20))
    assert own == 20 * step_of(v), (own, step_of(v))
    return sh, prior, own, len(tr), len(te)


def train(v, every, rnd, epochs_override=None, shuffle_override=None, tag=""):
    import deeplabcut
    from deeplabcut.generate_training_dataset.trainingsetmanipulation import merge_annotateddatasets
    unit, pu = unit_of(v), pseudo_unit(v, every, rnd)
    shuffle = shuffle_override or student_shuffle(v, every, rnd)
    name = model_name(every, rnd) + (f"_{tag}" if tag else "")
    out = mkdir(OUT / f"video{v:02d}" / name)
    loc = LOCAL / f"video{v:02d}" / pseudo_dirname(every, rnd)
    assert (loc / f"CollectedData_{SCORER}.h5").exists(), "run pseudo first"
    mdir = PROJECT / "dlc-models-pytorch" / "iteration-0" / f"{TASK}{DATE}-trainset*shuffle{shuffle}"
    assert not glob.glob(str(mdir)), f"shuffle {shuffle} exists already"
    tsh, prior, own, Nt, Nv = teacher_training_set(v)
    log(f"video {v}: teacher shuffle {tsh}: {Nt} train ({own} of this video, prior {dict(prior)}), {Nv} val")
    ensure_project([u for u, _ in prior] + [unit, pu])
    train_names = {}
    for u, last in prior:
        train_names[u], _ = stage(u, last, with_val=False)
    train_names[unit], val_names = stage(unit, step_of(v), with_val=True)
    dest = PROJECT / "labeled-data" / pu
    if dest.exists():
        shutil.rmtree(dest)
    shutil.copytree(loc, dest)
    T = pd.read_hdf(dest / f"CollectedData_{SCORER}.h5")
    train_names[pu] = [i[-1] for i in T.index]
    for dd in (PROJECT / "labeled-data").iterdir():
        if dd.is_dir() and dd.name not in train_names:
            shutil.rmtree(dd)
    n_human = sum(len(ns) for u, ns in train_names.items() if u != pu)
    assert n_human == Nt, (n_human, Nt)
    cfg = deeplabcut.auxiliaryfunctions.read_config(str(CONFIG))
    tsf = Path(deeplabcut.auxiliaryfunctions.get_training_set_folder(cfg))
    (PROJECT / tsf).mkdir(parents=True, exist_ok=True)
    merged = merge_annotateddatasets(cfg, PROJECT / tsf)
    keys = [(i[-2], i[-1]) if isinstance(i, tuple) else tuple(Path(str(i)).parts[-2:]) for i in merged.index]
    want_train = {(u, nm) for u, ns in train_names.items() for nm in ns}
    want_val = {(unit, nm) for nm in val_names}
    train_idx = [k for k, key in enumerate(keys) if key in want_train]
    val_idx = [k for k, key in enumerate(keys) if key in want_val]
    assert len(train_idx) == len(want_train) and len(val_idx) == len(want_val) == Nv and len(keys) == len(train_idx) + len(val_idx), (len(keys), len(train_idx), len(val_idx))
    Ns = len(train_idx)
    it_teacher = EPOCHS * math.ceil(Nt / BATCH)
    E = epochs_override or max(1, round(it_teacher / math.ceil(Ns / BATCH)))
    s = max(1, E // 12)
    E = max(s, round(E / s) * s)
    milestones = [max(1, round(0.8 * E)), max(1, round(0.95 * E))]
    fraction = round(Ns / len(keys), 2)
    tsi = ensure_fraction(fraction)
    pct = int(round(fraction * 100))
    mdir = PROJECT / "dlc-models-pytorch" / "iteration-0" / f"{TASK}{DATE}-trainset{pct}shuffle{shuffle}" / "train"
    log(f"student {name} shuffle {shuffle}: {Ns} train = {n_human} human + {Ns - n_human} pseudo, {Nv} val, fraction {fraction}, tsi {tsi}; "
        f"teacher {it_teacher} iterations -> {E} epochs x {math.ceil(Ns / BATCH)} = {E * math.ceil(Ns / BATCH)} iterations, milestones {milestones}, snapshot every {s}")
    t0 = time.time()
    deeplabcut.create_training_dataset(str(CONFIG), Shuffles=[shuffle], trainIndices=[train_idx], testIndices=[val_idx],
                                       net_type="resnet_50", userfeedback=False, engine=deeplabcut.Engine.PYTORCH)
    pc = mdir / "pytorch_config.yaml"
    c = yaml.safe_load(pc.read_text())
    c["train_settings"].update(batch_size=BATCH, epochs=E, seed=42)
    c["runner"]["scheduler"]["params"]["milestones"] = milestones
    c["runner"]["snapshots"].update(save_epochs=s, max_snapshots=14)
    pc.write_text(yaml.safe_dump(c, sort_keys=False))
    deeplabcut.train_network(str(CONFIG), shuffle=shuffle, trainingsetindex=tsi, epochs=E, save_epochs=s,
                             max_snapshots_to_keep=14, batch_size=BATCH, device="mps", snapshot_path=None)
    minutes = (time.time() - t0) / 60
    log(f"TRAINING DONE in {minutes:.0f} min")
    snaps = sorted(p.name for p in mdir.glob("snapshot-*.pt"))
    final = next(k for k, nm in enumerate(snaps) if nm in (f"snapshot-{E:03d}.pt", f"snapshot-best-{E:03d}.pt"))
    pred = mkdir(LOCAL / f"video{v:02d}" / name / "predictions")
    info = out / "train_info.csv"
    pd.DataFrame([{"video": v, "model": name, "shuffle": shuffle, "teacher_shuffle": tsh, "train_human": n_human, "train_pseudo": Ns - n_human, "val": Nv,
                   "teacher_iterations": it_teacher, "epochs": E, "iterations": E * math.ceil(Ns / BATCH), "milestones": "/".join(map(str, milestones)),
                   "snapshot_every": s, "final_snapshot": snaps[final], "train_minutes": round(minutes, 1), "predict_minutes": np.nan,
                   "prediction_h5": ""}]).to_csv(info, index=False)
    # the whole video is predicted by a fresh process that replaces this one (the training process keeps about 3 GB
    # after training, which on the 16 GB machine pushed the swap to its limit while the predictor ran as a child);
    # the predictor fills predict_minutes / prediction_h5 into train_info.csv when it is done
    log(f"handing over to the prediction process: {pred}")
    sys.stdout.flush(); sys.stderr.flush()
    os.chdir(HERE / "scripts")
    os.execv(sys.executable, [sys.executable, str(Path(__file__).resolve()), "predict", "--video", str(v), "--shuffle", str(shuffle),
                              "--out", str(pred), "--info", str(info)])


# ---------------------------------------------------------------------------------------------------- whole-video prediction
def predict_video(v, shuffle, out, limit=0, batch=32, device="mps"):
    """Predict every frame of the video with the final snapshot of a shuffle; streaming (frames read one by one with
    cv2, multithreading off), written as a DeepLabCut-style h5 (scorer / bodyparts / x, y, likelihood) that
    pupil_trace.load reads. A separate process from training, so memory stays at the model's size."""
    import cv2
    import torch
    from deeplabcut.pose_estimation_pytorch.apis.utils import get_pose_inference_runner
    unit = unit_of(v)
    folder = next((PROJECT / "dlc-models-pytorch" / "iteration-0").glob(f"*shuffle{shuffle}")) / "train"
    E = int(yaml.safe_load((folder / "pytorch_config.yaml").read_text())["train_settings"]["epochs"])
    snap = next(p for p in (folder / f"snapshot-{E:03d}.pt", folder / f"snapshot-best-{E:03d}.pt") if p.exists())
    runner = get_pose_inference_runner(model_config=folder / "pytorch_config.yaml", snapshot_path=snap, batch_size=batch, device=device,
                                       inference_cfg={"multithreading": {"enabled": False}})
    t0 = time.time()

    def frames():
        cap = cv2.VideoCapture(str(HERE / unit / f"{unit}.mp4"))
        i = 0
        while True:
            ok, im = cap.read()
            if not ok or (limit and i >= limit):
                break
            yield im[..., ::-1].copy()   # RGB, as DeepLabCut's video reader returns it
            i += 1
            if i % 5000 == 0:
                log(f"{i} frames  {(time.time() - t0) / 60:.1f} min")
        cap.release()

    with torch.no_grad():
        preds = [np.asarray(p["bodyparts"])[0, :, :3] for p in runner.inference(images=frames())]
    poses = np.stack(preds)
    scorer = f"DLC_Resnet50_{TASK}{DATE}shuffle{shuffle}_snapshot_{E:03d}"
    cols = pd.MultiIndex.from_product([[scorer], BPS, ["x", "y", "likelihood"]], names=["scorer", "bodyparts", "coords"])
    D = pd.DataFrame(poses.reshape(len(poses), -1), columns=cols)
    mkdir(out)
    f = out / f"{unit}{scorer}.h5"
    D.to_hdf(f, key="df_with_missing", mode="w")
    log(f"{len(D)} frames predicted with {snap.name} in {(time.time() - t0) / 60:.1f} min -> {f}")
    return f


# ---------------------------------------------------------------------------------------------------------- evaluation
def evaluate(v, every=None, rnd=None, teacher=False, tag="", save_flags=False, h5_override=None, name_override=None):
    unit, step = unit_of(v), step_of(v)
    name = name_override or (model_name(every, rnd, teacher) + (f"_{tag}" if tag else ""))
    out = mkdir(OUT / f"video{v:02d}" / name)
    rr, dt, P, H, truth, fwd, sig = load(v)
    Wm = teacher_eye_width(dt)
    if teacher:
        h5, shuffle = teacher_h5(v), teacher_shuffle(v)
        d = dt
    elif h5_override:                                       # any other model of the video (e.g. the 40-label model of the scaling curve)
        h5 = Path(h5_override)
        shuffle = int(re.search(r"shuffle(\d+)", h5.name).group(1))
        d = pt.load(str(h5))
    else:
        h5 = sorted((LOCAL / f"video{v:02d}" / name / "predictions").glob("*.h5"))[-1]
        shuffle = student_shuffle(v, every, rnd)
        d = pt.load(str(h5))
    n = len(d)
    assert n == len(dt)
    xy = {b: np.c_[d[b]["x"].to_numpy(float), d[b]["y"].to_numpy(float)] for b in BPS}
    # (a) truth frames
    rows = []
    for f, hr in truth.iterrows():
        pts = {b: xy[b][f] for b in BPS}
        r = {"frame": f, "set": hr.set, "error": frame_rmse(pts, hr) / Wm, "error_px": frame_rmse(pts, hr)}
        for b in BPS:
            r[b] = np.hypot(pts[b][0] - hr[f"human_{b}_x"], pts[b][1] - hr[f"human_{b}_y"]) / Wm
        rows.append(r)
    T = pd.DataFrame(rows)
    T.round(4).to_csv(out / "truth_errors.csv", index=False)
    # (c) whole video, both finders
    thr = float(pd.read_csv(RES / "fit_threshold" / "chosen.csv").iloc[0].threshold_eye_widths)
    cache = out / "pred200.npy"
    pm = np.load(cache) if cache.exists() else one_step_prediction(d)
    if not cache.exists():
        np.save(cache, pm)
    XY = np.stack([xy[b] for b in BPS], axis=1)
    dev_px = np.hypot(*(XY - pm).transpose(2, 0, 1))
    if teacher:
        diff = np.nanmax(np.abs(dev_px - fwd["200"]))
        assert diff < 1.0, f"teacher fitting deviations differ from the sweep's cache by {diff:.2f} px"
    flag_fit = np.nanmax(dev_px, axis=1) / Wm > thr
    z = np.load(RES / "finder_comparison" / "pca_cache" / f"video{v:02d}_pca.npz")
    res_px, nc = residuals(U.labeled_poses(unit, step), U.pose_matrix(d))
    score = np.sqrt(np.mean((res_px / float(z["eye_width_px"])) ** 2, axis=1))   # the PCA finder's own eye width (error_tier/eye_width.csv, 0.1 px), so the teacher reproduces its cache exactly
    med = float(np.nanmedian(score))
    if teacher:
        assert np.nanmax(np.abs(score - z["score"])) < 1e-6 and abs(med - float(z["median"])) < 1e-9
    mult = float(z["multiple"])
    flag_pca_own = score > mult * med
    flag_pca_teacher = score > float(z["threshold"])
    if save_flags:   # per-frame finder state (for the blind test set's strata); default outputs unchanged
        pd.DataFrame({"frame": np.arange(n), "fit_dev_eye": np.nanmax(dev_px, axis=1) / Wm, "pca_score": score,
                      "fit_flag": flag_fit, "pca_flag": flag_pca_own}).to_csv(out / "flags.csv.gz", index=False, float_format="%.5f")
    wrong = T.error > LINE
    R = {"video": v, "model": name, "shuffle": shuffle, "eye_width_px": round(Wm, 1), "truth_frames": len(T),
         "median_error_pct": round(100 * float(T.error.median()), 2), "p90_error_pct": round(100 * float(T.error.quantile(0.9)), 2),
         "wrong_frames": int(wrong.sum()), "wrong_val20": int((wrong & (T.set == "val20")).sum()), "wrong_test50": int((wrong & (T.set == "test50")).sum()),
         "test50_median_px": round(float(T.error_px[T.set == "test50"].median()), 2),
         **{f"median_{b}_pct": round(100 * float(T[b].median()), 2) for b in BPS},
         "flagged_fit_pct": round(100 * float(flag_fit.mean()), 2), "pca_median_pct": round(100 * med, 3), "pca_components": int(nc),
         "flagged_pca_pct": round(100 * float(flag_pca_own.mean()), 2), "flagged_pca_teacher_thr_pct": round(100 * float(flag_pca_teacher.mean()), 2),
         "flagged_either_pct": round(100 * float((flag_fit | flag_pca_own).mean()), 2), "prediction_file": Path(h5).name}
    if teacher:
        W = pd.read_csv(RES / "flag_and_repair" / FINAL.format(v=v) / "whole_video.csv").iloc[0]
        assert abs(R["flagged_fit_pct"] - W.flagged_fit_pct) < 0.011 and abs(R["flagged_pca_pct"] - W.flagged_pca_pct) < 0.011, (R["flagged_fit_pct"], W.flagged_fit_pct, R["flagged_pca_pct"], W.flagged_pca_pct)
        sc = pd.read_csv(HERE / unit / "results" / "scale_curve.csv")
        sc = sc[sc.label == f"x{step * 20:03d}_step{step:02d}_final"]
        if len(sc):
            print(f"test50 median px from the whole-video file {R['test50_median_px']} against scale_curve.csv {float(sc.median_frame_rmse_px.iloc[0])}")
    pd.DataFrame([R]).to_csv(out / "eval.csv", index=False)
    print(pd.Series(R).to_string())


def compare():
    rows = [pd.read_csv(f) for f in sorted(OUT.glob("video*/*/eval.csv"))]
    assert rows, "no eval.csv yet"
    E = pd.concat(rows, ignore_index=True)
    ref = []
    for v in sorted(E.video.unique()):
        T = pd.read_csv(RES / "flag_and_repair" / FINAL.format(v=int(v)) / "truth_frames.csv")
        W = pd.read_csv(RES / "flag_and_repair" / FINAL.format(v=int(v)) / "whole_video.csv").iloc[0]
        ref.append({"video": v, "model": "teacher + repair", "median_error_pct": round(100 * float(T.error_after.median()), 2),
                    "p90_error_pct": round(100 * float(T.error_after.quantile(0.9)), 2), "wrong_frames": int((T.error_after > LINE).sum()),
                    "flagged_not_repaired_pct": W.flagged_not_repaired_pct})
    E = pd.concat([E, pd.DataFrame(ref)], ignore_index=True)
    order = {"teacher": 0, "teacher + repair": 1, "human40": 90}
    E["o"] = E.model.map(lambda m: 2 + int(re.search(r"round(\d+)", m).group(1)) * 10 + VARIANT.get(int(re.search(r"every(\d+)", m).group(1)), 9) if "round" in m else order.get(m, 95))
    E = E.sort_values(["video", "o"]).drop(columns="o")
    E.to_csv(OUT / "comparison.csv", index=False)
    show = ["video", "model", "median_error_pct", "p90_error_pct", "wrong_frames", "test50_median_px", "flagged_fit_pct", "flagged_pca_pct", "flagged_either_pct"]
    print(E[show].to_string(index=False))

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    models = [m for m in ["teacher", "teacher + repair"] + sorted({m for m in E.model if "round" in m}) + ["human40"] if m in set(E.model)]
    cols = {"teacher": "0.55", "teacher + repair": "0.8"}
    palette = ["#2f6db3", "#e0a458", "#7fb069", "#d1495b", "#8e6c8a"]
    for m in models:
        if m not in cols:
            cols[m] = palette[len(cols) - 2]
    vids = sorted(E.video.unique())
    panels = [("median_error_pct", "median truth-frame error, % of eye width"), ("wrong_frames", "wrong truth frames (over the line)"),
              ("flagged_either_pct", "frames flagged by either finder, % of the video")]
    fig, axes = plt.subplots(1, 3, figsize=(max(15, 0.9 * len(vids) * len(models)), 4.4), constrained_layout=True)
    bw = 0.8 / len(models)
    for ax, (c, lab) in zip(axes, panels):
        for j, m in enumerate(models):
            S = E[E.model == m].set_index("video")
            vals = [S[c].get(v, np.nan) if c in S else np.nan for v in vids]
            ax.bar(np.arange(len(vids)) + (j - (len(models) - 1) / 2) * bw, vals, width=bw, color=cols[m], label=m)
            for x, y in zip(np.arange(len(vids)) + (j - (len(models) - 1) / 2) * bw, vals):
                if np.isfinite(y):
                    ax.text(x, y, f"{y:g}", ha="center", va="bottom", fontsize=7)
        ax.set_xticks(range(len(vids)))
        ax.set_xticklabels([f"video {int(v)}" for v in vids])
        ax.set_title(lab, fontsize=10)
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", color="0.92")
        ax.set_axisbelow(True)
    axes[0].legend(frameon=False, fontsize=8)
    fig.suptitle("Self-training pilot: teacher (plateau model) against students retrained on human + pseudo labels", fontsize=10)
    fig.savefig(OUT / "comparison.png", dpi=150)
    print("figure", OUT / "comparison.png")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["pseudo", "train", "predict", "evaluate", "compare"])
    ap.add_argument("--video", type=int)
    ap.add_argument("--every", type=int, choices=sorted(VARIANT), default=None)
    ap.add_argument("--round", type=int, default=1)
    ap.add_argument("--teacher", action="store_true")
    ap.add_argument("--epochs", type=int, default=None, help="override the iteration-matched epoch count (smoke tests only)")
    ap.add_argument("--shuffle", type=int, default=None, help="override the shuffle number (smoke tests only)")
    ap.add_argument("--tag", default="", help="suffix of the model folder (smoke tests only)")
    ap.add_argument("--out", default=None, help="predict: output folder")
    ap.add_argument("--limit", type=int, default=0, help="predict: first N frames only (tests)")
    ap.add_argument("--info", default=None, help="predict: train_info.csv to complete with predict_minutes / prediction_h5")
    ap.add_argument("--save-flags", action="store_true", help="evaluate: also write flags.csv.gz (per-frame finder state); default outputs unchanged")
    ap.add_argument("--h5", default=None, help="evaluate: whole-video prediction of another model of the video (with --model-name)")
    ap.add_argument("--model-name", default=None, help="evaluate: folder name under results/self_training/videoNN/ for --h5")
    a = ap.parse_args()
    if a.cmd == "pseudo":
        pseudo(a.video, a.every, a.round)
    elif a.cmd == "train":
        train(a.video, a.every, a.round, a.epochs, a.shuffle, a.tag)
    elif a.cmd == "predict":
        t1 = time.time()
        predict_video(a.video, a.shuffle, Path(a.out), a.limit)
        if a.info:
            h5 = sorted(Path(a.out).glob("*.h5"))[-1]
            log(f"whole video predicted in {(time.time() - t1) / 60:.0f} min: {h5.name}")
            T = pd.read_csv(a.info)
            T["predict_minutes"] = round((time.time() - t1) / 60, 1)
            T["prediction_h5"] = h5.name
            T.to_csv(a.info, index=False)
    elif a.cmd == "evaluate":
        evaluate(a.video, a.every, a.round, a.teacher, a.tag, a.save_flags, a.h5, a.model_name)
    else:
        compare()


if __name__ == "__main__":
    main()
