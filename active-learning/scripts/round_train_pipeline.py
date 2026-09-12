#!/usr/bin/env python3
"""Active learning: machine half of one round, for all three branches.

For each branch (uncertain, jump, fitting), sequentially:
1. Stage that branch's cumulative reviewed frames into the DLC project as
   labeled-data/face_first4min (index rewritten to standard form), so the
   combined table is seed 100 + branch's own frames only. Branches never see
   each other's frames.
2. Create that branch's training dataset: train = seed's 80 + all branch
   frames, validation = the frozen 20 (unchanged forever).
3. Train ResNet-50 from scratch, 100 epochs, batch 2, frozen schedule.
4. Evaluate once on the reviewed final-minute 100 frames -> one convergence
   point (report-only).
5. Analyze the first-4-minutes clip with the branch's new model and select the
   next round's 20 frames with the branch's frozen detector; collision-safe
   (already-reviewed frames are replaced by enlarging the k-means pick and
   dropping collisions).
6. Unstage.

Round-0 baseline (cumulative 80 training frames, external 20.10 px, from the
model-selection ResNet-50) is written once for every branch.
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[2]
UNIT = ROOT / "active-learning"
PROJECT = ROOT / "dlc_projects" / "EyePupilBlink-Zhiheng-2026-08-17"
CONFIG = PROJECT / "config.yaml"
STAGING = PROJECT / "labeled-data" / "face_first4min"
CLIP = UNIT / "training-data" / "face_first4min.mp4"
SEED_LABELS = PROJECT / "labeled-data" / "face" / "CollectedData_Zhiheng.h5"
SPLIT = json.loads((ROOT / "local_data/experiments/split_80_20.json").read_text())
PYTHON = sys.executable
SEED = 42
EPOCHS = 100
BATCH = 2
BRANCHES = {
    "uncertain": {"shuffle_base": 31, "params": {"outlieralgorithm": "uncertain", "p_bound": 0.6}},
    "jump": {"shuffle_base": 41, "params": {"outlieralgorithm": "jump", "epsilon": 20}},
    "fitting": {"shuffle_base": 51, "params": {"outlieralgorithm": "fitting", "epsilon": 20}},
}
CONVERGENCE = UNIT / "results" / "convergence.csv"


def log(msg: str) -> None:
    print(f"[{datetime.now().isoformat(timespec='seconds')}] {msg}", flush=True)


def run(cmd: list[str]) -> None:
    log("RUN " + " ".join(map(str, cmd)))
    import os
    env = {**os.environ, "PYTORCH_ENABLE_MPS_FALLBACK": "1"}
    r = subprocess.run(list(map(str, cmd)), cwd=ROOT, env=env)
    if r.returncode:
        raise RuntimeError(f"command failed ({r.returncode}): {cmd[:3]}...")


def branch_store(branch: str) -> Path:
    p = UNIT / "branches" / branch
    p.mkdir(parents=True, exist_ok=True)
    return p


def normalized_branch_table(branch: str, round_no: int) -> pd.DataFrame:
    """This round's freshly reviewed table, index rewritten for the project."""
    src = UNIT / "frames" / branch / "CollectedData_Zhiheng.h5"
    t = pd.read_hdf(src)
    t.index = pd.MultiIndex.from_tuples(
        [("labeled-data", "face_first4min", i[-1]) for i in t.index]
    )
    return t.sort_index()


def cumulative_table(branch: str, round_no: int) -> pd.DataFrame:
    """Seedless cumulative reviewed frames for the branch up to this round."""
    parts = []
    for r in range(1, round_no + 1):
        f = branch_store(branch) / f"round{r}_labels.h5"
        if not f.exists():
            raise RuntimeError(f"missing {f}")
        parts.append(pd.read_hdf(f))
    merged = pd.concat(parts)
    merged = merged[~merged.index.duplicated(keep="last")].sort_index()
    return merged


def stage(branch: str, round_no: int) -> pd.DataFrame:
    if STAGING.exists():
        shutil.rmtree(STAGING)
    STAGING.mkdir(parents=True)
    table = cumulative_table(branch, round_no)
    table.to_hdf(STAGING / "CollectedData_Zhiheng.h5", key="df_with_missing", mode="w")
    for _, _, img in table.index:
        found = False
        for r in range(1, round_no + 1):
            for cand in [UNIT / "frames" / branch / img,
                         branch_store(branch) / f"round{r}_frames" / img]:
                if cand.exists():
                    shutil.copy(cand, STAGING / img)
                    found = True
                    break
            if found:
                break
        if not found:
            raise RuntimeError(f"png not found for {img}")
    log(f"{branch}: staged {len(table)} cumulative frames")
    return table


def unstage() -> None:
    if STAGING.exists():
        shutil.rmtree(STAGING)


def ensure_fraction(fraction2: float) -> int:
    cfg = yaml.safe_load(CONFIG.read_text())
    fr = [round(float(x), 2) for x in cfg["TrainingFraction"]]
    f2 = round(fraction2, 2)
    if f2 not in fr:
        cfg["TrainingFraction"].append(f2)
        CONFIG.write_text(yaml.safe_dump(cfg, sort_keys=False))
        log(f"TrainingFraction += {f2}")
        cfg = yaml.safe_load(CONFIG.read_text())
        fr = [round(float(x), 2) for x in cfg["TrainingFraction"]]
    return fr.index(f2)


def combined_indices(n_new: int) -> tuple[list[int], list[int], int]:
    """Row indices in DLC's combined table: face block first, then staging."""
    seed = pd.read_hdf(SEED_LABELS)
    names = [str(v[-1] if isinstance(v, tuple) else v) for v in seed.index]
    val = set(SPLIT["validation_filenames"])
    val_rows = [i for i, n in enumerate(names) if n in val]
    train_rows = [i for i, n in enumerate(names) if n not in val]
    n_seed = len(names)
    new_rows = list(range(n_seed, n_seed + n_new))
    return train_rows + new_rows, val_rows, n_seed + n_new


def patch_model_config(fraction_pct: int, shuffle: int, batch: int) -> None:
    path = (PROJECT / "dlc-models-pytorch" / "iteration-0"
            / f"EyePupilBlinkAug17-trainset{fraction_pct}shuffle{shuffle}"
            / "train" / "pytorch_config.yaml")
    c = yaml.safe_load(path.read_text())
    c["train_settings"]["batch_size"] = batch
    c["train_settings"]["epochs"] = EPOCHS
    c["runner"]["scheduler"]["params"]["milestones"] = [80, 95]
    c["runner"]["snapshots"]["save_epochs"] = 10
    c["runner"]["snapshots"]["max_snapshots"] = 12
    path.write_text(yaml.safe_dump(c, sort_keys=False))
    check = yaml.safe_load(path.read_text())
    assert check["runner"]["scheduler"]["params"]["milestones"] == [80, 95]
    log(f"patched model config trainset{fraction_pct}shuffle{shuffle}")


def labeled_numbers(branch: str, round_no: int) -> set[int]:
    seed = pd.read_hdf(SEED_LABELS)
    nums = {int(str(v[-1] if isinstance(v, tuple) else v)[3:-4]) for v in seed.index}
    for r in range(1, round_no + 1):
        t = pd.read_hdf(branch_store(branch) / f"round{r}_labels.h5")
        nums |= {int(i[-1][3:-4]) for i in t.index}
    return nums


def select_next(branch: str, spec: dict, shuffle: int, tsi: int, round_no: int) -> None:
    import deeplabcut

    log(f"{branch}: analyzing clip with round-{round_no} model ...")
    deeplabcut.analyze_videos(str(CONFIG), [str(CLIP)], shuffle=shuffle,
                              trainingsetindex=tsi, snapshot_index=-1,
                              device="mps", save_as_csv=False)
    already = labeled_numbers(branch, round_no)
    pick = 20
    for attempt in range(4):
        if STAGING.exists():
            shutil.rmtree(STAGING)
        cfg = yaml.safe_load(CONFIG.read_text())
        cfg["numframes2pick"] = pick
        CONFIG.write_text(yaml.safe_dump(cfg, sort_keys=False))
        np.random.seed(SEED)
        deeplabcut.extract_outlier_frames(str(CONFIG), [str(CLIP)], shuffle=shuffle,
                                          trainingsetindex=tsi,
                                          extractionalgorithm="kmeans",
                                          automatic=True, savelabeled=False,
                                          **spec["params"])
        picked = sorted(int(p.stem[3:]) for p in STAGING.glob("img*.png"))
        clean = [f for f in picked if f not in already]
        if len(clean) >= 20:
            break
        log(f"{branch}: {len(picked)-len(clean)} collisions, enlarging pick")
        pick += (20 - len(clean)) + 3
    clean = clean[:20]
    dest = UNIT / "frames" / f"round{round_no + 1}" / branch
    if dest.exists():
        shutil.rmtree(dest)
    dest.mkdir(parents=True)
    for f in clean:
        shutil.copy(STAGING / f"img{f:05d}.png", dest / f"img{f:05d}.png")
    for extra in STAGING.glob("machinelabels*"):
        shutil.copy(extra, dest / extra.name)
    pd.DataFrame({"frame": clean,
                  "time_s": [round(f / 60, 2) for f in clean]}
                 ).to_csv(dest / "manifest.csv", index=False)
    cfg = yaml.safe_load(CONFIG.read_text())
    cfg["numframes2pick"] = 20
    CONFIG.write_text(yaml.safe_dump(cfg, sort_keys=False))
    unstage()
    log(f"{branch}: next-round selection -> {dest} ({len(clean)} frames)")


def record_point(branch: str, round_no: int, cum_train: int, label: str) -> None:
    candidates = [
        ROOT / "local_data/test_sets/eye_last_minute_100"
        / f"predictions_100train_{label}" / "eye_test_summary.csv",
        ROOT / "model-selection/predictions"
        / f"predictions_100train_{label}" / "eye_test_summary.csv",
    ]
    path = next((p for p in candidates if p.exists()), None)
    if path is None:
        raise RuntimeError(f"no evaluation summary found for {label}")
    t = pd.read_csv(path)
    g = lambda n: float(t.loc[t["metric"] == n, "value"].iloc[0])
    conf = int(t.loc[t["metric"] == "overall_keypoint_rmse_px_pcutoff_0.6", "n"].iloc[0])
    row = dict(branch=branch, round=round_no, cumulative_training_frames=cum_train,
               external_overall_rmse_px=g("overall_keypoint_rmse_px"),
               external_pupil_center_rmse_px=g("pupil_center_rmse_px"),
               external_pupil_width_mae_px=g("pupil_width_mae_px"),
               points_likelihood_ge_0_6=conf, evaluated_label=label,
               recorded_at=datetime.now().isoformat(timespec="seconds"))
    CONVERGENCE.parent.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(CONVERGENCE) if CONVERGENCE.exists() else pd.DataFrame()
    df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    df.to_csv(CONVERGENCE, index=False)
    log(f"{branch}: convergence point recorded (round {round_no}, "
        f"{row['external_overall_rmse_px']:.2f} px)")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--round", type=int, required=True)
    args = ap.parse_args()
    r = args.round

    # round-0 baseline: an MPS-trained twin of the selected model, so the
    # whole convergence curve is device-homogeneous (the CPU-trained
    # model-selection winner stays untouched as shuffle 27).
    if not CONVERGENCE.exists():
        sys.path.insert(0, str(ROOT / "scripts"))
        from prepare_eye_split_80_20 import build_split, create_shuffle, label_frame_names
        names = label_frame_names()
        tr, va, _ = build_split(names)
        create_shuffle("resnet_50", 30, BATCH, tr, va)
        run([PYTHON, ROOT / "scripts/train_eye_model.py",
             "--shuffle", 30, "--batch-size", BATCH, "--epochs", EPOCHS,
             "--device", "mps", "--trainset-fraction", 80,
             "--trainingsetindex", 1, "--save-epochs", 10,
             "--max-snapshots", 12, "--no-resume"])
        run([PYTHON, ROOT / "scripts/evaluate_eye_test_set.py",
             "--train-frames", 100, "--shuffle", 30,
             "--trainingsetindex", 1, "--snapshot-index", -1,
             "--model-label", "al_r0_baseline"])
        for b in BRANCHES:
            record_point(b, 0, 80, "al_r0_baseline")

    import deeplabcut

    for branch, spec in BRANCHES.items():
        shuffle = spec["shuffle_base"] + (r - 1)
        log(f"===== {branch} round {r} (shuffle {shuffle}) =====")

        # persist this round's reviewed table into the branch store
        fresh = normalized_branch_table(branch, r)
        store = branch_store(branch)
        fresh.to_hdf(store / f"round{r}_labels.h5", key="df_with_missing", mode="w")
        fdir = store / f"round{r}_frames"
        fdir.mkdir(exist_ok=True)
        src_dir = UNIT / "frames" / branch if r == 1 else UNIT / "frames" / f"round{r}" / branch
        for img in fresh.index:
            shutil.copy(src_dir / img[-1], fdir / img[-1])

        table = stage(branch, r)
        train_idx, val_idx, total = combined_indices(len(table))
        fraction2 = round(len(train_idx) / total, 2)
        tsi = ensure_fraction(fraction2)
        fraction_pct = int(round(fraction2 * 100))
        log(f"{branch}: {len(train_idx)} train / {len(val_idx)} val "
            f"(fraction {fraction2}, trainingsetindex {tsi})")

        deeplabcut.create_training_dataset(
            str(CONFIG), Shuffles=[shuffle],
            trainIndices=[train_idx], testIndices=[val_idx],
            net_type="resnet_50", userfeedback=False,
            engine=deeplabcut.Engine.PYTORCH)
        patch_model_config(fraction_pct, shuffle, BATCH)

        run([PYTHON, ROOT / "scripts/train_eye_model.py",
             "--shuffle", shuffle, "--batch-size", BATCH, "--epochs", EPOCHS,
             "--device", "mps", "--trainset-fraction", fraction_pct,
             "--trainingsetindex", tsi, "--save-epochs", 10,
             "--max-snapshots", 12, "--no-resume"])

        label = f"al_{branch}_r{r}"
        run([PYTHON, ROOT / "scripts/evaluate_eye_test_set.py",
             "--train-frames", 100, "--shuffle", shuffle,
             "--trainingsetindex", tsi, "--snapshot-index", -1,
             "--model-label", label])
        record_point(branch, r, len(train_idx), label)

        select_next(branch, spec, shuffle, tsi, r)
        unstage()

    (UNIT / "logs" / f"round{r}_train_DONE.txt").write_text(
        datetime.now().isoformat() + "\n", encoding="utf-8")
    log(f"ROUND {r} PIPELINE COMPLETE")


if __name__ == "__main__":
    main()
