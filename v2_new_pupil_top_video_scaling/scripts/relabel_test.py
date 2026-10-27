#!/usr/bin/env python3
"""Blind re-label test: how repeatable are the human labels that serve as truth?

40 of the 1330 step-8 truth frames (test50 + val20 of the 19 videos, never trained on) are labeled a second time by
the same person, without seeing the old labels or any prediction, under the unchanged rule (label a point only if
it can be placed with confidence, else leave it empty). The second label measures the labeler's own repeatability
per point; comparing both labels with the plateau model's prediction shows whether the old labels (which started
from model pre-labels) were anchored to the model.

    python relabel_test.py make   -> results/relabel_test/labeling/relabel_blind/img001..040.png (local only)
                                     results/relabel_test/selection_key.csv   (do not open before labeling)
    bash  label_folder.sh results/relabel_test/labeling/relabel_blind        (napari, empty label layer)
    python relabel_test.py eval   -> results/relabel_test/relabel_test.csv, per_point.csv, per_group.csv

Selection (seed 20261026): 20 frames with frame RMSE 20-50 px (3-4 from each of videos 1, 4, 8, 10, 15, 18, the videos
where moderate errors concentrate), 10 frames > 50 px, 10 frames < 10 px; shuffled, renamed img001-img040 so the
group and the video are not visible in the file name. Frame selection uses the truth only to balance the groups;
the labeler sees nothing but the image.
"""
import argparse
import glob
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parents[1]
RES = HERE / "results"
OUT = RES / "relabel_test"
LABEL_DIR = OUT / "labeling" / "relabel_blind"
KEY = OUT / "selection_key.csv"
SCORES = RES / "error_detector" / "scores_lovo.csv"
POINTS = ["pupil_top", "pupil_bottom", "pupil_left", "pupil_right",
          "eyelid_top", "eyelid_bottom", "eye_nasal_corner", "eye_temporal_corner"]
PUPIL = POINTS[:4]
SEED = 20261026
MID_VIDEOS = [1, 4, 8, 10, 15, 18]


def truth_frames():
    p = pd.read_csv(SCORES)
    p = p[p.plateau].drop_duplicates(["video", "frame"]).reset_index(drop=True)
    assert len(p) == 1330, len(p)
    assert (p.frame_rmse_px > 20).sum() == 155 and (p.frame_rmse_px > 50).sum() == 32, "truth does not reproduce step 8"
    return p


def png_path(r):
    return HERE / r.unit / "training-data" / "labels" / r.set / f"img{int(r.frame):06d}.png"


def make():
    p = truth_frames()
    rng = np.random.default_rng(SEED)
    pick = []
    low = p[p.frame_rmse_px < 10]
    pick.append(low.sample(10, random_state=rng.integers(1 << 31)).assign(group="low_lt10"))
    high = p[p.frame_rmse_px > 50]
    pick.append(high.sample(10, random_state=rng.integers(1 << 31)).assign(group="high_gt50"))
    mid = p[(p.frame_rmse_px > 20) & (p.frame_rmse_px <= 50)]
    quota = dict(zip(MID_VIDEOS, [4, 4, 3, 3, 3, 3]))
    for v, n in quota.items():
        pool = mid[mid.video == v]
        n = min(n, len(pool))
        pick.append(pool.sample(n, random_state=rng.integers(1 << 31)).assign(group="mid_20to50"))
    sel = pd.concat(pick).reset_index(drop=True)
    assert len(sel) == 40, len(sel)
    sel = sel.sample(frac=1, random_state=rng.integers(1 << 31)).reset_index(drop=True)
    sel["blind_name"] = [f"img{i + 1:03d}.png" for i in range(len(sel))]
    if LABEL_DIR.exists():
        raise SystemExit(f"{LABEL_DIR} exists; remove it by hand if you want a fresh set")
    LABEL_DIR.mkdir(parents=True)
    for r in sel.itertuples():
        src = png_path(r)
        assert src.exists(), src
        shutil.copy2(src, LABEL_DIR / r.blind_name)
    cols = ["blind_name", "group", "video", "unit", "set", "frame", "frame_rmse_px"]
    sel[cols].to_csv(KEY, index=False)
    write_empty_label_file(sel.blind_name.tolist())
    print(f"{len(sel)} frames copied to {LABEL_DIR}")
    print(sel.groupby("group").size().to_string())
    print(f"key written to {KEY} (do not open it before labeling)")


DUMMY = "img041.png"   # an all-black extra image with one placeholder point: napari-deeplabcut 0.3.1 cannot advance to
                       # the next key point when the label layer is empty (id becomes nan), so the layer must hold one
                       # real point; it sits on this black frame, which the labeler skips, and eval() ignores it


def write_empty_label_file(names):
    """A CollectedData_Zhiheng file in the same format as the existing label sets: all 40 test images NaN, plus the
    dummy frame with one point, so napari-deeplabcut opens the folder as a label set and saves back into the file."""
    template = HERE / "8_20251024_Pluto_spont1" / "training-data" / "labels" / "test50" / "CollectedData_Zhiheng.h5"
    cols = pd.read_hdf(template).columns
    # napari-deeplabcut addresses rows by the last three path parts of the image (parent of parent, parent, file)
    idx = pd.MultiIndex.from_tuples([(LABEL_DIR.parent.name, LABEL_DIR.name, n) for n in names + [DUMMY]])
    df = pd.DataFrame(np.nan, index=idx, columns=cols)
    df.loc[(LABEL_DIR.parent.name, LABEL_DIR.name, DUMMY), [("Zhiheng", "pupil_top", "x"), ("Zhiheng", "pupil_top", "y")]] = [20.0, 20.0]
    from PIL import Image
    Image.new("RGB", Image.open(LABEL_DIR / names[0]).size, (0, 0, 0)).save(LABEL_DIR / DUMMY)
    df.to_hdf(LABEL_DIR / "CollectedData_Zhiheng.h5", key="df_with_missing", mode="w")
    df.to_csv(LABEL_DIR / "CollectedData_Zhiheng.csv")


def read_new_labels():
    files = sorted(glob.glob(str(LABEL_DIR / "CollectedData_*.h5")))
    if files:
        df = pd.read_hdf(files[-1])
    else:
        files = sorted(glob.glob(str(LABEL_DIR / "CollectedData_*.csv")))
        if not files:
            raise SystemExit(f"no CollectedData file in {LABEL_DIR}: label first")
        df = pd.read_csv(files[-1], header=[0, 1, 2], index_col=[0, 1, 2])
    if df.columns.nlevels == 4:
        df.columns = df.columns.droplevel("individuals")
    names = [Path(i[-1] if isinstance(i, tuple) else i).name for i in df.index]
    rows = {}
    for name, (_, row) in zip(names, df.iterrows()):
        d = {}
        for bp in POINTS:
            sc = row.index.get_level_values(0)[0]
            d[f"new_{bp}_x"] = float(row[(sc, bp, "x")])
            d[f"new_{bp}_y"] = float(row[(sc, bp, "y")])
        rows[name] = d
    rows.pop(DUMMY, None)
    return pd.DataFrame.from_dict(rows, orient="index").rename_axis("blind_name").reset_index()


def area(df, pre):
    a = np.hypot(df[f"{pre}pupil_left_x"] - df[f"{pre}pupil_right_x"], df[f"{pre}pupil_left_y"] - df[f"{pre}pupil_right_y"])
    b = np.hypot(df[f"{pre}pupil_top_x"] - df[f"{pre}pupil_bottom_x"], df[f"{pre}pupil_top_y"] - df[f"{pre}pupil_bottom_y"])
    return np.pi * a * b / 4


def evaluate():
    key = pd.read_csv(KEY)
    p = truth_frames()
    new = read_new_labels()
    m = key.merge(p, on=["video", "unit", "set", "frame", "frame_rmse_px"]).merge(new, on="blind_name", how="left")
    assert len(m) == 40, len(m)
    n_labeled = m[[f"new_{bp}_x" for bp in POINTS]].notna().sum(axis=1)
    print("frames with a new label file row:", m.new_pupil_top_x.notna().sum() + m.new_pupil_top_x.isna().sum(),
          "| points labeled per frame (median):", int(n_labeled.median()))
    for bp in POINTS:
        for a, b, tag in (("human", "new", "old_new"), ("human", "pred", "old_pred"), ("new", "pred", "new_pred")):
            m[f"{tag}_{bp}"] = np.hypot(m[f"{a}_{bp}_x"] - m[f"{b}_{bp}_x"], m[f"{a}_{bp}_y"] - m[f"{b}_{bp}_y"])
    for tag in ("old_new", "old_pred", "new_pred"):
        m[f"{tag}_rmse8"] = np.sqrt((m[[f"{tag}_{bp}" for bp in POINTS]] ** 2).mean(axis=1))
        m[f"{tag}_rmse_pupil"] = np.sqrt((m[[f"{tag}_{bp}" for bp in PUPIL]] ** 2).mean(axis=1))
    m["area_old"], m["area_new"], m["area_pred"] = area(m, "human_"), area(m, "new_"), area(m, "pred_")
    m["area_old_new_pct"] = (m.area_new - m.area_old).abs() / m.area_old * 100
    m["area_old_pred_pct"] = (m.area_pred - m.area_old).abs() / m.area_old * 100
    m["area_new_pred_pct"] = (m.area_pred - m.area_new).abs() / m.area_new * 100
    m["new_points_labeled"] = n_labeled
    keep = ["blind_name", "group", "video", "set", "frame", "frame_rmse_px", "new_points_labeled",
            "old_new_rmse8", "old_new_rmse_pupil", "old_pred_rmse8", "old_pred_rmse_pupil", "new_pred_rmse8", "new_pred_rmse_pupil",
            "area_old", "area_new", "area_pred", "area_old_new_pct", "area_old_pred_pct", "area_new_pred_pct"] \
           + [f"{tag}_{bp}" for tag in ("old_new", "old_pred", "new_pred") for bp in POINTS]
    m[keep].round(2).to_csv(OUT / "relabel_test.csv", index=False)

    rows = []
    for bp in POINTS:
        for g, s in [("all", m)] + [(g, s) for g, s in m.groupby("group")]:
            d = s[f"old_new_{bp}"].dropna()
            rows.append(dict(point=bp, group=g, n=len(d), old_new_median=d.median(), old_new_p90=d.quantile(.9) if len(d) else np.nan,
                             old_pred_median=s[f"old_pred_{bp}"].median(), new_pred_median=s[f"new_pred_{bp}"].dropna().median(),
                             n_empty_new=int(s[f"new_{bp}_x"].isna().sum())))
    pp = pd.DataFrame(rows).round(2)
    pp.to_csv(OUT / "per_point.csv", index=False)

    rows = []
    for g, s in [("all", m)] + [(g, s) for g, s in m.groupby("group")]:
        rows.append(dict(group=g, n=len(s),
                         old_new_rmse_pupil_median=s.old_new_rmse_pupil.median(), old_new_rmse_pupil_p90=s.old_new_rmse_pupil.quantile(.9),
                         old_new_rmse8_median=s.old_new_rmse8.median(),
                         old_pred_rmse_pupil_median=s.old_pred_rmse_pupil.median(), new_pred_rmse_pupil_median=s.new_pred_rmse_pupil.median(),
                         frames_old_new_gt20=int((s.old_new_rmse8 > 20).sum()), frames_old_pred_gt20=int((s.old_pred_rmse8 > 20).sum()),
                         frames_new_pred_gt20=int((s.new_pred_rmse8 > 20).sum()),
                         frames_new_pred_gt50=int((s.new_pred_rmse8 > 50).sum()), frames_old_pred_gt50=int((s.old_pred_rmse8 > 50).sum()),
                         area_old_new_median_pct=s.area_old_new_pct.median(), area_old_pred_median_pct=s.area_old_pred_pct.median(),
                         area_new_pred_median_pct=s.area_new_pred_pct.median(),
                         model_closer_to_old=int((s.old_pred_rmse_pupil < s.new_pred_rmse_pupil).sum())))
    pg = pd.DataFrame(rows).round(2)
    pg.to_csv(OUT / "per_group.csv", index=False)
    pd.set_option("display.width", 220)
    print(pg.to_string(index=False))
    print(pp[pp.group == "all"].to_string(index=False))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["make", "eval"])
    a = ap.parse_args()
    make() if a.cmd == "make" else evaluate()
