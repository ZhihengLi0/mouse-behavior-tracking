"""Score the models on the second blind set (results/blind_test/videoNN/turn2, drawn by blind_turn2_set.py): the frames
labeled by the user (the subset marked in selection_key.csv; the placeholder img000.png, which napari-deeplabcut needs
to open an unlabeled folder, is ignored), same error and line as blind_test_set.py.

    python blind_turn2_score.py --video 1                          labels as they are now (after the user's review)
    python blind_turn2_score.py --video 1 --labels before_review   the labels before the review (labeling/before_review/)

Writes turn2/per_frame[_before_review].csv (model, frame, blind_name, stratum, error) and scores[_before_review].csv
(model x stratum). The user labeled blind (2026-11-14) and then reviewed every frame after being told that the nasal
corner lay further toward the nose than the models'; both label sets are kept."""
import argparse, glob, sys
from pathlib import Path
import numpy as np
import pandas as pd
import blind_test_set as B

MODELS = ["round1_formal", "round2_formal", "round3_formal", "round1_formal_seed43", "round1_both", "round2_both", "round3_both",
          "round1_repaired", "round2_repaired", "round3_repaired", "round1_reopened", "round2_reopened"]


def read_labels(v, which="blind"):
    d = B.OUT / f"video{v:02d}" / "turn2"
    key = pd.read_csv(d / "selection_key.csv")
    key = key[key.subset] if "subset" in key.columns else key
    key = key.set_index("blind_name")
    df = pd.read_csv(d / "labeling" / which / "CollectedData_Zhiheng.csv", header=[0, 1, 2], index_col=[0, 1, 2])   # the committed csv
    if df.columns.nlevels == 4:
        df.columns = df.columns.droplevel("individuals")
    sc = df.columns.get_level_values(0)[0]
    rows = []
    for i, row in df.iterrows():
        nm = Path(i[-1] if isinstance(i, tuple) else i).name
        if nm not in key.index:
            continue
        r = {"blind_name": nm, "frame": int(key.loc[nm, "frame"]), "stratum": key.loc[nm, "stratum"],
             "weight": float(key.loc[nm, "weight_subset"] if "weight_subset" in key.columns else key.loc[nm, "weight"])}
        for b in B.BPS:
            r[f"human_{b}_x"], r[f"human_{b}_y"] = float(row[(sc, b, "x")]), float(row[(sc, b, "y")])
        rows.append(r)
    L = pd.DataFrame(rows).set_index("frame")
    L["labeled_points"] = sum(np.isfinite(L[f"human_{b}_x"]).astype(int) for b in B.BPS)
    return L[L.labeled_points > 0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", type=int, required=True)
    ap.add_argument("--labels", default="blind", choices=["blind", "before_review"])
    a = ap.parse_args()
    v, suffix = a.video, ("" if a.labels == "blind" else "_before_review")
    L = read_labels(v, a.labels)
    models = {"teacher": B.teacher_h5(v)}
    for m in MODELS:
        p = sorted((B.LOCAL / f"video{v:02d}" / m / "predictions").glob("*.h5"))
        if p:
            models[m] = p[-1]
    E, _ = B.errors(v, L, models)
    E = E[np.isfinite(E.error)]
    out = B.OUT / f"video{v:02d}" / "turn2"
    keep = [c for c in ("model", "frame", "blind_name", "stratum", "weight", "labeled_points", "error", "error_px") if c in E.columns]
    E[keep].round(5).to_csv(out / f"per_frame{suffix}.csv", index=False)
    E["wrong"] = E.error > B.LINE
    S = E.groupby(["model", "stratum"], sort=False).agg(frames=("wrong", "size"), wrong=("wrong", "sum"), median_error_pct=("error", lambda x: round(100 * float(x.median()), 2))).reset_index()
    T = E.groupby("model", sort=False).agg(frames=("wrong", "size"), wrong=("wrong", "sum"), median_error_pct=("error", lambda x: round(100 * float(x.median()), 2))).reset_index()
    T.insert(1, "stratum", "all")
    pd.concat([S, T]).to_csv(out / f"scores{suffix}.csv", index=False)
    P = E.pivot_table(index="stratum", columns="model", values="wrong", aggfunc="sum").reindex(["T1", "TB", "T2", "TN"])
    P.loc["all"] = P.sum()
    P.insert(0, "frames", [int((L.stratum == s).sum()) for s in ["T1", "TB", "T2", "TN"]] + [len(L)])
    pd.set_option("display.width", 250)
    print(f"video {v} ({a.labels}): {len(L)} labeled frames; wrong frames (error over the {B.LINE:.2f} line) by stratum and model")
    print(P[["frames"] + [m for m in models if m in P.columns]].astype(int).to_string())
    a, b = E[E.model == "round1_formal"].set_index("frame"), E[E.model == "round2_formal"].set_index("frame")
    J = a[["stratum", "wrong", "error"]].join(b[["wrong", "error"]], rsuffix="_2")
    print("\n5b turn 1 -> turn 2, frame by frame:  wrong->right", int((J.wrong & ~J.wrong_2).sum()), "| right->wrong", int((~J.wrong & J.wrong_2).sum()),
          "| wrong in both", int((J.wrong & J.wrong_2).sum()), "| right in both", int((~J.wrong & ~J.wrong_2).sum()))
    print(J.groupby("stratum").apply(lambda g: pd.Series({"wrong->right": int((g.wrong & ~g.wrong_2).sum()), "right->wrong": int((~g.wrong & g.wrong_2).sum()),
                                                           "wrong both": int((g.wrong & g.wrong_2).sum()), "right both": int((~g.wrong & ~g.wrong_2).sum())})).reindex(["T1", "TB", "T2", "TN"]).to_string())
    d = (J.error_2 - J.error)
    print("\nframes better by more than 1 % of the eye width in turn 2:", int((d < -0.01).sum()), "| worse:", int((d > 0.01).sum()), "| median error turn 1 %.2f %%, turn 2 %.2f %%" % (100 * J.error.median(), 100 * J.error_2.median()))


if __name__ == "__main__":
    main()
