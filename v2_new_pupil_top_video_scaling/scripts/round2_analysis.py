"""Second turn of every experiment on videos 1, 8, 15, 6: which frames help (page 9.4, 2026-11-12).

Reads existing results only and changes none of them. Writes three small tables to results/self_training/round2_analysis/:

  summary.csv        one row per video x experiment x turn: frames fed back, median truth error, wrong truth frames (of 70),
                     flagged share, wrong blind frames (24 + 10)
  blind_changes.csv  turn 1 against turn 2 on the 34 blind frames, frame by frame, by stratum: wrong -> right,
                     right -> wrong, wrong in both, right in both (wrong = error over the 14 % line)
  stop_rule.csv      every stop decision of experiment 3 and of the final loop (5b) with the flagged ratio and the blind
                     change, next to what two other rules would have decided (flagged ratio alone; blind count must
                     rise by 2 or more to stop). The rule in use is not changed.

  python scripts/round2_analysis.py
"""
from pathlib import Path

import numpy as np
import pandas as pd

import blind_test_set as B

HERE = Path(__file__).resolve().parents[1]
RES = HERE / "results"
ST = RES / "self_training"
OUT = ST / "round2_analysis"
VIDEOS = (1, 8, 15, 6)
LINE = 0.14
# experiment -> (label, model name pattern, pseudo folder of turn 1, pseudo folder pattern of later turns)
EXPS = {"every30": "experiment 1 (grid frames only)", "repaired": "experiment 2 (repaired frames only)",
        "both": "experiment 3 (both kinds)", "reopened": "experiment 4c (reopened frames only)",
        "formal": "experiment 5b (final version)"}


def pseudo_kept(v, src, r):
    f = ST / f"video{v:02d}" / (f"pseudo_{src}" if r == 1 else f"pseudo_{src}_round{r}") / "pseudo_summary.csv"
    return int(pd.read_csv(f).iloc[0].kept_frames) if f.exists() else np.nan


def blind_frames(v):
    """Per-frame blind errors of every model scored in results/blind_test (24 scored + 10 supplement frames)."""
    P = [pd.read_csv(RES / "blind_test" / f"video{v:02d}" / f) for f in ("per_frame.csv", "supplement/per_frame.csv")]
    return pd.concat(P, ignore_index=True)[["model", "frame", "blind_name", "stratum", "error"]]


def blind_frames_model(v, models):
    """The same per-frame errors for models blind_test_set.py does not list (4c), computed with its own functions."""
    h5 = {}
    for m in models:
        p = sorted((B.LOCAL / f"video{v:02d}" / m / "predictions").glob("*.h5"))
        if p:
            h5[m] = p[-1]
    if not h5:
        return pd.DataFrame(columns=["model", "frame", "blind_name", "stratum", "error"])
    rows = []
    for sup in (False, True):
        B.SUP = sup
        E, _ = B.errors(v, B.read_labels(v), h5)
        rows.append(E[np.isfinite(E.error)][["model", "frame", "blind_name", "stratum", "error"]])
    B.SUP = False
    return pd.concat(rows, ignore_index=True)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    summ, chg = [], []
    for v in VIDEOS:
        P = blind_frames(v)
        extra = blind_frames_model(v, [f"round{r}_reopened" for r in (1, 2)] + ["round3_every30", "round3_repaired"])   # models blind_test_set.py does not list
        P = pd.concat([P, extra], ignore_index=True)
        for src, label in [("teacher", "starting model")] + list(EXPS.items()):
            for r in ((0,) if src == "teacher" else (1, 2, 3)):
                m = "teacher" if src == "teacher" else f"round{r}_{src}"
                e = ST / f"video{v:02d}" / m / "eval.csv"
                if not e.exists():
                    continue
                e = pd.read_csv(e).iloc[0]
                g = P[P.model == m]
                main_ = g[~g.stratum.astype(str).str.startswith(("E", "F"))]
                sup = g[g.stratum.astype(str).str.startswith(("E", "F"))]
                if m.endswith("_reopened") or m.startswith("round3_"):   # the committed one-model score must agree
                    bs = ST / f"video{v:02d}" / m / "blind_scores.csv"
                    if bs.exists() and len(g):
                        S = pd.read_csv(bs)
                        S = S[S.model == m]
                        assert int(S[S.set == "main"].wrong_frames.iloc[0]) == int((main_.error > LINE).sum()), (v, m)
                        assert int(S[S.set == "supplement"].wrong_frames.iloc[0]) == int((sup.error > LINE).sum()), (v, m)
                summ.append({"video": v, "experiment": src, "label": label, "turn": r, "model": m,
                             "frames_fed_back": np.nan if src == "teacher" else pseudo_kept(v, src, r),
                             "median_error_pct": float(e.median_error_pct), "wrong_truth_of_70": int(e.wrong_frames),
                             "flagged_either_pct": float(e.flagged_either_pct),
                             "blind_frames": len(g), "blind_wrong_24": int((main_.error > LINE).sum()) if len(g) else np.nan,
                             "blind_wrong_10": int((sup.error > LINE).sum()) if len(g) else np.nan,
                             "blind_wrong_34": int((g.error > LINE).sum()) if len(g) else np.nan})
        for src in EXPS:                                          # turn 1 against turn 2, frame by frame
            a, b = P[P.model == f"round1_{src}"], P[P.model == f"round2_{src}"]
            if not len(a) or not len(b):
                continue
            M = a.merge(b, on=["frame", "blind_name", "stratum"], suffixes=("_1", "_2"))
            T = P[P.model == "teacher"][["frame", "error"]].rename(columns={"error": "error_0"})
            M = M.merge(T, on="frame", how="left")
            M["w0"], M["w1"], M["w2"] = M.error_0 > LINE, M.error_1 > LINE, M.error_2 > LINE
            for s, g in M.groupby("stratum"):
                chg.append({"video": v, "experiment": src, "stratum": s, "frames": len(g),
                            "wrong_start": int(g.w0.sum()), "wrong_turn1": int(g.w1.sum()), "wrong_turn2": int(g.w2.sum()),
                            "wrong_to_right": int((g.w1 & ~g.w2).sum()), "right_to_wrong": int((~g.w1 & g.w2).sum()),
                            "wrong_both": int((g.w1 & g.w2).sum()), "right_both": int((~g.w1 & ~g.w2).sum()),
                            "right_to_wrong_frames": " ".join(g.blind_name[~g.w1 & g.w2]),
                            "wrong_to_right_frames": " ".join(g.blind_name[g.w1 & ~g.w2])})
    S = pd.DataFrame(summ)
    S.to_csv(OUT / "summary.csv", index=False)
    C = pd.DataFrame(chg)
    C.to_csv(OUT / "blind_changes.csv", index=False)

    stop = []
    for v in VIDEOS:
        for name, f in (("experiment 3", "loop_status.csv"), ("experiment 5b", "loop_status_formal.csv")):
            f = ST / f"video{v:02d}" / f
            if not f.exists():
                continue
            for _, r in pd.read_csv(f).drop_duplicates("round", keep="last").iterrows():
                ratio = r.flagged / r.flagged_prev
                d = int(r.blind_wrong - r.blind_wrong_prev)
                stop.append({"video": v, "loop": name, "turn": int(r["round"]), "flagged_prev": int(r.flagged_prev), "flagged": int(r.flagged),
                             "flagged_ratio": round(ratio, 2), "blind_prev": int(r.blind_wrong_prev), "blind": int(r.blind_wrong), "blind_change": d,
                             "rule_in_use_stops": bool(str(r.stop) == "True"),
                             "fired_flagged": bool(ratio >= 0.9), "fired_blind": bool(d >= 0),
                             "blind_change_within_seed_noise": bool(abs(d) <= 1),
                             "alt_flagged_only_stops": bool(ratio >= 0.9),
                             "alt_flagged_or_blind_up_2_stops": bool(ratio >= 0.9 or d >= 2)})
    R = pd.DataFrame(stop)
    R.to_csv(OUT / "stop_rule.csv", index=False)

    pd.set_option("display.width", 250)
    print(S.drop(columns=["label", "model"]).to_string(index=False))
    tot = S[S.blind_frames > 0].groupby(["experiment", "turn"]).agg(videos=("video", "count"), blind_wrong=("blind_wrong_34", "sum"),
                                                                   truth_wrong=("wrong_truth_of_70", "sum"))
    print("\ntotals over the videos that have the model\n", tot.to_string())
    print("\nturn 1 -> turn 2 on the blind frames, summed over videos and strata")
    print(C.groupby("experiment")[["frames", "wrong_turn1", "wrong_turn2", "wrong_to_right", "right_to_wrong", "wrong_both"]].sum().to_string())
    print("\nby stratum")
    print(C.groupby(["experiment", "stratum"])[["frames", "wrong_to_right", "right_to_wrong", "wrong_both"]].sum().to_string())
    print("\nstop rule\n", R.to_string(index=False))


if __name__ == "__main__":
    main()
