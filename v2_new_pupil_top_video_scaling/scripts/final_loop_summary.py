#!/usr/bin/env python3
"""One table for the final loop (experiment 5b) on all 19 videos, turn by turn (plan of 2026-11-24): does each turn
change the number of wrong frames, judged against the seed noise. Reads only committed result tables; nothing is
recomputed from predictions, so every number here is the number in those tables.

Comparisons (each on frames no compared model trained on):
  start -> turn 1   truth frames (70 per video); check frames: final/ (15 videos, 34 drawn after turn 2), blind set +
                    supplement (videos 1, 6, 8, 15)
  turn 1 -> turn 2  truth frames; check frames: final/ (15 videos), turn2/ (videos 1, 6, 8, 15)
  turn 2 -> turn 3  truth frames; check frames: turn3/ (drawn like turn2/, blind_turn2_set.py --turn 3), plus final/ and
                    turn2/ as supplementary rows
Seed noise: turn 1 with seed 42 against seed 43 on the turn2/ check frames of videos 1, 6, 8, 15 (same draw design as
turn3/): the share of frames on which two equally trained models disagree (one wrong, one right) = p_flip.

Decision rule, fixed before the turn-3 results (plan 2026-11-24): a turn "reduces wrong frames" when on its check frames
(a) the total drop is larger than the seed band 2 * sqrt(p_flip * N) (N = frames compared), and (b) more videos improve
than worsen with a two-sided sign test p < 0.05; otherwise "no effect seen". Also given: McNemar exact test on the
pooled frame pairs, a bootstrap 95 % interval over videos for the total change, weighted wrong share (stratum weights,
= the eligible part of the videos) for the stratified sets.

    python final_loop_summary.py            -> results/self_training/final_loop_summary.csv (+ printed tables)
"""
import math
from pathlib import Path

import numpy as np
import pandas as pd

H = Path(__file__).resolve().parents[1]
ST, BT = H / "results" / "self_training", H / "results" / "blind_test"
LINE = 0.14
BLIND4 = (1, 6, 8, 15)
MODEL = {0: "teacher", 1: "round1_formal", 2: "round2_formal", 3: "round3_formal"}


def truth_table():
    rows = []
    for v in range(19):
        for t, m in MODEL.items():
            p = ST / f"video{v:02d}" / m / "eval.csv"
            if p.exists():
                e = pd.read_csv(p)
                e = e[e.model == m].iloc[-1]
                rows.append(dict(video=v, turn=t, wrong=int(e.wrong_frames), frames=int(e.truth_frames),
                                 median_pct=float(e.median_error_pct), flagged_pct=float(e.flagged_either_pct)))
    return pd.DataFrame(rows)


def check_frames(v, sets):
    """Per-frame errors of every model on the given check sets of one video (committed per_frame.csv files)."""
    out = []
    for s in sets:
        p = {"final": BT / f"video{v:02d}" / "final" / "per_frame.csv", "turn2": BT / f"video{v:02d}" / "turn2" / "per_frame.csv",
             "turn3": BT / f"video{v:02d}" / "turn3" / "per_frame.csv", "blind": BT / f"video{v:02d}" / "per_frame.csv",
             "supplement": BT / f"video{v:02d}" / "supplement" / "per_frame.csv"}[s]
        if p.exists():
            E = pd.read_csv(p)
            E["set"] = s
            out.append(E)
    return pd.concat(out) if out else None


def pairs(E, a, b):
    P = E[E.model.isin([a, b])].pivot_table(index=["set", "frame"], columns="model", values="error")
    W = E[E.model == a].set_index(["set", "frame"])[["weight", "stratum"]]
    P = P.dropna().join(W)
    return P


def sign_test(k_up, k_down):
    n = k_up + k_down
    if n == 0:
        return 1.0
    k = min(k_up, k_down)
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n)


def mcnemar(b, c):
    return sign_test(b, c)        # exact McNemar = two-sided binomial on the discordant pairs


def compare(name, a, b, sets_by_video, p_flip, rng):
    rows = []
    for v, sets in sets_by_video.items():
        E = check_frames(v, sets)
        if E is None or not {a, b} <= set(E.model):
            continue
        P = pairs(E, a, b)
        wa, wb = P[a] > LINE, P[b] > LINE
        w = P.weight.fillna(0)
        rows.append(dict(comparison=name, video=v, sets="+".join(sorted(set(P.index.get_level_values(0)))), frames=len(P),
                         wrong_a=int(wa.sum()), wrong_b=int(wb.sum()), a_wrong_b_right=int((wa & ~wb).sum()), a_right_b_wrong=int((~wa & wb).sum()),
                         weighted_a=float((w * wa).sum() / w.sum()) if w.sum() > 0 else np.nan, weighted_b=float((w * wb).sum() / w.sum()) if w.sum() > 0 else np.nan))
    if not rows:
        return pd.DataFrame(), None
    T = pd.DataFrame(rows)
    d = T.wrong_b - T.wrong_a
    N = int(T.frames.sum())
    boot = [d.iloc[rng.integers(0, len(d), len(d))].sum() for _ in range(5000)]
    band = 2 * math.sqrt(p_flip * N)
    up, down = int((d < 0).sum()), int((d > 0).sum())
    ps = sign_test(up, down)
    S = dict(comparison=name, video="all", frames=N, wrong_a=int(T.wrong_a.sum()), wrong_b=int(T.wrong_b.sum()),
             a_wrong_b_right=int(T.a_wrong_b_right.sum()), a_right_b_wrong=int(T.a_right_b_wrong.sum()),
             videos_better=up, videos_worse=down, videos_same=int((d == 0).sum()), sign_p=round(ps, 4),
             mcnemar_p=round(mcnemar(int(T.a_wrong_b_right.sum()), int(T.a_right_b_wrong.sum())), 4),
             change=int(d.sum()), boot_lo=float(np.percentile(boot, 2.5)), boot_hi=float(np.percentile(boot, 97.5)),
             seed_band=round(band, 1), weighted_a=float(T.weighted_a.mean()), weighted_b=float(T.weighted_b.mean()),
             verdict="reduces wrong frames" if (d.sum() < -band and ps < 0.05 and up > down) else "no effect seen")
    return T, S


def main():
    rng = np.random.default_rng(42)
    tr = truth_table()
    # seed noise on the turn2/ check frames (turn 1, seed 42 vs seed 43)
    flips = frames = 0
    for v in BLIND4:
        E = check_frames(v, ["turn2"])
        if E is not None and "round1_formal_seed43" in set(E.model):
            P = pairs(E, "round1_formal", "round1_formal_seed43")
            a, b = P["round1_formal"] > LINE, P["round1_formal_seed43"] > LINE
            flips += int((a ^ b).sum()); frames += len(P)
    p_flip = flips / frames
    print(f"seed noise: {flips} of {frames} turn2/ check frames wrong for one seed only (p_flip {p_flip:.3f})")
    others = [v for v in range(19) if v not in BLIND4]
    plan = [("start -> turn 1", "teacher", "round1_formal", {**{v: ["final"] for v in others}, **{v: ["blind", "supplement"] for v in BLIND4}}),
            ("turn 1 -> turn 2", "round1_formal", "round2_formal", {**{v: ["final"] for v in others}, **{v: ["turn2"] for v in BLIND4}}),
            ("turn 2 -> turn 3", "round2_formal", "round3_formal", {v: ["turn3"] for v in range(19)}),
            ("turn 2 -> turn 3 (earlier check frames)", "round2_formal", "round3_formal", {**{v: ["final"] for v in others}, **{v: ["turn2"] for v in BLIND4}})]
    out, summ = [], []
    for name, a, b, sets in plan:
        T, S = compare(name, a, b, sets, p_flip, rng)
        if S is None:
            print(f"{name}: no check frames scored yet"); continue
        out.append(T); summ.append(S)
    # truth frames, turn by turn
    for t0, t1 in ((0, 1), (1, 2), (2, 3)):
        A = tr[tr.turn == t0].set_index("video"); B = tr[tr.turn == t1].set_index("video")
        J = A.join(B, lsuffix="_a", rsuffix="_b", how="inner")
        if len(J) == 0:
            continue
        d = J.wrong_b - J.wrong_a
        summ.append(dict(comparison=f"{['start', 'turn 1', 'turn 2'][t0]} -> turn {t1} (truth frames)", video="all", frames=int(J.frames_a.sum()),
                         wrong_a=int(J.wrong_a.sum()), wrong_b=int(J.wrong_b.sum()), videos_better=int((d < 0).sum()), videos_worse=int((d > 0).sum()),
                         videos_same=int((d == 0).sum()), sign_p=round(sign_test(int((d < 0).sum()), int((d > 0).sum())), 4), change=int(d.sum()),
                         median_a=round(float(J.median_pct_a.mean()), 2), median_b=round(float(J.median_pct_b.mean()), 2),
                         flagged_a=round(float(J.flagged_pct_a.mean()), 2), flagged_b=round(float(J.flagged_pct_b.mean()), 2), n_videos=len(J)))
    S = pd.DataFrame(summ)
    allrows = pd.concat([S] + out, ignore_index=True)
    allrows.insert(0, "p_flip_seed", round(p_flip, 4))
    allrows.to_csv(ST / "final_loop_summary.csv", index=False)
    pd.set_option("display.width", 250); pd.set_option("display.max_columns", 30)
    print(S.to_string(index=False))
    print(f"written {ST / 'final_loop_summary.csv'}")


if __name__ == "__main__":
    main()
