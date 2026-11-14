"""The two independent detectors for every model of the self-training experiments (page 9.4, 2026-11-14).

evaluate() has written them into eval.csv since 2026-11-08 (indep_lowconf_pct, indep_jump_pct); models evaluated
before that do not have them. This script computes them for every model that has a whole-video prediction, with the
same function (self_training.independent_detectors) and the same thresholds (where the video's starting model flags
5 %), and writes one table; it changes no existing file. Where eval.csv holds the values, they must be equal.

  results/self_training/round2_analysis/independent_detectors.csv
  python scripts/independent_detectors.py
"""
import re

import numpy as np
import pandas as pd

import pupil_trace as pt
import self_training as S

OUT = S.OUT / "round2_analysis"


def main():
    rows = []
    for vd in sorted(S.OUT.glob("video[0-9][0-9]")):
        v = int(vd.name[5:])
        if not (vd / "teacher" / "eval.csv").exists():
            continue
        rr, dt, P, H, truth, fwd, sig = S.load(v)
        Wm = S.teacher_eye_width(dt)
        t = S.independent_detectors(dt, dt, Wm)
        rows.append({"video": v, "model": "teacher", "experiment": "teacher", "turn": 0, "seed": 42, **t, "in_eval_csv": False})
        for md in sorted(vd.glob("round*_*")):
            h5 = sorted((S.LOCAL / vd.name / md.name / "predictions").glob("*.h5"))
            if not (md / "eval.csv").exists() or not h5:
                continue
            m = re.match(r"round(\d+)_([a-z0-9]+?)(_seed(\d+))?$", md.name)
            r = S.independent_detectors(pt.load(str(h5[-1])), dt, Wm)
            e = pd.read_csv(md / "eval.csv").iloc[0]
            have = "indep_lowconf_pct" in e.index and pd.notna(e.get("indep_lowconf_pct"))
            if have:
                assert abs(float(e.indep_lowconf_pct) - r["indep_lowconf_pct"]) < 0.011 and abs(float(e.indep_jump_pct) - r["indep_jump_pct"]) < 0.011, (v, md.name)
            rows.append({"video": v, "model": md.name, "experiment": m.group(2), "turn": int(m.group(1)), "seed": int(m.group(4) or 42), **r, "in_eval_csv": bool(have)})
            print(v, md.name, r["indep_lowconf_pct"], r["indep_jump_pct"], "(equal to eval.csv)" if have else "(new)", flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    D = pd.DataFrame(rows)
    D.to_csv(OUT / "independent_detectors.csv", index=False)
    print(len(D), "rows;", int((~D.in_eval_csv & (D.model != "teacher")).sum()), "models new;", int(D.in_eval_csv.sum()), "checked against eval.csv")


if __name__ == "__main__":
    main()
