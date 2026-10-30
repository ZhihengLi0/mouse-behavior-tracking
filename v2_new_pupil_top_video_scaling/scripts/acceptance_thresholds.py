#!/usr/bin/env python3
"""Acceptance tests for the second-peak repair that have a basis, and their cuts read off the truth frames.

    python acceptance_thresholds.py

Question (user, 2026-10-30): the geometry rule that accepts or refuses a repair was made by hand. What could accept a
repair with a basis? Three candidates, each evaluated on the same candidate repairs:
  conf      DeepLabCut's own confidence applied to the candidate point: the heatmap value of the second peak
            (smallest over the replaced points); accept if >= cut
  jump      DeepLabCut's jump criterion applied to the candidate point: its displacement to the model's position in
            the previous and in the next frame, the larger of the two, divided by the video's median eye width
            (largest over the replaced points); accept if <= cut
  labelgeo  the geometry region spanned by the human labels: in the frame's own eye frame (origin = midpoint of the
            corners, x along the corners, unit = that frame's eye width) twelve scale-free quantities of the four pupil
            points (along coordinate; position inside the eyelid band as a fraction; the four ratios of
            second_peak_repair.Geometry) must each lie inside the min-max range of the 1330 human labels; no constant

Candidate repairs = the truth frames flagged at the fitting threshold (results/fit_threshold/chosen.csv, in eye widths),
eye open, with at least one pupil point over the threshold; every pupil point over the threshold is replaced by its
second peak (the replacement of flag_and_repair.py --accept fit). Outcome, in eye widths: right if the frame error falls
by more than 0.01 of the eye width, wrong if it grows by more than 0.01, neutral otherwise. Cut for conf and jump = the value that maximises (share of right
repairs accepted) - (share of wrong repairs accepted), ties to the stricter cut; leave-one-video-out as for the
flagging threshold. For labelgeo the leave-one-video-out number is the coverage: how many human labels of the held-out
video fall outside the region built from the other 18 videos.

Margin for a new video (added the same evening): the region from 18 videos misses some labels of the 19th, so each
quantity's range is widened by a fraction m of its width on both sides; m = the smallest value (steps of 0.01) at which
no held-out label of any video falls outside the region built from the other 18 (label_region_margin.csv: m and the
widened ranges). flag_and_repair.py --accept labelregion uses the widened region in two steps, as the geometry rule
does: a flagged frame already inside the region is left alone; otherwise the points over the threshold are replaced
and the repair is accepted if the frame is then inside.

Outputs in results/acceptance_thresholds/: candidates.csv, chosen.csv, lovo.csv, label_region.csv, label_region_lovo.csv,
label_region_margin.csv.
Nothing existing is modified."""
import numpy as np
import pandas as pd

from second_peak_repair import BPS, P4, PL, RES, Geometry, frame_rmse, load, peak2_pixels, pooled_bands

OUT = RES / "acceptance_thresholds"
QUANTS = [f"{b}_along" for b in P4] + [f"{b}_lidfrac" for b in P4] + [f"{b}_ratio" for b in P4]


def eye_quantities(pts, sign_lr, sign_tb):
    """Twelve scale-free quantities of the four pupil points from a dict point -> (2,) pixel position (NaN allowed)."""
    N, T = pts["eye_nasal_corner"], pts["eye_temporal_corner"]
    C = (N + T) / 2
    u = T - N
    W = np.hypot(*u)
    if not np.isfinite(W) or W < 1e-6:
        return {q: np.nan for q in QUANTS}
    u = u / W
    v = np.array([-u[1], u[0]])
    E = {b: np.array([(pts[b] - C) @ u, (pts[b] - C) @ v]) / W for b in BPS}
    top, bot = E["eyelid_top"][1], E["eyelid_bottom"][1]
    h = abs(E["pupil_bottom"][1] - E["pupil_top"][1])
    w = abs(E["pupil_right"][0] - E["pupil_left"][0])
    mid_tb = (E["pupil_top"][0] + E["pupil_bottom"][0]) / 2
    mid_lr = (E["pupil_left"][1] + E["pupil_right"][1]) / 2
    ratio = {"pupil_left": sign_lr * (mid_tb - E["pupil_left"][0]) / max(h, 1e-6), "pupil_right": sign_lr * (E["pupil_right"][0] - mid_tb) / max(h, 1e-6),
             "pupil_top": sign_tb * (mid_lr - E["pupil_top"][1]) / max(w, 1e-6), "pupil_bottom": sign_tb * (E["pupil_bottom"][1] - mid_lr) / max(w, 1e-6)}
    out = {}
    for b in P4:
        out[f"{b}_along"] = E[b][0]
        out[f"{b}_lidfrac"] = (E[b][1] - top) / (bot - top) if abs(bot - top) > 1e-6 else np.nan
        out[f"{b}_ratio"] = ratio[b]
    return out


def label_quantities(truth, G):
    rows = []
    for f, hr in truth.iterrows():
        pts = {b: np.array([hr[f"human_{b}_x"], hr[f"human_{b}_y"]], float) for b in BPS}
        if not all(np.isfinite(pts[b]).all() for b in BPS):
            continue
        rows.append({"frame": f, **eye_quantities(pts, G.sign_lr, G.sign_tb)})
    return pd.DataFrame(rows)


def region(Q, m=0.0):
    lo, hi = np.array([Q[q].min() for q in QUANTS]), np.array([Q[q].max() for q in QUANTS])
    w = hi - lo
    return pd.DataFrame({"quantity": QUANTS, "min": lo - m * w, "max": hi + m * w})


def inside(quant, R):
    return all((quant[q] >= R.loc[q, "min"]) and (quant[q] <= R.loc[q, "max"]) for q in QUANTS)


def video_candidates(v, thr):
    rr, d, P, H, truth, fwd, sig = load(v)
    xy = {b: np.c_[d[b]["x"].to_numpy(float), d[b]["y"].to_numpy(float)] for b in BPS}
    p2, off = peak2_pixels(d, P)
    B, scale = pooled_bands(RES / "second_peak_repair")
    G = Geometry(xy, scale)
    dev = fwd["200"] / G.Wm
    n = len(d)
    peak2 = {b: H[f"{b}_peak2"].to_numpy(float) for b in BPS}
    rows = []
    for f, hr in truth.iterrows():
        if G.closed[f] or not np.nanmax(dev[f]) > thr:
            continue
        cand = [b for b in P4 if dev[f, BPS.index(b)] > thr]
        if not cand or any(not np.isfinite(p2[b][f]).all() for b in cand):
            continue
        before = {b: xy[b][f] for b in BPS}
        after = dict(before)
        for b in cand:
            after[b] = p2[b][f]
        jumps = []
        for b in cand:
            for g in (f - 1, f + 1):
                if 0 <= g < n and np.isfinite(xy[b][g]).all():
                    jumps.append(np.hypot(*(after[b] - xy[b][g])) / G.Wm)
        quant = eye_quantities(after, G.sign_lr, G.sign_tb)
        rows.append({"video": v, "frame": f, "set": hr.set, "points": "+".join(cand), "rmse_before": frame_rmse(before, hr) / G.Wm, "rmse_after": frame_rmse(after, hr) / G.Wm,
                     "conf": min(peak2[b][f] for b in cand), "jump": max(jumps) if jumps else np.nan, **quant})
    return pd.DataFrame(rows), label_quantities(truth, G)


def cut(C, score, higher_is_better):
    """Cut maximising (right accepted share) - (wrong accepted share); ties go to the stricter cut."""
    right, wrong = C[C.outcome == "right"], C[C.outcome == "wrong"]
    grid = np.unique(np.round(C[score].dropna(), 3))
    best = None
    for g in grid:
        acc = (lambda s: s >= g) if higher_is_better else (lambda s: s <= g)
        J = acc(right[score]).mean() - (acc(wrong[score]).mean() if len(wrong) else 0.0)
        key = (J, g if higher_is_better else -g)
        if best is None or key > best[0]:
            best = (key, g, J, int(acc(right[score]).sum()), int(acc(wrong[score]).sum()))
    return best[1], best[2], best[3], best[4]


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    thr = float(pd.read_csv(RES / "fit_threshold" / "chosen.csv").iloc[0].threshold_eye_widths)
    C, L = [], []
    for v in sorted(PL[PL.status == "final"].video.astype(int)):
        c, q = video_candidates(v, thr)
        q["video"] = v
        C.append(c)
        L.append(q)
        print(f"video {v}: {len(c)} candidate repairs, {len(q)} fully labeled frames")
    C, L = pd.concat(C, ignore_index=True), pd.concat(L, ignore_index=True)
    C["outcome"] = np.where(C.rmse_after < C.rmse_before - 0.01, "right", np.where(C.rmse_after > C.rmse_before + 0.01, "wrong", "neutral"))
    R = region(L).set_index("quantity")
    C["labelgeo_inside"] = [inside(r, R) for _, r in C.iterrows()]
    C.round(4).to_csv(OUT / "candidates.csv", index=False)
    R.reset_index().round(4).to_csv(OUT / "label_region.csv", index=False)
    line = float(pd.read_csv(RES / "error_tier" / "tier.csv").iloc[0].line_eye_widths)
    print(f"\n{len(C)} candidate repairs: {dict(C.outcome.value_counts())}; wrong frames (> {line} eye widths) before: {int((C.rmse_before > line).sum())}")

    rows, lovo = [], []
    for score, hib in (("conf", True), ("jump", False)):
        g, J, nr, nw = cut(C, score, hib)
        rows.append({"test": score, "cut": g, "accept_if": ">=" if hib else "<=", "right_accepted": nr, "right_total": int((C.outcome == "right").sum()),
                     "wrong_accepted": nw, "wrong_total": int((C.outcome == "wrong").sum()), "objective": round(J, 4)})
        for v in sorted(C.video.unique()):
            gv, _, _, _ = cut(C[C.video != v], score, hib)
            H = C[C.video == v]
            acc = (H[score] >= gv) if hib else (H[score] <= gv)
            lovo.append({"test": score, "held_out_video": v, "cut_from_other_18": gv, "right": int((H.outcome == "right").sum()), "right_accepted": int(((H.outcome == "right") & acc).sum()),
                         "wrong": int((H.outcome == "wrong").sum()), "wrong_accepted": int(((H.outcome == "wrong") & acc).sum())})
    ins = C.labelgeo_inside
    rows.append({"test": "labelgeo", "cut": np.nan, "accept_if": "inside label range", "right_accepted": int(((C.outcome == "right") & ins).sum()), "right_total": int((C.outcome == "right").sum()),
                 "wrong_accepted": int(((C.outcome == "wrong") & ins).sum()), "wrong_total": int((C.outcome == "wrong").sum()), "objective": np.nan})
    pd.DataFrame(rows).to_csv(OUT / "chosen.csv", index=False)
    print(pd.DataFrame(rows).to_string(index=False))
    Lv = pd.DataFrame(lovo)
    Lv.to_csv(OUT / "lovo.csv", index=False)
    for score in ("conf", "jump"):
        s = Lv[Lv.test == score]
        print(f"leave-one-video-out {score}: cuts {sorted(s.cut_from_other_18.unique())}, right accepted {int(s.right_accepted.sum())}/{int(s.right.sum())}, wrong accepted {int(s.wrong_accepted.sum())}/{int(s.wrong.sum())}")
    cov = []
    for v in sorted(L.video.unique()):
        Rv = region(L[L.video != v]).set_index("quantity")
        Hv = L[L.video == v]
        out = sum(not inside(r, Rv) for _, r in Hv.iterrows())
        Cv = C[C.video == v]
        insv = [inside(r, Rv) for _, r in Cv.iterrows()] if len(Cv) else []
        cov.append({"held_out_video": v, "labels": len(Hv), "labels_outside_region_of_other_18": out,
                    "right": int((Cv.outcome == "right").sum()), "right_accepted": int(((Cv.outcome == "right") & np.array(insv, bool)).sum()) if len(Cv) else 0,
                    "wrong": int((Cv.outcome == "wrong").sum()), "wrong_accepted": int(((Cv.outcome == "wrong") & np.array(insv, bool)).sum()) if len(Cv) else 0})
    Cv = pd.DataFrame(cov)
    Cv.to_csv(OUT / "label_region_lovo.csv", index=False)
    print(f"leave-one-video-out labelgeo: {int(Cv.labels_outside_region_of_other_18.sum())}/{int(Cv.labels.sum())} human labels outside the region of the other 18 videos; "
          f"right accepted {int(Cv.right_accepted.sum())}/{int(Cv.right.sum())}, wrong accepted {int(Cv.wrong_accepted.sum())}/{int(Cv.wrong.sum())}")
    m = 0.0
    while True:
        out = 0
        for v in sorted(L.video.unique()):
            Rv = region(L[L.video != v], m).set_index("quantity")
            out += sum(not inside(r, Rv) for _, r in L[L.video == v].iterrows())
        print(f"margin {m:.2f}: {out} held-out labels outside")
        if out == 0 or m >= 1.0:
            break
        m = round(m + 0.01, 2)
    Rm = region(L, m).reset_index(drop=True)
    Rm["margin"] = m
    Rm.round(4).to_csv(OUT / "label_region_margin.csv", index=False)
    print(f"margin m = {m:.2f} of each quantity's range: every held-out video's labels inside the region of the other 18")


if __name__ == "__main__":
    main()
