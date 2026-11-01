#!/usr/bin/env python3
"""Human-label range for the eye-outline points (eyelids and corners), so that a second-peak repair of those points can
be accepted the same way as a pupil-point repair.

    python outline_region.py

The pupil-point range (acceptance_thresholds.py) measures the four pupil points in the frame's own eye frame, which is
built from the two corners. That frame cannot check the corners themselves, and it does not look at the eyelids. The
eight quantities below do, all scale-free and label-free apart from the human labels that set their range:

  corners (against the video, from the plateau model's predictions, label-free)
    eye_width_ratio   corner-to-corner distance of the frame / the video's median corner distance
    eye_angle_dev     angle of the corner line minus the video's median angle, degrees
    eye_centre_shift  distance of the corner midpoint from the video's median midpoint, in eye widths
  eyelids (in the frame's own eye frame, units of the frame's corner distance)
    eyelid_top_along, eyelid_top_v, eyelid_bottom_along, eyelid_bottom_v   position along and across the corner line
    eye_opening       signed distance between the two eyelid points across the corner line

Range = min to max over the 1330 human labels, widened on each side by the smallest margin at which every held-out
video's labels fall inside the range of the other 18 (the same construction as label_region_margin.csv; the pupil
range keeps its own margin, the outline range gets its own). Outputs results/acceptance_thresholds/
label_region_outline.csv (min, max, margin), label_region_outline_lovo.csv. Nothing existing is modified."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from second_peak_repair import BPS, PL, RES, Geometry, load, pooled_bands  # noqa: E402

OUT = RES / "acceptance_thresholds"
OQUANTS = ["eye_width_ratio", "eye_angle_dev", "eye_centre_shift", "eyelid_top_along", "eyelid_top_v", "eyelid_bottom_along", "eyelid_bottom_v", "eye_opening"]


class VideoRef:
    """Label-free reference of a video: median corner distance, median corner-line angle, median corner midpoint."""

    def __init__(self, xy):
        N, T = xy["eye_nasal_corner"], xy["eye_temporal_corner"]
        u = T - N
        self.Wm = float(np.nanmedian(np.hypot(u[:, 0], u[:, 1])))
        self.ang = float(np.nanmedian(np.degrees(np.arctan2(u[:, 1], u[:, 0]))))
        self.C = np.nanmedian((N + T) / 2, axis=0)
        E_top = self._v(xy["eyelid_top"], N, T)
        E_bot = self._v(xy["eyelid_bottom"], N, T)
        self.sign_open = float(np.sign(np.nanmedian(E_bot - E_top)))

    @staticmethod
    def _v(p, N, T):
        u = T - N
        W = np.hypot(u[:, 0], u[:, 1])
        v = np.c_[-u[:, 1], u[:, 0]] / W[:, None]
        return ((p - (N + T) / 2) * v).sum(1) / W


def outline_quantities(pts, ref):
    """The eight outline quantities of one frame from a dict point -> (2,) pixel position."""
    N, T = pts["eye_nasal_corner"], pts["eye_temporal_corner"]
    u = T - N
    W = float(np.hypot(*u))
    if not np.isfinite(W) or W < 1e-6:
        return {q: np.nan for q in OQUANTS}
    ang = np.degrees(np.arctan2(u[1], u[0])) - ref.ang
    ang = (ang + 180) % 360 - 180
    C = (N + T) / 2
    uu = u / W
    vv = np.array([-uu[1], uu[0]])
    E = {b: np.array([(pts[b] - C) @ uu, (pts[b] - C) @ vv]) / W for b in ("eyelid_top", "eyelid_bottom")}
    return {"eye_width_ratio": W / ref.Wm, "eye_angle_dev": float(ang), "eye_centre_shift": float(np.hypot(*(C - ref.C)) / ref.Wm),
            "eyelid_top_along": E["eyelid_top"][0], "eyelid_top_v": E["eyelid_top"][1], "eyelid_bottom_along": E["eyelid_bottom"][0], "eyelid_bottom_v": E["eyelid_bottom"][1],
            "eye_opening": ref.sign_open * (E["eyelid_bottom"][1] - E["eyelid_top"][1])}


def inside_outline(q, R):
    return all(np.isfinite(q[k]) and R.loc[k, "min"] <= q[k] <= R.loc[k, "max"] for k in OQUANTS)


def region(Q, m=0.0):
    lo, hi = np.array([Q[q].min() for q in OQUANTS]), np.array([Q[q].max() for q in OQUANTS])
    w = hi - lo
    return pd.DataFrame({"quantity": OQUANTS, "min": lo - m * w, "max": hi + m * w}).set_index("quantity")


def main():
    L = []
    for v in sorted(PL[PL.status == "final"].video.astype(int)):
        rr, d, P, H, truth, fwd, sig = load(v)
        xy = {b: np.c_[d[b]["x"].to_numpy(float), d[b]["y"].to_numpy(float)] for b in BPS}
        ref = VideoRef(xy)
        for f, hr in truth.iterrows():
            pts = {b: np.array([hr[f"human_{b}_x"], hr[f"human_{b}_y"]], float) for b in BPS}
            if not all(np.isfinite(pts[b]).all() for b in ("eye_nasal_corner", "eye_temporal_corner", "eyelid_top", "eyelid_bottom")):
                continue
            L.append({"video": v, "frame": f, **outline_quantities(pts, ref)})
        print(f"video {v}: {sum(r['video'] == v for r in L)} labeled frames, eye width {ref.Wm:.0f} px, angle {ref.ang:.1f} deg")
    L = pd.DataFrame(L)
    L.round(4).to_csv(OUT / "label_outline_quantities.csv", index=False)
    m = 0.0
    while True:
        out, rows = 0, []
        for v in sorted(L.video.unique()):
            Rv = region(L[L.video != v], m)
            o = sum(not inside_outline(r, Rv) for _, r in L[L.video == v].iterrows())
            rows.append({"held_out_video": v, "labels": int((L.video == v).sum()), "labels_outside_region_of_other_18": o, "margin": m})
            out += o
        print(f"margin {m:.2f}: {out} held-out labels outside")
        if out == 0 or m >= 1.0:
            break
        m = round(m + 0.01, 2)
    pd.DataFrame(rows).to_csv(OUT / "label_region_outline_lovo.csv", index=False)
    R = region(L, m)
    R["margin"] = m
    R.round(4).to_csv(OUT / "label_region_outline.csv")
    print(R.round(4).to_string())
    print(f"margin m = {m:.2f}: every held-out video's labels inside the outline range of the other 18")


if __name__ == "__main__":
    main()
