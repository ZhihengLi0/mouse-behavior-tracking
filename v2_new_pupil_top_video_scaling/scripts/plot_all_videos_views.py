#!/usr/bin/env python3
"""All videos in one figure, drawn three times with a different emphasis each (adopted for the page 2026-10-04).

    python plot_all_videos_views.py
      -> results/all_videos_view_main.png       main line: each video's own test set while its own labels are added
      -> results/all_videos_view_fewer.png      first batch of 5 / 10 labels: the range between the two subsets (a, b)
      -> results/all_videos_view_backtest.png   back-test: each test set after its own block, one colour per video

Reads results/cross_video_matrix.csv and results/labels_to_plateau.csv (nothing is retrained or rescored). Blocks = the
videos in training order; inside a block x is the number of labels of that video (20 labels = one unit), starting with
a "0" slot = the score of the model the video started from. The 5- and 10-label models (two subsets each, a and b,
taken out of the 20 labels of the first batch) sit at x = 5 and x = 10; the band spans the two results. These models
were only trained and scored: no jump selection and no further labeling followed them."""
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.patches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import cross_video_eval as c  # noqa: E402

TITLE = {"main": "Main line: each video's own test set while its own labels are added (bold); the first 20 labels bring most videos close to their plateau",
         "fewer": "First batch of 5 or 10 labels instead of 20: shaded = range between the two subsets (a, b) of each size; these models were only trained and scored, no jump selection followed",
         "backtest": "Back-test: each video's test set after its own block (one colour per video); adding later videos does not make earlier ones worse"}
# one colour family per mouse, as in cross_video_eval.py: mouse A black, Pluto cool colours (purple -> green), Terra light warm colours
VCOL, _fam = {}, {}
for _v in c.VIDEOS:
    _fam.setdefault(_v[1], []).append(_v[0])
for _m, _us in _fam.items():
    for _i, _u in enumerate(_us):
        _f = _i / max(1, len(_us) - 1)
        VCOL[_u] = ("#111111" if "Pluto" not in _m and "Terra" not in _m else plt.cm.viridis(0.02 + 0.78 * _f) if "Pluto" in _m else plt.cm.Wistia(0.15 + 0.85 * _f))
ACCENT = ["#2a78d6", "#eb6834"]            # blue / orange, alternating between neighbouring blocks
GREY, INK, MUTED = "#c4c8cc", "#1c1f22", "#5d6368"

mods = c.models()
m = pd.read_csv(c.OUT)
m = m[m.shuffle.isin(mods.shuffle)].drop(columns=["model_idx", "subset"], errors="ignore").merge(mods[["shuffle", "model_idx", "subset"]], on="shuffle")
m["subset"] = m["subset"].fillna("")
units = c.test_units()
zero = pd.read_csv(c.HERE / "results" / "labels_to_plateau.csv").set_index("unit")["zero_label_px"]
blocks = [(u, mods[mods.model_unit == u]) for u, *_ in c.VIDEOS if (mods.model_unit == u).any()]
colour = {u: ACCENT[i % 2] for i, (u, _) in enumerate(blocks)}

# x positions: inside each block x is LINEAR in the video's own labels (20 labels = 1 unit, block starts at 0 labels),
# so slopes can be compared; the two 5-label subsets share x = 5 and the two 10-label subsets share x = 10
UNIT, GAP = 20.0, 0.9
pos, zero_slot, ticks, start_of, k = {}, {}, [], {}, 0.0
for u, g in blocks:
    reg = g[g.subset == ""]
    start_of[u] = k
    if u in zero.index and np.isfinite(zero[u]):
        zero_slot[u] = k
        ticks.append((k, "0"))
    for r in g.itertuples():
        pos[r.model_idx] = k + r.labels_this_video / UNIT
        if not r.subset:
            ticks.append((pos[r.model_idx], str(r.labels_this_video)))
    k += reg.labels_this_video.max() / UNIT + GAP
k -= GAP
m["x"] = m.model_idx.map(pos)

for FOCUS in ("main", "fewer", "backtest"):
    after_col = {}
    OUT = c.HERE / "results" / f"all_videos_view_{FOCUS}.png"
    fig, ax = plt.subplots(figsize=(26, 8.6), constrained_layout=True)
    ax.set_yscale("log")
    series = {}
    for tu in units:                                                # context: every test set, whole sequence, light grey
        s = m[(m.test_unit == tu) & (m.subset == "")].sort_values("x")
        series[tu] = s
        ax.plot(s.x, s.median_frame_rmse_px, "-", color=GREY, lw=1.1 if FOCUS != "fewer" else 0.7, zorder=1)
    prev_mouse = None
    for u, g in blocks:                                             # blocks: tint, header, own test set in the block colour
        xs = [pos[i] for i in g.model_idx] + ([zero_slot[u]] if u in zero_slot else [])
        x0, x1, col = start_of[u] - 0.3, max(xs) + 0.3, colour[u]
        v_ = next(v for v in c.VIDEOS if v[0] == u)
        short = v_[1].split("(")[-1].rstrip(")")                  # "mouse A", "Pluto", "Terra"
        if prev_mouse is not None and v_[1] != prev_mouse:          # change of mouse: thick line + label
            ax.axvline(x0 - GAP / 2 + 0.3, color="k", lw=2.2, zorder=6)
            ax.text(x0 - GAP / 2 + 0.3, 0.985, f" NEW MOUSE: {short} ", transform=ax.get_xaxis_transform(), ha="left", va="top", fontsize=10,
                    fontweight="bold", color="w", bbox=dict(fc="k", ec="none", pad=2), zorder=7)
        prev_mouse = v_[1]
        ax.axvspan(x0, x1, color=col, alpha=0.07, lw=0, zorder=0)
        ax.plot([x0 + 0.15, x1 - 0.15], [1.012, 1.012], color=col, lw=5, solid_capstyle="butt", transform=ax.get_xaxis_transform(), clip_on=False)
        v = next(v for v in c.VIDEOS if v[0] == u)
        head = f"video {u.split('_')[0]}" + ("\npoor quality" if u in c.POOR_QUALITY else "")
        carried = int(g[g.subset == ""].total_labels.iloc[0] - g[g.subset == ""].labels_this_video.iloc[0])
        ax.text((x0 + x1) / 2, 1.03, f"{head}\n{short}{'' if u.startswith('0_') else ', ' + v[2][5:]}\n+{carried} carried", transform=ax.get_xaxis_transform(), ha="center", va="bottom", fontsize=9, color=INK)
        if u not in series:
            continue
        own = series[u][series[u].model_unit == u]
        px, py = list(own.x), list(own.median_frame_rmse_px)
        if u in zero_slot:
            px, py = [zero_slot[u]] + px, [float(zero[u])] + py
        if FOCUS == "backtest":                                       # own block thin; the part after the block in the video's colour
            vc = VCOL[u]
            ax.plot(px, py, "-", color=vc, lw=1.2, zorder=3)
            after = series[u][series[u].x >= own.x.max()]
            ax.plot(after.x, after.median_frame_rmse_px, "-", color=vc, lw=2.4, zorder=4, solid_joinstyle="round")
            ax.plot([own.x.max()], [own.median_frame_rmse_px.iloc[-1]], "o", color=vc, ms=6, mec="white", mew=1.1, zorder=5)
            after_col[u] = vc
            continue
        ax.plot(px, py, "-", color=col, lw=2.6 if FOCUS == "main" else 1.3, zorder=4, solid_joinstyle="round")
        ax.plot(own.x, own.median_frame_rmse_px, "o", color=col, ms=6.5 if FOCUS == "main" else 4, mec="white", mew=1.2, zorder=5)
        if u in zero_slot:
            ax.plot([zero_slot[u]], [float(zero[u])], "o", color="white", ms=7, mec=col, mew=1.9, zorder=5)
        sub = m[(m.test_unit == u) & (m.subset != "") & (m.model_unit == u)]
        if FOCUS == "fewer" and len(sub) and len(own):                                     # 5 / 10-label subsets: the range between the two subsets as a band
            lo, hi = sub.groupby("x").median_frame_rmse_px.min(), sub.groupby("x").median_frame_rmse_px.max()
            x20, v20 = own.x.iloc[0], own.median_frame_rmse_px.iloc[0]
            top = [(x, hi[x]) for x in hi.index] + [(x20, v20)]
            bot = [(x, lo[x]) for x in lo.index][::-1]
            if u in zero_slot:
                top, bot = [(zero_slot[u], float(zero[u]))] + top, bot
            poly = top + bot
            ax.fill([p[0] for p in poly], [p[1] for p in poly], color=col, alpha=0.35, lw=0, zorder=2)
            ax.plot([p[0] for p in poly] + [poly[0][0]], [p[1] for p in poly] + [poly[0][1]], color=col, lw=0.9, alpha=0.8, zorder=3)
            ax.plot(sub.x, sub.median_frame_rmse_px, "o", color="white", ms=6, mec=col, mew=1.6, zorder=6)
    xr = k                                                      # names at the right end of every line (pushed apart in log space)
    ends = sorted(((series[tu].median_frame_rmse_px.iloc[-1], tu) for tu in units if len(series[tu])), key=lambda t: t[0])
    ys = [np.log10(v) for v, _ in ends]
    for i in range(1, len(ys)):
        ys[i] = max(ys[i], ys[i - 1] + 0.052)
    for (v, tu), y in zip(ends, ys):
        xe = series[tu].x.iloc[-1]
        ax.plot([xe + 0.1, xr + 0.9], [v, 10 ** y], color=GREY, lw=0.8, clip_on=False, zorder=1)
        ax.text(xr + 1.0, 10 ** y, f"video {tu.split('_')[0]}  {v:.1f} px", va="center", ha="left", fontsize=8.5, color=after_col.get(tu, INK) if FOCUS == "backtest" else INK, clip_on=False)
    ax.set_xlim(-0.8, xr + 4.6)
    ax.set_xticks([t for t, _ in ticks])
    ax.set_xticklabels([lab for _, lab in ticks], fontsize=7, color=MUTED)
    ax.set_xlabel("labels of the video itself, restarting at 0 in every block (same scale in all blocks: slopes are comparable); blocks in training order", color=MUTED)
    ax.set_ylabel("median frame RMSE on a video's 50 frozen test frames (px, log scale)", color=MUTED)
    ax.grid(alpha=0.25, which="major", color="#9aa0a6", lw=0.6)
    ax.grid(alpha=0.12, which="minor", color="#9aa0a6", lw=0.5)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.tick_params(colors=MUTED)
    h = {"main": [plt.Line2D([], [], color=ACCENT[0], lw=2.6, marker="o", mec="white", mew=1.2, ms=6.5, label="coloured: the test set of the video being trained in that block (colours alternate between blocks)"),
                  plt.Line2D([], [], color=ACCENT[0], lw=0, marker="o", mfc="white", mec=ACCENT[0], mew=1.9, ms=7, label="open circle at 0: that video before any label of its own (the model it started from)"),
                  plt.Line2D([], [], color=GREY, lw=1.1, label="grey: the same test sets outside their own block (named at the right end)")],
         "fewer": [matplotlib.patches.Patch(facecolor=ACCENT[0], alpha=0.4, edgecolor=ACCENT[0], label="shaded: range between the two subsets at 5 labels (5a, 5b) and at 10 labels (10a, 10b), from the 0-label point to the 20-label point"),
                   plt.Line2D([], [], color=ACCENT[0], lw=0, marker="o", mfc="white", mec=ACCENT[0], mew=1.6, ms=6, label="small circles: the four subset models of a video (x = 5 and x = 10)"),
                   plt.Line2D([], [], color=ACCENT[0], lw=1.3, marker="o", mec="white", ms=4, label="thin coloured line: the regular steps of 20 labels")],
         "backtest": [plt.Line2D([], [], color=VCOL["5_20251029_Pluto_spont_1"], lw=2.4, label="bold, one colour per video (black = mouse A, purple to green = Pluto, yellow = Terra): its test set scored by the models trained AFTER its own block"),
                      plt.Line2D([], [], color=VCOL["5_20251029_Pluto_spont_1"], lw=1.2, label="thin, same colour: inside its own block"),
                      plt.Line2D([], [], color=GREY, lw=1.1, label="grey: before its own block (the video not yet in the training set)")]}[FOCUS]
    ax.legend(handles=h, fontsize=9, loc="lower left", frameon=True, framealpha=0.92, edgecolor="#d8dbde", bbox_to_anchor=(0.012, 0.02))
    fig.suptitle(TITLE[FOCUS], fontsize=12.5, color=INK, x=0.012, ha="left")
    fig.savefig(OUT, dpi=130)
    plt.close(fig)
    print("saved", OUT)
