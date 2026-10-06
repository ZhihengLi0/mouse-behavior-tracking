#!/usr/bin/env python3
"""Build the project summary page.

    python report/build_report.py            # -> report/index.html (+ report/assets/*.jpg)

The page lists what was done, in order, with the result figures of each unit and a short reading of
each figure. The figures are the PNGs already in the units' results/ folders (copied here as JPG).
The tables that still change (labels per video, fewer-labels study) are generated from the result
CSVs, so the page can be rebuilt after every step. The few images that were cut from raw videos
(assets/keypoints.jpg, label_standard.jpg) are kept as built earlier.
"""
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
REP = ROOT / "report"
ASSETS = REP / "assets"
ASSETS.mkdir(exist_ok=True)
V1 = ROOT / "v1_old_pupil_top"                    # earlier label standards (5-minute video, first new mouse)
V2 = ROOT / "v2_new_pupil_top_video_scaling"      # ellipse label standard, one folder per video

# ---- result figures used on the page: (source PNG, asset name, max width in px) ------------------
FIGURES = [
    (V1 / "batch-size-selection/results/01_batch_size_overview.png", "fig_batch_size.jpg", 2000),
    (V1 / "batch-size-selection/results/overall_vs_median.png", "fig_batch_two_metrics.jpg", 2000),
    (V1 / "model-selection/results/overall_vs_median.png", "fig_model_two_metrics.jpg", 2000),
    (V1 / "model-selection/results/01_model_selection_overview.png", "fig_model_selection.jpg", 2000),
    (V1 / "active-learning-jump-selection/results/01_convergence_curves.png", "fig_active_learning.jpg", 2400),
    (V2 / "results/labels_to_plateau.png", "fig_labels_to_plateau.jpg", 2400),
    (V2 / "results/scale_curves_all_videos.png", "fig_all_videos.jpg", 5148),
    (V2 / "results/cross_video_curves.png", "cross_video_curves.jpg", 2600),
    (V2 / "results/all_videos_view_main.png", "all_view_main.jpg", 3380),
    (V2 / "results/all_videos_view_fewer.png", "all_view_fewer.jpg", 3380),
    (V2 / "results/all_videos_view_backtest.png", "all_view_backtest.jpg", 3380),
    (V2 / "results/cross_video_matrix.png", "cross_video_matrix.jpg", 2470),
    (V2 / "results/pupil_area_variants.png", "fig_area_variants.jpg", 2000),
    (V2 / "results/blink_area_consistency.png", "fig_blink_area.jpg", 2400),
    (V2 / "results/blink_and_area/blink_area_area_methods_all_videos.png", "fig_eye_area.jpg", 2400),
    (V2 / "results/blink_and_area/blink_area_closure_signals_all_videos.png", "fig_eye_blink.jpg", 2600),
    (V2 / "results/blink_and_area/blink_area_closed_frames_all_videos.jpg", "fig_closed_frames.jpg", 2784),
    (V2 / "results/blink_and_area/blink_area_timeseries_all_videos.png", "fig_eye_timeseries.jpg", 2200),
    (V2 / "results/keypoint_confidence_error/keypoint_confidence_error_all_videos.png", "fig_kp_conf_error.jpg", 2880),
    (V2 / "results/unreliable_frames/unreliable_frames_all_videos.png", "fig_unreliable.jpg", 2625),
    (V2 / "results/blink_eyelid_distance/blink_final_summary.png", "fig_blink_final.jpg", 2210),
    (ROOT / "keypoint_definitions/figures/fig1_all_keypoints.jpg", "kp_all_keypoints.jpg", 2000),
    (ROOT / "keypoint_definitions/figures/fig8_construction_pupil.jpg", "kp_construction_pupil.jpg", 2000),
    (ROOT / "keypoint_definitions/figures/fig7_construction_corners.jpg", "kp_construction_corners.jpg", 2000),
]
import cv2
try:

    for src, dst, width in FIGURES:
        im = cv2.imread(str(src))
        if im is None:
            print("figure missing, asset kept:", src)
            continue
        if im.shape[1] > width:
            sc = width / im.shape[1]
            im = cv2.resize(im, None, fx=sc, fy=sc, interpolation=cv2.INTER_AREA)
        cv2.imwrite(str(ASSETS / dst), im, [cv2.IMWRITE_JPEG_QUALITY, 86])
except ImportError as e:
    print("OpenCV not available, figure assets kept as they are:", repr(e))

# ---- tables that change while the experiment runs --------------------------------------------------
pl = pd.read_csv(V2 / "results/labels_to_plateau.csv")
CARRIED_0_3 = {0: 100, 1: 100, 2: 60, 3: 60}       # videos 0-3 carry the labels that existed when the next video started


def px(v):
    return "–" if pd.isna(v) else f"{float(v):.2f}"


rows, fewer, earlier, n_final, in_progress = [], [], 0, 0, None
for r in pl.itertuples():
    sc = pd.read_csv(V2 / r.unit / "results/scale_curve.csv")
    reg = sc[sc["label"].str.match(r"^x\d{3}_step\d{2}_final$") & (sc["training_frames_this_video"] > 0)].sort_values("training_frames_this_video")
    final = r.status == "final"
    series = [px(r.zero_label_px)] + [px(v) for v in reg["median_frame_rmse_px"]]
    labeled = int(reg["training_frames_this_video"].max())
    carried = CARRIED_0_3.get(int(r.video), int(r.labels_at_plateau) if final else None)
    zh_state = f"<b>{int(r.labels_at_plateau)} 帧，{float(r.plateau_median_px):.2f} px</b>" if final else f"进行中（目前最好：{int(r.labels_at_plateau)} 帧，{float(r.plateau_median_px):.2f} px）"
    en_state = f"<b>{int(r.labels_at_plateau)} labels, {float(r.plateau_median_px):.2f} px</b>" if final else f"in progress (best so far: {int(r.labels_at_plateau)} labels, {float(r.plateau_median_px):.2f} px)"
    rows.append(f"<tr><td>{int(r.video)}</td><td>{r.mouse}<br>{r.date}</td><td>{earlier}</td><td>{' / '.join(series)}</td>"
                f"<td><span lang=\"zh\">{zh_state}</span><span lang=\"en\">{en_state}</span></td>"
                f"<td>{labeled}</td><td>{'–' if carried is None else carried}</td></tr>")
    earlier += carried or 0
    n_final += int(final)
    if not final:
        in_progress = (int(r.video), series)
    sub = sc[sc["label"].str.contains(r"_final_sub\d+[ab]$")]
    if len(sub) and len(reg):
        g = lambda tag: (px(sub[sub["label"].str.endswith(tag)]["median_frame_rmse_px"].iloc[0]) if sub["label"].str.endswith(tag).any() else "–")
        fewer.append(f"<tr><td>{int(r.video)}</td><td>{px(r.zero_label_px)}</td><td>{g('sub05a')} / {g('sub05b')}</td>"
                     f"<td>{g('sub10a')} / {g('sub10b')}</td><td>{px(reg['median_frame_rmse_px'].iloc[0])}</td></tr>")

# every video's own scale curve, shown in one scrolling row (figure 5-3)
strip = []
for r in pl.itertuples():
    src = V2 / r.unit / "results/scale_curve.png"
    im = cv2.imread(str(src))
    if im is None:
        continue
    name = f"scale_video{int(r.video):02d}.jpg"
    sc = 1600 / im.shape[1]
    cv2.imwrite(str(ASSETS / name), cv2.resize(im, None, fx=sc, fy=sc, interpolation=cv2.INTER_AREA), [cv2.IMWRITE_JPEG_QUALITY, 86])
    fin = r.status == "final"
    zh = f"视频 {int(r.video)}（{r.mouse}，{r.date}）：" + (f"平台点 {int(r.labels_at_plateau)} 帧，{float(r.plateau_median_px):.2f} px" if fin else "进行中")
    en = f"Video {int(r.video)} ({r.mouse}, {r.date}): " + (f"plateau at {int(r.labels_at_plateau)} labels, {float(r.plateau_median_px):.2f} px" if fin else "in progress")
    strip.append(f'      <div class="item"><img src="assets/{name}" alt="Video {int(r.video)}: error vs own labels" loading="lazy">'
                 f'<div class="cap"><span lang="zh">{zh}</span><span lang="en">{en}</span></div></div>')

# every video's first jump-selected batch (selection sheet of batch 2), one scrolling row (figure 5-1)
sel_strip = []
for r in pl.itertuples():
    im = cv2.imread(str(V2 / r.unit / "results/selection_sheets/selection_040_frames.png"))
    if im is None:
        continue
    name = f"selection_video{int(r.video):02d}.jpg"
    sc = 1200 / im.shape[1]
    cv2.imwrite(str(ASSETS / name), cv2.resize(im, None, fx=sc, fy=sc, interpolation=cv2.INTER_AREA), [cv2.IMWRITE_JPEG_QUALITY, 84])
    sel_strip.append(f'      <div class="item"><img src="assets/{name}" alt="Video {int(r.video)}: batch 2 selection" loading="lazy">'
                     f'<div class="cap"><span lang="zh">视频 {int(r.video)}（{r.mouse}，{r.date}）：第 2 批</span><span lang="en">Video {int(r.video)} ({r.mouse}, {r.date}): batch 2</span></div></div>')

def strip_of(pattern, prefix, width, cap_zh, cap_en):
    """One scrolling row: the figure `pattern` (formatted with the video row) of every video."""
    out = []
    for r in pl.itertuples():
        im = cv2.imread(str(pattern(r)))
        if im is None:
            continue
        name = f"{prefix}_video{int(r.video):02d}.jpg"
        sc = width / im.shape[1]
        cv2.imwrite(str(ASSETS / name), cv2.resize(im, None, fx=sc, fy=sc, interpolation=cv2.INTER_AREA), [cv2.IMWRITE_JPEG_QUALITY, 84])
        out.append(f'      <div class="item"><img src="assets/{name}" alt="{prefix} video {int(r.video)}" loading="lazy">'
                   f'<div class="cap"><span lang="zh">视频 {int(r.video)}（{r.mouse}，{r.date}）{cap_zh}</span><span lang="en">Video {int(r.video)} ({r.mouse}, {r.date}){cap_en}</span></div></div>')
    return "\n".join(out)


fewer_strip = strip_of(lambda r: V2 / r.unit / "results/fewer_labels.png", "fewer", 1500, "", "")
kp_strip = strip_of(lambda r: V2 / f"results/keypoint_confidence_error/keypoint_confidence_error_video{int(r.video):02d}.png", "kpconf", 1700, "", "")
ts_strip = strip_of(lambda r: V2 / f"results/blink_and_area/blink_area_timeseries_video{int(r.video):02d}.png", "eyets", 1700, "", "")

# appendix: every selection sheet of every video, one scrolling row per video
appendix = []
for r in pl.itertuples():
    items = []
    for src in sorted((V2 / r.unit / "results/selection_sheets").glob("selection_*_frames.png")):
        im = cv2.imread(str(src))
        b = int(src.stem.split("_")[1]) // 20
        name = f"selection_video{int(r.video):02d}_batch{b:02d}.jpg"
        sc = 1000 / im.shape[1]
        cv2.imwrite(str(ASSETS / name), cv2.resize(im, None, fx=sc, fy=sc, interpolation=cv2.INTER_AREA), [cv2.IMWRITE_JPEG_QUALITY, 80])
        how_zh = "按画面外观挑，没有用模型" if b == 1 else f"由 {20 * (b - 1)} 帧模型的 jump 规则挑"
        how_en = "picked by appearance, no model" if b == 1 else f"picked by the jump rule of the {20 * (b - 1)}-label model"
        items.append(f'      <div class="item"><img src="assets/{name}" alt="Video {int(r.video)} batch {b}" loading="lazy">'
                     f'<div class="cap"><span lang="zh">视频 {int(r.video)} 第 {b} 批（{how_zh}）</span><span lang="en">Video {int(r.video)}, batch {b} ({how_en})</span></div></div>')
    if items:
        appendix.append(f'  <p class="text"><b><span lang="zh">视频 {int(r.video)}（{r.mouse}，{r.date}）：{len(items)} 批</span>'
                        f'<span lang="en">Video {int(r.video)} ({r.mouse}, {r.date}): {len(items)} batches</span></b></p>\n'
                        f'  <figure><div class="figbox scroll"><div class="strip tall">\n' + "\n".join(items) + '\n  </div></div></figure>')

if in_progress:
    zh_status = f"已完成 {n_final} 个视频（视频 0–{n_final - 1}）；视频 {in_progress[0]} 进行中，按 0 / 20 / 40 / … 帧的误差为 {' / '.join(in_progress[1])} px。"
    en_status = f"{n_final} videos finished (videos 0–{n_final - 1}); video {in_progress[0]} in progress, error at 0 / 20 / 40 / … labels: {' / '.join(in_progress[1])} px."
else:
    zh_status = f"已完成 {n_final} 个视频（视频 0–{n_final - 1}）。"
    en_status = f"{n_final} videos finished (videos 0–{n_final - 1})."

ts = pd.read_csv(V2 / "results/blink_and_area/blink_area_timeseries_all_videos.csv")
am = pd.read_csv(V2 / "results/blink_and_area/blink_area_area_methods_all_videos.csv")
cs = pd.read_csv(V2 / "results/blink_and_area/blink_area_closure_signals_all_videos.csv").astype({"video": str}).set_index("video")
eye = []
for r in ts.itertuples():
    a = am[am.video.astype(str) == str(r.video)].set_index("rule").median_abs_err_pct
    c = cs.loc[str(r.video)]
    eye.append(f"<tr><td>{r.video}</td><td>{r.mouse}</td><td>{a['four']:.1f} / {a['no_top']:.1f}</td><td>{int(c.closed_frames)} / {int(c.open_frames)}</td>"
               f"<td>{r.flagged_frames_pct:.2f}</td><td>{r.events} ({r.events_per_min:.1f})</td><td>{r.median_event_ms:.0f}</td>"
               f"<td>{r.production_flagged_pct:.1f}</td><td>{r.area_rel_spread_pct:.1f}</td><td>{r.corr_area_opening:.2f}</td></tr>")

FILL = {
    "{{FEWER_STRIP}}": fewer_strip,
    "{{KP_STRIP}}": kp_strip,
    "{{TS_STRIP}}": ts_strip,
    "{{APPENDIX_SELECTION}}": "\n".join(appendix),
    "{{SELECTION_STRIP}}": "\n".join(sel_strip),
    "{{SCALE_STRIP}}": "\n".join(strip),
    "{{EYE_ROWS}}": "\n".join(eye),
    "{{VIDEO_ROWS}}": "\n".join(rows),
    "{{FEWER_ROWS}}": "\n".join(fewer),
    "{{STATUS_ZH}}": zh_status,
    "{{STATUS_EN}}": en_status,
    "{{BUILT}}": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M"),
}
html = (REP / "template.html").read_text(encoding="utf-8")
for k, v in FILL.items():
    assert k in html, k
    html = html.replace(k, v)
(REP / "index.html").write_text(html, encoding="utf-8")

# ---- English-only version for the group: every lang="zh" element removed, no language switch ------
import re
from html.parser import HTMLParser


class DropZh(HTMLParser):
    VOID = {"img", "br", "meta", "link", "input", "mspace", "hr", "col"}

    def __init__(self):
        super().__init__(convert_charrefs=False)
        self.out, self.skip = [], 0

    def handle_starttag(self, tag, attrs):
        zh = dict(attrs).get("lang") == "zh"
        if self.skip or zh:
            self.skip += tag not in self.VOID
            return
        self.out.append(self.get_starttag_text())

    def handle_startendtag(self, tag, attrs):
        if not self.skip and dict(attrs).get("lang") != "zh":
            self.out.append(self.get_starttag_text())

    def handle_endtag(self, tag):
        if self.skip:
            self.skip -= tag not in self.VOID
            return
        self.out.append(f"</{tag}>")

    def handle_data(self, d):
        self.skip or self.out.append(d)

    def handle_entityref(self, n):
        self.skip or self.out.append(f"&{n};")

    def handle_charref(self, n):
        self.skip or self.out.append(f"&#{n};")

    def handle_comment(self, d):
        self.skip or self.out.append(f"<!--{d}-->")

    def handle_decl(self, d):
        self.out.append(f"<!{d}>")


body, script = html.split("<script>", 1)[0], None
dz = DropZh()
dz.feed(body)
en = "".join(dz.out)
en = re.sub(r'<div class="langbar".*?</div>', "", en, flags=re.S)
en += '<script>document.documentElement.setAttribute("data-lang","en");document.documentElement.lang="en";</script>\n'
left = re.findall(r"[\u4e00-\u9fff]+", en)
assert not left, left[:10]
(REP / "index_en.html").write_text(en, encoding="utf-8")
print("built", REP / "index_en.html", "| English only,", len(en) // 1024, "KB")
print("built", REP / "index.html", "|", len(rows), "videos,", len(fewer), "fewer-labels rows")
