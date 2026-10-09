# Archive index: DLC models moved to Kaiwen's Drive (2026-10-08 onwards)

Why: the Mac was down to 78 GB free with 125 trained models (133 GB, 12 snapshots each) of which only the final snapshot
(epoch 120) is ever read again (cross-video scoring, pre-labels). Every model folder is archived in full to Kaiwen's
personal Drive, round by round (the UMN Drive quota is 50 GB per round); after a round is copied by Kaiwen and the SSD
backup is verified, the 10 intermediate snapshots of those models are deleted locally. Folder structure, final snapshots,
configs and training logs stay on the Mac, so nothing in the pipeline changes.

Drive location (Kaiwen's personal Drive): `SNFM_archive/d1_all_main_face_videos/zhiheng_results/mouse_eye_models_archive_2026-10-08/roundN/`
Each round: `models_roundN.tar.part_*` (4 GB parts), `SHA256SUMS.txt`, `filelist.txt` (every archived path),
`model_folders.txt`. Kaiwen's copies carry a "Copy of " prefix: strip it before restoring.

Restore (in the folder that contains `dlc_projects/`): `shasum -a 256 -c SHA256SUMS.txt && cat models_roundN.tar.part_* | tar -xf -`
Paths inside the archive start at `dlc_projects/EyePupilEllipse-Zhiheng-2026-09-20/dlc-models-pytorch/iteration-0/`.

| round | models (shuffle numbers) | videos | size | packed | uploaded to the UMN Drive | copied by Kaiwen | local intermediate snapshots deleted |
|---|---|---|---|---|---|---|---|
| 1 | 111-115, 211-217, 225, 311, 321-324, 411, 412, 422, 423, 531-534, 631, 632, 711-713 (31 models, 473 files) | 0-7 | 31 GB | 2026-10-08 11:53 | 2026-10-08 12:55 | 2026-10-08 13:46 (copies renamed, originals deleted; all 8 parts SHA256-verified on the Drive 17:22) | 2026-10-08 15:08 (289 files, 27.7 GB) |
| 2 | 811, 812, 911-913, 951-974 (29 models, 464 files) | 8-9 | 31 GB | 2026-10-08 15:04 | 2026-10-08 15:45 | 2026-10-08 15:56 (copies renamed, originals deleted) | 2026-10-08 16:00 |
| 3 | 1011-1016, 1111-1114, 1211-1214, 1311, 1312, 1411-1413, 1511-1515 (24 models, 384 files) | 10-14 | 26 GB | 2026-10-08 16:02 | 2026-10-08 18:00 | 2026-10-08 20:45 (copies renamed, originals deleted) | 2026-10-08 23:24 (240 files, 23.0 GB; each file checked on the SSD and in the Drive file list) |
| 4 | 2061-2064, 2071-2074, 2081-2084, 2091-2094, 2101-2104, 2111-2114, 2121-2124 (fewer-labels subsets; 28 models, 448 files) | 6-12 | 30 GB | 2026-10-08 21:04 (packed straight into the Drive folder) | 2026-10-08 22:05 | 2026-10-08 23:20 (originals deleted) | 2026-10-08 23:24 (280 files, 26.8 GB; each file checked on the SSD and in the Drive file list) |
| 5 | 1611-1614, 2131-2134, 2141-2144, 2151-2154 (16 models, 256 files) | 13-15 | 17 GB | 2026-10-08 23:23 (packed straight into the Drive folder) | done by 2026-10-09 14:47 (all 8 files carry a Drive item id) | 2026-10-09 14:53 (copies renamed, originals deleted via the Drive mount; sizes equal) | waiting for the next SSD sync |

Shuffle numbering: video n step k = (n+1)*10 + k; 951-974 = the detector comparison on video 9; fewer-labels subsets =
2000 + 10*video + 1..4 (sub05a, sub05b, sub10a, sub10b). Scripts: `pack_models_round.sh` (tools folder of the Claude session).

## Other archives in `zhiheng_results/`
| folder | content | packed | copied by Kaiwen |
|---|---|---|---|
| `old_vs_new_test/` | `old_vs_new_test.tar` (511 MB, 801 files) = the local `local_data/` folder: the August test set `eye_last_minute_100` (100 frames + labels placed before the 2026-09-20 definitions) and the `label_review_2026-09-20` contact sheets (old vs new placement); README, SHA256SUMS, filelist inside | 2026-10-08 21:00 | 2026-10-08 23:25 (then the loose copy `archive/local_data` on the Stanford-account folder is deleted) |
