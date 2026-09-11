# Environment

Everything needed to recreate the compute environment for this project.

- `environment.yml` — conda environment definition. Create or update with:
  `conda env create -f environment/environment.yml` (env name: DEEPLABCUT).
- `01_local_setup.md` — local setup walkthrough: miniforge, DeepLabCut,
  napari labeling GUI, and the cache/config directories under `local_data/`.
- `check_setup.sh` — verifies the interpreter, DeepLabCut import, project
  paths, and video files.
- `inspect_videos.sh` — prints fps/frame-count/duration for the source videos.

The training interpreter used throughout:
`/Users/lizhiheng/miniforge3/envs/DEEPLABCUT/bin/python`
