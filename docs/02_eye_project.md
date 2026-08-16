# 02 Eye Project

Goal: use `face.mp4` to track pupil position, pupil size, and blink events.

## 1. Create A New DLC Project

Open the GUI:

```bash
conda activate DEEPLABCUT
python -m deeplabcut
```

In the GUI choose:

```text
Create New Project
```

Use:

```text
Project name: EyePupilBlink
Experimenter: Zhiheng
Video: /Users/lizhiheng/Desktop/research/生物/face.mp4
Working directory: /Users/lizhiheng/Desktop/research/生物/dlc_projects
Copy videos: yes
```

## 2. Bodyparts

After project creation, open the generated `config.yaml`.
Set the bodyparts to:

```yaml
bodyparts:
- pupil_top
- pupil_bottom
- pupil_left
- pupil_right
- eyelid_top
- eyelid_bottom
- eye_nasal_corner
- eye_temporal_corner
```

Definitions:

- `pupil_top`: top edge of the visible pupil.
- `pupil_bottom`: bottom edge of the visible pupil.
- `pupil_left`: left edge of the visible pupil.
- `pupil_right`: right edge of the visible pupil.
- `eyelid_top`: upper eyelid point at the eye opening.
- `eyelid_bottom`: lower eyelid point at the eye opening.
- `eye_nasal_corner`: inner/nasal eye corner.
- `eye_temporal_corner`: outer/temporal eye corner.

## 3. Extract Frames

In the GUI choose:

```text
Extract Frames
```

Recommended first pass:

```text
Mode: automatic
Algorithm: kmeans
Number of frames: 100
```

The first 100 frames should include open eye, half-closed eye, blink/closed eye, reflections, blur, and pupil occlusion.

## 4. Label Frames

In the GUI choose:

```text
Label Frames
```

Label carefully. Do not label reflections as pupil edges.
If the pupil is fully invisible during a blink, leave pupil points unlabeled if the GUI allows it, or be consistent and mark those frames as low-confidence cases in notes.

## 5. Check Labels

Run:

```python
import deeplabcut
config_path = "/Users/lizhiheng/Desktop/research/生物/dlc_projects/EyePupilBlink-Zhiheng-YYYY-MM-DD/config.yaml"
deeplabcut.check_labels(config_path)
```

Replace `YYYY-MM-DD` with the actual generated folder date.

Inspect the label check images manually before training.

## 6. Train First Model

```python
import deeplabcut
config_path = "/Users/lizhiheng/Desktop/research/生物/dlc_projects/EyePupilBlink-Zhiheng-YYYY-MM-DD/config.yaml"

deeplabcut.create_training_dataset(config_path)
deeplabcut.train_network(config_path, shuffle=1)
deeplabcut.evaluate_network(config_path, shuffle=[1])
```

The first model only needs to be good enough to reveal failure cases.

## 7. Analyze Video

```python
import deeplabcut
config_path = "/Users/lizhiheng/Desktop/research/生物/dlc_projects/EyePupilBlink-Zhiheng-YYYY-MM-DD/config.yaml"
video = "/Users/lizhiheng/Desktop/research/生物/face.mp4"

deeplabcut.analyze_videos(config_path, [video], save_as_csv=True)
deeplabcut.create_labeled_video(config_path, [video])
```

Watch the labeled video. Write down failure cases and add more labeled frames in the next iteration.
