# 03 Body Project

Goal: use `body.mp4` to track mouse facial features, ears, and forepaws.

Create a second DLC project only after the eye project has completed one full loop:

```text
label -> train -> analyze -> inspect -> improve
```

## Project Settings

```text
Project name: FaceBodyTracking
Experimenter: Zhiheng
Video: /Users/lizhiheng/Desktop/research/生物/body.mp4
Working directory: /Users/lizhiheng/Desktop/research/生物/dlc_projects
Copy videos: yes
```

## Bodyparts

Start with:

```yaml
bodyparts:
- nose_tip
- mouth
- whisker_pad
- jaw
- cheek
- left_ear
- right_ear
- left_forepaw
- right_forepaw
```

If some facial landmarks are ambiguous in the video, simplify before labeling many frames.
Reliable labels are more important than a long keypoint list.

## First Labeling Target

Extract 100 to 200 frames.
Cover:

- clear side view
- head movement
- grooming or paw movement
- forepaw visible/invisible transitions
- ear occlusion
- blur
- lighting changes

## Evaluation

For each keypoint, check:

- mean test error in pixels
- likelihood distribution
- labeled video stability
- common failure modes

Do not move to neural modeling until keypoint tracking is stable.
