# 04 Metrics And Outputs

The project should produce measurable outputs, not only labeled videos.

## Eye Outputs

From DLC predictions:

```text
pupil_top, pupil_bottom, pupil_left, pupil_right
eyelid_top, eyelid_bottom
eye_nasal_corner, eye_temporal_corner
```

Compute:

```text
pupil_center_x
pupil_center_y
pupil_width
pupil_height
eye_opening
blink_event
```

Suggested formulas:

```text
pupil_center_x = mean(pupil_left_x, pupil_right_x)
pupil_center_y = mean(pupil_top_y, pupil_bottom_y)
pupil_width = distance(pupil_left, pupil_right)
pupil_height = distance(pupil_top, pupil_bottom)
eye_opening = distance(eyelid_top, eyelid_bottom)
```

Blink detection should use:

```text
eye_opening < threshold
```

and a minimum duration in frames.

Do not choose the threshold blindly.
Plot the `eye_opening` trace first, then choose a threshold that separates open-eye frames from blink frames.

## Quality Metrics

Track these:

- Train/test pixel error.
- Percentage of frames with low likelihood.
- Manual review of labeled videos.
- Blink precision, recall, and F1 on a manually checked video segment.
- Pupil center and diameter error on sampled frames.
- Failure case categories: reflection, closed eye, occlusion, blur, out-of-view.
