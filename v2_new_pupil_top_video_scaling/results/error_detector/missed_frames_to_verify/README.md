# 101 frames > 20 px that fit_pupil misses at 5 % flagged — for human re-check

Source: `scores_image.csv` (frames with wrong20 and fit_pupil below its 95th percentile, 12.8 px). One jpg per
frame: eye crop around the predicted corners, green = human label, red = model prediction, orange line joins them.
`contact_sheet_*.jpg` show them 4 per row. `verdicts_template.csv` is to be filled by the labeler (verdict: model_wrong /
label_questionable / both_acceptable / other). The question: are these model errors, or 20–35 px disagreements where
the label itself is uncertain? Images are local only; the csv with verdicts is committed when filled.
