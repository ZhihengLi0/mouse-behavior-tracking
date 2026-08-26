#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path

import pandas as pd


PROJECT_DIR = Path(__file__).resolve().parents[1]
TEST_ROOT = PROJECT_DIR / "local_data" / "test_sets" / "eye_last_minute_100"
DELIVERABLE_TABLE_DIR = TEST_ROOT / "deliverables" / "04_tables"
OUT_CSV = DELIVERABLE_TABLE_DIR / "06_outlier_audit_template.csv"


def main() -> None:
    rows = []
    for train_frames in (20, 50, 100):
        source = TEST_ROOT / f"predictions_{train_frames}train" / "outlier_top12.csv"
        if not source.exists():
            raise FileNotFoundError(f"Missing outlier table: {source}")

        data = pd.read_csv(source)
        for rank, row in enumerate(data.itertuples(index=False), start=1):
            rows.append(
                {
                    "train_frames": train_frames,
                    "rank_within_model": rank,
                    "frame_name": row[0],
                    "worst_keypoint": row.max_keypoint,
                    "max_keypoint_error_px": row.max_keypoint_error_px,
                    "pupil_center_error_px": row.pupil_center_error_px,
                    "pupil_width_error_px": row.pupil_width_error_px,
                    "manual_label_issue": "",
                    "model_prediction_issue": "",
                    "ambiguous_or_minor": "",
                    "needs_relabel": "",
                    "notes": "",
                }
            )

    audit = pd.DataFrame(rows)
    DELIVERABLE_TABLE_DIR.mkdir(parents=True, exist_ok=True)
    audit.to_csv(OUT_CSV, index=False)
    print(f"saved_audit_template: {OUT_CSV}")
    print(audit.head(12).to_string(index=False))


if __name__ == "__main__":
    main()
