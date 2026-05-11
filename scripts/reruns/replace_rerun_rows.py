#!/usr/bin/env python3
"""Replace failed rows in a full rerun CSV with successful single-fix rerun rows."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def is_skipped_series(series: pd.Series) -> pd.Series:
    if series.dtype == bool:
        return series.fillna(False)
    return series.fillna(False).astype(str).str.lower().isin({"true", "1", "yes"})


def successful_fix_rows(fix: pd.DataFrame) -> pd.DataFrame:
    fix = fix.copy()
    fix["score_numeric"] = pd.to_numeric(fix["score"], errors="coerce")
    fix_skipped = is_skipped_series(fix["is_skipped"])
    valid = (~fix_skipped) & fix["score_numeric"].between(0.0, 1.0)
    return fix.loc[valid].copy()


def replace_rows(full_rerun_path: Path, fix_rerun_path: Path, output_path: Path) -> dict:
    full = pd.read_csv(full_rerun_path)
    fix = pd.read_csv(fix_rerun_path)

    required = {"job_id", "score", "is_skipped"}
    missing_full = required - set(full.columns)
    missing_fix = required - set(fix.columns)
    if missing_full:
        raise ValueError(f"Full rerun CSV missing required columns: {sorted(missing_full)}")
    if missing_fix:
        raise ValueError(f"Fix rerun CSV missing required columns: {sorted(missing_fix)}")

    if full["job_id"].duplicated().any():
        duplicates = sorted(full.loc[full["job_id"].duplicated(keep=False), "job_id"].unique())
        raise ValueError(f"Full rerun CSV has duplicate job_id values: {duplicates[:5]}")

    valid_fix = successful_fix_rows(fix)
    if valid_fix.empty:
        raise ValueError("Fix rerun CSV has no successful rows to apply.")
    if valid_fix["job_id"].duplicated().any():
        duplicates = sorted(valid_fix.loc[valid_fix["job_id"].duplicated(keep=False), "job_id"].unique())
        raise ValueError(f"Fix rerun CSV has duplicate successful job_id values: {duplicates[:5]}")

    full_by_job = full.set_index("job_id", drop=False)
    missing_jobs = sorted(set(valid_fix["job_id"]) - set(full_by_job.index))
    if missing_jobs:
        raise ValueError(f"Fix job_id(s) not found in full rerun CSV: {missing_jobs}")

    replaced = 0
    for _, fix_row in valid_fix.iterrows():
        job_id = fix_row["job_id"]
        full_idx = full.index[full_by_job.index.get_loc(job_id)]
        if not is_skipped_series(pd.Series([full.at[full_idx, "is_skipped"]])).iloc[0]:
            continue
        for column in full.columns:
            if column in fix_row.index:
                full.at[full_idx, column] = fix_row[column]
        replaced += 1

    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.resolve() == full_rerun_path.resolve():
        raise ValueError("Refusing to overwrite the full rerun CSV. Choose a different --output-path.")
    full.to_csv(output_path, index=False)

    return {
        "full_rows": len(full),
        "successful_fix_rows": len(valid_fix),
        "replaced_rows": replaced,
        "remaining_skipped_rows": int(is_skipped_series(full["is_skipped"]).sum()),
        "output_path": str(output_path),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Replace skipped rerun rows with successful single-fix rows.")
    parser.add_argument("--full-rerun-csv", type=Path, required=True)
    parser.add_argument("--fix-rerun-csv", type=Path, required=True)
    parser.add_argument("--output-path", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = replace_rows(args.full_rerun_csv, args.fix_rerun_csv, args.output_path)
    for key, value in summary.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
