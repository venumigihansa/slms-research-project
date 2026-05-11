#!/usr/bin/env python3
"""Merge Experiment 1 rerun scores into corrected CSV copies."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


DEFAULT_OUTPUT_DIR = Path("results /experiment 1 corrected")
KEY_COLUMNS = ["trace_id", "evaluator", "level"]


def is_skipped_series(series: pd.Series) -> pd.Series:
    if series.dtype == bool:
        return series.fillna(False)
    return series.fillna(False).astype(str).str.lower().isin({"true", "1", "yes"})


def add_job_identity(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["occurrence_index"] = df.groupby(KEY_COLUMNS).cumcount()
    df["job_id"] = (
        df["trace_id"].astype(str)
        + "|"
        + df["evaluator"].astype(str)
        + "|"
        + df["level"].astype(str)
        + "|"
        + df["occurrence_index"].astype(str)
    )
    if df["job_id"].duplicated().any():
        raise ValueError("Original CSV produced duplicate inferred job_id values.")
    return df


def valid_rerun_scores(rerun: pd.DataFrame) -> pd.DataFrame:
    rerun = rerun.copy()
    rerun["score_numeric"] = pd.to_numeric(rerun["score"], errors="coerce")
    rerun_skipped = is_skipped_series(rerun["is_skipped"])
    valid = (~rerun_skipped) & rerun["score_numeric"].between(0.0, 1.0)
    return rerun.loc[valid].copy()


def merge_reruns(original_path: Path, rerun_path: Path, output_path: Path) -> dict:
    original = add_job_identity(pd.read_csv(original_path))
    rerun = pd.read_csv(rerun_path)

    required = {"job_id", "score", "explanation", "latency_ms", "is_skipped", "skip_reason"}
    missing = required - set(rerun.columns)
    if missing:
        raise ValueError(f"Rerun CSV missing required columns: {sorted(missing)}")

    valid_reruns = valid_rerun_scores(rerun)
    duplicate_valid = valid_reruns["job_id"].duplicated(keep=False)
    if duplicate_valid.any():
        duplicates = sorted(valid_reruns.loc[duplicate_valid, "job_id"].unique())
        raise ValueError(f"Rerun CSV has duplicate valid rows for job_id(s): {duplicates[:5]}")

    valid_by_job = valid_reruns.set_index("job_id")
    rerun_by_job = rerun.drop_duplicates("job_id", keep="last").set_index("job_id")
    original_skipped = is_skipped_series(original["is_skipped"])
    original["original_skip_reason"] = original["skip_reason"].fillna("")
    original["rerun_skip_reason"] = ""
    original["score_source"] = "original"

    replaced = 0
    for idx, row in original.iterrows():
        job_id = row["job_id"]
        if job_id in rerun_by_job.index:
            original.at[idx, "rerun_skip_reason"] = rerun_by_job.loc[job_id].get("skip_reason", "")
        if not original_skipped.loc[idx] or job_id not in valid_by_job.index:
            continue
        rerun_row = valid_by_job.loc[job_id]
        original.at[idx, "score"] = rerun_row["score_numeric"]
        original.at[idx, "explanation"] = rerun_row.get("explanation", "")
        original.at[idx, "latency_ms"] = rerun_row.get("latency_ms", "")
        original.at[idx, "is_skipped"] = False
        original.at[idx, "skip_reason"] = ""
        original.at[idx, "score_source"] = "rerun"
        replaced += 1

    still_skipped = is_skipped_series(original["is_skipped"])
    original.loc[still_skipped, "score_source"] = "still_skipped"

    output_path.parent.mkdir(parents=True, exist_ok=True)
    original.to_csv(output_path, index=False)
    return {
        "original_rows": len(original),
        "valid_rerun_rows": len(valid_reruns),
        "replaced_rows": replaced,
        "still_skipped_rows": int(still_skipped.sum()),
        "output_path": str(output_path),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Merge Experiment 1 rerun scores into a corrected copy.")
    parser.add_argument("--original-csv", type=Path, required=True)
    parser.add_argument("--rerun-csv", type=Path, required=True)
    parser.add_argument("--output-path", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()
    if args.output_path is None:
        args.output_path = args.output_dir / args.original_csv.name
    if args.output_path.resolve() == args.original_csv.resolve():
        parser.error("Refusing to overwrite the original CSV. Choose a different --output-path.")
    return args


def main() -> None:
    args = parse_args()
    summary = merge_reruns(args.original_csv, args.rerun_csv, args.output_path)
    for key, value in summary.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
