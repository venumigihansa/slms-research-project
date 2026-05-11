#!/usr/bin/env python3
"""Build a manifest of skipped Experiment 1 evaluator jobs."""

from __future__ import annotations

import argparse
import os
import re
from pathlib import Path

import pandas as pd


DEFAULT_RESULTS_DIR = Path("results /experiment 1")
DEFAULT_OUTPUT_PATH = Path("results /experiment 1 reruns/experiment1_skip_manifest.csv")
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
    return df


def classify_skip(skip_reason: str) -> str:
    reason = str(skip_reason or "")
    memory = bool(re.search(r"out of memory|oom|cuda|mps|memory", reason, flags=re.IGNORECASE))
    invalid_json = bool(
        re.search(
            r"invalid judge json|json_invalid|invalid json|validation error for judgeoutput|judgeoutput",
            reason,
            flags=re.IGNORECASE,
        )
    )
    if memory and invalid_json:
        return "mixed_or_ambiguous"
    if memory:
        return "memory_oom"
    if invalid_json:
        return "invalid_json_schema"
    return "other"


def rerun_recommended(skip_category: str, raw_output_present: bool, skip_reason: str) -> bool:
    if skip_category == "memory_oom":
        return True
    if skip_category == "mixed_or_ambiguous":
        return bool(re.search(r"out of memory|oom|cuda|mps|memory", str(skip_reason), flags=re.IGNORECASE)) and not raw_output_present
    return False


def load_result_csvs(results_dir: Path) -> list[Path]:
    paths = sorted(path for path in results_dir.glob("*.csv") if path.is_file())
    if not paths:
        raise FileNotFoundError(f"No CSV files found in {results_dir}")
    return paths


def validate_job_key_sets(frames: dict[str, pd.DataFrame], reference_name: str = "gpt-5.4.csv") -> None:
    if reference_name not in frames:
        print(f"Warning: {reference_name} not found; skipping cross-model job_id set validation.")
        return

    reference_keys = set(frames[reference_name]["job_id"])
    for name, df in frames.items():
        duplicate_count = int(df["job_id"].duplicated().sum())
        if duplicate_count:
            raise ValueError(f"{name} has {duplicate_count} duplicate inferred job_id values.")

        keys = set(df["job_id"])
        missing = len(reference_keys - keys)
        extra = len(keys - reference_keys)
        if missing or extra:
            raise ValueError(
                f"{name} job_id set does not match {reference_name}: "
                f"missing={missing}, extra={extra}"
            )


def build_manifest(results_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    summary = []
    frames = {}

    for csv_path in load_result_csvs(results_dir):
        df = add_job_identity(pd.read_csv(csv_path))
        frames[csv_path.name] = df

        skipped = is_skipped_series(df["is_skipped"])
        raw_present = df["explanation"].fillna("").astype(str).str.strip().ne("")
        skip_categories = df["skip_reason"].fillna("").astype(str).map(classify_skip)

        summary.append(
            {
                "source_csv": csv_path.name,
                "rows": len(df),
                "unique_job_ids": df["job_id"].nunique(),
                "skipped": int(skipped.sum()),
                "memory_oom": int(((skip_categories == "memory_oom") & skipped).sum()),
                "invalid_json_schema": int(((skip_categories == "invalid_json_schema") & skipped).sum()),
                "mixed_or_ambiguous": int(((skip_categories == "mixed_or_ambiguous") & skipped).sum()),
                "other": int(((skip_categories == "other") & skipped).sum()),
            }
        )

        skipped_df = df.loc[skipped].copy()
        for idx, row in skipped_df.iterrows():
            raw_output_present = bool(raw_present.loc[idx])
            skip_category = str(skip_categories.loc[idx])
            rows.append(
                {
                    "model": row["model"],
                    "source_csv": csv_path.name,
                    "job_id": row["job_id"],
                    "trace_id": row["trace_id"],
                    "evaluator": row["evaluator"],
                    "level": row["level"],
                    "occurrence_index": int(row["occurrence_index"]),
                    "is_skipped": bool(skipped.loc[idx]),
                    "original_skip_reason": row.get("skip_reason", ""),
                    "original_raw_output_present": raw_output_present,
                    "skip_category": skip_category,
                    "rerun_recommended": rerun_recommended(skip_category, raw_output_present, row.get("skip_reason", "")),
                }
            )

    validate_job_key_sets(frames)
    return pd.DataFrame(rows), pd.DataFrame(summary)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build Experiment 1 skipped-job manifest.")
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    parser.add_argument("--output-path", type=Path, default=DEFAULT_OUTPUT_PATH)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest, summary = build_manifest(args.results_dir)
    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(args.output_path, index=False)

    print(f"Wrote skipped-job manifest: {args.output_path}")
    print(f"Skipped jobs: {len(manifest)}")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
