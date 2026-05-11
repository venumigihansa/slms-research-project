#!/usr/bin/env python3
"""Summarize groundedness prompt-format experiment results."""

from __future__ import annotations

import argparse
import json
from itertools import combinations
from pathlib import Path

import pandas as pd


def numeric_scores(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["score"] = pd.to_numeric(out["score"], errors="coerce")
    out["is_skipped"] = out["is_skipped"].astype(str).str.lower().isin({"true", "1", "yes"})
    return out


def summarize_by_format(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for prompt_format, group in df.groupby("prompt_format"):
        valid = group[group["score"].notna() & ~group["is_skipped"]]
        rows.append(
            {
                "prompt_format": prompt_format,
                "rows": len(group),
                "valid_scores": len(valid),
                "skipped_or_invalid": len(group) - len(valid),
                "valid_rate": round(len(valid) / len(group), 4) if len(group) else 0.0,
                "mean_score": round(valid["score"].mean(), 4) if len(valid) else None,
                "std_score": round(valid["score"].std(), 4) if len(valid) > 1 else None,
                "mean_latency_ms": round(group["latency_ms"].mean(), 2) if "latency_ms" in group else None,
            }
        )
    return pd.DataFrame(rows).sort_values("prompt_format")


def prompt_sensitivity(df: pd.DataFrame) -> pd.DataFrame:
    valid = df[df["score"].notna() & ~df["is_skipped"]]
    pivot = valid.pivot_table(index=["model", "trace_id"], columns="prompt_format", values="score", aggfunc="mean")
    if pivot.empty:
        return pd.DataFrame()
    pivot["available_formats"] = pivot.notna().sum(axis=1)
    score_columns = [col for col in pivot.columns if col != "available_formats"]
    pivot["prompt_sensitivity"] = pivot[score_columns].max(axis=1) - pivot[score_columns].min(axis=1)
    return pivot.reset_index()


def pairwise_correlations(df: pd.DataFrame) -> pd.DataFrame:
    valid = df[df["score"].notna() & ~df["is_skipped"]]
    rows = []
    for model, group in valid.groupby("model"):
        pivot = group.pivot_table(index="trace_id", columns="prompt_format", values="score", aggfunc="mean")
        for left, right in combinations(sorted(pivot.columns), 2):
            pair = pivot[[left, right]].dropna()
            rows.append(
                {
                    "model": model,
                    "left_format": left,
                    "right_format": right,
                    "shared_traces": len(pair),
                    "pearson": round(pair[left].corr(pair[right], method="pearson"), 4) if len(pair) > 1 else None,
                    "spearman": round(pair[left].corr(pair[right], method="spearman"), 4) if len(pair) > 1 else None,
                    "mean_abs_diff": round((pair[left] - pair[right]).abs().mean(), 4) if len(pair) else None,
                }
            )
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description="Analyze groundedness prompt-format CSV results.")
    parser.add_argument("results_csv", type=Path)
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()

    df = numeric_scores(pd.read_csv(args.results_csv))
    output_dir = args.output_dir or args.results_csv.parent
    output_dir.mkdir(parents=True, exist_ok=True)

    by_format = summarize_by_format(df)
    sensitivity = prompt_sensitivity(df)
    correlations = pairwise_correlations(df)

    by_format_path = output_dir / "prompt_format_summary.csv"
    sensitivity_path = output_dir / "prompt_sensitivity_by_trace.csv"
    correlations_path = output_dir / "prompt_format_pairwise_correlations.csv"

    by_format.to_csv(by_format_path, index=False)
    sensitivity.to_csv(sensitivity_path, index=False)
    correlations.to_csv(correlations_path, index=False)

    summary = {
        "input": str(args.results_csv),
        "rows": len(df),
        "prompt_format_summary": str(by_format_path),
        "prompt_sensitivity_by_trace": str(sensitivity_path),
        "prompt_format_pairwise_correlations": str(correlations_path),
        "mean_prompt_sensitivity": (
            round(sensitivity["prompt_sensitivity"].mean(), 4)
            if not sensitivity.empty and "prompt_sensitivity" in sensitivity
            else None
        ),
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
