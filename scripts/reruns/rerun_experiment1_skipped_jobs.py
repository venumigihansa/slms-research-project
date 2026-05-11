#!/usr/bin/env python3
"""Rerun selected skipped Experiment 1 evaluator jobs."""

from __future__ import annotations

import argparse
import csv
import gc
import json
import os
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Optional

import pandas as pd
from tqdm.auto import tqdm

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent.parent
AGREEMENT_DIR = REPO_ROOT / "scripts" / "agreement_experiment"
if str(AGREEMENT_DIR) not in sys.path:
    sys.path.insert(0, str(AGREEMENT_DIR))

DEFAULT_MANIFEST_PATH = Path("results /experiment 1 reruns/experiment1_skip_manifest.csv")
DEFAULT_RERUN_DIR = Path("results /experiment 1 reruns")
DEFAULT_TRACE_OUTPUT_DIR = Path("/workspace/model_agreement_experiment_full")
DEFAULT_MAX_SEQ_LENGTH = 50000
DEFAULT_MAX_NEW_TOKENS = 160
DEFAULT_MAX_RETRIES = 2
DEFAULT_TIME_LIMIT_SECONDS = None
RERUN_COLUMNS = [
    "job_id",
    "trace_id",
    "evaluator",
    "level",
    "occurrence_index",
    "target_label",
    "model",
    "score",
    "explanation",
    "latency_ms",
    "is_skipped",
    "skip_reason",
    "failure_type",
    "source_original_csv",
    "original_skip_reason",
]


def load_agreement_runner():
    """Import GPU/runtime-heavy Experiment 1 helpers only when executing a rerun."""
    import run_model_agreement_experiment as runner

    return runner


def add_job_id_to_jobs(jobs):
    mapping = {}
    for job in jobs:
        job_id = f"{job.trace_id}|{job.evaluator_name}|{job.level}|{job.occurrence_index}"
        if job_id in mapping:
            raise ValueError(f"Duplicate reconstructed job_id: {job_id}")
        mapping[job_id] = job
    return mapping


def selected_manifest_rows(args: argparse.Namespace) -> pd.DataFrame:
    manifest = pd.read_csv(args.manifest_path)
    if args.model:
        manifest = manifest[manifest["model"] == args.model]
    if args.source_csv:
        manifest = manifest[manifest["source_csv"] == args.source_csv]
    if args.job_id:
        requested_job_ids = set(args.job_id)
        manifest = manifest[manifest["job_id"].isin(requested_job_ids)]

    categories = {item.strip() for item in args.skip_categories.split(",") if item.strip()}
    if categories:
        manifest = manifest[manifest["skip_category"].isin(categories)]
    if args.recommended_only:
        manifest = manifest[manifest["rerun_recommended"].astype(str).str.lower().isin({"true", "1", "yes"})]
    if args.limit is not None:
        manifest = manifest.head(args.limit)

    if manifest.empty:
        raise ValueError("No manifest rows matched the requested filters.")
    if args.job_id:
        matched_job_ids = set(manifest["job_id"])
        missing_job_ids = sorted(set(args.job_id) - matched_job_ids)
        if missing_job_ids:
            raise ValueError(f"Requested job_id(s) not found after filters: {missing_job_ids}")
    if manifest["model"].nunique() != 1:
        raise ValueError("Rerun selection must contain exactly one model. Pass --model or --source-csv.")
    return manifest.copy()


def result_row(job, result, latency_ms: float, model_name: str, manifest_row: pd.Series, failure_type: Optional[str] = None) -> dict:
    return {
        "job_id": manifest_row["job_id"],
        "trace_id": job.trace_id,
        "evaluator": job.evaluator_name,
        "level": job.level,
        "occurrence_index": job.occurrence_index,
        "target_label": job.target_label,
        "model": model_name,
        "score": "" if result.is_skipped else result.score,
        "explanation": "" if result.is_skipped else (result.explanation or ""),
        "latency_ms": round(latency_ms, 2),
        "is_skipped": result.is_skipped,
        "skip_reason": result.skip_reason or "",
        "failure_type": failure_type or ("precomputed_skip" if result.is_skipped else ""),
        "source_original_csv": manifest_row["source_csv"],
        "original_skip_reason": manifest_row["original_skip_reason"],
    }


def fallback_row(job, manifest_row: pd.Series, latency_ms: float, model_name: str, raw_output: str, error: str, failure_type: str) -> dict:
    if failure_type == "invalid_json":
        skip_reason = f"Invalid judge JSON; saved raw output instead. Reason: {error}"
    elif failure_type == "generation_error":
        skip_reason = f"LLM generation failed before producing parseable output. Reason: {error}"
    else:
        skip_reason = f"{failure_type}. Reason: {error}"
    return {
        "job_id": manifest_row["job_id"],
        "trace_id": job.trace_id,
        "evaluator": job.evaluator_name,
        "level": job.level,
        "occurrence_index": job.occurrence_index,
        "target_label": job.target_label,
        "model": model_name,
        "score": "",
        "explanation": raw_output,
        "latency_ms": round(latency_ms, 2),
        "is_skipped": True,
        "skip_reason": skip_reason,
        "failure_type": failure_type,
        "source_original_csv": manifest_row["source_csv"],
        "original_skip_reason": manifest_row["original_skip_reason"],
    }


def append_row(path: Path, row: dict) -> None:
    write_header = not path.exists()
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=RERUN_COLUMNS)
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def build_runtime_args(args: argparse.Namespace, model_name: str) -> SimpleNamespace:
    return SimpleNamespace(
        model=model_name,
        dataset=args.dataset,
        dataset_split=args.dataset_split,
        output_dir=args.trace_output_dir,
        trace_limit=args.trace_limit,
        max_seq_length=args.max_seq_length,
        max_new_tokens=args.max_new_tokens,
        max_retries=args.max_retries,
        batch_size=args.batch_size,
        time_limit_seconds=args.time_limit_seconds,
        load_in_4bit=not args.no_4bit,
        hf_token=os.environ.get("HF_TOKEN"),
    )


def run_rerun(args: argparse.Namespace) -> dict:
    runner = load_agreement_runner()
    selected = selected_manifest_rows(args)
    model_name = str(selected["model"].iloc[0])
    runtime_args = build_runtime_args(args, model_name)
    traces = runner.load_traces_for_experiment(runtime_args)
    jobs = runner.build_jobs(traces, runner.build_evaluators(runtime_args))
    job_map = add_job_id_to_jobs(jobs)

    missing_job_ids = sorted(set(selected["job_id"]) - set(job_map))
    if missing_job_ids:
        raise ValueError(f"{len(missing_job_ids)} selected job_ids were not reconstructed. First missing: {missing_job_ids[0]}")

    selected = selected.set_index("job_id", drop=False)
    selected_jobs = [job_map[job_id] for job_id in selected["job_id"]]

    output_path = args.output_path
    if output_path is None:
        output_path = args.rerun_dir / f"{runner.sanitize_filename(model_name)}_rerun.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists() and not args.resume and not args.dry_run:
        output_path.unlink()

    print(f"Model: {model_name}")
    print(f"Selected rerun jobs: {len(selected_jobs)}")
    print(f"Output path: {output_path}")
    if args.dry_run:
        print("Dry run only; model inference was not executed.")
        print(selected[["job_id", "skip_category", "source_csv"]].head(args.preview_rows).to_string(index=False))
        return {"selected_jobs": len(selected_jobs), "output_path": str(output_path), "dry_run": True}

    model = None
    tokenizer = None
    success_count = 0
    skip_count = 0
    start_time = time.time()
    debug_log_path = output_path.with_suffix(".failures.jsonl")

    try:
        model, tokenizer = runner.load_local_judge(runtime_args)
        progress = tqdm(total=len(selected_jobs), desc="Rerunning skipped jobs", unit="job")
        for batch in runner.batched(selected_jobs, args.batch_size):
            remaining = [
                {
                    "job": job,
                    "manifest_row": selected.loc[f"{job.trace_id}|{job.evaluator_name}|{job.level}|{job.occurrence_index}"],
                    "last_error": None,
                    "total_latency_ms": 0.0,
                    "last_raw_output": "",
                    "last_normalized_output": "",
                    "last_failure_type": None,
                }
                for job in batch
            ]

            for attempt in range(args.max_retries + 1):
                prompts = [runner.build_retry_prompt(item["job"].prompt or "", attempt, item["last_error"]) for item in remaining]
                t0 = time.perf_counter()
                try:
                    raw_outputs = runner.run_generation_batch(model, tokenizer, prompts, runtime_args)
                    batch_latency_ms = (time.perf_counter() - t0) * 1000.0
                    per_item_latency_ms = batch_latency_ms / max(len(remaining), 1)
                    generation_error = None
                except Exception as exc:
                    raw_outputs = [""] * len(remaining)
                    batch_latency_ms = (time.perf_counter() - t0) * 1000.0
                    per_item_latency_ms = batch_latency_ms / max(len(remaining), 1)
                    generation_error = str(exc)

                failed = []
                for item, raw_output in zip(remaining, raw_outputs):
                    job = item["job"]
                    item["total_latency_ms"] += per_item_latency_ms
                    if generation_error is not None:
                        item["last_error"] = generation_error
                        item["last_raw_output"] = ""
                        item["last_normalized_output"] = ""
                        item["last_failure_type"] = "generation_error"
                        parsed_result = None
                    else:
                        item["last_raw_output"] = raw_output
                        parsed_result, error, normalized_output = runner.parse_generation_output(job, raw_output)
                        item["last_error"] = error
                        item["last_normalized_output"] = normalized_output
                        item["last_failure_type"] = None if parsed_result is not None else "invalid_json"

                    if parsed_result is not None:
                        append_row(output_path, result_row(job, parsed_result, item["total_latency_ms"], model_name, item["manifest_row"]))
                        success_count += 1
                        progress.update(1)
                        continue

                    runner.append_failure_log(
                        debug_log_path,
                        {
                            "job_id": item["manifest_row"]["job_id"],
                            "trace_id": job.trace_id,
                            "evaluator": job.evaluator_name,
                            "level": job.level,
                            "target_label": job.target_label,
                            "attempt": attempt + 1,
                            "error": item["last_error"],
                            "raw_output": item["last_raw_output"],
                            "normalized_output": item["last_normalized_output"],
                        },
                    )
                    failed.append(item)

                remaining = failed
                if not remaining:
                    break

            for item in remaining:
                job = item["job"]
                append_row(
                    output_path,
                    fallback_row(
                        job,
                        item["manifest_row"],
                        item["total_latency_ms"],
                        model_name,
                        item["last_raw_output"],
                        item["last_error"] or "Unknown error",
                        item["last_failure_type"] or "unknown_failure",
                    ),
                )
                skip_count += 1
                progress.update(1)

            gc.collect()
            try:
                import torch

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except ImportError:
                pass
        progress.close()
    finally:
        if model is not None or tokenizer is not None:
            runner.cleanup_model(model, tokenizer)

    return {
        "selected_jobs": len(selected_jobs),
        "success_count": success_count,
        "skip_count": skip_count,
        "elapsed_seconds": round(time.time() - start_time, 2),
        "output_path": str(output_path),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Rerun selected skipped Experiment 1 jobs.")
    parser.add_argument("--manifest-path", type=Path, default=DEFAULT_MANIFEST_PATH)
    parser.add_argument("--model", default=None, help="Exact model name from manifest, e.g. google/gemma-3-4b-it.")
    parser.add_argument("--source-csv", default=None, help="Source CSV name from manifest, e.g. google_gemma-3-4b-it.csv.")
    parser.add_argument(
        "--job-id",
        action="append",
        default=None,
        help="Exact inferred job_id to rerun. Can be passed multiple times.",
    )
    parser.add_argument("--skip-categories", default="memory_oom")
    parser.add_argument("--recommended-only", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--preview-rows", type=int, default=10)
    parser.add_argument("--rerun-dir", type=Path, default=DEFAULT_RERUN_DIR)
    parser.add_argument("--output-path", type=Path, default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dataset", default="PatronusAI/TRAIL")
    parser.add_argument("--dataset-split", default="gaia")
    parser.add_argument("--trace-output-dir", type=Path, default=DEFAULT_TRACE_OUTPUT_DIR)
    parser.add_argument("--trace-limit", type=int, default=None)
    parser.add_argument("--time-limit-seconds", type=int, default=DEFAULT_TIME_LIMIT_SECONDS)
    parser.add_argument("--max-seq-length", type=int, default=DEFAULT_MAX_SEQ_LENGTH)
    parser.add_argument("--max-new-tokens", type=int, default=DEFAULT_MAX_NEW_TOKENS)
    parser.add_argument("--max-retries", type=int, default=DEFAULT_MAX_RETRIES)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--no-4bit", action="store_true")
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error("--batch-size must be at least 1")
    return args


def main() -> None:
    summary = run_rerun(parse_args())
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
