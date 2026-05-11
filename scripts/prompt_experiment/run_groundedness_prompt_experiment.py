#!/usr/bin/env python3
"""Run the groundedness prompt-format experiment with a local HF judge model."""

from __future__ import annotations

import argparse
import csv
import gc
import json
import os
import sys
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import torch
from tqdm.auto import tqdm

SCRIPT_DIR = Path(__file__).resolve().parent
SCRIPTS_DIR = SCRIPT_DIR.parent
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from amp_evaluation.models import EvalResult
from amp_evaluation.trace.fetcher import _parse_trace
from amp_evaluation.trace.parser import parse_trace_for_evaluation
from groundedness_prompts import PROMPT_EVALUATORS
from run_gemma_runpod import (
    LOAD_IN_4BIT,
    MAX_NEW_TOKENS,
    MAX_RETRIES,
    MAX_SEQ_LENGTH,
    SHORT_OUTPUT_INSTRUCTIONS,
    build_retry_prompt,
    extract_json_object,
    flatten_trace_spans,
    infer_trace_io,
    load_and_prepare_traces as download_and_prepare_traces,
    load_local_judge,
    render_prompt_text,
)


DEFAULT_MODEL = "meta-llama/Llama-3.2-1B-Instruct"
DEFAULT_OUTPUT_DIR = Path("/workspace/groundedness_prompt_experiment_full")

CSV_COLUMNS = [
    "model",
    "trace_id",
    "evaluator",
    "prompt_format",
    "level",
    "score",
    "explanation",
    "latency_ms",
    "is_skipped",
    "skip_reason",
]


@dataclass(frozen=True)
class PromptEvalJob:
    trace_id: str
    evaluator_name: str
    prompt_format: str
    level: str
    prompt: Optional[str]
    evaluator: object
    precomputed_result: Optional[EvalResult] = None


def sanitize_filename(value: str) -> str:
    safe = "".join(ch if ch.isalnum() or ch in ("-", "_", ".") else "_" for ch in value)
    return safe.strip("_") or "model"


def load_preprocessed_traces(args):
    preprocessed_dir = args.output_dir / "preprocessed_traces"
    paths = sorted(preprocessed_dir.glob("*.json"))
    if args.trace_limit is not None:
        paths = paths[:args.trace_limit]

    traces = []
    for path in paths:
        with path.open("r", encoding="utf-8") as handle:
            record = json.load(handle)
        trace_obj = record["trace"]
        flat_spans = flatten_trace_spans(trace_obj.get("spans", []))
        if not flat_spans:
            continue
        trace_id, trace_input, trace_output = infer_trace_io(trace_obj, flat_spans)
        error_count = sum(
            1
            for span in flat_spans
            if ((span.get("ampAttributes") or {}).get("status") or {}).get("error") or span.get("status") == "ERROR"
        )
        api_trace = {
            "traceId": trace_id,
            "rootSpanId": flat_spans[0]["spanId"],
            "rootSpanName": flat_spans[0]["name"],
            "startTime": flat_spans[0]["startTime"],
            "endTime": max(span["endTime"] for span in flat_spans),
            "spans": flat_spans,
            "rootSpanKind": flat_spans[0]["kind"],
            "durationInNanos": sum(span["durationInNanos"] for span in flat_spans),
            "spanCount": len(flat_spans),
            "status": {"errorCount": error_count},
            "input": trace_input,
            "output": trace_output,
        }
        otel_trace = _parse_trace(api_trace)
        parsed_trace = parse_trace_for_evaluation(otel_trace)
        parsed_trace._labels = record.get("labels")
        traces.append(parsed_trace)

    print(f"Reused {len(traces)} AMP Trace objects from {preprocessed_dir}")
    return traces


def load_traces_for_experiment(args):
    preprocessed_dir = args.output_dir / "preprocessed_traces"
    if preprocessed_dir.exists() and any(preprocessed_dir.glob("*.json")):
        return load_preprocessed_traces(args)
    return download_and_prepare_traces(args)


def build_evaluators(args):
    evaluators = []
    for prompt_format in args.prompt_formats:
        evaluator = PROMPT_EVALUATORS[prompt_format](on_missing_context=args.on_missing_context)
        evaluator.model = args.model
        evaluator.max_retries = args.max_retries
        evaluator._OUTPUT_INSTRUCTIONS = SHORT_OUTPUT_INSTRUCTIONS
        evaluators.append(evaluator)
    return evaluators


def build_full_prompt(evaluator, trace) -> str:
    return evaluator._dispatch_build_prompt(trace, None) + evaluator._OUTPUT_INSTRUCTIONS


def precompute_groundedness_if_needed(evaluator, trace) -> Optional[EvalResult]:
    if trace.get_tool_calls() or trace.get_retrievals():
        return None
    if getattr(evaluator, "on_missing_context", "skip") == "zero":
        return EvalResult(
            score=0.0,
            passed=False,
            explanation="No tool or retrieval spans found; cannot assess groundedness",
        )
    return EvalResult.skip("No tool or retrieval spans found in this trace")


def build_jobs(traces, evaluators) -> List[PromptEvalJob]:
    jobs: List[PromptEvalJob] = []
    for trace in traces:
        for evaluator in evaluators:
            prompt_format = getattr(evaluator, "prompt_format", evaluator.name.replace("groundedness_", ""))
            precomputed = precompute_groundedness_if_needed(evaluator, trace)
            prompt = None if precomputed is not None else build_full_prompt(evaluator, trace)
            jobs.append(
                PromptEvalJob(
                    trace_id=trace.trace_id,
                    evaluator_name=evaluator.name,
                    prompt_format=prompt_format,
                    level=evaluator.level.value,
                    prompt=prompt,
                    evaluator=evaluator,
                    precomputed_result=precomputed,
                )
            )
    return jobs


def init_output_artifacts(results_path: Path, debug_log_path: Path, resume: bool):
    results_path.parent.mkdir(parents=True, exist_ok=True)
    if not resume:
        for path in (results_path, debug_log_path):
            if path.exists():
                path.unlink()
    if not results_path.exists():
        with results_path.open("w", newline="", encoding="utf-8") as handle:
            csv.DictWriter(handle, fieldnames=CSV_COLUMNS).writeheader()
    debug_log_path.touch(exist_ok=True)


def append_result_row(results_path: Path, row: Dict[str, object]):
    with results_path.open("a", newline="", encoding="utf-8") as handle:
        csv.DictWriter(handle, fieldnames=CSV_COLUMNS).writerow(row)


def append_failure_log(debug_log_path: Path, payload: Dict[str, object]):
    with debug_log_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False) + "\n")


def result_to_row(job: PromptEvalJob, result: EvalResult, latency_ms: float, model_name: str) -> Dict[str, object]:
    return {
        "model": model_name,
        "trace_id": job.trace_id,
        "evaluator": job.evaluator_name,
        "prompt_format": job.prompt_format,
        "level": job.level,
        "score": "" if result.is_skipped else result.score,
        "explanation": "" if result.is_skipped else (result.explanation or ""),
        "latency_ms": round(latency_ms, 2),
        "is_skipped": result.is_skipped,
        "skip_reason": result.skip_reason or "",
    }


def raw_output_fallback_row(
    job: PromptEvalJob,
    raw_output: str,
    latency_ms: float,
    model_name: str,
    error: str,
) -> Dict[str, object]:
    return {
        "model": model_name,
        "trace_id": job.trace_id,
        "evaluator": job.evaluator_name,
        "prompt_format": job.prompt_format,
        "level": job.level,
        "score": "",
        "explanation": raw_output,
        "latency_ms": round(latency_ms, 2),
        "is_skipped": True,
        "skip_reason": f"Invalid judge JSON; saved raw output instead. {error}",
    }


def batched(items: List[PromptEvalJob], batch_size: int):
    for start in range(0, len(items), batch_size):
        yield items[start:start + batch_size]


def run_generation_batch(model, tokenizer, prompts: List[str], args) -> List[str]:
    rendered_prompts = [render_prompt_text(tokenizer, prompt) for prompt in prompts]
    encoded_inputs = [
        tokenizer(text=prompt, return_tensors=None, truncation=True)
        for prompt in rendered_prompts
    ]
    input_id_rows = []
    for encoded in encoded_inputs:
        input_ids = encoded["input_ids"]
        if input_ids and isinstance(input_ids[0], list):
            input_ids = input_ids[0]
        input_id_rows.append(torch.tensor(input_ids, dtype=torch.long))

    pad_token_id = tokenizer.pad_token_id
    if pad_token_id is None:
        pad_token_id = tokenizer.eos_token_id
    if pad_token_id is None:
        raise ValueError("Tokenizer must define pad_token_id or eos_token_id for batched generation.")

    max_prompt_length = max(row.shape[0] for row in input_id_rows)
    padded_rows = []
    attention_rows = []
    for row in input_id_rows:
        pad_length = max_prompt_length - row.shape[0]
        if pad_length <= 0:
            padded = row
            attention = torch.ones_like(row)
        else:
            padding = torch.full((pad_length,), pad_token_id, dtype=torch.long)
            attention_padding = torch.zeros((pad_length,), dtype=torch.long)
            attention_content = torch.ones_like(row)
            if getattr(tokenizer, "padding_side", "right") == "left":
                padded = torch.cat([padding, row])
                attention = torch.cat([attention_padding, attention_content])
            else:
                padded = torch.cat([row, padding])
                attention = torch.cat([attention_content, attention_padding])

        padded_rows.append(padded)
        attention_rows.append(attention)

    model_inputs = {
        "input_ids": torch.stack(padded_rows).to(model.device),
        "attention_mask": torch.stack(attention_rows).to(model.device),
    }
    with torch.no_grad():
        outputs = model.generate(
            **model_inputs,
            max_new_tokens=args.max_new_tokens,
            do_sample=False,
            use_cache=True,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    prompt_length = model_inputs["input_ids"].shape[1]
    decoded = []
    for row in outputs:
        generated = row[prompt_length:]
        decoded.append(tokenizer.decode(generated, skip_special_tokens=True).strip())
    return decoded


def parse_generation_output(job: PromptEvalJob, raw_output: str):
    normalized_output = extract_json_object(raw_output)
    parsed_result, error = job.evaluator._parse_and_validate(normalized_output)
    return parsed_result, error, normalized_output


def run_experiment(traces, args):
    model_slug = sanitize_filename(args.model)
    results_path = args.results_path or (args.output_dir / f"{model_slug}_groundedness_prompt_results.csv")
    debug_log_path = args.debug_log_path or (args.output_dir / f"{model_slug}_groundedness_prompt_failures.jsonl")
    evaluators = build_evaluators(args)
    jobs = build_jobs(traces, evaluators)
    init_output_artifacts(results_path, debug_log_path, resume=args.resume)

    print(f"Local judge model: {args.model}")
    print(f"Prompt formats: {', '.join(args.prompt_formats)}")
    print("Evaluator criterion: groundedness")
    print(f"Traces available: {len(traces)}")
    print(f"Jobs queued: {len(jobs)}")
    print(f"Batch size: {args.batch_size}")
    print(f"Results path: {results_path}")
    print(f"Debug log path: {debug_log_path}")

    start_time = time.time()
    attempted_jobs = 0
    success_count = 0
    skip_count = 0
    per_format = Counter()
    per_format_skips = Counter()
    model = None
    tokenizer = None
    progress = tqdm(total=len(jobs), desc="Groundedness prompt experiment", unit="job")

    try:
        model, tokenizer = load_local_judge(args)
        pending_jobs: List[PromptEvalJob] = []

        for job in jobs:
            elapsed = time.time() - start_time
            if args.time_limit_seconds is not None and elapsed >= args.time_limit_seconds:
                print(f"Reached time limit after {elapsed:.1f}s; stopping.")
                break

            attempted_jobs += 1
            per_format[job.prompt_format] += 1

            if job.precomputed_result is not None:
                result = job.precomputed_result
                append_result_row(results_path, result_to_row(job, result, 0.0, args.model))
                skip_count += int(result.is_skipped)
                success_count += int(not result.is_skipped)
                per_format_skips[job.prompt_format] += int(result.is_skipped)
                progress.update(1)
                continue

            pending_jobs.append(job)

        for batch in batched(pending_jobs, args.batch_size):
            elapsed = time.time() - start_time
            if args.time_limit_seconds is not None and elapsed >= args.time_limit_seconds:
                print(f"Reached time limit after {elapsed:.1f}s; stopping.")
                break

            remaining = [
                {
                    "job": job,
                    "last_error": None,
                    "total_latency_ms": 0.0,
                    "last_raw_output": "",
                    "last_normalized_output": "",
                }
                for job in batch
            ]

            for attempt in range(args.max_retries + 1):
                prompts = [
                    build_retry_prompt(item["job"].prompt or "", attempt, item["last_error"])
                    for item in remaining
                ]
                t0 = time.perf_counter()
                try:
                    raw_outputs = run_generation_batch(model, tokenizer, prompts, args)
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
                        parsed_result = None
                    else:
                        item["last_raw_output"] = raw_output
                        parsed_result, error, normalized_output = parse_generation_output(job, raw_output)
                        item["last_error"] = error
                        item["last_normalized_output"] = normalized_output

                    if parsed_result is not None:
                        append_result_row(
                            results_path,
                            result_to_row(job, parsed_result, item["total_latency_ms"], args.model),
                        )
                        skip_count += int(parsed_result.is_skipped)
                        success_count += int(not parsed_result.is_skipped)
                        per_format_skips[job.prompt_format] += int(parsed_result.is_skipped)
                        progress.update(1)
                        continue

                    append_failure_log(
                        debug_log_path,
                        {
                            "trace_id": job.trace_id,
                            "evaluator": job.evaluator_name,
                            "prompt_format": job.prompt_format,
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
                append_result_row(
                    results_path,
                    raw_output_fallback_row(
                        job,
                        item["last_raw_output"],
                        item["total_latency_ms"],
                        args.model,
                        item["last_error"] or "Unknown error",
                    ),
                )
                skip_count += 1
                per_format_skips[job.prompt_format] += 1
                progress.update(1)

            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    finally:
        progress.close()
        if model is not None or tokenizer is not None:
            try:
                del model
                del tokenizer
            except Exception:
                pass
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return {
        "attempted_jobs": attempted_jobs,
        "success_count": success_count,
        "skip_count": skip_count,
        "elapsed_seconds": round(time.time() - start_time, 2),
        "per_format": dict(per_format),
        "per_format_skips": dict(per_format_skips),
        "batch_size": args.batch_size,
        "results_path": str(results_path),
        "debug_log_path": str(debug_log_path),
    }


def parse_prompt_formats(raw: str) -> List[str]:
    if raw.strip().lower() == "all":
        return ["line", "json", "bullet"]
    formats = [item.strip().lower() for item in raw.split(",") if item.strip()]
    invalid = [item for item in formats if item not in PROMPT_EVALUATORS]
    if invalid:
        raise argparse.ArgumentTypeError(
            f"Unknown prompt format(s): {', '.join(invalid)}. Valid values: all, line, json, bullet"
        )
    return formats


def parse_args():
    parser = argparse.ArgumentParser(description="Run groundedness prompt-format experiment on Runpod.")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="HF model id for the local judge.")
    parser.add_argument("--prompt-formats", type=parse_prompt_formats, default=["line", "json", "bullet"])
    parser.add_argument("--dataset", default="PatronusAI/TRAIL")
    parser.add_argument("--dataset-split", default="gaia")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--results-path", type=Path, default=None)
    parser.add_argument("--debug-log-path", type=Path, default=None)
    parser.add_argument("--trace-limit", type=int, default=None)
    parser.add_argument("--time-limit-seconds", type=int, default=None)
    parser.add_argument("--max-seq-length", type=int, default=MAX_SEQ_LENGTH)
    parser.add_argument("--max-new-tokens", type=int, default=MAX_NEW_TOKENS)
    parser.add_argument("--max-retries", type=int, default=MAX_RETRIES)
    parser.add_argument("--batch-size", type=int, default=4, help="Number of prompts to generate in one batch.")
    parser.add_argument("--on-missing-context", choices=["skip", "zero"], default="skip")
    parser.add_argument("--no-4bit", action="store_true", help="Disable 4-bit loading.")
    parser.add_argument("--resume", action="store_true", help="Append to existing output files instead of recreating them.")
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error("--batch-size must be at least 1")
    args.hf_token = os.environ.get("HF_TOKEN")
    args.load_in_4bit = not args.no_4bit
    args.output_dir.mkdir(parents=True, exist_ok=True)
    return args


def main():
    args = parse_args()
    traces = load_traces_for_experiment(args)
    summary = run_experiment(traces, args)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
