#!/usr/bin/env python3
"""Run the Experiment 1 model-agreement evaluation with a local HF judge model."""

from __future__ import annotations

import argparse
import csv
import gc
import json
import os
import re
import sys
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import torch
from datasets import load_dataset
from tqdm.auto import tqdm
from unsloth import FastLanguageModel

SCRIPT_DIR = Path(__file__).resolve().parent
SCRIPTS_DIR = SCRIPT_DIR.parent
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from amp_evaluation import builtin
from amp_evaluation.models import EvalResult
from amp_evaluation.trace import AgentTrace
from amp_evaluation.trace.fetcher import _parse_trace
from amp_evaluation.trace.parser import parse_trace_for_evaluation


DEFAULT_OUTPUT_DIR = Path("/workspace/model_agreement_experiment_full")

CSV_COLUMNS = [
    "model",
    "trace_id",
    "evaluator",
    "level",
    "score",
    "explanation",
    "latency_ms",
    "is_skipped",
    "skip_reason",
]

LOCAL_JUDGE_MODEL = "meta-llama/Llama-3.2-1B-Instruct"
LOAD_IN_4BIT = True
MAX_SEQ_LENGTH = 50000
MAX_NEW_TOKENS = 160
MAX_RETRIES = 2
TRIAL_TIME_LIMIT_SECONDS = None

EVALUATOR_NAMES = [
    "helpfulness",
    "accuracy",
    "groundedness",
    "instruction_following",
    "reasoning_quality",
]

SHORT_OUTPUT_INSTRUCTIONS = """
Respond with ONLY a JSON object:
{
  "explanation": "<brief explanation in 1-3 short sentences>",
  "score": <float between 0.0 and 1.0>
}
Do not include markdown fences.
Do not include any extra text before or after the JSON.
Keep the explanation concise.
"""


@dataclass(frozen=True)
class EvalJob:
    trace_id: str
    evaluator_name: str
    level: str
    occurrence_index: int
    prompt: Optional[str]
    evaluator: object
    trace: object
    target: object
    target_label: str
    precomputed_result: Optional[EvalResult] = None


def to_int(value: Any) -> Optional[int]:
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    if isinstance(value, str):
        try:
            return int(float(value.strip()))
        except ValueError:
            return None
    return None


def to_float(value: Any) -> Optional[float]:
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value.strip())
        except ValueError:
            return None
    return None


def maybe_parse_json(value: Any) -> Any:
    if not isinstance(value, str):
        return value
    stripped = value.strip()
    if not stripped or stripped[0] not in '[{"':
        return value
    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        return value


def infer_kind(span_name: str, attrs: Dict[str, Any]) -> str:
    kind = str(attrs.get("openinference.span.kind", "")).strip().lower()
    if kind in {"llm", "tool", "embedding", "retriever", "agent"}:
        return kind
    if kind in {"chain", "task", "workflow", "crewaitask"}:
        return "chain"

    lower_name = span_name.lower()
    if "retriev" in lower_name or "vector" in lower_name:
        return "retriever"
    if "agent" in lower_name:
        return "agent"
    if "tool" in lower_name or "finalanswertool" in lower_name:
        return "tool"
    if "litellmmodel" in lower_name or "chat" in lower_name or "completion" in lower_name:
        return "llm"
    if "step" in lower_name or "chain" in lower_name:
        return "chain"
    return "unknown"


def extract_status_and_error(span: Dict[str, Any], attrs: Dict[str, Any]) -> Tuple[Dict[str, Any], Optional[Dict[str, Any]]]:
    status_code = str(span.get("status_code", "")).strip().lower()
    status_message = str(span.get("status_message", "")).strip()
    error_type = attrs.get("error.type")
    error_message = attrs.get("error.message") or status_message or None

    has_error = bool(isinstance(error_type, str) and error_type.strip()) or status_code in {"error", "failed"}
    status = {"error": has_error}
    if has_error and isinstance(error_type, str) and error_type.strip():
        status["errorType"] = error_type.strip()
    elif has_error and status_message:
        status["errorType"] = status_message
    elif has_error:
        status["errorType"] = "StatusCodeError"

    error_obj = {"message": str(error_message)} if has_error and error_message else None
    return status, error_obj


def extract_token_usage(attrs: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    input_tokens = to_int(attrs.get("llm.token_count.prompt"))
    output_tokens = to_int(attrs.get("llm.token_count.completion"))
    total_tokens = to_int(attrs.get("llm.token_count.total"))
    if input_tokens is None and output_tokens is None and total_tokens is None:
        return None
    input_tokens = input_tokens or 0
    output_tokens = output_tokens or 0
    total_tokens = total_tokens if total_tokens is not None else input_tokens + output_tokens
    return {"inputTokens": input_tokens, "outputTokens": output_tokens, "totalTokens": total_tokens}


def parse_temperature(attrs: Dict[str, Any]) -> Optional[float]:
    raw = maybe_parse_json(attrs.get("llm.invocation_parameters"))
    if isinstance(raw, dict):
        for key in ("temperature", "temp"):
            value = to_float(raw.get(key))
            if value is not None:
                return value
    return None


def extract_llm_tools(attrs: Dict[str, Any]) -> List[Dict[str, Any]]:
    pattern = re.compile(r"^llm\.tools\.(\d+)\.tool\.json_schema$")
    bucket: Dict[int, Dict[str, Any]] = {}
    for key, value in attrs.items():
        match = pattern.match(key)
        if not match:
            continue
        parsed = maybe_parse_json(value)
        if not isinstance(parsed, dict):
            continue
        source = parsed.get("function") if isinstance(parsed.get("function"), dict) else parsed
        parameters_obj = source.get("parameters")
        parameters = ""
        if parameters_obj is not None:
            try:
                parameters = json.dumps(parameters_obj, ensure_ascii=False)
            except TypeError:
                parameters = str(parameters_obj)
        bucket[int(match.group(1))] = {
            "name": str(source.get("name", "")).strip(),
            "description": str(source.get("description", "")).strip(),
            "parameters": parameters,
        }
    return [{k: v for k, v in bucket[index].items() if v not in ("", None)} for index in sorted(bucket)]


def extract_message_blocks(attrs: Dict[str, Any], prefix: str) -> List[Dict[str, Any]]:
    pattern = re.compile(rf"^{re.escape(prefix)}\.(\d+)\.message\.(.+)$")
    bucket: Dict[int, Dict[str, Any]] = {}
    for key, value in attrs.items():
        match = pattern.match(key)
        if match:
            bucket.setdefault(int(match.group(1)), {})[match.group(2)] = value

    messages: List[Dict[str, Any]] = []
    for index in sorted(bucket):
        item = bucket[index]
        msg: Dict[str, Any] = {}
        if "role" in item:
            msg["role"] = str(item["role"])
        if "content" in item:
            msg["content"] = str(item["content"])
        raw_tool_calls = item.get("tool_calls") or item.get("toolCalls")
        if raw_tool_calls is not None:
            parsed_tool_calls = maybe_parse_json(raw_tool_calls)
            if parsed_tool_calls:
                msg["tool_calls"] = parsed_tool_calls
        if msg:
            messages.append(msg)
    return messages


def llm_amp(attrs: Dict[str, Any]) -> Tuple[Any, Any, Dict[str, Any]]:
    input_messages = extract_message_blocks(attrs, "llm.input_messages")
    output_messages = extract_message_blocks(attrs, "llm.output_messages")
    amp_input = input_messages if input_messages else maybe_parse_json(attrs.get("input.value"))
    amp_output = output_messages if output_messages else maybe_parse_json(attrs.get("output.value"))

    data: Dict[str, Any] = {}
    for attr_key, out_key in (("llm.model_name", "model"), ("llm.vendor", "vendor")):
        value = attrs.get(attr_key)
        if isinstance(value, str) and value.strip():
            data[out_key] = value.strip()
    temperature = parse_temperature(attrs)
    if temperature is not None:
        data["temperature"] = temperature
    tools = extract_llm_tools(attrs)
    if tools:
        data["tools"] = tools
    token_usage = extract_token_usage(attrs)
    if token_usage:
        data["tokenUsage"] = token_usage
    return amp_input, amp_output, data


def tool_amp(attrs: Dict[str, Any]) -> Tuple[Any, Any, Dict[str, Any]]:
    data: Dict[str, Any] = {}
    for attr_key, out_key in (("tool.name", "name"), ("tool.description", "description")):
        value = attrs.get(attr_key)
        if isinstance(value, str) and value.strip():
            data[out_key] = value.strip()
    return maybe_parse_json(attrs.get("input.value")), maybe_parse_json(attrs.get("output.value")), data


def agent_amp(span: Dict[str, Any], attrs: Dict[str, Any]) -> Tuple[Any, Any, Dict[str, Any]]:
    data: Dict[str, Any] = {"name": span.get("span_name", "") or "", "framework": "openinference"}
    model = attrs.get("llm.model_name")
    if isinstance(model, str) and model.strip():
        data["model"] = model.strip()
    token_usage = extract_token_usage(attrs)
    if token_usage:
        data["tokenUsage"] = token_usage
    tools_names = maybe_parse_json(attrs.get("smolagents.tools_names"))
    if isinstance(tools_names, list):
        tools = [{"name": str(name).strip()} for name in tools_names if str(name).strip()]
        if tools:
            data["tools"] = tools
    max_steps = to_int(attrs.get("smolagents.max_steps"))
    if max_steps is not None:
        data["maxIter"] = max_steps
    return maybe_parse_json(attrs.get("input.value")), maybe_parse_json(attrs.get("output.value")), data


def retriever_amp(attrs: Dict[str, Any]) -> Tuple[Any, Any, Dict[str, Any]]:
    data: Dict[str, Any] = {}
    vector_db = attrs.get("retrieval.vector_db") or attrs.get("vector_db")
    if isinstance(vector_db, str) and vector_db.strip():
        data["vectorDB"] = vector_db.strip()
    top_k = to_int(attrs.get("retrieval.top_k") or attrs.get("top_k"))
    if top_k is not None:
        data["topK"] = top_k
    return maybe_parse_json(attrs.get("input.value")), maybe_parse_json(attrs.get("output.value")), data


def compute_amp_attributes(span: Dict[str, Any]) -> Dict[str, Any]:
    attrs = span.get("span_attributes")
    attrs = attrs if isinstance(attrs, dict) else {}
    kind = infer_kind(str(span.get("span_name", "")), attrs)
    status, error_obj = extract_status_and_error(span, attrs)

    if kind == "llm":
        amp_input, amp_output, data = llm_amp(attrs)
    elif kind == "tool":
        amp_input, amp_output, data = tool_amp(attrs)
    elif kind == "agent":
        amp_input, amp_output, data = agent_amp(span, attrs)
    elif kind == "retriever":
        amp_input, amp_output, data = retriever_amp(attrs)
    else:
        amp_input, amp_output, data = maybe_parse_json(attrs.get("input.value")), maybe_parse_json(attrs.get("output.value")), {}

    result: Dict[str, Any] = {"kind": kind, "status": status}
    if error_obj:
        result["error"] = error_obj
    if amp_input not in (None, "", [], {}):
        result["input"] = amp_input
    if amp_output not in (None, "", [], {}):
        result["output"] = amp_output
    if data:
        result["data"] = data
    return result


def process_span_recursive(span: Dict[str, Any]) -> int:
    span["ampAttributes"] = compute_amp_attributes(span)
    count = 1
    children = span.get("child_spans")
    if isinstance(children, list):
        for child in children:
            if isinstance(child, dict):
                count += process_span_recursive(child)
    return count


def _to_iso_z(ts) -> str:
    if ts is None:
        return pd.Timestamp.utcnow().isoformat().replace("+00:00", "Z")
    return pd.Timestamp(ts).isoformat().replace("+00:00", "Z")


def _duration_to_nanos(raw_duration) -> int:
    if not raw_duration:
        return 0
    try:
        return int(pd.Timedelta(raw_duration).value)
    except Exception:
        return 0


def flatten_trace_spans(root_spans):
    flat = []

    def walk(span):
        if not isinstance(span, dict):
            return
        start_time = _to_iso_z(span.get("timestamp"))
        duration_ns = _duration_to_nanos(span.get("duration"))
        end_time = (pd.Timestamp(start_time) + pd.to_timedelta(duration_ns, unit="ns")).isoformat().replace("+00:00", "Z")
        flat.append({
            "traceId": str(span.get("trace_id", "")),
            "spanId": str(span.get("span_id", "")),
            "parentSpanId": span.get("parent_span_id"),
            "name": span.get("span_name", "") or "",
            "service": span.get("service_name", "") or "",
            "startTime": start_time,
            "endTime": end_time,
            "durationInNanos": duration_ns,
            "kind": str(span.get("span_kind", "INTERNAL")).upper(),
            "status": str(span.get("status_code", "UNSET")).upper(),
            "attributes": span.get("span_attributes", {}) or {},
            "ampAttributes": span.get("ampAttributes", {}) or {},
        })
        for child in span.get("child_spans") or []:
            walk(child)

    for span in root_spans or []:
        walk(span)
    flat.sort(key=lambda s: (s.get("startTime") or "", s.get("spanId") or ""))
    return flat


def infer_trace_io(trace_obj: dict, flat_spans: list[dict]) -> tuple[str, str, str]:
    trace_id = str(trace_obj.get("trace_id") or (flat_spans[0]["traceId"] if flat_spans else ""))
    trace_input = trace_obj.get("input", trace_obj.get("question", ""))
    trace_output = trace_obj.get("output", trace_obj.get("final_answer", ""))

    if not trace_input:
        for span in flat_spans:
            raw_input = (span.get("ampAttributes") or {}).get("input")
            if isinstance(raw_input, str) and raw_input.strip():
                trace_input = raw_input
                break
            if isinstance(raw_input, list):
                for msg in raw_input:
                    if isinstance(msg, dict) and msg.get("role") == "user" and msg.get("content"):
                        trace_input = msg["content"]
                        break
                if trace_input:
                    break

    if not trace_output:
        for span in reversed(flat_spans):
            raw_output = (span.get("ampAttributes") or {}).get("output")
            if isinstance(raw_output, str) and raw_output.strip():
                trace_output = raw_output
                break
            if isinstance(raw_output, dict) and raw_output.get("content"):
                trace_output = raw_output["content"]
                break
            if isinstance(raw_output, list) and raw_output:
                last = raw_output[-1]
                if isinstance(last, dict) and last.get("content"):
                    trace_output = last["content"]
                    break
    return trace_id, str(trace_input or ""), str(trace_output or "")


def download_and_prepare_traces(args) -> list:
    if not args.hf_token:
        raise RuntimeError("HF_TOKEN is required. Export it in Runpod before running this script.")

    print(f"Loading {args.dataset} split '{args.dataset_split}'")
    trail_dataset = load_dataset(args.dataset, token=args.hf_token)
    df = trail_dataset[args.dataset_split].to_pandas().reset_index(drop=True)
    if args.trace_limit is not None:
        df = df.head(args.trace_limit)

    preprocessed_dir = args.output_dir / "preprocessed_traces"
    preprocessed_dir.mkdir(parents=True, exist_ok=True)

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Preprocessing traces", unit="trace"):
        trace_obj = maybe_parse_json(row["trace"])
        if not isinstance(trace_obj, dict):
            continue
        for span in trace_obj.get("spans") or []:
            if isinstance(span, dict):
                process_span_recursive(span)
        trace_id = str(trace_obj.get("trace_id") or row.name)
        record = {
            "trace_id": trace_id,
            "labels": maybe_parse_json(row["labels"]) if "labels" in row and row["labels"] is not None else None,
            "trace": trace_obj,
        }
        with (preprocessed_dir / f"{trace_id}.json").open("w", encoding="utf-8") as handle:
            json.dump(record, handle, ensure_ascii=False, indent=2)
            handle.write("\n")

    return load_preprocessed_traces(args)


def load_local_judge(args):
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.model,
        max_seq_length=args.max_seq_length,
        load_in_4bit=args.load_in_4bit,
    )
    FastLanguageModel.for_inference(model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    return model, tokenizer


def cleanup_model(model=None, tokenizer=None):
    try:
        del model
    except Exception:
        pass
    try:
        del tokenizer
    except Exception:
        pass
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def render_prompt_text(tokenizer, prompt: str) -> str:
    messages = [{"role": "user", "content": prompt}]
    if hasattr(tokenizer, "apply_chat_template"):
        try:
            return tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
        except TypeError:
            return tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
    return prompt


def build_evaluators(args):
    evaluators = []
    for name in EVALUATOR_NAMES:
        ev = builtin(name)
        ev.model = args.model
        ev.max_retries = args.max_retries
        ev._OUTPUT_INSTRUCTIONS = SHORT_OUTPUT_INSTRUCTIONS
        evaluators.append(ev)
    return evaluators


def build_full_prompt(evaluator, target, task=None) -> str:
    return evaluator._dispatch_build_prompt(target, task) + evaluator._OUTPUT_INSTRUCTIONS


def precompute_result_if_needed(evaluator, trace) -> Optional[EvalResult]:
    if evaluator.name != "groundedness" or trace.get_tool_calls() or trace.get_retrievals():
        return None
    if getattr(evaluator, "on_missing_context", "skip") == "zero":
        return EvalResult(score=0.0, passed=False, explanation="No tool or retrieval spans found; cannot assess groundedness")
    return EvalResult.skip("No tool or retrieval spans found in this trace")


def build_jobs(traces, evaluators) -> List[EvalJob]:
    jobs = []
    occurrences: Dict[Tuple[str, str, str], int] = defaultdict(int)
    for trace in traces:
        for ev in evaluators:
            level = ev.level.value
            if level == "trace":
                precomputed = precompute_result_if_needed(ev, trace)
                prompt = None if precomputed is not None else build_full_prompt(ev, trace)
                base_key = (trace.trace_id, ev.name, level)
                occurrence_index = occurrences[base_key]
                occurrences[base_key] += 1
                jobs.append(EvalJob(trace.trace_id, ev.name, level, occurrence_index, prompt, ev, trace, trace, "trace", precomputed))
                continue
            if level != "agent":
                raise ValueError(f"Unsupported evaluator level: {level}")
            targets = []
            agent_spans = trace.get_agents()
            if not agent_spans:
                root_span = trace._get_root_span()
                fallback_agent_id = root_span.span_id if root_span else trace.trace_id
                fallback = AgentTrace(
                    agent_id=fallback_agent_id,
                    input=trace.input,
                    output=trace.output,
                    steps=trace._get_agent_steps(deduplicate_messages=True),
                    metrics=trace.metrics,
                )
                targets.append((f"fallback-agent:{fallback_agent_id}", fallback))
            else:
                for agent_span in agent_spans:
                    agent_trace = trace._create_agent_trace(agent_span.span_id)
                    targets.append((f"agent:{agent_trace.agent_id}", agent_trace))
            for target_label, agent_trace in targets:
                base_key = (trace.trace_id, ev.name, level)
                occurrence_index = occurrences[base_key]
                occurrences[base_key] += 1
                jobs.append(EvalJob(trace.trace_id, ev.name, level, occurrence_index, build_full_prompt(ev, agent_trace), ev, trace, agent_trace, target_label))
    return jobs


def build_retry_prompt(base_prompt: str, attempt: int, last_error: Optional[str]) -> str:
    if attempt <= 0 or not last_error:
        return base_prompt
    retry_ctx = (
        f"\n\n[IMPORTANT: Your previous response was invalid: {last_error}. "
        "You MUST respond with ONLY a JSON object containing exactly two fields:\n"
        '{"explanation": "<your analysis>", "score": <float between 0.0 and 1.0>}\n'
        "The 'score' MUST be a top-level numeric field in the JSON, NOT embedded in the explanation text.]"
    )
    return base_prompt + retry_ctx


def extract_json_object(raw_output: str) -> str:
    text = raw_output.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
        text = re.sub(r"\s*```$", "", text).strip()

    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        return text[start:end + 1].strip()
    return text


def append_result_row(results_path: Path, row: Dict[str, object]):
    with results_path.open("a", newline="", encoding="utf-8") as handle:
        csv.DictWriter(handle, fieldnames=CSV_COLUMNS).writerow(row)


def append_failure_log(debug_log_path: Path, payload: Dict[str, object]):
    with debug_log_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False) + "\n")


def result_to_row(job: EvalJob, result: EvalResult, latency_ms: float, model_name: str) -> Dict[str, object]:
    return {
        "model": model_name,
        "trace_id": job.trace_id,
        "evaluator": job.evaluator_name,
        "level": job.level,
        "score": "" if result.is_skipped else result.score,
        "explanation": "" if result.is_skipped else (result.explanation or ""),
        "latency_ms": round(latency_ms, 2),
        "is_skipped": result.is_skipped,
        "skip_reason": result.skip_reason or "",
    }


def raw_output_fallback_row(
    job: EvalJob,
    raw_output: str,
    latency_ms: float,
    model_name: str,
    error: str,
    failure_type: Optional[str] = None,
) -> Dict[str, object]:
    if failure_type is None:
        failure_type = "invalid_json" if raw_output.strip() else "generation_error"

    if failure_type == "invalid_json":
        skip_reason = f"Invalid judge JSON; saved raw output instead. Reason: {error}"
    elif failure_type == "generation_error":
        skip_reason = f"LLM generation failed before producing parseable output. Reason: {error}"
    else:
        skip_reason = f"{failure_type}. Reason: {error}"

    return {
        "model": model_name,
        "trace_id": job.trace_id,
        "evaluator": job.evaluator_name,
        "level": job.level,
        "score": "",
        "explanation": raw_output,
        "latency_ms": round(latency_ms, 2),
        "is_skipped": True,
        "skip_reason": skip_reason,
    }


def sanitize_filename(value: str) -> str:
    safe = "".join(ch if ch.isalnum() or ch in ("-", "_", ".") else "_" for ch in value)
    return safe.strip("_") or "model"


def load_preprocessed_traces(args):
    preprocessed_dir = args.output_dir / "preprocessed_traces"
    paths = sorted(preprocessed_dir.glob("*.json"))
    if args.trace_limit is not None:
        paths = paths[: args.trace_limit]

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


def init_outputs(results_path: Path, debug_log_path: Path, resume: bool):
    results_path.parent.mkdir(parents=True, exist_ok=True)
    if not resume:
        for path in (results_path, debug_log_path):
            if path.exists():
                path.unlink()
    if not results_path.exists():
        with results_path.open("w", newline="", encoding="utf-8") as handle:
            csv.DictWriter(handle, fieldnames=CSV_COLUMNS).writeheader()
    debug_log_path.touch(exist_ok=True)


def batched(items: list, batch_size: int):
    for start in range(0, len(items), batch_size):
        yield items[start : start + batch_size]


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


def parse_generation_output(job, raw_output: str):
    normalized_output = extract_json_object(raw_output)
    parsed_result, error = job.evaluator._parse_and_validate(normalized_output)
    return parsed_result, error, normalized_output


def run_experiment(traces, args):
    model_slug = sanitize_filename(args.model)
    results_path = args.results_path or (args.output_dir / f"{model_slug}.csv")
    debug_log_path = args.debug_log_path or (args.output_dir / f"{model_slug}_failures.jsonl")
    evaluators = build_evaluators(args)
    jobs = build_jobs(traces, evaluators)
    init_outputs(results_path, debug_log_path, resume=args.resume)

    print(f"Local judge model: {args.model}")
    print("Experiment: model agreement")
    print(f"Traces available: {len(traces)}")
    print(f"Jobs queued: {len(jobs)}")
    print(f"Batch size: {args.batch_size}")
    print(f"Results path: {results_path}")
    print(f"Debug log path: {debug_log_path}")

    start_time = time.time()
    attempted_jobs = 0
    success_count = 0
    skip_count = 0
    per_evaluator = Counter()
    per_evaluator_skips = Counter()
    model = None
    tokenizer = None
    progress = tqdm(total=len(jobs), desc="Model agreement experiment", unit="job")

    try:
        model, tokenizer = load_local_judge(args)
        pending_jobs = []

        for job in jobs:
            elapsed = time.time() - start_time
            if args.time_limit_seconds is not None and elapsed >= args.time_limit_seconds:
                print(f"Reached time limit after {elapsed:.1f}s; stopping.")
                break

            attempted_jobs += 1
            per_evaluator[job.evaluator_name] += 1

            if job.precomputed_result is not None:
                result = job.precomputed_result
                append_result_row(results_path, result_to_row(job, result, 0.0, args.model))
                skip_count += int(result.is_skipped)
                success_count += int(not result.is_skipped)
                per_evaluator_skips[job.evaluator_name] += int(result.is_skipped)
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
                    "last_failure_type": None,
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
                        item["last_failure_type"] = "generation_error"
                        parsed_result = None
                    else:
                        item["last_raw_output"] = raw_output
                        parsed_result, error, normalized_output = parse_generation_output(job, raw_output)
                        item["last_error"] = error
                        item["last_normalized_output"] = normalized_output
                        item["last_failure_type"] = None if parsed_result is not None else "invalid_json"

                    if parsed_result is not None:
                        append_result_row(
                            results_path,
                            result_to_row(job, parsed_result, item["total_latency_ms"], args.model),
                        )
                        skip_count += int(parsed_result.is_skipped)
                        success_count += int(not parsed_result.is_skipped)
                        per_evaluator_skips[job.evaluator_name] += int(parsed_result.is_skipped)
                        progress.update(1)
                        continue

                    append_failure_log(
                        debug_log_path,
                        {
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
                append_result_row(
                    results_path,
                    raw_output_fallback_row(
                        job,
                        item["last_raw_output"],
                        item["total_latency_ms"],
                        args.model,
                        item["last_error"] or "Unknown error",
                        item["last_failure_type"],
                    ),
                )
                skip_count += 1
                per_evaluator_skips[job.evaluator_name] += 1
                progress.update(1)

            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    finally:
        progress.close()
        if model is not None or tokenizer is not None:
            cleanup_model(model, tokenizer)

    return {
        "attempted_jobs": attempted_jobs,
        "success_count": success_count,
        "skip_count": skip_count,
        "elapsed_seconds": round(time.time() - start_time, 2),
        "per_evaluator": dict(per_evaluator),
        "per_evaluator_skips": dict(per_evaluator_skips),
        "batch_size": args.batch_size,
        "results_path": str(results_path),
        "debug_log_path": str(debug_log_path),
    }


def parse_args():
    parser = argparse.ArgumentParser(description="Run Experiment 1 model-agreement evaluation on Runpod.")
    parser.add_argument("--model", default=LOCAL_JUDGE_MODEL, help="HF model id for the local judge.")
    parser.add_argument("--dataset", default="PatronusAI/TRAIL")
    parser.add_argument("--dataset-split", default="gaia")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--results-path", type=Path, default=None)
    parser.add_argument("--debug-log-path", type=Path, default=None)
    parser.add_argument("--trace-limit", type=int, default=None)
    parser.add_argument("--time-limit-seconds", type=int, default=TRIAL_TIME_LIMIT_SECONDS)
    parser.add_argument("--max-seq-length", type=int, default=MAX_SEQ_LENGTH)
    parser.add_argument("--max-new-tokens", type=int, default=MAX_NEW_TOKENS)
    parser.add_argument("--max-retries", type=int, default=MAX_RETRIES)
    parser.add_argument("--batch-size", type=int, default=1, help="Number of prompts to generate in one batch.")
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
