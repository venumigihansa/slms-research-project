# Experiment 2 Results

This folder is reserved for the groundedness prompt-format sensitivity experiment.

## Purpose

Experiment 2 tests whether small LLM judges are sensitive to prompt format when evaluating a single criterion: `groundedness`.

The experiment compares three semantically equivalent prompt formats:

- `line`: SDK default groundedness prompt, used as the line-by-line baseline
- `json`: groundedness rubric represented as a JSON-like prompt body
- `bullet`: groundedness rubric represented as bullet-point sections

The output schema is held constant across all prompt formats.

## Main Research Question

If the trace, evidence, scoring scale, evaluator criterion, model, and decoding settings are fixed, does changing only the prompt format change the judge's scores, output validity, latency, or trace ranking?

## Expected Files

Raw run outputs should be copied here after each model run.

Recommended filename format:

```text
<model>.csv
```

Use filesystem-safe model names by replacing `/` with `_`.

Examples:

- `Qwen_Qwen3.5-2B.csv`
- `meta-llama_Meta-Llama-3.1-8B-Instruct.csv`
- `google_gemma-3-12b-it.csv`
- `gpt-5.4.csv`

## Expected Columns

Prompt experiment result CSVs should contain:

- `model`
- `trace_id`
- `evaluator`
- `prompt_format`
- `level`
- `score`
- `explanation`
- `latency_ms`
- `is_skipped`
- `skip_reason`

## Analysis Outputs

The analysis script may also produce:

- `prompt_format_summary.csv`
- `prompt_sensitivity_by_trace.csv`
- `prompt_format_pairwise_correlations.csv`

If running multiple models, prefer saving per-model analysis outputs in separate subfolders or with model-specific filenames to avoid overwriting summaries.
