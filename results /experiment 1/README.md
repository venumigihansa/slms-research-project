# Experiment 1 Results

This folder contains the CSV outputs from the first evaluation experiment: model agreement across multiple LLM-as-judge evaluators.

## Purpose

Experiment 1 compares judge outputs across several models for the AMP trace evaluation setup. The results are intended for agreement analysis across evaluator dimensions such as:

- `helpfulness`
- `accuracy`
- `groundedness`
- `instruction_following`
- `reasoning_quality`

## File Naming

Each CSV is named from the exact `model` value found inside the file. Provider/model separators are replaced with `_` so the names are valid local filenames.

Examples:

- `Qwen_Qwen3.5-2B.csv` represents `Qwen/Qwen3.5-2B`
- `meta-llama_Meta-Llama-3.1-8B-Instruct.csv` represents `meta-llama/Meta-Llama-3.1-8B-Instruct`
- `gpt-5.4.csv` represents `gpt-5.4`

## Expected Columns

The result CSVs should contain:

- `model`
- `trace_id`
- `evaluator`
- `level`
- `score`
- `explanation`
- `latency_ms`
- `is_skipped`
- `skip_reason`

## Notes

These files are the baseline agreement results. Do not mix prompt-format experiment outputs into this folder; use `results /experiment 2` for the prompt experiment.
