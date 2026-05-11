# LLM Judge Evaluation Experiments

This repository contains the code and outputs for evaluating LLM judges on AMP agent traces.

It supports two workflows:

- **Experiment 1:** compare judge models against GPT-5.4 across multiple evaluator dimensions.
- **Experiment 2:** test whether groundedness scores change when the same rubric is written in different prompt formats.

## Quick Start

Install the Runpod/runtime dependencies:

```bash
python -m pip install -r scripts/requirements-runpod.txt
```

Set credentials in `.env` or export them in the shell:

```bash
HF_TOKEN=...
OPENAI_API_KEY=...
```

`HF_TOKEN` is needed for Hugging Face models and datasets. `OPENAI_API_KEY` is only needed for explanation-similarity analysis in `agreement_analysis.ipynb`.

Important: the results directory is named `results ` with a trailing space. Always quote result paths:

```bash
ls "results /experiment 1 corrected"
```

## Repository Layout

```text
.
├── inference notebooks/
│   └── experiment 1/
├── results /
│   ├── analysis/
│   ├── cache/
│   ├── experiment 1/
│   ├── experiment 1 corrected/
│   ├── experiment 2/
│   └── figures/
├── scripts/
│   ├── agreement_experiment/
│   ├── prompt_experiment/
│   ├── reruns/
│   ├── requirements-runpod.txt
│   └── run_gemma_runpod.py
├── agreement_analysis.ipynb
├── experiment2_prompt_format_repeated_measures_analysis.ipynb
├── .env.example
└── README.md
```

## Experiment 1: Model Agreement

Experiment 1 compares judge model scores against GPT-5.4 for these evaluator dimensions:

- `helpfulness`
- `accuracy`
- `groundedness`
- `instruction_following`
- `reasoning_quality`

Use this experiment to check:

- agreement with GPT-5.4
- evaluator-specific stability
- score bias and variance
- invalid JSON / skipped outputs
- explanation similarity against GPT-5.4

### Run A Judge Model

Use the model-generic runner:

```bash
python scripts/agreement_experiment/run_model_agreement_experiment.py \
  --model "google/gemma-3-4b-it" \
  --dataset "PatronusAI/TRAIL" \
  --dataset-split gaia \
  --batch-size 1 \
  --output-dir /workspace/model_agreement_experiment_full \
  --results-path "results /experiment 1/google_gemma-3-4b-it.csv" \
  --debug-log-path "results /experiment 1/google_gemma-3-4b-it_failures.jsonl"
```

On Runpod, you can use the wrapper:

```bash
bash scripts/agreement_experiment/run_model_agreement_experiment.sh \
  --model "google/gemma-3-4b-it" \
  --batch-size 1
```

More details are in:

```text
scripts/agreement_experiment/README.md
```

### Result Folders

Raw Experiment 1 model outputs:

```text
results /experiment 1
```

Final repaired outputs:

```text
results /experiment 1 corrected
```

Use `results /experiment 1 corrected` for final analysis.

## Experiment 1: Rerun Skipped Rows

Use this when a model output has skipped rows from OOM, invalid JSON, or schema failures.

Create the rerun folder:

```bash
mkdir -p "results /experiment 1 reruns"
```

Build the skip manifest:

```bash
python scripts/reruns/build_experiment1_skip_manifest.py \
  --results-dir "results /experiment 1" \
  --output-path "results /experiment 1 reruns/experiment1_skip_manifest.csv"
```

Rerun skipped rows for one model:

```bash
python scripts/reruns/rerun_experiment1_skipped_jobs.py \
  --manifest-path "results /experiment 1 reruns/experiment1_skip_manifest.csv" \
  --model "google/gemma-3-4b-it" \
  --skip-categories memory_oom,invalid_json_schema \
  --batch-size 1 \
  --trace-output-dir /workspace/model_agreement_experiment_full \
  --output-path "results /experiment 1 reruns/google_gemma-3-4b-it_rerun.csv"
```

Merge successful reruns into corrected CSVs:

```bash
python scripts/reruns/merge_experiment1_reruns.py \
  --original-dir "results /experiment 1" \
  --rerun-dir "results /experiment 1 reruns" \
  --output-dir "results /experiment 1 corrected"
```

## Experiment 1 Analysis

Open:

```text
agreement_analysis.ipynb
```

This notebook reads from:

```text
results /experiment 1 corrected
```

It generates:

- Spearman agreement heatmaps against GPT-5.4
- score-difference density plots
- discrete score-difference distributions
- explanation-similarity density plots
- score-disagreement vs explanation-similarity plots

Figures are saved to:

```text
results /figures
```

Explanation similarity results are saved to:

```text
results /analysis/explanation_similarity_text-embedding-3-small.csv
```

Embedding cache:

```text
results /cache/explanation_embeddings_text-embedding-3-small.jsonl
```

## Experiment 2: Prompt-Format Sensitivity

Experiment 2 tests whether groundedness judgments change when the same rubric is written in different formats.

Prompt formats:

- `line`
- `json`
- `bullet`

Run:

```bash
python scripts/prompt_experiment/run_groundedness_prompt_experiment.py \
  --model "Qwen/Qwen3.5-2B" \
  --batch-size 4
```

On Runpod:

```bash
bash scripts/prompt_experiment/run_groundedness_prompt_experiment.sh \
  --model "Qwen/Qwen3.5-2B" \
  --batch-size 4
```

Analyze one result CSV:

```bash
python scripts/prompt_experiment/analyze_prompt_results.py \
  "results /experiment 2/Qwen_Qwen3.5-2B.csv"
```

Experiment 2 result CSVs belong in:

```text
results /experiment 2
```

The main notebook is:

```text
experiment2_prompt_format_repeated_measures_analysis.ipynb
```

More details are in:

```text
scripts/prompt_experiment/README.md
```

## Script Reference

```text
scripts/agreement_experiment/run_model_agreement_experiment.py
```

Runs Experiment 1 for one judge model.

```text
scripts/reruns/build_experiment1_skip_manifest.py
```

Creates a CSV manifest of skipped Experiment 1 rows.

```text
scripts/reruns/rerun_experiment1_skipped_jobs.py
```

Reruns selected skipped Experiment 1 jobs.

```text
scripts/reruns/merge_experiment1_reruns.py
```

Merges valid rerun rows into corrected Experiment 1 outputs.

```text
scripts/reruns/replace_rerun_rows.py
```

Replaces specific rows inside a rerun CSV when a one-off repair was run separately.

```text
scripts/prompt_experiment/run_groundedness_prompt_experiment.py
```

Runs Experiment 2.

```text
scripts/prompt_experiment/analyze_prompt_results.py
```

Summarizes Experiment 2 output.

```text
scripts/run_gemma_runpod.py
```

Shared helper module used by the prompt experiment. Do not remove it unless those imports are refactored.

## Output CSV Columns

Experiment 1 CSVs contain:

```text
model, trace_id, evaluator, level, score, explanation, latency_ms, is_skipped, skip_reason
```

Rerun CSVs may include additional columns such as:

```text
job_id, occurrence_index, target_label, failure_type, source_original_csv, original_skip_reason
```

Experiment 2 CSVs contain:

```text
model, trace_id, evaluator, prompt_format, level, score, explanation, latency_ms, is_skipped, skip_reason
```
