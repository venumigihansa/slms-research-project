# Groundedness Prompt-Format Experiment

This experiment tests whether small LLM judges are sensitive to prompt format when evaluating the same criterion: groundedness.

## Experiment Outline

### Research Question

Are small LLM judges sensitive to prompt format when judging groundedness of agent traces?

More specifically, if the criterion, trace, evidence, scoring scale, and output schema stay fixed, does changing the rubric format change the judge's score, validity, latency, or trace ranking?

### Hypothesis

Small LLM judges will show measurable prompt-format sensitivity. The same model may assign different groundedness scores to the same trace depending on whether the rubric is presented as line-by-line text, JSON-like structure, or bullet points.

### Evaluator

This experiment uses only one evaluator:

- `groundedness`

Groundedness is used because it depends directly on the evidence available in the trace. The judge must compare factual claims in the agent response against tool or retrieval evidence, which makes the task more concrete than broad criteria such as helpfulness or reasoning quality.

### Prompt Conditions

The three prompt conditions are:

- `line`: SDK default groundedness prompt, renamed as the line-by-line baseline.
- `json`: same groundedness task represented as a JSON-like rubric.
- `bullet`: same groundedness task represented as bullet-point sections.

The output schema is held constant because the runner appends the same short JSON output instruction to every prompt.

### Controlled Variables

The experiment keeps these fixed:

- same trace set
- same groundedness criterion
- same evidence formatting from the AMP trace object
- same 0.0 to 1.0 scoring scale
- same JSON output instruction
- same model checkpoint per run
- same decoding setup, using deterministic generation
- same batch size within a run, if batched inference is enabled

Only the rubric format changes.

### Measurements

The runner records one row per model, trace, and prompt format. The analysis script computes:

- score distribution by prompt format
- valid score rate by prompt format
- skipped or invalid output rate
- mean latency by prompt format
- prompt sensitivity per trace
- pairwise Pearson and Spearman correlations between prompt formats
- mean absolute score difference between prompt formats

Prompt sensitivity is defined as:

```text
max(score across prompt formats) - min(score across prompt formats)
```

Higher prompt sensitivity means the judge's conclusion changes more when only the prompt format changes.

### Expected Interpretation

If the three prompt formats produce highly similar scores and rankings, the judge is robust to prompt format for groundedness.

If the formats produce different scores, low correlations, or large per-trace sensitivity, then prompt format is an uncontrolled source of variance in LLM-as-judge evaluation.

The stronger research claim is not that one prompt is always best. The main claim is whether prompt format changes evaluation outcomes enough to affect reliability.

## Run

```bash
export HF_TOKEN="..."

bash scripts/prompt_experiment/run_groundedness_prompt_experiment.sh \
  --model meta-llama/Llama-3.2-1B-Instruct \
  --trace-limit 150 \
  --batch-size 4
```

Use a different model by changing `--model`:

```bash
bash scripts/prompt_experiment/run_groundedness_prompt_experiment.sh \
  --model google/gemma-3-12b-it \
  --trace-limit 150 \
  --batch-size 2
```

Run only selected prompt formats:

```bash
bash scripts/prompt_experiment/run_groundedness_prompt_experiment.sh \
  --model Qwen/Qwen2.5-3B-Instruct \
  --prompt-formats json,bullet \
  --trace-limit 50 \
  --batch-size 4
```

Batch size controls how many prompts are sent through `model.generate()` at once. Use `--batch-size 1` for the original one-prompt-at-a-time behavior. Increase it only when GPU memory allows it; long traces can make larger batches run out of memory.

By default, the runner uses:

```text
/workspace/groundedness_prompt_experiment_full
```

If that folder already contains `preprocessed_traces/`, the runner reuses those preprocessed traces instead of downloading and preprocessing the dataset again.

## Outputs

By default, results are written to:

```text
/workspace/groundedness_prompt_experiment_full/<model>_groundedness_prompt_results.csv
```

The CSV includes:

- `model`
- `trace_id`
- `evaluator`
- `prompt_format`
- `score`
- `explanation`
- `latency_ms`
- `is_skipped`
- `skip_reason`

Failures and invalid JSON outputs are written to a matching `.jsonl` debug file.

## Analyze Results

```bash
python scripts/prompt_experiment/analyze_prompt_results.py \
  /workspace/groundedness_prompt_experiment_full/<model>_groundedness_prompt_results.csv
```

This writes:

- `prompt_format_summary.csv`
- `prompt_sensitivity_by_trace.csv`
- `prompt_format_pairwise_correlations.csv`
