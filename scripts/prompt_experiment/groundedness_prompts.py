"""Groundedness prompt variants for the prompt-format experiment."""

from __future__ import annotations

import json

from amp_evaluation.evaluators.builtin.llm_judge import GroundednessEvaluator
from amp_evaluation.trace import Trace


class GroundednessLinePromptEvaluator(GroundednessEvaluator):
    """SDK default groundedness prompt, renamed for experiment bookkeeping."""

    name = "groundedness_line"
    prompt_format = "line"

    def build_prompt(self, trace: Trace) -> str:
        return super().build_prompt(trace)


class GroundednessJsonPromptEvaluator(GroundednessEvaluator):
    """Groundedness rubric represented as a JSON-like prompt body."""

    name = "groundedness_json"
    prompt_format = "json"

    def build_prompt(self, trace: Trace) -> str:
        payload = {
            "role": "expert evaluator",
            "criterion": "GROUNDEDNESS",
            "evaluation_question": (
                "Are the factual claims in the agent response grounded in the "
                "evidence that was available to the agent?"
            ),
            "inputs": {
                "user_query": trace.input,
                "agent_response": trace.output,
                "evidence_available_to_agent": trace.format_evidence(),
            },
            "evaluation_steps": [
                "Identify each factual claim in the response, including specific facts, numbers, references, or assertions presented as true.",
                "For each claim, check whether the evidence directly supports it.",
                "Classify each claim as SUPPORTED, UNSUPPORTED, or CONTRADICTED.",
                "Score based on the proportion of supported claims.",
                "Penalize contradictions more heavily than unsupported claims.",
            ],
            "claim_rules": {
                "assess": "specific factual claims",
                "do_not_penalize": [
                    "opinions",
                    "hedged statements",
                    "general knowledge that does not need source evidence",
                ],
            },
            "scoring_rubric": {
                "0.0": "Most claims are fabricated or contradict the available evidence",
                "0.25": "Many claims lack support; one or more are contradicted by evidence",
                "0.5": "Mixed: some claims are supported, others are not; no major contradictions",
                "0.75": "Most claims are supported by evidence; only minor unsupported details",
                "1.0": "Every factual claim is grounded in the provided evidence",
            },
        }
        return "Evaluate the following JSON-like groundedness task:\n" + json.dumps(
            payload,
            ensure_ascii=False,
            indent=2,
        )


class GroundednessBulletPromptEvaluator(GroundednessEvaluator):
    """Groundedness rubric represented as bullet-point sections."""

    name = "groundedness_bullet"
    prompt_format = "bullet"

    def build_prompt(self, trace: Trace) -> str:
        return f"""You are an expert evaluator.

- Criterion: GROUNDEDNESS
- Evaluation question: Are the factual claims in the agent response grounded in the evidence that was available to the agent?

User Query:
- {trace.input}

Agent Response:
- {trace.output}

Evidence Available to the Agent:
- {trace.format_evidence()}

Evaluation Steps:
- Identify each factual claim in the response, including specific facts, numbers, references, or assertions presented as true.
- For each claim, check whether the evidence directly supports it.
- Classify each claim as SUPPORTED, UNSUPPORTED, or CONTRADICTED.
- Score based on the proportion of supported claims.
- Penalize contradictions more heavily than unsupported claims.

Claim Rules:
- Assess only specific factual claims.
- Do not penalize opinions.
- Do not penalize hedged statements.
- Do not penalize general knowledge that does not need source evidence.

Scoring Rubric:
- 0.0: Most claims are fabricated or contradict the available evidence.
- 0.25: Many claims lack support; one or more are contradicted by evidence.
- 0.5: Mixed: some claims are supported, others are not; no major contradictions.
- 0.75: Most claims are supported by evidence; only minor unsupported details.
- 1.0: Every factual claim is grounded in the provided evidence."""


PROMPT_EVALUATORS = {
    "line": GroundednessLinePromptEvaluator,
    "json": GroundednessJsonPromptEvaluator,
    "bullet": GroundednessBulletPromptEvaluator,
}
