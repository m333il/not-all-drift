from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Any, cast

from .evaluation import evaluate_examples
from .metrics import evaluate_multilabel
from .prompts import describe_output_contract, format_labels
from .providers import TextProvider
from .schemas import Example

COMPONENT = "instructions"


@dataclass(frozen=True)
class GepaRunResult:
    selected_instruction: str
    candidates: tuple[dict[str, Any], ...]
    raw_result: Any

    def result_snapshot(self) -> dict[str, Any]:
        return cast(dict[str, Any], self.raw_result.to_dict())

    def candidate_tree_html(self) -> str:
        return str(self.raw_result.candidate_tree_html())


class MultilabelGepaAdapter:
    """Scores candidates on optimizer splits only; GEPA is imported lazily in ``optimize``."""

    def __init__(self, task: TextProvider, labels: Sequence[str]):
        self.task = task
        self.labels = tuple(labels)

    def score(
        self, examples: Sequence[Example], instruction: str
    ) -> tuple[list[float], list[dict[str, Any]]]:
        predictions = evaluate_examples(
            self.task, examples, condition="candidate", instruction=instruction, labels=self.labels
        )
        scores: list[float] = []
        traces: list[dict[str, Any]] = []
        for example, prediction in zip(examples, predictions, strict=True):
            metric = evaluate_multilabel([example.labels], [prediction.labels], self.labels)
            metric_value = metric["f1_samples"]
            assert isinstance(metric_value, float)
            score = metric_value if prediction.parse_ok else 0.0
            scores.append(score)
            traces.append(
                {
                    "text": example.text,
                    "gold": list(example.labels),
                    "predicted": list(prediction.labels),
                    "raw": prediction.raw_response,
                    "error": prediction.parse_error,
                    "score": score,
                }
            )
        return scores, traces

    def feedback(self, trace: dict[str, Any]) -> str:
        # Same wire format as the parser expects, so the reflector does not learn another one.
        gold, predicted = set(trace["gold"]), set(trace["predicted"])
        show = partial(format_labels, self.labels)
        return (
            f"Text: {trace['text']}\nExpected: {show(gold)}\nPredicted: {show(predicted)}\n"
            f"Missed: {show(gold - predicted)}\nExtra: {show(predicted - gold)}\n"
            f"Parse error: {trace['error']}"
        )


def optimize(
    *,
    train: Sequence[Example],
    validation: Sequence[Example],
    seed_instruction: str,
    task_provider: TextProvider,
    reflection_provider: TextProvider,
    labels: Sequence[str],
    seed: int,
    max_metric_calls: int,
    reflection_minibatch_size: int,
    reflection_max_tokens: int,
    reflection_instruction_budget_tokens: int,
    run_dir: Path,
) -> GepaRunResult:
    try:
        from gepa.api import optimize as gepa_optimize
        from gepa.core.adapter import EvaluationBatch
        from gepa.core.result import GEPAResult
        from gepa.strategies.instruction_proposal import InstructionProposalSignature
    except ImportError as exc:
        raise RuntimeError("install the 'gepa' extra to optimize prompts") from exc

    if reflection_instruction_budget_tokens >= reflection_max_tokens:
        raise ValueError("instruction budget must leave headroom under the reflection ceiling")
    # GEPA has no length limit: state the budget and the output contract in the
    # meta-prompt. 4.4 characters per token, measured on this corpus.
    output_contract = describe_output_contract(labels)
    reflection_prompt_template = InstructionProposalSignature.default_prompt_template.replace(
        "Provide the new instructions within ``` blocks.",
        f"{output_contract}\n\n"
        f"Keep the new instruction under {reflection_instruction_budget_tokens} tokens "
        f"(roughly {reflection_instruction_budget_tokens * 44 // 10} characters). Spend that "
        "budget on decision rules and examples that change the label.\n\n"
        "Provide the new instructions within ``` blocks.",
    )

    core = MultilabelGepaAdapter(task_provider, labels)

    class Adapter:
        propose_new_texts: Any = None

        def evaluate(
            self, batch: list[Example], candidate: dict[str, str], capture_traces: bool = False
        ) -> EvaluationBatch[dict[str, Any], dict[str, float]]:
            scores, traces = core.score(batch, candidate[COMPONENT])
            return EvaluationBatch(
                outputs=[{"score": x} for x in scores],
                scores=scores,
                trajectories=traces if capture_traces else None,
            )

        def make_reflective_dataset(
            self,
            candidate: dict[str, str],
            eval_batch: EvaluationBatch[dict[str, Any], dict[str, float]],
            components_to_update: list[str],
        ) -> dict[str, list[dict[str, Any]]]:
            del candidate
            if COMPONENT not in components_to_update:
                return {}
            return {
                COMPONENT: [
                    {
                        "Inputs": {"text": t["text"]},
                        "Generated Outputs": t["predicted"],
                        "Feedback": core.feedback(t),
                    }
                    for t in eval_batch.trajectories or ()
                ]
            }

    class ReflectionLM:
        def __call__(self, prompt: str | list[dict[str, Any]]) -> str:
            messages = (
                [{"role": "user", "content": prompt}]
                if isinstance(prompt, str)
                else [
                    {"role": str(message["role"]), "content": str(message["content"])}
                    for message in prompt
                ]
            )
            return reflection_provider.complete(messages, max_tokens=reflection_max_tokens).text

    result: GEPAResult[dict[str, float], int] = gepa_optimize(
        seed_candidate={COMPONENT: seed_instruction},
        trainset=list(train),
        valset=list(validation),
        adapter=cast(Any, Adapter()),
        reflection_lm=ReflectionLM(),
        reflection_prompt_template=reflection_prompt_template,
        max_metric_calls=max_metric_calls,
        reflection_minibatch_size=reflection_minibatch_size,
        run_dir=str(run_dir),
        seed=seed,
    )
    best_candidate = result.best_candidate
    if isinstance(best_candidate, str):
        raise RuntimeError("GEPA returned a scalar candidate for a component-mapping seed")
    selected = best_candidate[COMPONENT]
    candidates = tuple(
        {"index": i, "instruction": c[COMPONENT]} for i, c in enumerate(result.candidates)
    )
    return GepaRunResult(selected, candidates, result)
