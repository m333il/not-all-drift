#!/usr/bin/env python3
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import time
from typing import Any
from urllib.parse import urlsplit
from urllib.request import urlopen

import gepa
from gepa.core.adapter import EvaluationBatch
from gepa.strategies.instruction_proposal import InstructionProposalSignature

from mrd.civil_v2_contract import (
    CONTRACT_SOURCE_COMMIT,
    CONTRACT_SOURCE_REPOSITORY,
    PROMPT_CONTRACT_ID,
    SEED_INSTRUCTION,
    describe_output_contract,
    reflection_feedback,
    render_user_prompt,
    score_response,
)
from mrd.jsonl import read_jsonl
from mrd.llm_client import FileQueueLLMClient, LLMConfig, OpenAICompatibleClient


COMPONENT = "instructions"
REFLECTION_MODEL = "openai/gpt-5.6-luna-pro-20260709"
REFLECTION_BASE_URL = "https://openrouter.ai/api/v1"
REFLECTION_TEMPERATURE = 0.7
REFLECTION_MAX_TOKENS = 32768
REFLECTION_INSTRUCTION_BUDGET_TOKENS = 8192
REFLECTION_MINIBATCH_SIZE = 5
REFLECTION_FAILURE_LIMIT = 20
REFLECTION_MAX_RETRIES = 8
TASK_MAX_TOKENS = int(os.environ.get("MRD_TASK_MAX_TOKENS", "4096"))
TASK_MAX_TOKENS_AMENDMENT = os.environ.get("MRD_TASK_MAX_TOKENS_AMENDMENT")
TASK_CONCURRENCY = int(os.environ.get("MRD_TASK_CONCURRENCY", "1"))
TASK_CONCURRENCY_AMENDMENT = os.environ.get("MRD_TASK_CONCURRENCY_AMENDMENT")


def sha256_path(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def rows_sha256(rows: list[dict[str, Any]]) -> str:
    digest = hashlib.sha256()
    for row in rows:
        digest.update(json.dumps(row, ensure_ascii=False, sort_keys=True).encode())
        digest.update(b"\n")
    return digest.hexdigest()


def normalize_example(row: dict[str, Any]) -> dict[str, Any]:
    key = str(row.get("key") or row.get("id"))
    if key == "None":
        raise ValueError("Civil example has neither key nor id")
    group = row.get("split_group") or row.get("group_id") or key
    return {**row, "key": key, "split_group": str(group)}


def redact_proxy(proxy: str | None) -> str | None:
    if proxy is None:
        return None
    parsed = urlsplit(proxy)
    host = parsed.hostname or ""
    port = f":{parsed.port}" if parsed.port is not None else ""
    return f"{parsed.scheme}://{host}{port}"


def task_server_contract(health: dict[str, Any], model: str) -> dict[str, Any]:
    from mrd.models.registry import MODEL_SPECS

    spec = next(spec for spec in MODEL_SPECS.values() if spec.repo_id == model)
    if health["model"] != model or health["revision"] != spec.revision:
        raise ValueError("Task server differs from the pinned model revision")
    if health["backends"] != {"attention": "eager", "experts": "grouped_mm"}:
        raise ValueError("Task server differs from the validated main-grid backends")
    keys = [
        "model", "revision", "backends", "torch", "transformers", "date_in_template",
        "reasoning_effort", "device", "decoding",
        "batch_size",
    ]
    return {key: health[key] for key in keys}


def reflection_prompt_template(instruction_budget_tokens: int = REFLECTION_INSTRUCTION_BUDGET_TOKENS) -> str:
    return InstructionProposalSignature.default_prompt_template.replace(
        "Provide the new instructions within ``` blocks.",
        f"{describe_output_contract()}\n\n"
        f"Keep the new instruction under {instruction_budget_tokens} tokens "
        f"(roughly {instruction_budget_tokens * 44 // 10} characters). Spend that "
        "budget on decision rules and examples that change the label.\n\n"
        "Provide the new instructions within ``` blocks.",
    )


class CivilV2Adapter:
    propose_new_texts: Any = None

    def __init__(self, client: OpenAICompatibleClient, evaluation_log) -> None:
        self.client = client
        self.evaluation_log = evaluation_log

    def evaluate(
        self,
        batch: list[dict[str, Any]],
        candidate: dict[str, str],
        capture_traces: bool = False,
    ) -> EvaluationBatch[dict[str, Any], dict[str, Any]]:
        instruction = candidate[COMPONENT]
        def rollout(example: dict[str, Any]):
            prompt = render_user_prompt(instruction, example["text"])
            response = self.client.generate(prompt, max_tokens=TASK_MAX_TOKENS)
            score, predicted, error = score_response(example, response.text)
            trajectory = {
                "text": example["text"],
                "gold": list(example["labels"]),
                "predicted": list(predicted),
                "raw": response.text,
                "error": error,
                "score": score,
                "finish_reason": response.finish_reason,
                "usage": response.usage,
                "raw_usage": response.raw_usage,
                "model": response.model,
                "request_attempts": response.request_attempts,
                "retry_events": response.retry_events,
            }
            return response, score, predicted, trajectory

        outputs: list[dict[str, Any]] = []
        scores: list[float] = []
        trajectories: list[dict[str, Any]] = []
        with ThreadPoolExecutor(max_workers=min(TASK_CONCURRENCY, len(batch))) as executor:
            futures = [executor.submit(rollout, example) for example in batch]
            for example, future in zip(batch, futures, strict=True):
                try:
                    response, score, predicted, trajectory = future.result()
                except Exception as error:
                    self.evaluation_log.write(json.dumps({
                        "key": example["key"], "candidate": candidate,
                        "error": f"api: {type(error).__name__}: {error}",
                    }, ensure_ascii=False) + "\n")
                    raise
                self.evaluation_log.write(json.dumps({
                    "key": example["key"], "candidate": candidate, **trajectory,
                }, ensure_ascii=False) + "\n")
                if response.finish_reason != "stop":
                    raise RuntimeError("Task-model response must finish before the token ceiling")
                outputs.append({"predicted": list(predicted), "raw": response.text})
                scores.append(score)
                trajectories.append(trajectory)
        return EvaluationBatch(
            outputs=outputs,
            scores=scores,
            trajectories=trajectories if capture_traces else None,
        )

    def make_reflective_dataset(
        self,
        candidate: dict[str, str],
        eval_batch: EvaluationBatch[dict[str, Any], dict[str, Any]],
        components_to_update: list[str],
    ) -> dict[str, list[dict[str, Any]]]:
        del candidate
        if COMPONENT not in components_to_update:
            return {}
        assert eval_batch.trajectories is not None
        return {
            COMPONENT: [
                {
                    "Inputs": {"text": trajectory["text"]},
                    "Generated Outputs": trajectory["predicted"],
                    "Feedback": reflection_feedback(
                        {"text": trajectory["text"], "labels": trajectory["gold"]},
                        trajectory["predicted"],
                        trajectory["error"],
                    ),
                }
                for trajectory in eval_batch.trajectories
            ]
        }


def summarize_jsonl(path: Path) -> dict[str, Any]:
    rows = read_jsonl(path) if path.exists() else []
    prompt_tokens = completion_tokens = reasoning_tokens = 0
    cost = 0.0
    unknown_usage = 0
    truncated = 0
    attempts = 0
    for row in rows:
        usage = row.get("raw_usage") or row.get("usage") or {}
        if usage.get("prompt_tokens") is None or usage.get("completion_tokens") is None:
            unknown_usage += 1
        prompt_tokens += usage.get("prompt_tokens") or 0
        completion_tokens += usage.get("completion_tokens") or 0
        reasoning_tokens += (usage.get("completion_tokens_details") or {}).get("reasoning_tokens") or 0
        cost += usage.get("cost") or 0.0
        truncated += int(row.get("finish_reason") == "length")
        attempts += row.get("request_attempts") or 1
    return {
        "calls": len(rows),
        "request_attempts": attempts,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "reasoning_tokens": reasoning_tokens,
        "truncated_calls": truncated,
        "unknown_usage_calls": unknown_usage,
        "provider_reported_cost_usd": cost,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--validation", type=Path, required=True)
    parser.add_argument("--data-manifest", type=Path)
    parser.add_argument("--resolved-dataset-revision")
    parser.add_argument("--train-limit", type=int)
    parser.add_argument("--validation-limit", type=int)
    parser.add_argument("--model", required=True)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--reflection-proxy")
    parser.add_argument("--reflection-queue", type=Path)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--budget", type=int, default=10000)
    args = parser.parse_args()
    if args.train_limit is not None and args.train_limit < 1:
        parser.error("--train-limit must be positive")
    if args.validation_limit is not None and args.validation_limit < 1:
        parser.error("--validation-limit must be positive")
    if args.budget < 1:
        parser.error("--budget must be positive")

    if bool(args.data_manifest) != bool(args.resolved_dataset_revision):
        parser.error("--data-manifest and --resolved-dataset-revision must be supplied together")
    all_train = [normalize_example(row) for row in read_jsonl(args.train)]
    all_validation = [normalize_example(row) for row in read_jsonl(args.validation)]
    train = all_train[:args.train_limit]
    validation = all_validation[:args.validation_limit]
    if not train or not validation:
        raise ValueError("Train and validation must both contain examples")
    if {row["split_group"] for row in train} & {row["split_group"] for row in validation}:
        raise ValueError("Train and validation share content clusters")
    with urlopen(args.base_url.rstrip("/").removesuffix("/v1") + "/health", timeout=30) as response:
        server = task_server_contract(json.load(response), args.model)

    template = reflection_prompt_template()
    prompt_contract = {
        "id": PROMPT_CONTRACT_ID,
        "source_repository": CONTRACT_SOURCE_REPOSITORY,
        "source_commit": CONTRACT_SOURCE_COMMIT,
        "component": COMPONENT,
        "placement": "first and only user message; no system message",
        "seed_instruction": SEED_INSTRUCTION,
        "rendered_template": render_user_prompt("{instructions}", "{text}"),
        "task_generation": {"temperature": 0.0, "max_new_tokens": TASK_MAX_TOKENS},
        "strict_parser": "exact labels, canonical order, comma-separated; NONE for empty",
        "metric": "empty-aware per-example label-set F1; invalid answer scores zero",
    }
    if TASK_MAX_TOKENS_AMENDMENT:
        if TASK_MAX_TOKENS_AMENDMENT != "4096:8192:10471" or TASK_MAX_TOKENS != 8192:
            raise ValueError("Unknown task-token amendment")
        prompt_contract["task_generation"].update({
            "prior_max_new_tokens": 4096,
            "amendment_effective_after_logged_calls": 10471,
        })
    manifest = {
        "prompt_contract": prompt_contract,
        "model": args.model,
        "seed": args.seed,
        "max_metric_calls": args.budget,
        "task_server": server,
        "task_concurrency": TASK_CONCURRENCY,
        "gepa": {
            "version": importlib.metadata.version("gepa"),
            "candidate_selection_strategy": "pareto",
            "skip_perfect_score": True,
            "perfect_score": 1.0,
            "use_merge": False,
            "cache_evaluation": False,
            "reflection_minibatch_size": REFLECTION_MINIBATCH_SIZE,
        },
        "reflector": {
            "kind": "openai",
            "model": REFLECTION_MODEL,
            "base_url": REFLECTION_BASE_URL,
            "temperature": REFLECTION_TEMPERATURE,
            "max_tokens": REFLECTION_MAX_TOKENS,
            "instruction_budget_tokens": REFLECTION_INSTRUCTION_BUDGET_TOKENS,
            "failure_limit": REFLECTION_FAILURE_LIMIT,
            "max_retries": REFLECTION_MAX_RETRIES,
            "timeout_seconds": 900.0,
            "extra_body": {"reasoning": {"effort": "medium"}},
            "transport": "nfs_file_queue" if args.reflection_queue else "direct_http",
            "proxy": redact_proxy(args.reflection_proxy),
        },
        "reflection_prompt_template_sha256": hashlib.sha256(template.encode()).hexdigest(),
        "data_alignment": (
            "v2 split generator and contract of dense/discrete at the resolved source revision"
            if args.data_manifest else
            "local frozen v3 subset; not claimed hash-identical to the dense/discrete v2 splits"
        ),
        "train": {
            "path": str(args.train), "source_sha256": sha256_path(args.train),
            "selected_rows": len(train), "selected_sha256": rows_sha256(train),
            "keys": [row["key"] for row in train],
        },
        "validation": {
            "path": str(args.validation), "source_sha256": sha256_path(args.validation),
            "selected_rows": len(validation), "selected_sha256": rows_sha256(validation),
            "keys": [row["key"] for row in validation],
        },
    }
    if args.data_manifest:
        source_manifest = json.loads(args.data_manifest.read_text())
        manifest["data_manifest"] = {
            "path": str(args.data_manifest),
            "sha256": sha256_path(args.data_manifest),
            "declared_dataset_revision": source_manifest["dataset_revision"],
            "resolved_dataset_revision": args.resolved_dataset_revision,
            "sources": source_manifest["sources"],
        }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = args.out_dir / "manifest.json"
    if manifest_path.exists() and json.loads(manifest_path.read_text()) != manifest:
        existing_manifest = json.loads(manifest_path.read_text())
        prior_manifest = json.loads(json.dumps(manifest))
        prior_manifest["prompt_contract"]["task_generation"] = {
            "temperature": 0.0,
            "max_new_tokens": 4096,
        }
        prior_rows = read_jsonl(args.out_dir / "evaluations.jsonl")
        truncated_rows = [row for row in prior_rows if row.get("finish_reason") == "length"]
        token_amendment = (
            bool(TASK_MAX_TOKENS_AMENDMENT)
            and existing_manifest == prior_manifest
            and len(prior_rows) == 10471
            and len(truncated_rows) == 1
        )
        concurrency_prior = json.loads(json.dumps(manifest))
        concurrency_prior.pop("task_concurrency")
        concurrency_prior["task_server"].pop("batch_size", None)
        concurrency_prior["task_server"]["decoding"] = (
            "native chat, one request at a time, explicit final channel, BF16"
        )
        concurrency_amendment = (
            TASK_CONCURRENCY_AMENDMENT == "1"
            and TASK_CONCURRENCY == 4
            and existing_manifest == concurrency_prior
            and bool(prior_rows)
            and not truncated_rows
        )
        if not (token_amendment or concurrency_amendment):
            raise ValueError("Resume requires an identical manifest")
        amendment = (
            {
                "kind": "task_generation_ceiling",
                "previous_max_new_tokens": 4096,
                "new_max_new_tokens": 8192,
                "effective_after_logged_calls": 10471,
                "reason": "one GPT-OSS response reached 4096 tokens without final content",
                "scope": "gepa-gpt-oss-20b-n200-s42 targeted resume only",
            }
            if token_amendment else
            {
                "kind": "task_microbatching",
                "previous_task_concurrency": 1,
                "new_task_concurrency": TASK_CONCURRENCY,
                "effective_after_logged_calls": len(prior_rows),
                "reason": "restore batched task inference for the unfinished Qwen grid",
                "scope": "unfinished Qwen Civil-v2 GEPA cells only",
            }
        )
        amendment_path = args.out_dir / "protocol_amendment.json"
        if amendment_path.exists() and json.loads(amendment_path.read_text()) != amendment:
            raise ValueError("Existing protocol amendment differs")
        amendment_path.write_text(json.dumps(amendment, indent=2) + "\n")
    if (args.out_dir / "summary.json").exists():
        raise ValueError("This run has already completed")
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    (args.out_dir / "prompt_contract.json").write_text(json.dumps(prompt_contract, indent=2) + "\n")

    task_client = OpenAICompatibleClient(LLMConfig(
        model_name=args.model,
        api_base_url=args.base_url,
        max_tokens=TASK_MAX_TOKENS,
        temperature=0.0,
        timeout=900.0,
    ))
    if args.reflection_queue:
        reflection_client = FileQueueLLMClient(
            REFLECTION_MODEL, args.reflection_queue, timeout=900.0
        )
    else:
        api_key = os.environ.get("OPENROUTER_API_KEY")
        if not api_key:
            raise ValueError("OPENROUTER_API_KEY is required")
        reflection_client = OpenAICompatibleClient(LLMConfig(
            model_name=REFLECTION_MODEL,
            api_key=api_key,
            api_base_url=REFLECTION_BASE_URL,
            max_tokens=REFLECTION_MAX_TOKENS,
            temperature=REFLECTION_TEMPERATURE,
            timeout=900.0,
            max_retries=REFLECTION_MAX_RETRIES,
            extra_body={"reasoning": {"effort": "medium"}},
            proxy=args.reflection_proxy,
        ))
    reflection_path = args.out_dir / "reflection.jsonl"
    prior_reflections = read_jsonl(reflection_path) if reflection_path.exists() else []
    reflection_cache = {
        row["prompt_sha256"]: row["text"]
        for row in prior_reflections
        if row.get("finish_reason") == "stop" and row.get("text", "").strip()
    }
    reflection_log = reflection_path.open("a", buffering=1)
    evaluation_path = args.out_dir / "evaluations.jsonl"
    evaluation_log = evaluation_path.open("a", buffering=1)
    consecutive_reflection_failures = 0

    def reflection(prompt: str | list[dict[str, Any]]) -> str:
        nonlocal consecutive_reflection_failures
        if not isinstance(prompt, str):
            raise TypeError("Civil v2 reflection prompt must be text")
        prompt_hash = hashlib.sha256(prompt.encode()).hexdigest()
        if prompt_hash in reflection_cache:
            return reflection_cache[prompt_hash]
        try:
            response = reflection_client.generate(prompt)
        except Exception:
            consecutive_reflection_failures += 1
            if consecutive_reflection_failures >= REFLECTION_FAILURE_LIMIT:
                raise RuntimeError(
                    f"Reflection endpoint failed {consecutive_reflection_failures} calls in a row"
                ) from None
            raise
        consecutive_reflection_failures = 0
        reflection_log.write(json.dumps({
            "prompt_sha256": prompt_hash, "prompt": prompt, **asdict(response),
        }, ensure_ascii=False) + "\n")
        if response.finish_reason != "stop" or not response.text.strip():
            raise RuntimeError("Reflection must finish with nonempty final content")
        reflection_cache[prompt_hash] = response.text
        return response.text

    adapter = CivilV2Adapter(task_client, evaluation_log)
    started = time.monotonic()
    try:
        result = gepa.optimize(
            seed_candidate={COMPONENT: SEED_INSTRUCTION},
            trainset=train,
            valset=validation,
            adapter=adapter,
            reflection_lm=reflection,
            reflection_prompt_template=template,
            max_metric_calls=args.budget,
            candidate_selection_strategy="pareto",
            batch_sampler="epoch_shuffled",
            reflection_minibatch_size=REFLECTION_MINIBATCH_SIZE,
            perfect_score=1.0,
            skip_perfect_score=True,
            use_merge=False,
            seed=args.seed,
            run_dir=str(args.out_dir / "gepa_logs"),
            display_progress_bar=True,
            track_best_outputs=True,
            cache_evaluation=False,
            raise_on_exception=True,
        )
    finally:
        reflection_log.close()
        evaluation_log.close()

    index = result.best_idx
    instructions = result.candidates[index][COMPONENT]
    rendered_prompt = render_user_prompt(instructions, "{text}")
    (args.out_dir / "optimized_instructions.txt").write_text(instructions)
    (args.out_dir / "optimized_prompt.txt").write_text(rendered_prompt)
    candidates = [
        {
            "index": i,
            "instruction": candidate[COMPONENT],
            "validation_score": float(result.val_aggregate_scores[i]),
            "parents": result.parents[i],
            "discovery_metric_calls": result.discovery_eval_counts[i],
        }
        for i, candidate in enumerate(result.candidates)
    ]
    (args.out_dir / "candidates.json").write_text(json.dumps(candidates, indent=2) + "\n")
    (args.out_dir / "gepa_logs" / "result.json").write_text(
        json.dumps(result.to_dict(), indent=2) + "\n"
    )
    (args.out_dir / "gepa_logs" / "candidate_tree.html").write_text(result.candidate_tree_html())
    task_usage = summarize_jsonl(evaluation_path)
    reflection_usage = summarize_jsonl(reflection_path)
    reflection_usage["list_price_estimate_usd"] = (
        reflection_usage["prompt_tokens"] * 0.20 / 1_000_000
        + reflection_usage["completion_tokens"] * 1.20 / 1_000_000
    )
    summary = {
        "status": "PASS",
        "best_idx": index,
        "validation_score": float(result.val_aggregate_scores[index]),
        "seed_validation_score": float(result.val_aggregate_scores[0]),
        "n_candidates": len(result.candidates),
        "total_metric_calls": result.total_metric_calls,
        "task_usage": task_usage,
        "reflection_usage": reflection_usage,
        "this_process_optimization_seconds": time.monotonic() - started,
    }
    (args.out_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2), flush=True)
    print("GEPA_RESULT=PASS", flush=True)


if __name__ == "__main__":
    main()
