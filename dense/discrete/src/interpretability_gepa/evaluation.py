from __future__ import annotations

from collections.abc import Callable, Sequence

from .prompts import parse_labels, render_condition_messages
from .providers import TextProvider
from .schemas import Example, Prediction

Parser = Callable[..., tuple[str, ...]]


def build_prediction(
    example_id: str,
    condition: str,
    response: str,
    labels: Sequence[str],
    parser: Parser = parse_labels,
) -> Prediction:
    """Parse one response with both the strict and the set (order-lenient) parser."""
    try:
        lenient = parser(response, labels, enforce_order=False)
        lenient_ok = True
    except Exception:  # preserve failures as data
        lenient, lenient_ok = (), False
    try:
        parsed = parser(response, labels)
        return Prediction(example_id, condition, response, parsed, True, None, lenient, False)
    except Exception as exc:  # preserve failures as data
        return Prediction(example_id, condition, response, (), False, str(exc), lenient, lenient_ok)


def evaluate_examples(
    provider: TextProvider,
    examples: Sequence[Example],
    *,
    condition: str,
    instruction: str,
    labels: Sequence[str],
    max_tokens: int = 128,
) -> list[Prediction]:
    predictions: list[Prediction] = []
    for example in examples:
        rendered = render_condition_messages(condition, instruction, labels, example.text)
        response = provider.complete(rendered.messages, max_tokens=max_tokens)
        predictions.append(build_prediction(example.id, condition, response.text, labels))
    return predictions


def evaluate_hf_examples(
    model: object,
    tokenizer: object,
    examples: Sequence[Example],
    *,
    condition: str,
    instruction: str,
    labels: Sequence[str],
    non_thinking: bool,
    max_tokens: int = 128,
    batch_size: int = 8,
) -> list[Prediction]:
    import torch

    from .prompts import apply_chat_template

    tokenizer.padding_side = "left"  # type: ignore[attr-defined]
    predictions: list[Prediction] = []
    for start in range(0, len(examples), batch_size):
        batch = examples[start : start + batch_size]
        prompts = [
            apply_chat_template(
                tokenizer,
                render_condition_messages(condition, instruction, labels, example.text),
                non_thinking=non_thinking,
            )
            for example in batch
        ]
        # The chat template already emits <bos>.
        encoded = tokenizer(  # type: ignore[operator]
            prompts, padding=True, return_tensors="pt", add_special_tokens=False
        )
        device = next(model.parameters()).device  # type: ignore[attr-defined]
        encoded = {key: value.to(device) for key, value in encoded.items()}
        with torch.no_grad():
            generated = model.generate(  # type: ignore[attr-defined]
                **encoded,
                do_sample=False,
                temperature=None,
                max_new_tokens=max_tokens,
                pad_token_id=tokenizer.pad_token_id,  # type: ignore[attr-defined]
            )
        continuation = generated[:, encoded["input_ids"].shape[1] :]
        responses = tokenizer.batch_decode(  # type: ignore[attr-defined]
            continuation, skip_special_tokens=True
        )
        for example, response in zip(batch, responses, strict=True):
            predictions.append(build_prediction(example.id, condition, response, labels))
    return predictions
