from dataclasses import dataclass


CHAT_DATE = "2026-09-07"
REASONING_EFFORT = "low"


def chat_ids(tokenizer, messages, *, add_generation_prompt):
    return tokenizer.apply_chat_template(
        messages, tokenize=True, return_dict=False,
        add_generation_prompt=add_generation_prompt,
        reasoning_effort=REASONING_EFFORT,
        strftime_now=lambda _format: CHAT_DATE,
    )


def chat_messages(system, user):
    return ([{"role": "system", "content": system}] if system else []) + [
        {"role": "user", "content": user},
    ]


def encode_user_prompt(tokenizer, system, user):
    messages = chat_messages(system, user)
    options = dict(tokenize=False, add_generation_prompt=True,
                   reasoning_effort=REASONING_EFFORT, strftime_now=lambda _format: CHAT_DATE)
    sentinel = "MRD_USER_SPAN_2a1c674934f84a5782fe"
    if sentinel in system or sentinel in user:
        raise ValueError("User-span sentinel occurs in the input")
    marked = tokenizer.apply_chat_template(chat_messages(system, sentinel), **options)
    if marked.count(sentinel) != 1:
        raise ValueError("Chat template must contain the user content exactly once")
    prefix, suffix = marked.split(sentinel)
    rendered = tokenizer.apply_chat_template(messages, **options)
    if rendered != prefix + user + suffix:
        raise ValueError("Chat template transforms user content; explicit alignment is required")
    encoded = tokenizer(rendered, add_special_tokens=False, return_offsets_mapping=True)
    ids = chat_ids(tokenizer, messages, add_generation_prompt=True)
    if encoded["input_ids"] != ids:
        raise ValueError("Offset tokenizer disagrees with native chat tokenization")
    start, end = len(prefix), len(prefix) + len(user)
    positions = [i for i, (a, b) in enumerate(encoded["offset_mapping"]) if start <= a < b <= end]
    if not positions:
        raise ValueError("No complete tokens inside the user content")
    return ids, positions


def encode_assistant_turn(tokenizer, system, user, target, thinking=None):
    messages = chat_messages(system, user)
    prompt = chat_ids(tokenizer, messages, add_generation_prompt=True)
    assistant = {"role": "assistant", "content": target}
    if thinking is not None:
        assistant["thinking"] = thinking
    full = chat_ids(tokenizer, messages + [assistant], add_generation_prompt=False)
    if full[:len(prompt)] != prompt:
        raise ValueError("Native assistant serialization does not preserve the generation prefix")
    return full, [-100] * len(prompt) + full[len(prompt):]


def encode_final_content(tokenizer, system, user, target, thinking=None):
    sentinel = "MRD_FINAL_SPAN_8cb17a239d454f4aaa1e"
    if any(sentinel in text for text in [system, user, target, thinking or ""]):
        raise ValueError("Final-span sentinel occurs in the input")
    assistant = {"role": "assistant", "content": sentinel}
    if thinking is not None:
        assistant["thinking"] = thinking
    messages = chat_messages(system, user) + [assistant]
    options = dict(tokenize=False, add_generation_prompt=False,
                   reasoning_effort=REASONING_EFFORT, strftime_now=lambda _format: CHAT_DATE)
    marked = tokenizer.apply_chat_template(messages, **options)
    if marked.count(sentinel) != 1:
        raise ValueError("Chat template must contain final content exactly once")
    prefix, suffix = marked.split(sentinel)
    assistant["content"] = target
    rendered = tokenizer.apply_chat_template(messages, **options)
    if rendered != prefix + target + suffix:
        raise ValueError("Chat template transforms final content; explicit alignment is required")
    encoded = tokenizer(rendered, add_special_tokens=False, return_offsets_mapping=True)
    ids, _ = encode_assistant_turn(tokenizer, system, user, target, thinking)
    if encoded["input_ids"] != ids:
        raise ValueError("Offset tokenizer disagrees with native chat tokenization")
    start, end = len(prefix), len(prefix) + len(target)
    positions = [i for i, (a, b) in enumerate(encoded["offset_mapping"]) if start <= a < b <= end]
    if not positions:
        raise ValueError("No complete tokens inside final content")
    return ids, positions


@dataclass
class Completion:
    final: str
    analysis: str
    raw: str
    finished: bool


def decode_completion(tokenizer, generated_ids, model_family):
    ids = list(generated_ids)
    raw = tokenizer.decode(ids, skip_special_tokens=False)
    finished = bool(ids and ids[-1] == tokenizer.eos_token_id)
    if model_family != "gpt_oss":
        return Completion(tokenizer.decode(ids, skip_special_tokens=True).strip(), "", raw, finished)

    final, analysis = "", ""
    for part in ("<|start|>assistant" + raw).split("<|start|>"):
        if part.startswith("assistant<|channel|>final<|message|>"):
            final = part.split("<|message|>", 1)[1].split("<|return|>", 1)[0]
        elif part.startswith("assistant<|channel|>analysis<|message|>"):
            analysis = part.split("<|message|>", 1)[1].split("<|end|>", 1)[0]
    return Completion(final.strip(), analysis.strip(), raw, finished)
