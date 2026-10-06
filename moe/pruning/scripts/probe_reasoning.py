"""Does gpt-oss actually reason on this task, and does the effort knob matter?

Everything downstream assumes an answer either opens the `analysis` channel or
does not. This asks the model directly: same prompts, no prefill, one run per
`reasoning_effort`, raw decode printed with the special tokens left in.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from mrd_pruning.arms import ArmSpec, load_arm  # noqa: E402
from mrd_pruning.task import (  # noqa: E402
    V25_SEED_INSTRUCTION, reasoning_on_kwargs, render_user_prompt_v25,
    pin_template_date,
)

logger = logging.getLogger("probe")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="openai/gpt-oss-20b")
    parser.add_argument("--revision", default="6cee5e81ee83917806bbde320786a8fb61efebee")
    parser.add_argument("--data", type=Path, default=Path("data/calib_val_n500.jsonl"))
    parser.add_argument("--n", type=int, default=5)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--efforts", default="low,medium,high")
    parser.add_argument("--adapter", type=Path, default=None,
                        help="optional: probe an arm instead of the bare model")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    import torch

    rows = [json.loads(l) for l in args.data.read_text().splitlines() if l.strip()][: args.n]
    spec = (
        ArmSpec(name="base", kind="base", system_prompt_text=None, prompt_in_user_turn=True)
        if args.adapter is None
        else ArmSpec(name="prompt_tuning", kind="prompt_tuning", adapter_path=args.adapter,
                     system_prompt_text=None, prompt_in_user_turn=True,
                     allow_unpinned_checkpoint=True)
    )
    model, tokenizer = load_arm(spec, model_id=args.model, revision=args.revision,
                                dtype="bfloat16")

    for effort in args.efforts.split(","):
        effort = effort.strip()
        kwargs = reasoning_on_kwargs(tokenizer, effort=effort)
        logger.info("\n===== reasoning_effort=%s | template kwargs: %s =====", effort, kwargs)
        for row in rows:
            user = render_user_prompt_v25(row.get("comment", row.get("text")),
                                          V25_SEED_INSTRUCTION)
            text = tokenizer.apply_chat_template(
                [{"role": "user", "content": user}],
                tokenize=False, add_generation_prompt=True, **kwargs,
            )
            text = pin_template_date(text)
            ids = torch.tensor(
                [tokenizer(text, add_special_tokens=False)["input_ids"]], device=model.device
            )
            with torch.inference_mode():
                out = model.generate(input_ids=ids, attention_mask=torch.ones_like(ids),
                                     max_new_tokens=args.max_new_tokens, do_sample=False)
            raw = tokenizer.decode(out[0, ids.shape[1]:], skip_special_tokens=False)
            has_analysis = "analysis" in raw.lower()
            logger.info("  Tokens %3d | analysis: %-5s | %s",
                        out.shape[1] - ids.shape[1], has_analysis, repr(raw[:220]))


if __name__ == "__main__":
    main()
