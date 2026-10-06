"""The arms job wraps one already-loaded backbone once per PEFT type, and the
profiler has to attribute positions correctly under each.

That is the only part of the arms job that cannot be read off the source. peft
0.20.0 refuses to hold a prompt-tuning and a prefix-tuning adapter in the same
``PeftModel``, so they take turns over the same weights; and PEFT installs prompt
tuning by prepending embeddings while prefix tuning writes ``past_key_values``,
so the sequence the experts see is longer in one case and unchanged in the other.
Getting either wrong would silently compare position 0 of one arm against
position 100 of another.

CPU, tiny random model, randomly initialised adapters -- a wiring test.
"""
import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from se_gepa.profiler import FusedExpertProfiler
from test_profiler import tiny_model

# The gate stage does not install peft; only the arms stage needs it.
pytest.importorskip("peft")

VIRTUAL = 6


def configs(hidden_size):
    from peft import PrefixTuningConfig, PromptTuningConfig

    return (
        PromptTuningConfig(task_type="CAUSAL_LM", num_virtual_tokens=VIRTUAL),
        PrefixTuningConfig(task_type="CAUSAL_LM", num_virtual_tokens=VIRTUAL,
                           prefix_projection=True, encoder_hidden_size=hidden_size),
    )


def profile(model, ids, offset):
    with FusedExpertProfiler(model) as profiler:
        profiler.begin_example(0, ids[0].tolist(), virtual_tokens=offset)
        with torch.no_grad():
            model(input_ids=ids)
    return profiler.records


def test_mixing_peft_types_in_one_wrapper_is_still_refused():
    """Pinned so an upgrade that lifts the restriction is noticed, not assumed."""
    from peft import get_peft_model

    model, config = tiny_model("grouped_mm", seed=30)
    prompt, prefix = configs(config.hidden_size)
    wrapped = get_peft_model(model, prompt, adapter_name="soft-arm")
    with pytest.raises(ValueError, match="different peft types"):
        wrapped.add_adapter("kv-arm", prefix)


def test_each_type_wraps_the_same_backbone_in_turn():
    from peft import get_peft_model

    model, config = tiny_model("grouped_mm", seed=31)
    prompt, prefix = configs(config.hidden_size)
    ids = torch.randint(0, config.vocab_size, (1, 10))
    base = profile(model, ids, offset=0)

    soft = get_peft_model(model, prompt, adapter_name="soft-arm").eval()
    with pytest.raises(RuntimeError, match="one full sequence"):
        profile(soft, ids, offset=0)
    soft_records = profile(soft, ids, offset=VIRTUAL)
    backbone = soft.get_base_model()
    del soft

    kv = get_peft_model(backbone, prefix, adapter_name="kv-arm").eval()
    with pytest.raises(RuntimeError, match="one full sequence"):
        profile(kv, ids, offset=VIRTUAL)
    kv_records = profile(kv, ids, offset=0)
    backbone = kv.get_base_model()
    del kv

    assert soft_records and kv_records
    # Prefix tuning inserts no positions but must still move the expert inputs;
    # identical maxima would mean the adapter was never installed.
    shared = set(base) & set(kv_records)
    assert shared and any(base[key].output_max != kv_records[key].output_max for key in shared)
    # Unwrapping has to leave the backbone exactly as it was found.
    after = profile(backbone, ids, offset=0)
    assert {key: record.output_max for key, record in after.items()} == \
           {key: record.output_max for key, record in base.items()}


def test_virtual_positions_carry_no_token_id():
    from peft import get_peft_model

    model, config = tiny_model("grouped_mm", seed=32)
    prompt, _prefix = configs(config.hidden_size)
    ids = torch.randint(0, config.vocab_size, (1, 10))
    soft = get_peft_model(model, prompt, adapter_name="soft-arm").eval()
    records = profile(soft, ids, offset=VIRTUAL)
    text = ids[0].tolist()
    assert any(record.position < VIRTUAL for record in records.values()), (
        "no expert took its maximum on a virtual position, so the case is untested")
    for record in records.values():
        if record.position < VIRTUAL:
            assert record.token_id == -1
        else:
            assert record.token_id == text[record.position - VIRTUAL]
