"""A Super Expert set belongs to the model it was identified on.

Qwen's set is (1,68), (2,92), (3,82); GPT-OSS-20B has 32 experts per layer, so
those indices do not exist there at all. The set used to be a CLI default, which
meant a GPT-OSS ablation would be asked for and fail on an index range rather
than using the model's own replication-gated set.
"""
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

SETS = json.loads((ROOT / "configs/super_experts.json").read_text())
QWEN = SETS["Qwen/Qwen3-30B-A3B-Instruct-2507"]
GPT_OSS = SETS["openai/gpt-oss-20b"]


def test_each_spec_states_its_own_set():
    assert QWEN["super_experts"] == ["1:68", "2:92", "3:82"]
    assert GPT_OSS["super_experts"] == ["17:5", "6:5"]
    assert "wikitext-2" in GPT_OSS["super_experts_note"].lower()


def test_no_script_hardcodes_the_set_as_a_default():
    for path in sorted((ROOT / "scripts").glob("*.py")):
        assert 'default="1:68,2:92,3:82"' not in path.read_text(), f"{path.name} defaults the set"


@pytest.mark.parametrize("intervention", ["ExpertAblation", "RouterMask"])
def test_both_interventions_refuse_an_out_of_range_expert(intervention):
    import torch
    from se_gepa import ablation
    from test_profiler import tiny_model

    model, config = tiny_model("eager")
    cls = getattr(ablation, intervention)
    with pytest.raises(ValueError, match="Experts outside this model"):
        cls(model, {(1, config.num_experts + 10)})
