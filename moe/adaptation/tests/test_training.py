import json
from pathlib import Path

import pytest
import torch
from peft import get_peft_model
from transformers import Qwen3MoeConfig, Qwen3MoeForCausalLM, get_scheduler

from mrd.peft_training import PeftTrainConfig, _build_peft_config
import mrd.training as training
from mrd.training import train_encoded


@pytest.mark.parametrize("method", ["prompt_tuning", "prefix_tuning", "p_tuning_v2"])
@pytest.mark.parametrize("pause", [2, 3])
@pytest.mark.parametrize("checkpoint_every", [1, 2])
@pytest.mark.parametrize("schedule", ["linear", "cosine"])
def test_resumed_adapter_matches_uninterrupted_adam_and_data_order(tmp_path, method, pause, checkpoint_every, schedule):
    encoded = [([1, 2] + [3] * n + [0], [-100, -100] + [3] * n + [0]) for n in range(1, 6)]
    keys = [str(i) for i in range(5)]
    contract = {"data": keys, "epochs": 2, "method": method}

    def setup():
        torch.manual_seed(45)
        config = Qwen3MoeConfig(hidden_size=16, intermediate_size=32, moe_intermediate_size=8,
                               num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=1,
                               num_experts=4, num_experts_per_tok=2, vocab_size=16, head_dim=8,
                               norm_topk_prob=True, attention_dropout=0.1)
        config._attn_implementation = "eager"
        base = Qwen3MoeForCausalLM(config)
        cfg = PeftTrainConfig(method=method, sft_targets=Path("unused"), num_virtual_tokens=2)
        model = get_peft_model(base, _build_peft_config(cfg, base, "unused"))
        optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=0.003)
        scheduler = get_scheduler(schedule, optimizer, num_warmup_steps=2 if schedule == "cosine" else 0,
                                  num_training_steps=6)
        return model, optimizer, scheduler

    settings = dict(pad_id=0, epochs=2, batch_size=1, accumulation=2, seed=19, contract=contract,
                    checkpoint_every=checkpoint_every)
    complete, opt, scheduler = setup()
    train_encoded(complete, encoded, keys, opt, scheduler, tmp_path / "full", **settings)
    if checkpoint_every == 2:
        assert {p.name for p in (tmp_path / "full").glob("step_*")} == {
            "step_000000", "step_000002", "step_000003", "step_000004", "step_000006"}
    partial, opt, scheduler = setup()
    train_encoded(partial, encoded, keys, opt, scheduler, tmp_path / "resume", stop_after=pause, **settings)
    # A crashed process can leave an uncheckpointed event and a partial JSON line.
    with (tmp_path / "resume" / "steps.jsonl").open("a") as stream:
        stream.write(json.dumps({"step": pause + 1, "loss": -1, "keys": ["uncommitted"]}) + "\n")
        stream.write('{"step":')
    resumed, opt, scheduler = setup()
    torch.manual_seed(8912)
    final = train_encoded(resumed, encoded, keys, opt, scheduler, tmp_path / "resume",
                          resume=tmp_path / "resume" / f"step_{pause:06d}", **settings)
    assert final["step"] == 6 and final["epoch"] == 2
    for expected, actual in zip(complete.parameters(), resumed.parameters()):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    records = [[json.loads(line) for line in (tmp_path / name / "steps.jsonl").read_text().splitlines()]
               for name in ("full", "resume")]
    for field in ("keys", "loss", "tokens", "lr", "gradient_norm"):
        assert [r[field] for r in records[0]] == [r[field] for r in records[1]]


@pytest.mark.parametrize("method", ["prompt_tuning", "prefix_tuning"])
def test_epoch_adapter_only_keeps_selection_snapshots_and_resumes(tmp_path, monkeypatch, method):
    encoded = [([1, 2] + [3] * n + [0], [-100, -100] + [3] * n + [0]) for n in range(1, 6)]
    keys = [str(i) for i in range(5)]
    contract = {"data": keys, "epochs": 2, "method": method, "epoch_adapter_only": True}

    def setup():
        torch.manual_seed(45)
        config = Qwen3MoeConfig(hidden_size=16, intermediate_size=32, moe_intermediate_size=8,
                               num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=1,
                               num_experts=4, num_experts_per_tok=2, vocab_size=16, head_dim=8,
                               norm_topk_prob=True, attention_dropout=0.1)
        config._attn_implementation = "eager"
        base = Qwen3MoeForCausalLM(config)
        cfg = PeftTrainConfig(method=method, sft_targets=Path("unused"), num_virtual_tokens=2)
        model = get_peft_model(base, _build_peft_config(cfg, base, "unused"))
        optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=0.003)
        scheduler = get_scheduler("linear", optimizer, num_warmup_steps=0, num_training_steps=6)
        return model, optimizer, scheduler

    settings = dict(pad_id=0, epochs=2, batch_size=1, accumulation=2, seed=19, contract=contract,
                    checkpoint_every=2, epoch_adapter_only=True)
    complete, optimizer, scheduler = setup()
    train_encoded(complete, encoded, keys, optimizer, scheduler, tmp_path / "full", **settings)
    snapshot = tmp_path / "full/step_000003"
    assert (snapshot / "ADAPTER_ONLY").is_file()
    assert (snapshot / "adapter/adapter_model.safetensors").is_file()
    assert not (snapshot / "training.pt").exists()
    assert {p.parent.name for p in (tmp_path / "full").glob("step_*/COMPLETE")} == {
        "step_000000", "step_000002", "step_000004", "step_000006"}

    original_save = training.save_adapter_snapshot

    def crash_after_snapshot(model, path):
        original_save(model, path)
        raise RuntimeError("simulated interruption after adapter snapshot")

    monkeypatch.setattr(training, "save_adapter_snapshot", crash_after_snapshot)
    partial, optimizer, scheduler = setup()
    with pytest.raises(RuntimeError, match="simulated interruption"):
        train_encoded(partial, encoded, keys, optimizer, scheduler, tmp_path / "resume", **settings)
    monkeypatch.setattr(training, "save_adapter_snapshot", original_save)
    resumed, optimizer, scheduler = setup()
    train_encoded(resumed, encoded, keys, optimizer, scheduler, tmp_path / "resume",
                  resume=tmp_path / "resume/step_000002", **settings)
    for expected, actual in zip(complete.parameters(), resumed.parameters()):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
