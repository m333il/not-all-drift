import hashlib
import json
from pathlib import Path

import pytest
import torch
from peft import get_peft_model
from transformers import Qwen3MoeConfig, Qwen3MoeForCausalLM

from mrd.peft_training import PeftTrainConfig, _build_peft_config
from mrd.training import train_encoded


def setup_model(method, start_step=0, total_steps=6):
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
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda step: max(0.0, min(1.0, 1 - (step - start_step) / (total_steps - start_step))))
    return model, optimizer, scheduler


@pytest.mark.parametrize("method", ["prompt_tuning", "prefix_tuning"])
def test_extension_preserves_parent_adam_and_data_then_resumes_exactly(tmp_path, method):
    encoded = [([1, 2] + [3] * n + [0], [-100, -100] + [3] * n + [0]) for n in range(1, 6)]
    keys = [str(i) for i in range(5)]
    parent_contract = {"data": keys, "epochs": 2, "total_steps": 6, "method": method, "lr": 0.003}
    settings = dict(pad_id=0, batch_size=1, accumulation=2, seed=19)
    parent_model, opt, scheduler = setup_model(method)
    parent_dir = tmp_path / "parent"
    train_encoded(parent_model, encoded, keys, opt, scheduler, parent_dir,
                  epochs=2, contract=parent_contract, **settings)
    checkpoint = parent_dir / "step_000006"
    hashes = {str(p.relative_to(parent_dir)): hashlib.sha256(p.read_bytes()).hexdigest()
              for p in parent_dir.rglob("*") if p.is_file()}
    source = torch.load(checkpoint / "training.pt", weights_only=False)
    assert source["optimizer"]["param_groups"][0]["lr"] == 0
    contract = {**parent_contract, "epochs": 4, "total_steps": 12,
                "extension": {"start_step": 6, "completed_epochs": 2}}
    full, full_opt, scheduler = setup_model(method, 6, 12)
    train_encoded(full, encoded, keys, full_opt, scheduler, tmp_path / "full",
                  epochs=4, contract=contract, extend_from=checkpoint, **settings)
    boundary = torch.load(tmp_path / "full/step_000006/training.pt", weights_only=False)
    for name, expected in source["trainable"].items():
        torch.testing.assert_close(boundary["trainable"][name], expected, rtol=0, atol=0)
    for index, expected in source["optimizer"]["state"].items():
        for name, value in expected.items():
            torch.testing.assert_close(boundary["optimizer"]["state"][index][name], value, rtol=0, atol=0)
    assert boundary["optimizer"]["param_groups"][0]["lr"] == 0.003
    assert boundary["state"]["rng"] == source["state"]["rng"]
    torch.testing.assert_close(boundary["torch_rng"], source["torch_rng"], rtol=0, atol=0)

    partial, opt, scheduler = setup_model(method, 6, 12)
    train_encoded(partial, encoded, keys, opt, scheduler, tmp_path / "resumed",
                  epochs=4, contract=contract, extend_from=checkpoint, stop_after=7, **settings)
    resumed, opt, scheduler = setup_model(method, 6, 12)
    torch.manual_seed(999)
    final = train_encoded(resumed, encoded, keys, opt, scheduler, tmp_path / "resumed",
                          epochs=4, contract=contract, resume=tmp_path / "resumed/step_000007", **settings)
    assert final["step"] == 12 and final["epoch"] == 4
    for expected, actual in zip(full.parameters(), resumed.parameters()):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    logs = {name: [json.loads(line) for line in (tmp_path / name / "steps.jsonl").read_text().splitlines()]
            for name in ["parent", "full", "resumed"]}
    assert logs["full"][:6] == logs["parent"]
    for field in ["keys", "loss", "tokens", "lr", "gradient_norm"]:
        assert [r[field] for r in logs["full"]] == [r[field] for r in logs["resumed"]]
    assert [r["lr"] for r in logs["full"]][6:] == pytest.approx([0.003, 0.0025, 0.002, 0.0015, 0.001, 0.0005])
    assert opt.param_groups[0]["lr"] == 0
    assert {str(p.relative_to(parent_dir)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in parent_dir.rglob("*") if p.is_file()} == hashes

    invalid = {**contract, "lr": 0.1}
    model, opt, scheduler = setup_model(method, 6, 12)
    with pytest.raises(ValueError, match="may not change"):
        train_encoded(model, encoded, keys, opt, scheduler, tmp_path / "invalid",
                      epochs=4, contract=invalid, extend_from=checkpoint, **settings)
    model, opt, scheduler = setup_model(method, 6, 12)
    with pytest.raises(ValueError, match="completed parent"):
        train_encoded(model, encoded, keys, opt, scheduler, tmp_path / "mid_parent",
                      epochs=4, contract=contract, extend_from=parent_dir / "step_000005", **settings)
