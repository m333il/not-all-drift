from __future__ import annotations

from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from interpretability_gepa.errors import ArtifactError  # noqa: E402
from interpretability_gepa.tuned_lens import (  # noqa: E402
    TunedLensConfig,
    batched,
    build_translators,
    load_translators,
    sample_positions,
    save_translators,
    translator_kl,
)


def _config(**overrides: object) -> TunedLensConfig:
    base = {"hidden_size": 8, "layers": 3, "positions_per_sequence": 2, "max_steps": 5}
    base.update(overrides)
    return TunedLensConfig(**base)  # type: ignore[arg-type]


def test_translators_start_as_the_identity() -> None:
    # An untrained tuned lens must reproduce the logit lens exactly, otherwise a
    # difference in results cannot be attributed to the fitting.
    translators = build_translators(_config(), device="cpu", dtype=torch.float32)
    states = torch.randn(2, 4, 8)
    for translator in translators:
        assert torch.allclose(translator(states), states, atol=1e-5)


def test_config_rejects_degenerate_geometry() -> None:
    for bad in ({"hidden_size": 0}, {"layers": 0}, {"positions_per_sequence": 0}, {"max_steps": 0}):
        with pytest.raises(ArtifactError):
            build_translators(_config(**bad), device="cpu", dtype=torch.float32)


def test_sampled_positions_stay_inside_the_sequence_and_skip_the_last_token() -> None:
    mask = torch.tensor([[1, 1, 1, 0], [1, 1, 1, 1]])
    picked = sample_positions(mask, 2)
    assert picked.shape == (2, 2)
    # Row 0 has three real tokens, so only 0 and 1 predict a following token.
    assert set(picked[0].tolist()) <= {0, 1}
    assert set(picked[1].tolist()) <= {0, 1, 2}


def test_sampling_repeats_when_the_sequence_is_shorter_than_the_request() -> None:
    picked = sample_positions(torch.tensor([[1, 1, 0, 0]]), 5)
    assert picked.shape == (1, 5)
    assert set(picked[0].tolist()) == {0}


def test_empty_sequence_is_refused_rather_than_silently_scored() -> None:
    with pytest.raises(ArtifactError):
        sample_positions(torch.zeros(1, 4, dtype=torch.long), 2)


def test_identity_translator_gives_zero_kl_against_its_own_distribution() -> None:
    class FakeModel:
        def __init__(self) -> None:
            self.lm_head = torch.nn.Linear(8, 11, bias=False)

    model = FakeModel()
    translators = build_translators(_config(), device="cpu", dtype=torch.float32)
    hidden = torch.randn(2, 5, 8)
    positions = torch.tensor([[0, 1], [2, 3]])
    gathered = torch.gather(hidden, 1, positions[..., None].expand(-1, -1, 8))
    target = torch.log_softmax(model.lm_head(gathered).float(), dim=-1)
    loss = translator_kl(
        model=model,
        translator=translators[0],
        final_norm=None,
        hidden=hidden,
        target_log_probs=target,
        positions=positions,
    )
    assert float(loss) == pytest.approx(0.0, abs=1e-5)


def test_translators_round_trip_through_disk(tmp_path: Path) -> None:
    config = _config(learning_rate=5e-4, warmup_steps=7, seed=3)
    translators = build_translators(config, device="cpu", dtype=torch.float32)
    with torch.no_grad():
        translators[1].delta.bias.fill_(0.25)
    path = tmp_path / "translators.pt"
    save_translators(path, translators, config)

    restored, restored_config = load_translators(path, device="cpu", dtype=torch.float32)
    assert restored_config == config
    assert torch.allclose(restored[1].delta.bias, torch.full((8,), 0.25))


def test_batched_covers_every_item_exactly_once() -> None:
    chunks = list(batched(list(range(7)), 3))
    assert [len(c) for c in chunks] == [3, 3, 1]
    assert [x for chunk in chunks for x in chunk] == list(range(7))


def test_weight_decay_pulls_translators_toward_identity_not_toward_zero() -> None:
    # A full matrix initialised to I would be decayed away from the identity, which
    # destroys the upper layers that are already correct. Storing the deviation makes
    # decay pull back to identity instead.
    translators = build_translators(_config(), device="cpu", dtype=torch.float32)
    optimizer = torch.optim.AdamW(translators.parameters(), lr=0.0, weight_decay=0.5)
    states = torch.randn(2, 3, 8)
    loss = translators[0](states).sum()
    loss.backward()
    optimizer.step()

    assert torch.allclose(translators[0](states), states, atol=1e-6)
