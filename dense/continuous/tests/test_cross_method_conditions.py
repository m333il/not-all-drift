from argparse import Namespace
from pathlib import Path

from scripts.sae.analyze_civil_comments_sae_all_layer_similarity import parse_run_spec
from scripts.sae.evaluate_cross_method_dense_carrier_screen import (
    condition_namespace,
    method_names,
)


def _args(gepa_prompt_file: Path | None) -> Namespace:
    return Namespace(
        prompt_run_dir=Path("prompt-run"),
        prefix_run_dir=Path("prefix-run"),
        gepa_prompt_file=gepa_prompt_file,
    )


def test_gepa_is_opt_in_for_cross_method_comparisons() -> None:
    assert method_names(_args(None)) == ("prompt", "prefix")
    assert method_names(_args(Path("gepa.txt"))) == ("prompt", "prefix", "gepa")


def test_gepa_uses_the_frozen_text_condition_loader() -> None:
    namespace = condition_namespace(_args(Path("gepa.txt")), "gepa")
    assert namespace.condition == "manual"
    assert namespace.run_dir is None
    assert namespace.prompt_file == Path("gepa.txt")


def test_all_layer_similarity_accepts_gepa_runs() -> None:
    spec = parse_run_spec("gepa:42=/tmp/gepa-s42")
    assert spec.method == "gepa"
    assert spec.seed == 42
    assert spec.directory == Path("/tmp/gepa-s42")
