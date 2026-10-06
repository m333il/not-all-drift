"""Keep saved-artifact checks optional in the source-only distribution."""
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SPLITS = ("data/civil_v2_val_seed42_n200.jsonl", "data/civil_v2_test_head2000.jsonl")
ARTIFACT_INPUTS = {
    "test_the_shipped_gepa_instructions_pass": ("prompts",),
    "test_final_reserved_data_counts_and_prior_exclusions": SPLITS,
    "test_frozen_data_and_matched_checkpoint_spec": SPLITS,
    "test_frozen_objects_authenticate_and_modified_file_is_rejected": (
        "data/prefill_template_frozen_20260924",),
    "test_fresh_role_confirmation_rows_are_disjoint": SPLITS,
    "test_frozen_confirmation_ids_and_previous_interventions": SPLITS,
    "test_frozen_rows_exclude_prior_eval_and_validation": SPLITS,
    "test_wording_rows_are_disclosed_subset_of_confirmation": (
        "data/civil_v2_test_head2000.jsonl",),
    "test_real_frozen_split_is_disjoint_and_overlap_rejected": SPLITS,
    "test_existing_wording_render_gate_is_reused_and_no_input_slice_changes": (
        "data/prefill_wording_20260924.json",),
}


def pytest_collection_modifyitems(items):
    for item in items:
        required = ARTIFACT_INPUTS.get(item.originalname or item.name, ())
        missing = [name for name in required if not (ROOT / name).exists()]
        if missing:
            item.add_marker(pytest.mark.skip(
                reason="External research artifacts must be regenerated locally: "
                + ", ".join(missing)))
