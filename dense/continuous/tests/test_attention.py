import numpy as np
import torch

from prompt_optimization.attention import (
    aggregate_segment_mass,
    compute_prompt_spans,
    key_segment_ids,
    unmasked_position_ids,
)


def test_prompt_spans_partition_real_tokens_and_virtual_keys():
    prompt = "Task: classify Text: hello Answer:"
    offsets = [(0, 4), (4, 14), (15, 20), (20, 26), (26, 34)]
    spans = compute_prompt_spans(prompt, "hello", offsets, virtual_tokens=2)
    assert spans["virtual"] == [[0, 2]]
    assert spans["text"] == [[3, 4]]
    segment_ids = key_segment_ids(
        spans, virtual_tokens=2, left_padding=1, key_count=8
    )
    assert segment_ids.tolist() == [1, 1, -1, 0, 2, 2, 3, 2]


def test_segment_mass_and_position_ids():
    rows = np.asarray([[0.1, 0.2, 0.3, 0.4]], dtype=np.float32)
    segment_ids = np.asarray([0, 1, 2, 3], dtype=np.int8)
    assert np.allclose(aggregate_segment_mass(rows, segment_ids), rows)
    mask = torch.tensor([[0, 1, 1, 1]])
    assert unmasked_position_ids(mask).tolist() == [[0, 0, 1, 2]]
