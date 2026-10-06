import json

import pytest

from mrd.jsonl import read_jsonl


def test_unicode_separators_are_content_not_record_boundaries(tmp_path):
    rows = [{"key": "first", "prompt": "a\u2028b\u2029c\x85d\nnext"}, {"key": "second"}]
    path = tmp_path / "examples.jsonl"
    path.write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in rows), encoding="utf-8")
    assert read_jsonl(path) == rows


def test_malformed_record_is_not_silently_skipped(tmp_path):
    path = tmp_path / "examples.jsonl"
    path.write_text('{"key": "first"}\n{"key":\n', encoding="utf-8")
    with pytest.raises(json.JSONDecodeError):
        read_jsonl(path)
