import importlib.util
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]


def test_score_stats_pairs_rows_by_key(tmp_path, monkeypatch, capsys):
    score = tmp_path / "score"
    score.mkdir()
    rows = [
        {"arm": "base", "condition": "intact", "key": "a", "score": 1.0},
        {"arm": "base", "condition": "intact", "key": "b", "score": 0.5},
        {"arm": "base", "condition": "ablated", "key": "b", "score": 0.0},
        {"arm": "base", "condition": "ablated", "key": "a", "score": 0.5},
    ]
    (score / "rows.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
    spec = importlib.util.spec_from_file_location("analyze_score", ROOT / "scripts" / "analyze_score.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(sys, "argv", ["analyze_score.py", "--run", str(tmp_path), "--bootstrap", "100"])
    module.main()
    result = json.loads(capsys.readouterr().out.removeprefix("SCORE_STATS="))[0]
    assert result["n"] == 2
    assert result["paired_delta"] == -0.5
    assert result["bootstrap_95"] == [-0.5, -0.5]
    assert result["worsened"] == 2
