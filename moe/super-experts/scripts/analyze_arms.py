#!/usr/bin/env python3
"""Compact report over a finished arms run, printed rather than copied.

The per-example placement files are long, so this reduces them to the quantities
the comparison is about: for each arm, corpus and tracked Super Expert, how large its
maximum was, where it landed, and what the gate scored it there.

Positions are reported raw. A distribution clustered at small indices means
something different from one spread across the sequence, and collapsing it to a
single "fires at the sink" rate would hide exactly that.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import median


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True, help="Run directory containing arms/")
    parser.add_argument("--corpora", default="civil,wikitext2")
    return parser.parse_args()


def mean(values):
    values = [value for value in values if value is not None]
    return sum(values) / len(values) if values else None


def show(value, spec=".3g"):
    return "-" if value is None else format(value, spec)


def main() -> None:
    args = parse_args()
    arms_dir = args.run / "arms"
    summaries = json.loads((arms_dir / "summary.json").read_text())
    order = [summary["arm"] for summary in summaries if summary["corpus"] == summaries[0]["corpus"]]

    for corpus in args.corpora.split(","):
        rows = [summary for summary in summaries if summary["corpus"] == corpus]
        if not rows:
            continue
        print(f"\n########## corpus: {corpus} ##########", flush=True)
        for summary in sorted(rows, key=lambda row: order.index(row["arm"])):
            label = f"{summary['arm']}@{corpus}"
            placement = [json.loads(line) for line in
                         (arms_dir / f"{label}.placement.jsonl").read_text().splitlines() if line.strip()]
            identified = {(row["layer"], row["expert"]) for row in summary["super_experts"]}
            profile = json.loads((arms_dir / f"{label}.profile.json").read_text())
            maxima = [row["output_max"] for row in profile["records"]]
            top = max(profile["records"], key=lambda row: row["output_max"])
            print(f"\n--- {summary['arm']}  offset={summary['virtual_offset']}  "
                  f"examples={summary['examples']}  criterion_super_experts={len(identified)} "
                  f"{sorted(identified)}", flush=True)
            # The largest value anywhere is what says whether massive activations
            # still exist under this arm at all, as opposed to having moved expert.
            print(f"    all experts: max={top['output_max']:.4g} at L{top['layer']}E{top['expert']}"
                  f" pos={top['position']} | median over routed experts={median(maxima):.4g}"
                  f" | experts routed={len(maxima)}", flush=True)
            for key, record in summary["tracked"].items():
                layer, expert = (int(part) for part in key.split(":"))
                mine = [row for row in placement if (row["layer"], row["expert"]) == (layer, expert)]
                positions = [row["argmax_position"] for row in mine]
                print(
                    f"  L{layer}E{expert}"
                    f" in_criterion={(layer, expert) in identified}"
                    f" corpus_max={show(record['output_max'] if record else None, '.4g')}"
                    f" at_pos={record['position'] if record else '-'}"
                    f" hits={record['hits'] if record else 0}"
                    f" routed_examples={len(mine)}", flush=True)
                if not mine:
                    continue
                print(
                    f"      argmax position: min={min(positions)} median={median(positions):g}"
                    f" max={max(positions)}"
                    f" at_stream_start={sum(row['argmax_at_stream_start'] for row in mine) / len(mine):.2f}"
                    f" at_chat_start={sum(row['argmax_at_chat_start'] for row in mine) / len(mine):.2f}",
                    flush=True)
                print(
                    f"      activation: max={show(mean(row['activation_at_max'] for row in mine), '.4g')}"
                    f" stream_start={show(mean(row['activation_at_stream_start'] for row in mine), '.4g')}"
                    f" chat_start={show(mean(row['activation_at_chat_start'] for row in mine), '.4g')}"
                    f" elsewhere={show(mean(row['activation_max_elsewhere'] for row in mine), '.4g')}"
                    " (means over examples)", flush=True)
                print(
                    f"      router p: stream_start={show(mean(row['router_probability_stream_start'] for row in mine))}"
                    f" chat_start={show(mean(row['router_probability_chat_start'] for row in mine))}"
                    f" selected_at_stream_start="
                    f"{show(mean(float(row['router_selected_stream_start']) for row in mine if row['router_selected_stream_start'] is not None), '.2f')}"
                    f" positions_routed_mean={show(mean(row['positions_routed'] for row in mine), '.4g')}",
                    flush=True)
    print("\nANALYSIS_COMPLETE", flush=True)


if __name__ == "__main__":
    main()
