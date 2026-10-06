"""Command line entry point for generating the v2 Civil Comments splits."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from .contract import BINARY_CONTRACT, MULTILABEL_CONTRACT, SplitContractV2
from .generator import generate_splits_v2

logger = logging.getLogger(__name__)

CONTRACTS: dict[str, SplitContractV2] = {
    "multilabel": MULTILABEL_CONTRACT,
    "binary": BINARY_CONTRACT,
}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate the v2 Civil Comments splits")
    parser.add_argument("--train-parquet", action="append", type=Path, required=True)
    parser.add_argument("--test-parquet", action="append", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument(
        "--setup",
        action="append",
        choices=sorted(CONTRACTS),
        help="repeatable; defaults to both setups",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    setups = args.setup or sorted(CONTRACTS)
    for setup in setups:
        contract = CONTRACTS[setup]
        output = args.output_root / f"civil_comments_splits_v2_{setup}"
        logger.info("generating the %s setup into %s", setup, output)
        manifest = generate_splits_v2(
            tuple(args.train_parquet), tuple(args.test_parquet), output, contract
        )
        shares = manifest["splits"]["test"]["label_share_of_positives"]
        logger.info(
            "%s: %d splits, test label shares %s",
            setup,
            len(manifest["splits"]),
            {label: round(share, 4) for label, share in shares.items()},
        )


if __name__ == "__main__":
    main()
