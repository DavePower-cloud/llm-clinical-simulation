from __future__ import annotations

import argparse
from pathlib import Path

from .simulation import SimulationRunner


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run multi-agent LLM clinical conversation simulations."
    )
    parser.add_argument(
        "--batch",
        type=int,
        default=0,
        help="Number of simulations to run in batch mode. If omitted or 0, runs one simulation.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print generated conversation turns to stdout.",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    runner = SimulationRunner()

    if args.batch and args.batch > 0:
        paths = runner.run_batch(num_runs=args.batch, verbose=args.verbose)
        print("\nSaved files:")
        for path in paths:
            print(f" - {path}")
    else:
        path = runner.run_and_save(verbose=args.verbose)
        print(f"\nConversation saved to: {path}")


if __name__ == "__main__":
    main()
