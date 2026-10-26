from __future__ import annotations

import argparse
import json
from pathlib import Path

from .pretraining import calculate_pretraining_baselines, evaluate_masked_chord_checkpoint


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate simple masked-chord baselines")
    parser.add_argument("--dataset-root", type=Path, default=Path("data/pretraining"))
    parser.add_argument("--split", choices=["validation", "test"], default="test")
    parser.add_argument("--checkpoint", type=Path)
    args = parser.parse_args()
    result = (
        evaluate_masked_chord_checkpoint(
            args.dataset_root, args.checkpoint, split=args.split
        )
        if args.checkpoint else
        calculate_pretraining_baselines(args.dataset_root, split=args.split)
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
