from __future__ import annotations

import argparse
import json
from pathlib import Path

from .pretraining import calculate_pretraining_baselines


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate simple masked-chord baselines")
    parser.add_argument("--dataset-root", type=Path, default=Path("data/pretraining"))
    parser.add_argument("--split", choices=["validation", "test"], default="test")
    args = parser.parse_args()
    print(json.dumps(
        calculate_pretraining_baselines(args.dataset_root, split=args.split), indent=2
    ))


if __name__ == "__main__":
    main()
