from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from .pretraining import MaskedChordModel

_ROOT_NAMES = ("C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B")


def _symbol(root: int, quality: str) -> str:
    if root == 12:
        return "N"
    name = _ROOT_NAMES[root]
    return name if quality == "maj" else f"{name}:{quality}"


def _valid_candidates(qualities: list[str]) -> list[tuple[int, int, str]]:
    candidates: list[tuple[int, int, str]] = []
    for root in range(13):
        for quality_id, quality in enumerate(qualities):
            if (root == 12) != (quality == "N"):
                continue
            candidates.append((root, quality_id, _symbol(root, quality)))
    return candidates


def generate_chord_suggestions(
    dataset_root: str | Path,
    checkpoint_path: str | Path,
    output_path: str | Path,
    *,
    track_id: str,
    top_k: int = 5,
    include_source: bool = False,
    device: str = "auto",
) -> dict[str, object]:
    """Generate reviewable chord alternatives for every segment of one cached track."""
    if top_k < 1:
        raise ValueError("top_k must be positive")
    dataset_root = Path(dataset_root).expanduser().resolve()
    checkpoint_path = Path(checkpoint_path).expanduser().resolve()
    output_path = Path(output_path).expanduser().resolve()
    vocabulary = json.loads((dataset_root / "vocabulary.json").read_text())
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if checkpoint.get("vocabulary") != vocabulary:
        raise ValueError("Checkpoint vocabulary does not match this dataset")
    if device == "auto":
        device = (
            "mps" if torch.backends.mps.is_available()
            else "cuda" if torch.cuda.is_available()
            else "cpu"
        )
    torch_device = torch.device(device)
    model = MaskedChordModel(
        len(vocabulary["qualities"]), int(vocabulary["sequence_length"])
    ).to(torch_device)
    model.load_state_dict(checkpoint["model"])
    model.eval()

    selected: dict[str, list[np.ndarray]] = {}
    selected_split: str | None = None
    for split in ("train", "validation", "test"):
        path = dataset_root / f"{split}.npz"
        if not path.exists():
            continue
        with np.load(path) as values:
            mask = values["track_id"] == track_id
            if not np.any(mask):
                continue
            selected_split = split
            for name in (
                "root_input", "quality_input", "melody", "scalars",
                "target_root", "target_quality", "work_id", "start",
            ):
                selected.setdefault(name, []).append(values[name][mask])
    if selected_split is None:
        raise ValueError(f"track_id {track_id!r} was not found in the pretraining dataset")
    arrays = {name: np.concatenate(parts) for name, parts in selected.items()}
    qualities = list(vocabulary["qualities"])
    candidates = _valid_candidates(qualities)
    candidate_roots = np.asarray([item[0] for item in candidates], dtype=np.int64)
    candidate_qualities = np.asarray([item[1] for item in candidates], dtype=np.int64)

    with torch.inference_mode():
        root_logits, quality_logits = model(
            torch.from_numpy(arrays["root_input"]).long().to(torch_device),
            torch.from_numpy(arrays["quality_input"]).long().to(torch_device),
            torch.from_numpy(arrays["melody"]).float().to(torch_device),
            torch.from_numpy(arrays["scalars"]).float().to(torch_device),
        )
        root_probability = root_logits.softmax(dim=1).cpu().numpy()
        quality_probability = quality_logits.softmax(dim=1).cpu().numpy()

    rows: list[dict[str, object]] = []
    for index in range(len(arrays["target_root"])):
        source = _symbol(
            int(arrays["target_root"][index]),
            qualities[int(arrays["target_quality"][index])],
        )
        probability = (
            root_probability[index, candidate_roots]
            * quality_probability[index, candidate_qualities]
        )
        probability /= max(float(probability.sum()), 1e-12)
        order = np.argsort(probability)[::-1]
        rank = 0
        for candidate_index in order:
            candidate = candidates[int(candidate_index)][2]
            if not include_source and candidate == source:
                continue
            rank += 1
            rows.append({
                "track_id": track_id,
                "work_id": str(arrays["work_id"][index]),
                "split": selected_split,
                "start": round(float(arrays["start"][index]), 3),
                "source_chord": source,
                "rank": rank,
                "candidate_chord": candidate,
                "probability": float(probability[int(candidate_index)]),
            })
            if rank == top_k:
                break
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(output_path, index=False)
    return {
        "track_id": track_id,
        "split": selected_split,
        "segments": len(arrays["target_root"]),
        "suggestions": len(rows),
        "top_k": top_k,
        "include_source": include_source,
        "checkpoint": str(checkpoint_path),
        "output": str(output_path),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate top-k masked-chord alternatives")
    parser.add_argument("--dataset-root", type=Path, default=Path("data/pretraining"))
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--track-id", required=True)
    parser.add_argument("--output", type=Path, default=Path("artifacts/chord-suggestions.csv"))
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--include-source", action="store_true")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "mps", "cuda"])
    args = parser.parse_args()
    summary = generate_chord_suggestions(
        args.dataset_root, args.checkpoint, args.output, track_id=args.track_id,
        top_k=args.top_k, include_source=args.include_source, device=args.device,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
