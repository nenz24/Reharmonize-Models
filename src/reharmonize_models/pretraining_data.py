from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .features import chord_root_pitch_class

NO_CHORD_ROOT = 12
PAD_ROOT = 13
MASK_ROOT = 14


def chord_quality(symbol: str) -> str:
    """Return a stable quality label while ignoring slash-bass inversions."""
    symbol = str(symbol).strip()
    if symbol in {"", "N", "X"}:
        return "N"
    head = symbol.split("/", 1)[0]
    cursor = 1
    if len(head) > 1 and head[1] in {"#", "b"}:
        cursor = 2
    quality = head[cursor:]
    if quality.startswith(":"):
        quality = quality[1:]
    return quality or "maj"


def _read_track_features(path: Path) -> dict[str, np.ndarray]:
    with np.load(path) as values:
        return {name: values[name] for name in values.files}


def _track_records(manifest: pd.DataFrame, processed_root: Path) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    for row in manifest.itertuples(index=False):
        track_dir = processed_root / str(row.track_id)
        metadata_path = track_dir / "metadata.json"
        features_path = track_dir / "features.npz"
        if not metadata_path.is_file() or not features_path.is_file():
            continue
        metadata = json.loads(metadata_path.read_text())
        if metadata.get("status") != "complete":
            continue
        features = _read_track_features(features_path)
        required = {"start", "end", "chord", "melody_profile", "voiced_frames"}
        missing = required - features.keys()
        if missing:
            raise ValueError(f"{features_path} is missing arrays: {sorted(missing)}")
        length = len(features["chord"])
        if any(len(features[name]) != length for name in required):
            raise ValueError(f"Feature arrays have different lengths in {features_path}")
        records.append({
            "track_id": str(row.track_id),
            "work_id": str(row.work_id),
            "split": str(metadata["split"]),
            "features": features,
        })
    return records


def build_pretraining_dataset(
    manifest_path: str | Path,
    processed_root: str | Path,
    output_root: str | Path,
    *,
    sequence_length: int = 9,
) -> dict[str, object]:
    """Create masked-chord sequences from cached BTC and vocal-melody features."""
    if sequence_length < 3 or sequence_length % 2 == 0:
        raise ValueError("sequence_length must be an odd integer of at least 3")
    manifest_path = Path(manifest_path).expanduser().resolve()
    processed_root = Path(processed_root).expanduser().resolve()
    output_root = Path(output_root).expanduser().resolve()
    manifest = pd.read_csv(manifest_path, dtype=str).fillna("")
    required_columns = {"track_id", "work_id", "status"}
    missing_columns = required_columns - set(manifest.columns)
    if missing_columns:
        raise ValueError(f"Manifest is missing columns: {sorted(missing_columns)}")
    manifest = manifest[manifest.status == "ready"]
    records = _track_records(manifest, processed_root)
    if not records:
        raise ValueError("No complete processed tracks were found")

    qualities = sorted({
        chord_quality(symbol)
        for record in records
        for symbol in record["features"]["chord"]
    })
    quality_to_id = {quality: index for index, quality in enumerate(qualities)}
    pad_quality = len(qualities)
    mask_quality = len(qualities) + 1
    radius = sequence_length // 2
    work_examples: dict[str, int] = {}
    for record in records:
        work = str(record["work_id"])
        work_examples[work] = work_examples.get(work, 0) + len(record["features"]["chord"])
    total_examples = sum(work_examples.values())
    work_count = len(work_examples)

    split_rows: dict[str, dict[str, list[object]]] = {}
    for record in records:
        split = str(record["split"])
        rows = split_rows.setdefault(split, {
            "root_input": [], "quality_input": [], "melody": [], "scalars": [],
            "target_root": [], "target_quality": [], "sample_weight": [],
            "track_id": [], "work_id": [], "start": [],
        })
        features = record["features"]
        symbols = [str(value) for value in features["chord"]]
        roots = [
            NO_CHORD_ROOT if chord_root_pitch_class(symbol) is None
            else int(chord_root_pitch_class(symbol))
            for symbol in symbols
        ]
        quality_ids = [quality_to_id[chord_quality(symbol)] for symbol in symbols]
        durations = np.maximum(
            np.asarray(features["end"], dtype=np.float32)
            - np.asarray(features["start"], dtype=np.float32),
            1e-3,
        )
        voiced = np.asarray(features["voiced_frames"], dtype=np.float32)
        melody_profiles = np.asarray(features["melody_profile"], dtype=np.float32)
        if melody_profiles.shape != (len(symbols), 12):
            raise ValueError(f"Expected 12-bin melody profiles for {record['track_id']}")

        for target_index in range(len(symbols)):
            root_sequence = np.full(sequence_length, PAD_ROOT, dtype=np.int64)
            quality_sequence = np.full(sequence_length, pad_quality, dtype=np.int64)
            melody_sequence = np.zeros((sequence_length, 12), dtype=np.float32)
            scalar_sequence = np.zeros((sequence_length, 2), dtype=np.float32)
            for position, source_index in enumerate(
                range(target_index - radius, target_index + radius + 1)
            ):
                if source_index < 0 or source_index >= len(symbols):
                    continue
                root_sequence[position] = roots[source_index]
                quality_sequence[position] = quality_ids[source_index]
                melody_sequence[position] = melody_profiles[source_index]
                scalar_sequence[position, 0] = np.clip(np.log1p(durations[source_index]) / np.log(11), 0, 1)
                scalar_sequence[position, 1] = np.clip(
                    voiced[source_index] / (durations[source_index] * 100.0), 0, 1
                )
            root_sequence[radius] = MASK_ROOT
            quality_sequence[radius] = mask_quality
            rows["root_input"].append(root_sequence)
            rows["quality_input"].append(quality_sequence)
            rows["melody"].append(melody_sequence)
            rows["scalars"].append(scalar_sequence)
            rows["target_root"].append(roots[target_index])
            rows["target_quality"].append(quality_ids[target_index])
            rows["sample_weight"].append(
                total_examples / (work_count * work_examples[str(record["work_id"])])
            )
            rows["track_id"].append(str(record["track_id"]))
            rows["work_id"].append(str(record["work_id"]))
            rows["start"].append(float(features["start"][target_index]))

    output_root.mkdir(parents=True, exist_ok=True)
    split_summary: list[dict[str, object]] = []
    for split, rows in sorted(split_rows.items()):
        arrays = {
            "root_input": np.stack(rows["root_input"]),
            "quality_input": np.stack(rows["quality_input"]),
            "melody": np.stack(rows["melody"]),
            "scalars": np.stack(rows["scalars"]),
            "target_root": np.asarray(rows["target_root"], dtype=np.int64),
            "target_quality": np.asarray(rows["target_quality"], dtype=np.int64),
            "sample_weight": np.asarray(rows["sample_weight"], dtype=np.float32),
            "track_id": np.asarray(rows["track_id"], dtype=str),
            "work_id": np.asarray(rows["work_id"], dtype=str),
            "start": np.asarray(rows["start"], dtype=np.float32),
        }
        np.savez_compressed(output_root / f"{split}.npz", **arrays)
        split_summary.append({
            "split": split,
            "examples": len(rows["target_root"]),
            "tracks": len(set(rows["track_id"])),
            "works": len(set(rows["work_id"])),
        })

    vocabulary = {
        "qualities": qualities,
        "root_classes": 13,
        "no_chord_root": NO_CHORD_ROOT,
        "pad_root": PAD_ROOT,
        "mask_root": MASK_ROOT,
        "pad_quality": pad_quality,
        "mask_quality": mask_quality,
        "sequence_length": sequence_length,
        "feature_version": 1,
    }
    (output_root / "vocabulary.json").write_text(json.dumps(vocabulary, indent=2))
    summary: dict[str, object] = {
        "tracks": len(records),
        "works": len(work_examples),
        "examples": total_examples,
        "quality_classes": len(qualities),
        "splits": split_summary,
    }
    (output_root / "summary.json").write_text(json.dumps(summary, indent=2))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Build masked-chord pretraining examples")
    parser.add_argument("--manifest", type=Path, default=Path("data/manifest.csv"))
    parser.add_argument("--processed-root", type=Path, default=Path("data/processed"))
    parser.add_argument("--output-root", type=Path, default=Path("data/pretraining"))
    parser.add_argument("--sequence-length", type=int, default=9)
    args = parser.parse_args()
    summary = build_pretraining_dataset(
        args.manifest, args.processed_root, args.output_root,
        sequence_length=args.sequence_length,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
