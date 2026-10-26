from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .btc import BTCChordDetector, save_lab, smooth_chord_intervals
from .features import aggregate_melody_by_chord, melody_pitch_track
from .separation import VocalSeparator


def assign_work_splits(manifest: pd.DataFrame, *, seed: int = 42) -> dict[str, str]:
    works = sorted(manifest.work_id.astype(str).unique())
    if len(works) < 3:
        return {work: "train" for work in works}
    rng = np.random.default_rng(seed)
    shuffled = list(rng.permutation(works))
    validation_count = max(1, round(len(works) * 0.1))
    test_count = max(1, round(len(works) * 0.1))
    validation = set(shuffled[:validation_count])
    test = set(shuffled[validation_count:validation_count + test_count])
    return {
        work: "validation" if work in validation else "test" if work in test else "train"
        for work in works
    }


def _resolve_audio_path(value: str, manifest_path: Path) -> Path:
    path = Path(value).expanduser()
    if path.is_absolute():
        return path
    project_candidate = Path.cwd() / path
    if project_candidate.exists():
        return project_candidate.resolve()
    return (manifest_path.parent / path).resolve()


def _save_melody(melody: pd.DataFrame, path: Path) -> None:
    np.savez_compressed(path, **{column: melody[column].to_numpy() for column in melody.columns})


def _save_segment_features(segments: pd.DataFrame, path: Path) -> None:
    np.savez_compressed(
        path,
        start=segments.start.to_numpy(dtype=np.float32),
        end=segments.end.to_numpy(dtype=np.float32),
        chord=segments.chord.to_numpy(dtype=str),
        melody_profile=np.stack(segments.melody_profile),
        sustained_pitch_classes=np.stack(segments.sustained_pitch_classes),
        voiced_frames=segments.voiced_frames.to_numpy(dtype=np.int32),
    )


def preprocess_manifest(
    manifest_path: str | Path,
    output_root: str | Path,
    *,
    work_limit: int | None = None,
    max_duration: float | None = None,
    force: bool = False,
    seed: int = 42,
) -> dict[str, object]:
    """Preprocess a resumable subset of a manifest into model-ready cached features."""
    manifest_path = Path(manifest_path).expanduser().resolve()
    output_root = Path(output_root).expanduser().resolve()
    manifest = pd.read_csv(manifest_path, dtype=str).fillna("")
    required = {"track_id", "work_id", "audio_path", "sha256", "status"}
    missing = required - set(manifest.columns)
    if missing:
        raise ValueError(f"Manifest is missing columns: {sorted(missing)}")
    manifest = manifest[manifest.status == "ready"].copy()
    splits = assign_work_splits(manifest, seed=seed)
    selected_works = sorted(manifest.work_id.unique())
    if work_limit is not None:
        if work_limit < 1:
            raise ValueError("work_limit must be positive")
        selected_works = selected_works[:work_limit]
    manifest = manifest[manifest.work_id.isin(selected_works)]

    output_root.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        [{"work_id": work, "split": split} for work, split in sorted(splits.items())]
    ).to_csv(output_root / "splits.csv", index=False)

    detector: BTCChordDetector | None = None
    separator: VocalSeparator | None = None
    completed = skipped = failed = 0
    errors: list[dict[str, str]] = []
    windows: list[dict[str, object]] = []

    total_tracks = len(manifest)
    for position, row in enumerate(manifest.itertuples(index=False), start=1):
        print(f"[{position}/{total_tracks}] {row.track_id} ({row.work_id})", flush=True)
        track_dir = output_root / str(row.track_id)
        metadata_path = track_dir / "metadata.json"
        signature = {
            "sha256": str(row.sha256),
            "max_duration": max_duration,
            "pipeline_version": 1,
        }
        if metadata_path.exists() and not force:
            old = json.loads(metadata_path.read_text())
            if old.get("signature") == signature and old.get("status") == "complete":
                print("  cached: complete", flush=True)
                skipped += 1
                processed_duration = float(old["processed_duration"])
                for start in np.arange(0, processed_duration, 30.0):
                    windows.append({
                        "track_id": row.track_id, "work_id": row.work_id,
                        "split": splits[str(row.work_id)], "start": float(start),
                        "end": float(min(start + 30, processed_duration)),
                    })
                continue
        track_dir.mkdir(parents=True, exist_ok=True)
        audio_path = _resolve_audio_path(str(row.audio_path), manifest_path)
        try:
            print("  BTC: detecting chords", flush=True)
            if detector is None:
                detector = BTCChordDetector()
            raw_chords = detector.detect(audio_path, duration=max_duration)
            chords = smooth_chord_intervals(raw_chords)
            save_lab(raw_chords, track_dir / "chords.raw.lab")
            save_lab(chords, track_dir / "chords.lab")

            print("  HDemucs: separating vocals", flush=True)
            if separator is None:
                separator = VocalSeparator()
            vocals_path = separator.separate(
                audio_path, track_dir / "vocals.wav", duration=max_duration
            )
            processed_duration = float(chords.end.max())
            print("  CREPE: extracting melody", flush=True)
            melody = melody_pitch_track(vocals_path, duration=processed_duration)
            _save_melody(melody, track_dir / "melody.npz")
            segments = aggregate_melody_by_chord(chords, melody)
            _save_segment_features(segments, track_dir / "features.npz")

            metadata = {
                "track_id": row.track_id,
                "work_id": row.work_id,
                "split": splits[str(row.work_id)],
                "audio_path": str(row.audio_path),
                "processed_duration": processed_duration,
                "chord_intervals": len(chords),
                "voiced_frames": int(segments.voiced_frames.sum()),
                "signature": signature,
                "status": "complete",
            }
            metadata_path.write_text(json.dumps(metadata, indent=2))
            print("  complete", flush=True)
            for start in np.arange(0, processed_duration, 30.0):
                windows.append({
                    "track_id": row.track_id, "work_id": row.work_id,
                    "split": splits[str(row.work_id)], "start": float(start),
                    "end": float(min(start + 30, processed_duration)),
                })
            completed += 1
        except Exception as error:
            print(f"  failed: {error!r}", flush=True)
            failed += 1
            errors.append({"track_id": str(row.track_id), "error": repr(error)})
            metadata_path.write_text(json.dumps({
                "track_id": row.track_id, "work_id": row.work_id,
                "signature": signature, "status": "failed", "error": repr(error),
            }, indent=2))

    pd.DataFrame(windows).to_csv(output_root / "windows.csv", index=False)
    summary: dict[str, object] = {
        "selected_works": len(selected_works),
        "selected_tracks": len(manifest),
        "completed": completed,
        "skipped_cached": skipped,
        "failed": failed,
        "errors": errors,
    }
    (output_root / "summary.json").write_text(json.dumps(summary, indent=2))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Batch preprocess the Reharmonize audio dataset")
    parser.add_argument("--manifest", type=Path, default=Path("data/manifest.csv"))
    parser.add_argument("--output-root", type=Path, default=Path("data/processed"))
    parser.add_argument("--work-limit", type=int)
    parser.add_argument("--max-duration", type=float)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    result = preprocess_manifest(
        args.manifest, args.output_root, work_limit=args.work_limit,
        max_duration=args.max_duration, force=args.force,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
