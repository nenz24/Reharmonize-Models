from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def _read_chords(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, sep=r"\s+", names=["start", "end", "chord"])


def build_quality_report(
    manifest_path: str | Path,
    processed_root: str | Path,
    output_path: str | Path,
) -> dict[str, object]:
    """Audit cached perception features before model training."""
    manifest_path = Path(manifest_path).expanduser().resolve()
    processed_root = Path(processed_root).expanduser().resolve()
    output_path = Path(output_path).expanduser().resolve()
    manifest = pd.read_csv(manifest_path)
    rows: list[dict[str, object]] = []

    for track in manifest.itertuples(index=False):
        track_dir = processed_root / str(track.track_id)
        metadata_path = track_dir / "metadata.json"
        chords_path = track_dir / "chords.lab"
        features_path = track_dir / "features.npz"
        issues: list[str] = []
        if not metadata_path.is_file() or not chords_path.is_file() or not features_path.is_file():
            rows.append({
                "track_id": track.track_id, "work_id": track.work_id,
                "title": getattr(track, "title", ""), "split": "",
                "processed_duration": np.nan, "manifest_duration": getattr(track, "duration_seconds", np.nan),
                "duration_difference": np.nan, "chord_intervals": 0,
                "chord_changes_per_minute": np.nan, "no_chord_ratio": np.nan,
                "voiced_fraction": np.nan, "review": True, "issues": "missing_cache",
            })
            continue

        metadata = json.loads(metadata_path.read_text())
        chords = _read_chords(chords_path)
        features = np.load(features_path)
        duration = float(metadata["processed_duration"])
        manifest_duration = float(getattr(track, "duration_seconds", duration))
        durations = chords.end.to_numpy() - chords.start.to_numpy()
        no_chord = chords.chord.astype(str).isin(["N", "X"]).to_numpy()
        no_chord_ratio = float(durations[no_chord].sum() / duration) if duration else 0.0
        chord_changes_per_minute = max(len(chords) - 1, 0) / max(duration / 60.0, 1e-8)
        voiced_frames = int(np.asarray(features["voiced_frames"]).sum())
        voiced_fraction = voiced_frames / max(duration * 100.0, 1.0)
        duration_difference = abs(duration - manifest_duration)

        if duration_difference > 2.0:
            issues.append("duration_mismatch")
        if no_chord_ratio > 0.4:
            issues.append("high_no_chord_ratio")
        if voiced_fraction < 0.03:
            issues.append("low_vocal_evidence")
        if chord_changes_per_minute > 45:
            issues.append("excessive_chord_changes")
        if chord_changes_per_minute < 1:
            issues.append("sparse_chord_changes")
        rows.append({
            "track_id": track.track_id,
            "work_id": track.work_id,
            "title": getattr(track, "title", ""),
            "split": metadata.get("split", ""),
            "processed_duration": round(duration, 3),
            "manifest_duration": round(manifest_duration, 3),
            "duration_difference": round(duration_difference, 3),
            "chord_intervals": len(chords),
            "chord_changes_per_minute": round(chord_changes_per_minute, 3),
            "no_chord_ratio": round(no_chord_ratio, 4),
            "voiced_fraction": round(voiced_fraction, 4),
            "review": bool(issues),
            "issues": ";".join(issues),
        })

    report = pd.DataFrame(rows)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    report.to_csv(output_path, index=False)
    split_summary = (
        report.groupby("split").agg(tracks=("track_id", "size"), hours=("processed_duration", lambda x: round(x.sum() / 3600, 3)))
        .reset_index().to_dict("records")
        if not report.empty else []
    )
    summary: dict[str, object] = {
        "tracks_expected": len(manifest),
        "tracks_checked": len(report),
        "passed": int((~report.review).sum()) if not report.empty else 0,
        "needs_review": int(report.review.sum()) if not report.empty else 0,
        "missing_cache": int(report.issues.str.contains("missing_cache").sum()) if not report.empty else 0,
        "split_summary": split_summary,
        "report": str(output_path),
    }
    (output_path.parent / f"{output_path.stem}-summary.json").write_text(json.dumps(summary, indent=2))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit processed Reharmonize dataset features")
    parser.add_argument("--manifest", type=Path, default=Path("data/manifest.csv"))
    parser.add_argument("--processed-root", type=Path, default=Path("data/processed"))
    parser.add_argument("--output", type=Path, default=Path("data/qc.csv"))
    args = parser.parse_args()
    print(json.dumps(build_quality_report(args.manifest, args.processed_root, args.output), indent=2))


if __name__ == "__main__":
    main()
