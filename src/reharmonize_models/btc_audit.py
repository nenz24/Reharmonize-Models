from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .btc import BTCChordDetector


def _resolve_audio_path(value: str, manifest_path: Path) -> Path:
    path = Path(value).expanduser()
    if path.is_absolute():
        return path
    project_candidate = Path.cwd() / path
    if project_candidate.exists():
        return project_candidate.resolve()
    return (manifest_path.parent / path).resolve()


def audit_btc_predictions(
    manifest_path: str | Path,
    output_root: str | Path,
    *,
    track_ids: list[str] | None = None,
    duration: float | None = None,
) -> dict[str, object]:
    """Run BTC without other perception stages and retain uncertainty diagnostics."""
    manifest_path = Path(manifest_path).expanduser().resolve()
    output_root = Path(output_root).expanduser().resolve()
    manifest = pd.read_csv(manifest_path, dtype=str).fillna("")
    manifest = manifest[manifest.status == "ready"].copy()
    if track_ids:
        missing = sorted(set(track_ids) - set(manifest.track_id))
        if missing:
            raise ValueError(f"Unknown track IDs: {missing}")
        manifest = manifest[manifest.track_id.isin(track_ids)]
    if manifest.empty:
        raise ValueError("No ready tracks selected")

    output_root.mkdir(parents=True, exist_ok=True)
    detector = BTCChordDetector()
    rows: list[dict[str, object]] = []
    errors: list[dict[str, str]] = []
    total = len(manifest)
    for position, track in enumerate(manifest.itertuples(index=False), start=1):
        print(f"[{position}/{total}] {track.track_id}: BTC confidence audit", flush=True)
        try:
            chords = detector.detect_with_confidence(
                _resolve_audio_path(str(track.audio_path), manifest_path), duration=duration
            )
            chords.to_csv(output_root / f"{track.track_id}.csv", index=False)
            interval_duration = chords.end.to_numpy() - chords.start.to_numpy()
            total_duration = max(float(interval_duration.sum()), 1e-8)
            weights = interval_duration / total_duration
            uncertain = (chords.confidence < 0.25) | (chords.margin < 0.05)
            rows.append({
                "track_id": track.track_id,
                "work_id": track.work_id,
                "title": getattr(track, "title", ""),
                "duration": round(total_duration, 3),
                "intervals": len(chords),
                "mean_confidence": round(float(np.sum(chords.confidence * weights)), 4),
                "mean_margin": round(float(np.sum(chords.margin * weights)), 4),
                "mean_entropy": round(float(np.sum(chords.entropy * weights)), 4),
                "uncertain_time_ratio": round(float(np.sum(weights[uncertain])), 4),
            })
        except Exception as error:
            errors.append({"track_id": str(track.track_id), "error": repr(error)})
            print(f"  failed: {error!r}", flush=True)

    report = pd.DataFrame(rows)
    report.to_csv(output_root / "audit.csv", index=False)
    summary: dict[str, object] = {
        "selected_tracks": total,
        "completed": len(rows),
        "failed": len(errors),
        "errors": errors,
        "report": str(output_root / "audit.csv"),
        "segment_files": str(output_root / "track-*.csv"),
    }
    (output_root / "summary.json").write_text(json.dumps(summary, indent=2))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit BTC chord uncertainty without rerunning stems")
    parser.add_argument("--manifest", type=Path, default=Path("data/manifest.csv"))
    parser.add_argument("--output-root", type=Path, default=Path("data/btc-audit"))
    parser.add_argument("--track-id", action="append", dest="track_ids")
    parser.add_argument("--duration", type=float)
    args = parser.parse_args()
    print(json.dumps(audit_btc_predictions(
        args.manifest, args.output_root, track_ids=args.track_ids, duration=args.duration,
    ), indent=2))


if __name__ == "__main__":
    main()
