from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import librosa

from .btc import BTCChordDetector
from .features import chord_root_pitch_class


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
    compare_small_vocabulary: bool = False,
    compare_chroma: bool = False,
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
    small_detector = (
        BTCChordDetector(large_vocabulary=False) if compare_small_vocabulary else None
    )
    rows: list[dict[str, object]] = []
    errors: list[dict[str, str]] = []
    total = len(manifest)
    for position, track in enumerate(manifest.itertuples(index=False), start=1):
        print(f"[{position}/{total}] {track.track_id}: BTC confidence audit", flush=True)
        try:
            audio_path = _resolve_audio_path(str(track.audio_path), manifest_path)
            chords = detector.detect_with_confidence(
                audio_path, duration=duration
            )
            chords.to_csv(output_root / f"{track.track_id}.csv", index=False)
            interval_duration = chords.end.to_numpy() - chords.start.to_numpy()
            total_duration = max(float(interval_duration.sum()), 1e-8)
            weights = interval_duration / total_duration
            uncertain = (chords.confidence < 0.25) | (chords.margin < 0.05)
            row: dict[str, object] = {
                "track_id": track.track_id,
                "work_id": track.work_id,
                "title": getattr(track, "title", ""),
                "duration": round(total_duration, 3),
                "intervals": len(chords),
                "mean_confidence": round(float(np.sum(chords.confidence * weights)), 4),
                "mean_margin": round(float(np.sum(chords.margin * weights)), 4),
                "mean_entropy": round(float(np.sum(chords.entropy * weights)), 4),
                "uncertain_time_ratio": round(float(np.sum(weights[uncertain])), 4),
            }
            if small_detector is not None:
                small = small_detector.detect_with_confidence(
                    audio_path, duration=duration
                )
                small.to_csv(output_root / f"{track.track_id}.small.csv", index=False)
                comparison = compare_btc_vocabularies(chords, small)
                comparison.to_csv(
                    output_root / f"{track.track_id}.vocabulary-comparison.csv", index=False
                )
                comparison_duration = comparison.end - comparison.start
                comparison_weights = comparison_duration / max(
                    float(comparison_duration.sum()), 1e-8
                )
                agreement = comparison.root_agreement.to_numpy(dtype=bool)
                confident = (
                    (comparison.large_confidence >= 0.5)
                    & (comparison.small_confidence >= 0.5)
                ).to_numpy(dtype=bool)
                row.update({
                    "small_mean_confidence": round(float(np.sum(
                        comparison.small_confidence * comparison_weights
                    )), 4),
                    "root_agreement_ratio": round(float(np.sum(
                        comparison_weights[agreement]
                    )), 4),
                    "confident_disagreement_ratio": round(float(np.sum(
                        comparison_weights[confident & ~agreement]
                    )), 4),
                })
            if compare_chroma:
                chroma = validate_btc_with_chroma(audio_path, chords, duration=duration)
                chroma.to_csv(output_root / f"{track.track_id}.chroma.csv", index=False)
                pitched = chroma.large_root.notna() & chroma.chroma_root.notna()
                pitched_duration = (chroma.end - chroma.start).where(pitched, 0.0)
                pitched_total = max(float(pitched_duration.sum()), 1e-8)
                row.update({
                    "chroma_root_agreement_ratio": round(float(
                        pitched_duration[chroma.root_agreement.fillna(False)].sum()
                        / pitched_total
                    ), 4),
                    "chroma_mean_margin": round(float(
                        np.sum(chroma.chroma_margin * pitched_duration) / pitched_total
                    ), 4),
                })
            rows.append(row)
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
        "compared_small_vocabulary": compare_small_vocabulary,
        "compared_chroma": compare_chroma,
    }
    (output_root / "summary.json").write_text(json.dumps(summary, indent=2))
    return summary


def compare_btc_vocabularies(
    large: pd.DataFrame, small: pd.DataFrame,
) -> pd.DataFrame:
    """Align two interval streams and compare their root predictions."""
    rows: list[dict[str, object]] = []
    large_index = small_index = 0
    while large_index < len(large) and small_index < len(small):
        large_row = large.iloc[large_index]
        small_row = small.iloc[small_index]
        start = max(float(large_row.start), float(small_row.start))
        end = min(float(large_row.end), float(small_row.end))
        if end > start:
            large_root = chord_root_pitch_class(str(large_row.chord))
            small_root = chord_root_pitch_class(str(small_row.chord))
            rows.append({
                "start": start,
                "end": end,
                "large_chord": large_row.chord,
                "small_chord": small_row.chord,
                "large_confidence": float(large_row.confidence),
                "small_confidence": float(small_row.confidence),
                "large_root": large_root,
                "small_root": small_root,
                "root_agreement": large_root == small_root,
            })
        if float(large_row.end) <= float(small_row.end):
            large_index += 1
        if float(small_row.end) <= float(large_row.end):
            small_index += 1
    return pd.DataFrame(rows)


def validate_btc_with_chroma(
    audio_path: str | Path,
    chords: pd.DataFrame,
    *,
    duration: float | None = None,
    sample_rate: int = 22_050,
    hop_length: int = 512,
) -> pd.DataFrame:
    """Compare BTC roots with an independent major/minor chroma-template validator."""
    audio, _ = librosa.load(
        Path(audio_path).expanduser().resolve(), sr=sample_rate, mono=True, duration=duration
    )
    harmonic = librosa.effects.harmonic(audio, margin=4.0)
    chroma = librosa.feature.chroma_cqt(
        y=harmonic, sr=sample_rate, hop_length=hop_length
    )
    frame_times = librosa.frames_to_time(
        np.arange(chroma.shape[1]), sr=sample_rate, hop_length=hop_length
    )
    templates: list[np.ndarray] = []
    labels: list[str] = []
    for root in range(12):
        for quality, intervals in (("maj", (0, 4, 7)), ("min", (0, 3, 7))):
            template = np.zeros(12, dtype=np.float64)
            template[[(root + interval) % 12 for interval in intervals]] = (1.2, 1.0, 1.0)
            template /= np.linalg.norm(template)
            templates.append(template)
            labels.append(f"{root}:{quality}")
    template_matrix = np.stack(templates)
    rows: list[dict[str, object]] = []
    for chord in chords.itertuples(index=False):
        mask = (frame_times >= float(chord.start)) & (frame_times < float(chord.end))
        profile = chroma[:, mask].mean(axis=1) if np.any(mask) else np.zeros(12)
        norm = float(np.linalg.norm(profile))
        if norm <= 1e-8:
            chroma_root = np.nan
            chroma_quality = ""
            score = margin = 0.0
        else:
            scores = template_matrix @ (profile / norm)
            order = np.argsort(scores)[::-1]
            best, second = int(order[0]), int(order[1])
            chroma_root = best // 2
            chroma_quality = labels[best].split(":", 1)[1]
            score = float(scores[best])
            margin = float(scores[best] - scores[second])
        large_root = chord_root_pitch_class(str(chord.chord))
        rows.append({
            "start": float(chord.start),
            "end": float(chord.end),
            "large_chord": str(chord.chord),
            "large_confidence": float(chord.confidence),
            "large_root": large_root,
            "chroma_root": chroma_root,
            "chroma_quality": chroma_quality,
            "chroma_score": score,
            "chroma_margin": margin,
            "root_agreement": (
                large_root == int(chroma_root)
                if large_root is not None and not np.isnan(chroma_root) else np.nan
            ),
        })
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit BTC chord uncertainty without rerunning stems")
    parser.add_argument("--manifest", type=Path, default=Path("data/manifest.csv"))
    parser.add_argument("--output-root", type=Path, default=Path("data/btc-audit"))
    parser.add_argument("--track-id", action="append", dest="track_ids")
    parser.add_argument("--duration", type=float)
    parser.add_argument("--compare-small-vocabulary", action="store_true")
    parser.add_argument("--compare-chroma", action="store_true")
    args = parser.parse_args()
    print(json.dumps(audit_btc_predictions(
        args.manifest, args.output_root, track_ids=args.track_ids, duration=args.duration,
        compare_small_vocabulary=args.compare_small_vocabulary,
        compare_chroma=args.compare_chroma,
    ), indent=2))


if __name__ == "__main__":
    main()
