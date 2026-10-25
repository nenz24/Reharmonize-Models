from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pandas as pd

from .features import (
    chord_pitch_classes,
    chord_root_pitch_class,
    melody_compatibility_features,
)

_PITCH_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
_MAJOR_PROFILE = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
_MINOR_PROFILE = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])


def estimate_key(chords: pd.DataFrame) -> str:
    """Estimate a major/minor key from duration-weighted detected chord tones."""
    profile = np.zeros(12, dtype=np.float64)
    for row in chords.itertuples(index=False):
        tones = chord_pitch_classes(str(row.chord))
        if not tones:
            continue
        weight = max(float(row.end) - float(row.start), 0.0) / len(tones)
        profile[list(tones)] += weight
    if profile.sum() == 0:
        raise ValueError("Cannot estimate a key without pitched chord intervals")
    profile = (profile - profile.mean()) / (profile.std() + 1e-8)

    scores: list[tuple[float, int, str]] = []
    for tonic in range(12):
        for quality, template in (("maj", _MAJOR_PROFILE), ("min", _MINOR_PROFILE)):
            normalized = (template - template.mean()) / template.std()
            scores.append((float(profile @ np.roll(normalized, tonic)), tonic, quality))
    _, tonic, quality = max(scores)
    return f"{_PITCH_NAMES[tonic]}:{quality}"


def key_candidates(key: str) -> list[str]:
    """Build a compact candidate space; a learned ranker will choose within it."""
    root = chord_root_pitch_class(key)
    if root is None:
        raise ValueError(f"Invalid key: {key}")
    quality = key.split(":", 1)[1] if ":" in key else "maj"
    if quality == "min":
        degrees = [0, 2, 3, 5, 7, 8, 10]
        triads = ["min", "dim", "", "min", "", "", ""]
        sevenths = ["min7", "hdim7", "maj7", "min7", "7", "maj7", "7"]
    else:
        degrees = [0, 2, 4, 5, 7, 9, 11]
        triads = ["", "min", "min", "", "", "min", "dim"]
        sevenths = ["maj7", "min7", "min7", "maj7", "7", "min7", "hdim7"]

    candidates: list[str] = []
    for degree, triad, seventh in zip(degrees, triads, sevenths):
        name = _PITCH_NAMES[(root + degree) % 12]
        candidates.extend([name if not triad else f"{name}:{triad}", f"{name}:{seventh}"])
        dominant = _PITCH_NAMES[(root + degree + 7) % 12]
        candidates.append(f"{dominant}:7")
    return list(dict.fromkeys(candidates))


def _tone_overlap(left: str, right: str) -> float:
    left_tones, right_tones = chord_pitch_classes(left), chord_pitch_classes(right)
    union = left_tones | right_tones
    return len(left_tones & right_tones) / len(union) if union else 0.0


def create_candidate_sheet(
    segments: pd.DataFrame,
    *,
    key: str | None = None,
    max_candidates: int = 8,
    extra_candidates: Iterable[str] = (),
) -> pd.DataFrame:
    """Create an unlabeled table for collecting one preferred chord per segment."""
    if max_candidates < 2:
        raise ValueError("max_candidates must be at least 2")
    key = key or estimate_key(segments)
    base_candidates = key_candidates(key) + list(extra_candidates)
    rows: list[dict[str, object]] = []

    for segment_id, segment in segments.reset_index(drop=True).iterrows():
        source = str(segment.chord)
        if source in {"N", "X"}:
            continue
        duration = float(segment.end) - float(segment.start)
        pool = list(dict.fromkeys([source, *base_candidates]))
        scored: list[tuple[float, str, dict[str, float]]] = []
        for candidate in pool:
            features = melody_compatibility_features(
                segment.melody_profile,
                candidate,
                voiced_frames=int(segment.voiced_frames),
                duration=duration,
            )
            diagnostic_order = (
                1.5 * features["chord_tone_ratio"]
                - features["semitone_proximity_ratio"]
                - 0.2 * features["extension_ratio"]
                + 0.15 * _tone_overlap(source, candidate)
            )
            scored.append((diagnostic_order, candidate, features))
        selected = sorted(scored, reverse=True)[:max_candidates]
        if source not in {candidate for _, candidate, _ in selected}:
            source_item = next(item for item in scored if item[1] == source)
            selected[-1] = source_item

        previous = str(segments.iloc[segment_id - 1].chord) if segment_id > 0 else "N"
        following = str(segments.iloc[segment_id + 1].chord) if segment_id + 1 < len(segments) else "N"
        for display_order, (_, candidate, features) in enumerate(selected, start=1):
            rows.append({
                "segment_id": segment_id,
                "start": float(segment.start),
                "end": float(segment.end),
                "key": key,
                "previous_chord": previous,
                "source_chord": source,
                "next_chord": following,
                "candidate": candidate,
                "display_order": display_order,
                **{
                    f"melody_pc_{pitch}": float(segment.melody_profile[pitch])
                    for pitch in range(12)
                },
                **features,
                "preferred": "",
            })
    return pd.DataFrame(rows)
