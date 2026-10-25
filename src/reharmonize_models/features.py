from __future__ import annotations

from pathlib import Path
import librosa
import numpy as np
import pandas as pd
import torch
import torchcrepe

_ROOTS = {"C": 0, "D": 2, "E": 4, "F": 5, "G": 7, "A": 9, "B": 11}
_QUALITIES = {
    "": (0, 4, 7), "maj": (0, 4, 7), "min": (0, 3, 7),
    "7": (0, 4, 7, 10), "maj7": (0, 4, 7, 11), "min7": (0, 3, 7, 10),
    "maj6": (0, 4, 7, 9), "min6": (0, 3, 7, 9), "sus2": (0, 2, 7),
    "sus4": (0, 5, 7), "dim": (0, 3, 6), "aug": (0, 4, 8),
    "dim7": (0, 3, 6, 9), "hdim7": (0, 3, 6, 10), "minmaj7": (0, 3, 7, 11),
}


def chord_root_pitch_class(symbol: str) -> int | None:
    """Return the root pitch class of a Harte-style chord symbol."""
    if symbol in {"N", "X"} or not symbol or symbol[0] not in _ROOTS:
        return None
    root = _ROOTS[symbol[0]]
    if len(symbol) > 1 and symbol[1] in {"#", "b"}:
        root += 1 if symbol[1] == "#" else -1
    return root % 12


def melody_pitch_track(audio_path: str | Path, *, duration: float | None = None,
                       sample_rate: int = 16_000, hop_length: int = 160,
                       periodicity_threshold: float = 0.5) -> pd.DataFrame:
    """Estimate dominant F0. Separate vocals first when accompaniment dominates the mix."""
    audio, _ = librosa.load(audio_path, sr=sample_rate, mono=True, duration=duration)
    with torch.inference_mode():
        frequency, periodicity = torchcrepe.predict(
            torch.from_numpy(audio).unsqueeze(0), sample_rate, hop_length, 50, 1_000, "tiny",
            batch_size=256, device="cpu", return_periodicity=True,
        )
    hz = frequency.squeeze(0).cpu().numpy()
    confidence = periodicity.squeeze(0).cpu().numpy()
    midi = np.full_like(hz, np.nan, dtype=np.float64)
    voiced = confidence >= periodicity_threshold
    midi[voiced] = librosa.hz_to_midi(hz[voiced])
    return pd.DataFrame({
        "time": np.arange(len(hz)) * hop_length / sample_rate,
        "frequency_hz": hz, "confidence": confidence, "midi": midi,
        "pitch_class": np.mod(np.rint(midi), 12),
    })


def aggregate_melody_by_chord(chords: pd.DataFrame, melody: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for chord in chords.itertuples(index=False):
        frames = melody[(melody.time >= chord.start) & (melody.time < chord.end)].dropna(subset=["pitch_class"])
        profile = np.zeros(12, dtype=np.float32)
        for frame in frames.itertuples(index=False):
            profile[int(frame.pitch_class)] += float(frame.confidence)
        if profile.sum() > 0:
            profile /= profile.sum()
        rows.append({"start": chord.start, "end": chord.end, "chord": chord.chord,
                     "melody_profile": profile,
                     "sustained_pitch_classes": (profile >= 0.2).astype(np.float32),
                     "voiced_frames": len(frames)})
    return pd.DataFrame(rows)


def chord_pitch_classes(symbol: str) -> set[int]:
    if symbol in {"N", "X"} or not symbol:
        return set()
    head = symbol.split("/", 1)[0]
    if head[0] not in _ROOTS:
        return set()
    cursor, accidental = 1, 0
    if cursor < len(head) and head[cursor] in {"#", "b"}:
        accidental = 1 if head[cursor] == "#" else -1
        cursor += 1
    quality = head[cursor + 1:] if cursor < len(head) and head[cursor] == ":" else head[cursor:]
    intervals = _QUALITIES.get(quality)
    if intervals is None:
        return set()
    root = (_ROOTS[head[0]] + accidental) % 12
    return {(root + interval) % 12 for interval in intervals}


def melody_clash_score(profile: np.ndarray, candidate_symbol: str) -> float:
    """Weighted share of melody outside the chord (a diagnostic, not a rejection rule)."""
    tones, values = chord_pitch_classes(candidate_symbol), np.asarray(profile)
    if not tones or values.sum() <= 0:
        return 0.0
    return float(np.clip(1 - values[list(tones)].sum() / values.sum(), 0, 1))


def melody_compatibility_features(
    profile: np.ndarray,
    candidate_symbol: str,
    *,
    voiced_frames: int = 0,
    duration: float | None = None,
    frame_rate: float = 100.0,
) -> dict[str, float]:
    """Describe melody/chord interaction without treating every extension as a clash."""
    values = np.asarray(profile, dtype=np.float64)
    if values.shape != (12,):
        raise ValueError("profile must contain 12 pitch-class weights")
    total = float(values.sum())
    if total > 0:
        values = values / total

    tones = chord_pitch_classes(candidate_symbol)
    if not tones or total <= 0:
        chord_tone = semitone = extension = 0.0
    else:
        tone_indexes = np.array(sorted(tones), dtype=np.int64)
        distances = np.array([
            min(min((pitch - tone) % 12, (tone - pitch) % 12) for tone in tones)
            for pitch in range(12)
        ])
        chord_tone = float(values[tone_indexes].sum())
        semitone = float(values[distances == 1].sum())
        extension = float(values[distances >= 2].sum())

    voiced_fraction = 0.0
    if duration is not None and duration > 0:
        voiced_fraction = float(np.clip(voiced_frames / (duration * frame_rate), 0, 1))
    evidence = float(np.clip(voiced_frames / 20.0, 0, 1))
    return {
        "chord_tone_ratio": chord_tone,
        "semitone_proximity_ratio": semitone,
        "extension_ratio": extension,
        "voiced_fraction": voiced_fraction,
        "evidence": evidence,
    }
