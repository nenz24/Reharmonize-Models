from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from .features import chord_pitch_classes, chord_root_pitch_class
from .ranker import ChordCandidateRanker, pairwise_ranking_loss

CONTEXT_SIZE = 66
CANDIDATE_SIZE = 28


def _chord_vector(symbol: str) -> np.ndarray:
    vector = np.zeros(24, dtype=np.float32)
    root = chord_root_pitch_class(symbol)
    if root is not None:
        vector[root] = 1
    tones = chord_pitch_classes(symbol)
    if tones:
        vector[12 + np.array(sorted(tones))] = 1
    return vector


def _context_vector(row: pd.Series) -> np.ndarray:
    melody = np.array([row[f"melody_pc_{pitch}"] for pitch in range(12)], dtype=np.float32)
    previous_tones = _chord_vector(str(row.previous_chord))[12:]
    source_tones = _chord_vector(str(row.source_chord))[12:]
    next_tones = _chord_vector(str(row.next_chord))[12:]
    key_root = np.zeros(12, dtype=np.float32)
    root = chord_root_pitch_class(str(row.key))
    if root is not None:
        key_root[root] = 1
    key_quality = np.array(
        [float(str(row.key).endswith(":maj")), float(str(row.key).endswith(":min"))],
        dtype=np.float32,
    )
    scalars = np.array([
        min(max(float(row.end) - float(row.start), 0.0), 10.0) / 10.0,
        min(max(float(row.start), 0.0), 600.0) / 600.0,
        float(row.voiced_fraction),
        float(row.evidence),
    ], dtype=np.float32)
    vector = np.concatenate([
        melody, previous_tones, source_tones, next_tones, key_root, key_quality, scalars,
    ])
    if vector.size != CONTEXT_SIZE:
        raise AssertionError(f"Expected {CONTEXT_SIZE} context features, got {vector.size}")
    return vector


def _candidate_vector(row: pd.Series) -> np.ndarray:
    chord = _chord_vector(str(row.candidate))
    source_tones = chord_pitch_classes(str(row.source_chord))
    candidate_tones = chord_pitch_classes(str(row.candidate))
    union = source_tones | candidate_tones
    overlap = len(source_tones & candidate_tones) / len(union) if union else 0.0
    diagnostics = np.array([
        float(row.chord_tone_ratio),
        float(row.semitone_proximity_ratio),
        float(row.extension_ratio),
        overlap,
    ], dtype=np.float32)
    vector = np.concatenate([chord, diagnostics])
    if vector.size != CANDIDATE_SIZE:
        raise AssertionError(f"Expected {CANDIDATE_SIZE} candidate features, got {vector.size}")
    return vector


def preference_pairs(sheet: pd.DataFrame) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Convert labeled candidate groups into context/preferred/rejected tensors."""
    required = {
        "segment_id", "preferred", "candidate", "previous_chord", "source_chord",
        "next_chord", "key", "start", "end", "voiced_fraction", "evidence",
        "chord_tone_ratio", "semitone_proximity_ratio", "extension_ratio",
        *{f"melody_pc_{pitch}" for pitch in range(12)},
    }
    missing = required - set(sheet.columns)
    if missing:
        raise ValueError(f"Preference CSV is missing columns: {sorted(missing)}")

    labeled = sheet.copy()
    labeled["preferred"] = pd.to_numeric(labeled["preferred"], errors="coerce")
    contexts: list[np.ndarray] = []
    preferred: list[np.ndarray] = []
    rejected: list[np.ndarray] = []
    grouping = ["_song_id", "segment_id"] if "_song_id" in labeled.columns else "segment_id"
    for segment_id, group in labeled.groupby(grouping):
        positive = group[group.preferred == 1]
        negative = group[group.preferred == 0]
        if len(positive) == 0 and len(negative) == 0:
            continue
        if len(positive) != 1 or len(negative) == 0:
            raise ValueError(
                f"segment_id {segment_id} must contain exactly one preferred=1 and at least one preferred=0"
            )
        positive_row = positive.iloc[0]
        for _, negative_row in negative.iterrows():
            contexts.append(_context_vector(positive_row))
            preferred.append(_candidate_vector(positive_row))
            rejected.append(_candidate_vector(negative_row))
    if not contexts:
        raise ValueError("No labeled preferred/rejected pairs were found")
    return tuple(torch.from_numpy(np.stack(values)) for values in (contexts, preferred, rejected))


def train_ranker(
    preference_files: list[str | Path],
    output_path: str | Path,
    *,
    epochs: int = 100,
    learning_rate: float = 1e-3,
    seed: int = 42,
) -> dict[str, float | int]:
    """Train and save the first pairwise chord-candidate ranker."""
    if epochs < 1:
        raise ValueError("epochs must be positive")
    torch.manual_seed(seed)
    sheets = []
    for path in preference_files:
        sheet = pd.read_csv(path)
        sheet["_song_id"] = Path(path).stem
        sheets.append(sheet)
    contexts, preferred, rejected = preference_pairs(pd.concat(sheets, ignore_index=True))
    model = ChordCandidateRanker(CONTEXT_SIZE, CANDIDATE_SIZE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    model.train()
    initial_loss = final_loss = 0.0
    for epoch in range(epochs):
        optimizer.zero_grad()
        preferred_scores = model(contexts, preferred)
        rejected_scores = model(contexts, rejected)
        loss = pairwise_ranking_loss(preferred_scores, rejected_scores)
        if epoch == 0:
            initial_loss = float(loss.detach())
        loss.backward()
        optimizer.step()
        final_loss = float(loss.detach())

    model.eval()
    with torch.inference_mode():
        accuracy = float((model(contexts, preferred) > model(contexts, rejected)).float().mean())
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model": model.state_dict(),
        "context_size": CONTEXT_SIZE,
        "candidate_size": CANDIDATE_SIZE,
        "training_pairs": len(contexts),
        "feature_version": 1,
    }, output_path)
    return {
        "training_pairs": len(contexts),
        "initial_loss": initial_loss,
        "final_loss": final_loss,
        "training_accuracy": accuracy,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the chord candidate ranker")
    parser.add_argument("preferences", nargs="+", type=Path)
    parser.add_argument("--output", "-o", type=Path, default=Path("artifacts/chord-ranker.pt"))
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    args = parser.parse_args()
    metrics = train_ranker(
        args.preferences, args.output, epochs=args.epochs, learning_rate=args.learning_rate
    )
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
