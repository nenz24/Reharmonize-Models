from __future__ import annotations

import argparse
import importlib
import sys
from pathlib import Path
from typing import Sequence

import librosa
import numpy as np
import pandas as pd
import torch
import yaml


def _large_vocabulary() -> dict[int, str]:
    roots = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
    qualities = [
        "min", "maj", "dim", "aug", "min6", "maj6", "min7", "minmaj7",
        "maj7", "7", "dim7", "hdim7", "sus2", "sus4",
    ]
    labels = {
        root_index * len(qualities) + quality_index: (
            root if quality == "maj" else f"{root}:{quality}"
        )
        for root_index, root in enumerate(roots)
        for quality_index, quality in enumerate(qualities)
    }
    labels[168] = "X"
    labels[169] = "N"
    return labels


_SMALL_VOCABULARY = [
    chord
    for root in ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
    for chord in (root, f"{root}:min")
] + ["N"]


def _collapse_predictions(
    predictions: Sequence[int], labels: dict[int, str] | list[str], time_unit: float,
    duration: float,
) -> pd.DataFrame:
    """Collapse frame labels into contiguous, timestamped chord intervals."""
    if not predictions or duration <= 0:
        return pd.DataFrame(columns=["start", "end", "chord"])

    rows: list[tuple[float, float, str]] = []
    start_frame = 0
    previous = int(predictions[0])
    for frame, prediction in enumerate(predictions[1:], start=1):
        prediction = int(prediction)
        if prediction != previous:
            end = min(frame * time_unit, duration)
            if end > start_frame * time_unit:
                rows.append((start_frame * time_unit, end, labels[previous]))
            start_frame, previous = frame, prediction

    start = start_frame * time_unit
    if start < duration:
        rows.append((start, duration, labels[previous]))
    return pd.DataFrame(rows, columns=["start", "end", "chord"])


class BTCChordDetector:
    """Run the published BTC-ISMIR19 checkpoint without modifying its submodule."""

    def __init__(
        self,
        btc_root: str | Path = "vendor/BTC-ISMIR19",
        *,
        large_vocabulary: bool = True,
        device: str | torch.device = "cpu",
    ) -> None:
        self.root = Path(btc_root).expanduser().resolve()
        self.large_vocabulary = large_vocabulary
        self.device = torch.device(device)

        config_path = self.root / "run_config.yaml"
        checkpoint_name = "btc_model_large_voca.pt" if large_vocabulary else "btc_model.pt"
        self.checkpoint_path = self.root / "test" / checkpoint_name
        if not config_path.is_file() or not self.checkpoint_path.is_file():
            raise FileNotFoundError(
                "BTC files are missing. Run `git submodule update --init --recursive` "
                f"and check {self.root}"
            )

        with config_path.open() as stream:
            config = yaml.safe_load(stream)
        config["feature"]["large_voca"] = large_vocabulary
        config["model"]["num_chords"] = 170 if large_vocabulary else 25
        self.config = config
        self.labels = _large_vocabulary() if large_vocabulary else _SMALL_VOCABULARY
        self.model = self._load_model()

    def _load_model(self) -> torch.nn.Module:
        # BTC predates packaging. Add only its root so its original absolute imports resolve.
        # NumPy 1.24 removed this alias, but BTC's timing-signal helper still references it.
        if "float" not in np.__dict__:
            np.float = float  # type: ignore[attr-defined]
        root_string = str(self.root)
        if root_string not in sys.path:
            sys.path.insert(0, root_string)
        previous_bytecode_setting = sys.dont_write_bytecode
        try:
            # Keep the Git submodule clean when this package imports the legacy modules.
            sys.dont_write_bytecode = True
            btc_module = importlib.import_module("btc_model")
        finally:
            sys.dont_write_bytecode = previous_bytecode_setting
        model = btc_module.BTC_model(config=self.config["model"]).to(self.device)
        checkpoint = torch.load(
            self.checkpoint_path, map_location=self.device, weights_only=False
        )
        model.load_state_dict(checkpoint["model"])
        model.eval()
        self.mean = float(checkpoint["mean"])
        self.std = float(checkpoint["std"])
        return model

    def detect(self, audio_path: str | Path, *, duration: float | None = None) -> pd.DataFrame:
        """Return `start`, `end`, and `chord` columns for one audio file."""
        audio_path = Path(audio_path).expanduser().resolve()
        if not audio_path.is_file():
            raise FileNotFoundError(audio_path)

        mp3 = self.config["mp3"]
        feature_config = self.config["feature"]
        audio, sample_rate = librosa.load(
            audio_path, sr=mp3["song_hz"], mono=True, duration=duration
        )
        if audio.size == 0:
            raise ValueError(f"Audio file is empty: {audio_path}")

        features = librosa.cqt(
            audio,
            sr=sample_rate,
            n_bins=feature_config["n_bins"],
            bins_per_octave=feature_config["bins_per_octave"],
            hop_length=feature_config["hop_length"],
        )
        features = (np.log(np.abs(features) + 1e-6).T - self.mean) / self.std

        timestep = int(self.config["model"]["timestep"])
        padding = (-features.shape[0]) % timestep
        if padding:
            features = np.pad(features, ((0, padding), (0, 0)), mode="constant")

        batch = torch.as_tensor(features, dtype=torch.float32, device=self.device).unsqueeze(0)
        predictions: list[int] = []
        with torch.inference_mode():
            for start in range(0, batch.shape[1], timestep):
                encoded, _ = self.model.self_attn_layers(batch[:, start:start + timestep, :])
                predicted, _ = self.model.output_layer(encoded)
                predictions.extend(predicted.squeeze(0).cpu().tolist())

        valid_frames = features.shape[0] - padding
        predictions = predictions[:valid_frames]
        time_unit = mp3["inst_len"] / timestep
        audio_duration = len(audio) / sample_rate
        return _collapse_predictions(predictions, self.labels, time_unit, audio_duration)


def save_lab(chords: pd.DataFrame, output_path: str | Path) -> Path:
    """Write chord intervals in the standard three-column `.lab` format."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w") as stream:
        for row in chords.itertuples(index=False):
            stream.write(f"{row.start:.3f} {row.end:.3f} {row.chord}\n")
    return output_path


def detect_chords(
    audio_path: str | Path,
    *,
    btc_root: str | Path = "vendor/BTC-ISMIR19",
    large_vocabulary: bool = True,
    duration: float | None = None,
    device: str | torch.device = "cpu",
) -> pd.DataFrame:
    detector = BTCChordDetector(
        btc_root, large_vocabulary=large_vocabulary, device=device
    )
    return detector.detect(audio_path, duration=duration)


def main() -> None:
    parser = argparse.ArgumentParser(description="Detect chord intervals with BTC-ISMIR19")
    parser.add_argument("audio", type=Path)
    parser.add_argument("--output", "-o", type=Path)
    parser.add_argument("--btc-root", type=Path, default=Path("vendor/BTC-ISMIR19"))
    parser.add_argument("--duration", type=float)
    parser.add_argument("--small-vocabulary", action="store_true")
    args = parser.parse_args()

    chords = detect_chords(
        args.audio,
        btc_root=args.btc_root,
        large_vocabulary=not args.small_vocabulary,
        duration=args.duration,
    )
    if args.output:
        print(save_lab(chords, args.output))
    else:
        print(chords.to_string(index=False))


if __name__ == "__main__":
    main()
