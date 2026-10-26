from __future__ import annotations

import argparse
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf
import torch
import torchaudio


class VocalSeparator:
    """Reusable, chunked HDemucs vocal separator for long recordings."""

    def __init__(self, *, device: str | torch.device = "cpu") -> None:
        self.bundle = torchaudio.pipelines.HDEMUCS_HIGH_MUSDB_PLUS
        self.device = torch.device(device)
        self.model = self.bundle.get_model().to(self.device).eval()
        self.vocal_index = self.model.sources.index("vocals")

    def separate(
        self,
        audio_path: str | Path,
        output_path: str | Path,
        *,
        duration: float | None = None,
        segment_seconds: float = 30.0,
        overlap_seconds: float = 1.0,
    ) -> Path:
        audio_path = Path(audio_path).expanduser().resolve()
        output_path = Path(output_path).expanduser().resolve()
        if not audio_path.is_file():
            raise FileNotFoundError(audio_path)
        if segment_seconds <= overlap_seconds or overlap_seconds < 0:
            raise ValueError("segment_seconds must be greater than overlap_seconds >= 0")

        audio, sample_rate = librosa.load(
            audio_path, sr=self.bundle.sample_rate, mono=False, duration=duration
        )
        if audio.size == 0:
            raise ValueError(f"Audio file is empty: {audio_path}")
        if audio.ndim == 1:
            audio = np.stack([audio, audio])
        elif audio.shape[0] != 2:
            mono = librosa.to_mono(audio)
            audio = np.stack([mono, mono])

        segment = int(segment_seconds * sample_rate)
        overlap = int(overlap_seconds * sample_rate)
        hop = segment - overlap
        output = np.zeros_like(audio, dtype=np.float32)
        weights = np.zeros(audio.shape[-1], dtype=np.float32)
        for start in range(0, audio.shape[-1], hop):
            end = min(start + segment, audio.shape[-1])
            waveform = torch.as_tensor(
                audio[:, start:end], dtype=torch.float32, device=self.device
            ).unsqueeze(0)
            with torch.inference_mode():
                separated = self.model(waveform)
            vocals = separated[0, self.vocal_index].detach().cpu().numpy()
            weight = np.ones(end - start, dtype=np.float32)
            fade = min(overlap, end - start)
            if start > 0 and fade:
                weight[:fade] = np.linspace(0, 1, fade, dtype=np.float32)
            if end < audio.shape[-1] and fade:
                weight[-fade:] = np.minimum(
                    weight[-fade:], np.linspace(1, 0, fade, dtype=np.float32)
                )
            output[:, start:end] += vocals * weight
            weights[start:end] += weight
        output /= np.maximum(weights, 1e-8)[None, :]

        output_path.parent.mkdir(parents=True, exist_ok=True)
        sf.write(output_path, output.T, sample_rate)
        return output_path


def separate_vocals(
    audio_path: str | Path,
    output_path: str | Path,
    *,
    duration: float | None = None,
    device: str | torch.device = "cpu",
    segment_seconds: float = 30.0,
    overlap_seconds: float = 1.0,
) -> Path:
    """Extract a vocal stem with torchaudio's pretrained HDemucs model."""
    return VocalSeparator(device=device).separate(
        audio_path, output_path, duration=duration,
        segment_seconds=segment_seconds, overlap_seconds=overlap_seconds,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract vocals with pretrained HDemucs")
    parser.add_argument("audio", type=Path)
    parser.add_argument("--output", "-o", type=Path, required=True)
    parser.add_argument("--duration", type=float)
    parser.add_argument("--segment-seconds", type=float, default=30.0)
    args = parser.parse_args()
    print(separate_vocals(
        args.audio, args.output, duration=args.duration, segment_seconds=args.segment_seconds
    ))


if __name__ == "__main__":
    main()
