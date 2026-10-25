from __future__ import annotations

import argparse
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf
import torch
import torchaudio


def separate_vocals(
    audio_path: str | Path,
    output_path: str | Path,
    *,
    duration: float | None = None,
    device: str | torch.device = "cpu",
) -> Path:
    """Extract a vocal stem with torchaudio's pretrained HDemucs model."""
    audio_path = Path(audio_path).expanduser().resolve()
    output_path = Path(output_path).expanduser().resolve()
    if not audio_path.is_file():
        raise FileNotFoundError(audio_path)

    bundle = torchaudio.pipelines.HDEMUCS_HIGH_MUSDB_PLUS
    audio, sample_rate = librosa.load(
        audio_path, sr=bundle.sample_rate, mono=False, duration=duration
    )
    if audio.size == 0:
        raise ValueError(f"Audio file is empty: {audio_path}")
    if audio.ndim == 1:
        audio = np.stack([audio, audio])
    elif audio.shape[0] != 2:
        mono = librosa.to_mono(audio)
        audio = np.stack([mono, mono])

    target_device = torch.device(device)
    model = bundle.get_model().to(target_device).eval()
    waveform = torch.as_tensor(audio, dtype=torch.float32, device=target_device).unsqueeze(0)
    with torch.inference_mode():
        separated = model(waveform)
    vocal_index = model.sources.index("vocals")
    vocals = separated[0, vocal_index].detach().cpu().numpy().T

    output_path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(output_path, vocals, sample_rate)
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract vocals with pretrained HDemucs")
    parser.add_argument("audio", type=Path)
    parser.add_argument("--output", "-o", type=Path, required=True)
    parser.add_argument("--duration", type=float)
    args = parser.parse_args()
    print(separate_vocals(args.audio, args.output, duration=args.duration))


if __name__ == "__main__":
    main()
