# Reharmonize Models

Python research workspace for SongMapApp's audio-aware chord suggestions.

The prototype separates three stages:

1. **BTC chord recognition** produces timestamped chord intervals from reference audio.
2. **CREPE melody estimation** produces frame-level pitch and voicing confidence.
3. **ChordCandidateRanker** learns to rank reharmonization candidates from chord context, melody,
   key, genre, structure, edit type, and timing.

BTC and CREPE are pretrained perception models. The candidate ranker starts untrained; meaningful
scores require preferred/rejected candidates from licensed arrangements or musician feedback.

## Setup and validation

```bash
git clone --recurse-submodules https://github.com/nenz24/Reharmonize-Models.git
cd Reharmonize-Models
brew install uv # macOS only; skip when uv is already installed
UV_CACHE_DIR=/tmp/reharmonize-uv-cache uv sync
uv run pytest
uv run jupyter nbconvert --to notebook --execute --inplace \
  notebooks/01_audio_to_features.ipynb --ExecutePreprocessor.timeout=180
```

The notebook uses BTC's bundled example recording and limits melody extraction to 30 seconds for a
fast CPU smoke test. BTC retains its upstream MIT license. Confirm the terms of any audio,
annotations, arrangements, or feedback before using them for training or distribution.

## Detect chords from audio

Put local recordings in `data/` (the directory is ignored by Git), then run:

```bash
uv run reharmonize-detect data/song.mp3 --output artifacts/song.lab
```

The command loads the published large-vocabulary BTC checkpoint and writes timestamped chord
intervals. From Python or a notebook, the same operation is available as:

```python
from reharmonize_models import detect_chords

chords = detect_chords("data/song.mp3")
```

For a visual end-to-end analysis, open `notebooks/02_btc_audio_analysis.ipynb`. To choose a local
recording without editing the notebook, start Jupyter with:

```bash
AUDIO_PATH=data/song.mp3 uv run jupyter lab
```

## Isolate vocals before melody extraction

CREPE can follow bass or lead instruments when it receives a full mix. Extract a vocal stem first:

```bash
uv run reharmonize-separate data/song.mp3 --output artifacts/song-vocals.wav --duration 30
```

The first run downloads the 319 MB pretrained HDemucs checkpoint. BTC should continue to analyze
the original mix; pass the resulting vocal WAV to `melody_pitch_track`. Short BTC label flicker can
be cleaned before melody alignment with:

```python
from reharmonize_models import smooth_chord_intervals

chords = smooth_chord_intervals(chords, minimum_duration=0.35)
```

`notebooks/03_vocal_aware_analysis.ipynb` combines these steps for `data/song.mp3` and writes its
vocal stem and smoothed chord labels under `artifacts/`.
