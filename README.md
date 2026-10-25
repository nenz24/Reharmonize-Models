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
