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

## Collect chord preferences

After notebook 03, run `notebooks/04_candidate_labeling.ipynb`. It estimates the song key, separates
chord tones from extensions and semitone proximity, and exports `data/song-preferences.csv`. For
each `segment_id`, mark exactly one candidate as `preferred=1` and the remaining candidates as
`preferred=0`. These labels become preferred/rejected pairs for training the neural ranker.

Train a checkpoint after labeling one or more songs:

```bash
uv run reharmonize-train data/song-preferences.csv --output artifacts/chord-ranker.pt
```

The command rejects incomplete labeled segments, builds pairwise preferred/rejected examples, and
prints training loss and accuracy. Training on a single song is only a pipeline check; a useful
model needs labels from multiple songs and evaluation split by song and artist.

## Build an audio dataset manifest

Place audio files under one folder per composition. Multiple arrangements of the same composition
belong in the same folder:

```text
data/audio/amazing-grace/acoustic.wav
data/audio/amazing-grace/gospel.mp3
data/audio/how-great-thou-art/original.mp3
```

Then generate the manifest:

```bash
uv run reharmonize-dataset
```

The command creates `data/manifest.csv` with stable track/work IDs, initial title and version,
relative path, SHA-256, duration, and `unknown` genre. Re-running it preserves manual metadata edits,
detects exact duplicate files, and adds newly discovered audio. Folder names group arrangements;
the title itself is never used as the dataset identity.

## Batch preprocessing

Start with a disposable 60-second smoke dataset for the first two compositions:

```bash
uv run reharmonize-preprocess \
  --manifest data/manifest.csv \
  --output-root data/processed-smoke \
  --work-limit 2 \
  --max-duration 60
```

The command caches BTC chords, a chunked HDemucs vocal stem, CREPE melody, chord-level features,
30-second windows, and work-level train/validation/test splits. Re-running the same command skips
completed tracks. After reviewing the smoke output, process full recordings with
`--output-root data/processed` and without `--max-duration` or `--work-limit`.

Audit a completed dataset before training:

```bash
uv run reharmonize-qc \
  --manifest data/manifest.csv \
  --processed-root data/processed \
  --output data/qc.csv
```

The report flags missing caches, duration mismatches, excessive `N` labels, unusually sparse or
dense chord changes, and low vocal evidence. Flags request review; they do not automatically delete
or exclude a track.

## Self-supervised chord pretraining

Once QC is reviewed, convert the cached features into masked-chord examples:

```bash
uv run reharmonize-build-pretraining \
  --manifest data/manifest.csv \
  --processed-root data/processed \
  --output-root data/pretraining
```

Each example hides one chord and retains its vocal melody profile, duration, voicing evidence, and
four neighboring chords on each side. Examples stay in the work-level split created during
preprocessing. Inverse work-frequency weights stop compositions with more versions or chord
segments from dominating the objective.

Train the first self-supervised checkpoint with random pitch transposition:

```bash
uv run reharmonize-pretrain \
  --dataset-root data/pretraining \
  --output artifacts/masked-chord-model.pt \
  --epochs 20
```

The trainer reports root, quality, and complete-chord accuracy on validation data after every
epoch and saves the state with the lowest validation loss. Its final report also evaluates the
held-out test works. This checkpoint learns a general harmony-and-melody representation; genre or
arrangement controls can be added in a later fine-tuning stage without changing the base dataset.

Compare test accuracy with work-balanced majority and neighboring-chord baselines:

```bash
uv run reharmonize-baselines --dataset-root data/pretraining --split test
```

Pass a checkpoint to report its aggregate and macro class accuracy on the same split:

```bash
uv run reharmonize-baselines \
  --dataset-root data/pretraining \
  --split test \
  --checkpoint artifacts/masked-chord-model.pt
```

When the best validation epoch is still the final epoch, continue from the saved model instead of
restarting its learned weights:

```bash
uv run reharmonize-pretrain \
  --dataset-root data/pretraining \
  --output artifacts/masked-chord-model-continued.pt \
  --resume artifacts/masked-chord-model.pt \
  --quality-balance sqrt \
  --epochs 20
```

Square-root class balancing reduces majority-major bias while avoiding the instability of full
inverse-frequency weighting. Compare both aggregate chord accuracy and macro quality accuracy
against the original checkpoint before selecting the model for downstream chord ranking.

## Inspect model-generated chord alternatives

Generate five alternatives per segment for one processed track. The source BTC chord is excluded
so the CSV shows actual alternatives rather than reconstruction accuracy:

```bash
uv run reharmonize-suggest \
  --dataset-root data/pretraining \
  --checkpoint artifacts/masked-chord-model-balanced.pt \
  --track-id track-0001 \
  --top-k 5 \
  --output artifacts/track-0001-suggestions.csv \
  --midi-output artifacts/track-0001-variation.mid \
  --source-midi-output artifacts/track-0001-source.mid \
  --change-rate 0.1
```

By default, silent `N` source segments are skipped and `N` is not offered as an alternative. The
CSV includes model probability plus chord-tone, semitone-proximity, extension, and vocal-evidence
diagnostics. The source MIDI renders the BTC progression. The variation MIDI replaces roughly ten
percent of its segments with rank one, prioritizes model confidence and low semitone proximity, and
keeps changed segments more than four chord boundaries apart. This preserves the original context
used for each masked prediction. Import both MIDI files into a DAW with the original recording for
an A/B listening review. These alternatives are not yet conditioned on a requested genre or
arrangement style.
