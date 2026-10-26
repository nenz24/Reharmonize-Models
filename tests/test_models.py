import numpy as np
import pandas as pd
import torch
import soundfile as sf

from reharmonize_models import (
    ChordCandidateRanker, aggregate_melody_by_chord, assign_work_splits, build_manifest,
    build_quality_report, chord_pitch_classes,
    build_pretraining_dataset, chord_quality, MaskedChordModel,
    create_candidate_sheet, estimate_key, melody_clash_score,
    melody_compatibility_features, pairwise_ranking_loss, save_lab,
    preprocess_manifest, smooth_chord_intervals, train_masked_chord_model, train_ranker,
    transpose_batch,
)
from reharmonize_models.btc import _collapse_predictions


def test_chord_parser_and_clash_score():
    assert chord_pitch_classes("C:maj7") == {0, 4, 7, 11}
    assert chord_pitch_classes("F#:7") == {1, 4, 6, 10}
    profile = np.zeros(12, dtype=np.float32)
    profile[[0, 4, 7]] = 1 / 3
    assert melody_clash_score(profile, "C") < 1e-6
    assert melody_clash_score(profile, "C#:maj") > 0.99


def test_melody_alignment():
    chords = pd.DataFrame([(0.0, 1.0, "C"), (1.0, 2.0, "G")], columns=["start", "end", "chord"])
    melody = pd.DataFrame([(0.1, 1.0, 0), (0.8, 0.5, 4), (1.2, 0.8, 7)],
                          columns=["time", "confidence", "pitch_class"])
    result = aggregate_melody_by_chord(chords, melody)
    assert list(result.voiced_frames) == [2, 1]
    assert np.isclose(result.iloc[0].melody_profile.sum(), 1)


def test_ranker_backpropagates():
    model = ChordCandidateRanker(16, 8)
    context = torch.randn(4, 16)
    preferred = model(context, torch.randn(4, 8))
    rejected = model(context, torch.randn(4, 8))
    loss = pairwise_ranking_loss(preferred, rejected)
    loss.backward()
    assert preferred.shape == (4,)
    assert torch.isfinite(loss)


def test_btc_predictions_become_intervals(tmp_path):
    labels = {0: "C", 1: "G"}
    result = _collapse_predictions([0, 0, 1, 1, 1], labels, 0.1, 0.45)
    assert result.to_dict("records") == [
        {"start": 0.0, "end": 0.2, "chord": "C"},
        {"start": 0.2, "end": 0.45, "chord": "G"},
    ]

    output = save_lab(result, tmp_path / "nested" / "result.lab")
    assert output.read_text() == "0.000 0.200 C\n0.200 0.450 G\n"


def test_smoothing_removes_short_chord_flicker():
    chords = pd.DataFrame(
        [(0.0, 1.0, "E"), (1.0, 1.2, "E:maj7"), (1.2, 2.0, "E")],
        columns=["start", "end", "chord"],
    )
    result = smooth_chord_intervals(chords, minimum_duration=0.35)
    assert result.to_dict("records") == [{"start": 0.0, "end": 2.0, "chord": "E"}]


def test_compatibility_separates_extensions_from_semitones():
    extension = np.zeros(12, dtype=np.float32)
    extension[6] = 1  # F# is the ninth over E.
    nearby = np.zeros(12, dtype=np.float32)
    nearby[7] = 1  # G is one semitone from E's major third.
    extension_features = melody_compatibility_features(extension, "E", voiced_frames=20, duration=1)
    nearby_features = melody_compatibility_features(nearby, "E", voiced_frames=20, duration=1)
    assert extension_features["extension_ratio"] == 1
    assert extension_features["semitone_proximity_ratio"] == 0
    assert nearby_features["semitone_proximity_ratio"] == 1
    assert extension_features["evidence"] == 1


def test_key_estimation_and_candidate_sheet():
    chords = pd.DataFrame(
        [(0.0, 4.0, "E"), (4.0, 6.0, "A"), (6.0, 8.0, "B:7"), (8.0, 12.0, "E")],
        columns=["start", "end", "chord"],
    )
    assert estimate_key(chords) == "E:maj"
    segments = chords.copy()
    segments["melody_profile"] = [np.eye(12, dtype=np.float32)[4]] * len(segments)
    segments["voiced_frames"] = 40
    sheet = create_candidate_sheet(segments, max_candidates=6)
    assert set(sheet.groupby("segment_id").size()) == {6}
    assert set(segments.chord) - {"N", "X"} <= set(sheet.candidate)
    assert set(sheet.key) == {"E:maj"}


def test_labeled_candidates_train_checkpoint(tmp_path):
    chords = pd.DataFrame(
        [(0.0, 2.0, "E"), (2.0, 4.0, "A")], columns=["start", "end", "chord"]
    )
    segments = chords.copy()
    segments["melody_profile"] = [np.eye(12, dtype=np.float32)[4]] * 2
    segments["voiced_frames"] = 40
    sheet = create_candidate_sheet(segments, key="E:maj", max_candidates=4)
    sheet["preferred"] = 0
    sheet.loc[sheet.groupby("segment_id").head(1).index, "preferred"] = 1
    csv_path = tmp_path / "preferences.csv"
    checkpoint = tmp_path / "ranker.pt"
    sheet.to_csv(csv_path, index=False)
    metrics = train_ranker([csv_path], checkpoint, epochs=3)
    saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
    assert metrics["training_pairs"] == 6
    assert np.isfinite(metrics["final_loss"])
    assert saved["context_size"] == 66
    assert saved["candidate_size"] == 28


def test_manifest_generation_groups_versions_and_preserves_metadata(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    audio_root = tmp_path / "data" / "audio"
    work = audio_root / "amazing-grace"
    work.mkdir(parents=True)
    sample_rate = 8_000
    sf.write(work / "acoustic.wav", np.zeros(sample_rate), sample_rate)
    sf.write(work / "gospel.wav", np.ones(sample_rate) * 0.01, sample_rate)
    manifest_path = tmp_path / "data" / "manifest.csv"

    summary = build_manifest(audio_root, manifest_path)
    manifest = pd.read_csv(manifest_path)
    assert summary == {
        "audio_files": 2, "manifest_tracks": 2, "works": 1,
        "duplicates_skipped": 0, "unreadable": 0,
    }
    assert manifest.work_id.nunique() == 1
    assert manifest.track_id.nunique() == 2
    assert set(manifest.primary_genre) == {"unknown"}

    manifest.loc[manifest.index[0], "primary_genre"] = "worship"
    manifest.to_csv(manifest_path, index=False)
    build_manifest(audio_root, manifest_path)
    refreshed = pd.read_csv(manifest_path)
    assert refreshed.loc[refreshed.track_id == manifest.iloc[0].track_id, "primary_genre"].item() == "worship"


def test_work_splits_never_separate_arrangements():
    manifest = pd.DataFrame({"work_id": [f"work-{index:04d}" for index in range(20) for _ in range(2)]})
    splits = assign_work_splits(manifest, seed=7)
    assert set(splits.values()) == {"train", "validation", "test"}
    assert len(splits) == 20


def test_batch_preprocessing_is_resumable(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    audio = tmp_path / "data" / "audio" / "song" / "take.wav"
    audio.parent.mkdir(parents=True)
    sf.write(audio, np.zeros(16_000), 16_000)
    manifest = pd.DataFrame([{
        "track_id": "track-0001", "work_id": "work-0001",
        "audio_path": "data/audio/song/take.wav", "sha256": "abc", "status": "ready",
    }])
    manifest_path = tmp_path / "data" / "manifest.csv"
    manifest.to_csv(manifest_path, index=False)

    class FakeDetector:
        def detect(self, path, duration=None):
            return pd.DataFrame([(0.0, 1.0, "C")], columns=["start", "end", "chord"])

    class FakeSeparator:
        def separate(self, source, output, duration=None):
            sf.write(output, np.zeros(16_000), 16_000)
            return output

    melody = pd.DataFrame({
        "time": [0.1], "frequency_hz": [261.6], "confidence": [0.9],
        "midi": [60.0], "pitch_class": [0.0],
    })
    monkeypatch.setattr("reharmonize_models.preprocess.BTCChordDetector", FakeDetector)
    monkeypatch.setattr("reharmonize_models.preprocess.VocalSeparator", FakeSeparator)
    monkeypatch.setattr("reharmonize_models.preprocess.melody_pitch_track", lambda *args, **kwargs: melody)

    output = tmp_path / "data" / "processed-smoke"
    first = preprocess_manifest(manifest_path, output, max_duration=60)
    second = preprocess_manifest(manifest_path, output, max_duration=60)
    assert first["completed"] == 1
    assert second["skipped_cached"] == 1
    assert (output / "track-0001" / "features.npz").is_file()
    assert pd.read_csv(output / "windows.csv").to_dict("records") == [{
        "track_id": "track-0001", "work_id": "work-0001", "split": "train",
        "start": 0.0, "end": 1.0,
    }]

    summary = build_quality_report(manifest_path, output, tmp_path / "data" / "qc.csv")
    report = pd.read_csv(tmp_path / "data" / "qc.csv")
    assert summary["tracks_checked"] == 1
    assert summary["missing_cache"] == 0
    assert report.loc[0, "no_chord_ratio"] == 0
    assert report.loc[0, "voiced_fraction"] == 0.01


def test_pretraining_dataset_masks_targets_and_balances_works(tmp_path):
    manifest = pd.DataFrame([
        {"track_id": "track-a", "work_id": "work-a", "status": "ready"},
        {"track_id": "track-b", "work_id": "work-b", "status": "ready"},
    ])
    manifest_path = tmp_path / "manifest.csv"
    manifest.to_csv(manifest_path, index=False)
    processed = tmp_path / "processed"
    for track, work, split, chords in [
        ("track-a", "work-a", "train", ["C", "G:7", "C"]),
        ("track-b", "work-b", "validation", ["F:min", "N"]),
    ]:
        track_dir = processed / track
        track_dir.mkdir(parents=True)
        (track_dir / "metadata.json").write_text(__import__("json").dumps({
            "status": "complete", "split": split,
        }))
        count = len(chords)
        np.savez_compressed(
            track_dir / "features.npz",
            start=np.arange(count, dtype=np.float32),
            end=np.arange(1, count + 1, dtype=np.float32),
            chord=np.asarray(chords),
            melody_profile=np.eye(12, dtype=np.float32)[:count],
            sustained_pitch_classes=np.zeros((count, 12), dtype=np.float32),
            voiced_frames=np.full(count, 50, dtype=np.int32),
        )

    output = tmp_path / "pretraining"
    summary = build_pretraining_dataset(manifest_path, processed, output, sequence_length=5)
    assert summary["examples"] == 5
    assert chord_quality("C#/G#") == "maj"
    assert chord_quality("Bb:min7/Db") == "min7"
    with np.load(output / "train.npz") as train:
        assert train["root_input"].shape == (3, 5)
        assert set(train["root_input"][:, 2]) == {14}
        assert np.isclose(train["sample_weight"].sum(), 2.5)
    with np.load(output / "validation.npz") as validation:
        assert np.isclose(validation["sample_weight"].sum(), 2.5)
    checkpoint = tmp_path / "masked-chord.pt"
    metrics = train_masked_chord_model(output, checkpoint, epochs=1, batch_size=2)
    assert metrics["training_examples"] == 3
    assert checkpoint.is_file()


def test_masked_chord_model_and_transposition_are_shape_safe():
    root = torch.tensor([[0, 14, 7], [12, 14, 2]])
    quality = torch.tensor([[0, 3, 1], [2, 3, 0]])
    melody = torch.zeros(2, 3, 12)
    melody[0, :, 0] = 1
    melody[1, :, 2] = 1
    target = torch.tensor([0, 12])
    shifted_root, shifted_melody, shifted_target = transpose_batch(
        root, melody, target, torch.tensor([2, 5])
    )
    assert shifted_root.tolist() == [[2, 14, 9], [12, 14, 7]]
    assert shifted_target.tolist() == [2, 12]
    assert shifted_melody[0, 0, 2] == 1
    assert shifted_melody[1, 0, 7] == 1

    model = MaskedChordModel(2, 3, hidden_size=16, attention_heads=4, layers=1)
    root_logits, quality_logits = model(
        shifted_root, quality, shifted_melody, torch.zeros(2, 3, 2)
    )
    assert root_logits.shape == (2, 13)
    assert quality_logits.shape == (2, 2)
