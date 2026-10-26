from .candidates import create_candidate_sheet, estimate_key, key_candidates
from .dataset import build_manifest
from .features import (
    aggregate_melody_by_chord, chord_pitch_classes, chord_root_pitch_class,
    melody_clash_score, melody_compatibility_features, melody_pitch_track,
)
from .btc import BTCChordDetector, detect_chords, save_lab, smooth_chord_intervals
from .ranker import ChordCandidateRanker, pairwise_ranking_loss
from .preprocess import assign_work_splits, preprocess_manifest
from .quality import build_quality_report
from .pretraining import (
    MaskedChordModel, calculate_pretraining_baselines, evaluate_masked_chord_checkpoint,
    train_masked_chord_model, transpose_batch,
)
from .pretraining_data import build_pretraining_dataset, chord_quality
from .separation import VocalSeparator, separate_vocals
from .suggestions import (
    generate_chord_suggestions, render_mixed_audio_preview, render_suggestion_midi,
)
from .training import preference_pairs, train_ranker

__all__ = [
    "BTCChordDetector", "ChordCandidateRanker", "MaskedChordModel", "VocalSeparator",
    "aggregate_melody_by_chord", "assign_work_splits",
    "build_manifest", "build_pretraining_dataset", "build_quality_report",
    "calculate_pretraining_baselines", "chord_pitch_classes",
    "chord_quality", "chord_root_pitch_class",
    "create_candidate_sheet",
    "detect_chords", "estimate_key", "key_candidates", "melody_clash_score",
    "generate_chord_suggestions",
    "melody_compatibility_features", "melody_pitch_track", "pairwise_ranking_loss",
    "preference_pairs", "preprocess_manifest", "save_lab", "separate_vocals",
    "evaluate_masked_chord_checkpoint", "render_mixed_audio_preview",
    "render_suggestion_midi", "smooth_chord_intervals",
    "train_masked_chord_model", "train_ranker", "transpose_batch",
]
