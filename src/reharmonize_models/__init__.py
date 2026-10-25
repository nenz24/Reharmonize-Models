from .candidates import create_candidate_sheet, estimate_key, key_candidates
from .features import (
    aggregate_melody_by_chord, chord_pitch_classes, chord_root_pitch_class,
    melody_clash_score, melody_compatibility_features, melody_pitch_track,
)
from .btc import BTCChordDetector, detect_chords, save_lab, smooth_chord_intervals
from .ranker import ChordCandidateRanker, pairwise_ranking_loss
from .separation import separate_vocals
from .training import preference_pairs, train_ranker

__all__ = [
    "BTCChordDetector", "ChordCandidateRanker", "aggregate_melody_by_chord",
    "chord_pitch_classes", "chord_root_pitch_class", "create_candidate_sheet",
    "detect_chords", "estimate_key", "key_candidates", "melody_clash_score",
    "melody_compatibility_features", "melody_pitch_track", "pairwise_ranking_loss",
    "preference_pairs", "save_lab", "separate_vocals", "smooth_chord_intervals",
    "train_ranker",
]
