import numpy as np
import pandas as pd
import torch

from reharmonize_models import (
    ChordCandidateRanker, aggregate_melody_by_chord, chord_pitch_classes,
    melody_clash_score, pairwise_ranking_loss,
)


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
