from __future__ import annotations

import argparse
import html
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pretty_midi
import librosa
import soundfile as sf
import torch

from .features import chord_pitch_classes, chord_root_pitch_class, melody_compatibility_features
from .pretraining import MaskedChordModel

_ROOT_NAMES = ("C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B")


def _symbol(root: int, quality: str) -> str:
    if root == 12:
        return "N"
    name = _ROOT_NAMES[root]
    return name if quality == "maj" else f"{name}:{quality}"


def _valid_candidates(
    qualities: list[str], *, include_no_chord: bool = False,
) -> list[tuple[int, int, str]]:
    candidates: list[tuple[int, int, str]] = []
    for root in range(13):
        for quality_id, quality in enumerate(qualities):
            if (root == 12) != (quality == "N"):
                continue
            if root == 12 and not include_no_chord:
                continue
            candidates.append((root, quality_id, _symbol(root, quality)))
    return candidates


def generate_chord_suggestions(
    dataset_root: str | Path,
    checkpoint_path: str | Path,
    output_path: str | Path,
    *,
    track_id: str,
    top_k: int = 5,
    include_source: bool = False,
    include_no_chord: bool = False,
    device: str = "auto",
) -> dict[str, object]:
    """Generate reviewable chord alternatives for every segment of one cached track."""
    if top_k < 1:
        raise ValueError("top_k must be positive")
    dataset_root = Path(dataset_root).expanduser().resolve()
    checkpoint_path = Path(checkpoint_path).expanduser().resolve()
    output_path = Path(output_path).expanduser().resolve()
    vocabulary = json.loads((dataset_root / "vocabulary.json").read_text())
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if checkpoint.get("vocabulary") != vocabulary:
        raise ValueError("Checkpoint vocabulary does not match this dataset")
    if device == "auto":
        device = (
            "mps" if torch.backends.mps.is_available()
            else "cuda" if torch.cuda.is_available()
            else "cpu"
        )
    torch_device = torch.device(device)
    model = MaskedChordModel(
        len(vocabulary["qualities"]), int(vocabulary["sequence_length"])
    ).to(torch_device)
    model.load_state_dict(checkpoint["model"])
    model.eval()

    selected: dict[str, list[np.ndarray]] = {}
    selected_split: str | None = None
    for split in ("train", "validation", "test"):
        path = dataset_root / f"{split}.npz"
        if not path.exists():
            continue
        with np.load(path) as values:
            mask = values["track_id"] == track_id
            if not np.any(mask):
                continue
            selected_split = split
            for name in (
                "root_input", "quality_input", "melody", "scalars",
                "target_root", "target_quality", "work_id", "start",
            ):
                selected.setdefault(name, []).append(values[name][mask])
    if selected_split is None:
        raise ValueError(f"track_id {track_id!r} was not found in the pretraining dataset")
    arrays = {name: np.concatenate(parts) for name, parts in selected.items()}
    qualities = list(vocabulary["qualities"])
    candidates = _valid_candidates(qualities, include_no_chord=include_no_chord)
    candidate_roots = np.asarray([item[0] for item in candidates], dtype=np.int64)
    candidate_qualities = np.asarray([item[1] for item in candidates], dtype=np.int64)

    with torch.inference_mode():
        root_logits, quality_logits = model(
            torch.from_numpy(arrays["root_input"]).long().to(torch_device),
            torch.from_numpy(arrays["quality_input"]).long().to(torch_device),
            torch.from_numpy(arrays["melody"]).float().to(torch_device),
            torch.from_numpy(arrays["scalars"]).float().to(torch_device),
        )
        root_probability = root_logits.softmax(dim=1).cpu().numpy()
        quality_probability = quality_logits.softmax(dim=1).cpu().numpy()

    rows: list[dict[str, object]] = []
    center = arrays["root_input"].shape[1] // 2
    for index in range(len(arrays["target_root"])):
        if int(arrays["target_root"][index]) == 12 and not include_no_chord:
            continue
        source = _symbol(
            int(arrays["target_root"][index]),
            qualities[int(arrays["target_quality"][index])],
        )
        probability = (
            root_probability[index, candidate_roots]
            * quality_probability[index, candidate_qualities]
        )
        probability /= max(float(probability.sum()), 1e-12)
        order = np.argsort(probability)[::-1]
        start = float(arrays["start"][index])
        if index + 1 < len(arrays["start"]):
            end = float(arrays["start"][index + 1])
        else:
            normalized_duration = float(arrays["scalars"][index, center, 0])
            end = start + float(np.expm1(normalized_duration * np.log(11.0)))
        duration = max(end - start, 1e-3)
        voiced_fraction = float(arrays["scalars"][index, center, 1])
        voiced_frames = int(round(voiced_fraction * duration * 100.0))
        melody_profile = arrays["melody"][index, center]
        rank = 0
        for candidate_index in order:
            candidate = candidates[int(candidate_index)][2]
            if not include_source and candidate == source:
                continue
            rank += 1
            compatibility = melody_compatibility_features(
                melody_profile, candidate, voiced_frames=voiced_frames, duration=duration
            )
            rows.append({
                "track_id": track_id,
                "work_id": str(arrays["work_id"][index]),
                "split": selected_split,
                "start": round(start, 3),
                "end": round(end, 3),
                "source_chord": source,
                "rank": rank,
                "candidate_chord": candidate,
                "probability": float(probability[int(candidate_index)]),
                **compatibility,
            })
            if rank == top_k:
                break
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(output_path, index=False)
    return {
        "track_id": track_id,
        "split": selected_split,
        "segments": len(arrays["target_root"]),
        "suggested_segments": len(rows) // top_k,
        "suggestions": len(rows),
        "top_k": top_k,
        "include_source": include_source,
        "include_no_chord": include_no_chord,
        "checkpoint": str(checkpoint_path),
        "output": str(output_path),
    }


def render_suggestion_midi(
    suggestions_path: str | Path,
    output_path: str | Path,
    *,
    rank: int = 1,
    velocity: int = 72,
    change_rate: float = 0.1,
    minimum_segment_gap: int = 4,
    source_only: bool = False,
) -> Path:
    """Render a sparse variation while keeping the model's original chord context intact."""
    if not 0 <= change_rate <= 1:
        raise ValueError("change_rate must be between 0 and 1")
    if minimum_segment_gap < 0:
        raise ValueError("minimum_segment_gap cannot be negative")
    suggestions_path = Path(suggestions_path).expanduser().resolve()
    output_path = Path(output_path).expanduser().resolve()
    suggestions = pd.read_csv(suggestions_path)
    selected = suggestions[suggestions["rank"] == rank].sort_values("start")
    if selected.empty:
        raise ValueError(f"No rank {rank} suggestions were found")
    changed_positions = _select_change_positions(
        selected, change_rate=change_rate, minimum_segment_gap=minimum_segment_gap
    ) if not source_only else set()
    midi = pretty_midi.PrettyMIDI()
    piano = pretty_midi.Instrument(
        program=pretty_midi.instrument_name_to_program("Acoustic Grand Piano")
    )
    for position, row in enumerate(selected.itertuples(index=False)):
        symbol = (
            str(row.candidate_chord)
            if position in changed_positions else str(row.source_chord)
        )
        root = chord_root_pitch_class(symbol)
        tones = chord_pitch_classes(symbol)
        if root is None or not tones:
            continue
        root_midi = 48 + root
        pitches = [root_midi + ((pitch_class - root) % 12) for pitch_class in tones]
        for pitch in sorted(pitches):
            piano.notes.append(pretty_midi.Note(
                velocity=velocity, pitch=int(pitch),
                start=float(row.start), end=max(float(row.end), float(row.start) + 0.05),
            ))
    midi.instruments.append(piano)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    midi.write(str(output_path))
    return output_path


def _selection_scores(selected: pd.DataFrame) -> np.ndarray:
    scores = np.log(np.maximum(selected.probability.to_numpy(), 1e-12))
    if "semitone_proximity_ratio" in selected:
        scores -= selected.semitone_proximity_ratio.to_numpy()
    return scores


def _select_change_positions(
    selected: pd.DataFrame, *, change_rate: float, minimum_segment_gap: int,
) -> set[int]:
    if change_rate <= 0:
        return set()
    target_changes = max(1, round(len(selected) * change_rate))
    changed_positions: set[int] = set()
    for position in np.argsort(_selection_scores(selected))[::-1]:
        position = int(position)
        if any(abs(position - other) <= minimum_segment_gap for other in changed_positions):
            continue
        changed_positions.add(position)
        if len(changed_positions) == target_changes:
            break
    return changed_positions


def build_change_plan(
    suggestions_path: str | Path,
    *,
    rank: int = 1,
    change_rate: float = 0.1,
    minimum_segment_gap: int = 4,
) -> pd.DataFrame:
    """Return the exact source-to-candidate replacements used by the variation MIDI."""
    suggestions = pd.read_csv(Path(suggestions_path).expanduser().resolve())
    selected = suggestions[suggestions["rank"] == rank].sort_values("start").reset_index(drop=True)
    positions = _select_change_positions(
        selected, change_rate=change_rate, minimum_segment_gap=minimum_segment_gap
    )
    plan = selected.iloc[sorted(positions)].copy()
    scores = _selection_scores(selected)
    plan["selection_score"] = [scores[position] for position in sorted(positions)]
    return plan


def render_mixed_audio_preview(
    audio_path: str | Path,
    midi_path: str | Path,
    output_path: str | Path,
    *,
    sample_rate: int = 44_100,
    audio_gain: float = 0.72,
    chord_gain: float = 0.22,
) -> Path:
    """Mix an original recording with a lightweight synthesized chord MIDI preview."""
    audio_path = Path(audio_path).expanduser().resolve()
    midi_path = Path(midi_path).expanduser().resolve()
    output_path = Path(output_path).expanduser().resolve()
    audio, _ = librosa.load(audio_path, sr=sample_rate, mono=True)
    chord_audio = pretty_midi.PrettyMIDI(str(midi_path)).synthesize(fs=sample_rate)
    length = max(len(audio), len(chord_audio))
    audio = np.pad(audio, (0, length - len(audio)))
    chord_audio = np.pad(chord_audio, (0, length - len(chord_audio)))
    chord_peak = float(np.max(np.abs(chord_audio))) if len(chord_audio) else 0.0
    if chord_peak > 0:
        chord_audio = chord_audio / chord_peak
    mixed = audio_gain * audio + chord_gain * chord_audio
    peak = float(np.max(np.abs(mixed))) if len(mixed) else 0.0
    if peak > 0.98:
        mixed = mixed * (0.98 / peak)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(output_path, mixed, sample_rate, subtype="PCM_16")
    return output_path


def render_ab_review(
    audio_path: str | Path,
    source_midi_path: str | Path,
    variation_midi_path: str | Path,
    suggestions_path: str | Path,
    output_path: str | Path,
    *,
    change_rate: float = 0.1,
    review_count: int = 5,
    context_seconds: float = 1.5,
    sample_rate: int = 44_100,
    html_output: str | Path | None = None,
) -> tuple[Path, Path]:
    """Create a compact, loud A/B reel for the strongest sparse chord replacements."""
    if review_count < 1:
        raise ValueError("review_count must be positive")
    audio, _ = librosa.load(Path(audio_path).expanduser().resolve(), sr=sample_rate, mono=True)
    source = pretty_midi.PrettyMIDI(str(Path(source_midi_path).expanduser().resolve())).synthesize(
        fs=sample_rate
    )
    variation = pretty_midi.PrettyMIDI(
        str(Path(variation_midi_path).expanduser().resolve())
    ).synthesize(fs=sample_rate)
    length = max(len(audio), len(source), len(variation))
    audio = np.pad(audio, (0, length - len(audio)))
    source = np.pad(source, (0, length - len(source)))
    variation = np.pad(variation, (0, length - len(variation)))
    audio_peak = float(np.max(np.abs(audio))) if len(audio) else 0.0
    if audio_peak > 0:
        audio /= audio_peak
    for chord_audio in (source, variation):
        peak = float(np.max(np.abs(chord_audio))) if len(chord_audio) else 0.0
        if peak > 0:
            chord_audio /= peak

    plan = build_change_plan(suggestions_path, change_rate=change_rate)
    plan = plan.sort_values("selection_score", ascending=False).head(review_count).copy()
    plan.insert(0, "review_order", np.arange(1, len(plan) + 1))
    chunks: list[np.ndarray] = []
    html_cards: list[str] = []
    html_path = Path(html_output).expanduser().resolve() if html_output else None
    clips_dir = html_path.parent / f"{html_path.stem}-clips" if html_path else None
    if clips_dir:
        clips_dir.mkdir(parents=True, exist_ok=True)

    def cue(beeps: int) -> np.ndarray:
        beep_length = round(0.12 * sample_rate)
        gap_length = round(0.10 * sample_rate)
        wave = 0.25 * np.sin(2 * np.pi * 880 * np.arange(beep_length) / sample_rate)
        parts: list[np.ndarray] = []
        for _ in range(beeps):
            parts.extend([wave, np.zeros(gap_length)])
        parts.append(np.zeros(round(0.18 * sample_rate)))
        return np.concatenate(parts)

    def mix_clip(chords: np.ndarray, start: int, end: int) -> np.ndarray:
        mixed = 0.52 * audio[start:end] + 0.46 * chords[start:end]
        peak = float(np.max(np.abs(mixed))) if len(mixed) else 0.0
        return mixed * (0.95 / peak) if peak > 0.95 else mixed

    for row in plan.itertuples(index=False):
        start_seconds = max(0.0, float(row.start) - context_seconds)
        end_seconds = min(length / sample_rate, float(row.end) + context_seconds)
        start = round(start_seconds * sample_rate)
        end = round(end_seconds * sample_rate)
        source_clip = mix_clip(source, start, end)
        variation_clip = mix_clip(variation, start, end)
        chunks.extend([
            cue(1), source_clip, np.zeros(round(0.45 * sample_rate)),
            cue(2), variation_clip, np.zeros(round(0.8 * sample_rate)),
        ])
        if html_path and clips_dir:
            order = int(row.review_order)
            source_path = clips_dir / f"{order:02d}-source.wav"
            variation_path = clips_dir / f"{order:02d}-variation.wav"
            sf.write(source_path, source_clip, sample_rate, subtype="PCM_16")
            sf.write(variation_path, variation_clip, sample_rate, subtype="PCM_16")
            source_url = source_path.relative_to(html_path.parent).as_posix()
            variation_url = variation_path.relative_to(html_path.parent).as_posix()
            source_chord = html.escape(str(row.source_chord))
            candidate_chord = html.escape(str(row.candidate_chord))
            html_cards.append(f"""
<section class="card" data-order="{order}" data-start="{float(row.start):.3f}"
 data-end="{float(row.end):.3f}" data-source="{source_chord}"
 data-candidate="{candidate_chord}">
  <h2>Bagian {order} <small>{float(row.start):.1f}s · {source_chord} → {candidate_chord}</small></h2>
  <div class="players">
    <label>Source<audio controls preload="metadata" src="{source_url}"></audio></label>
    <label>Variation<audio controls preload="metadata" src="{variation_url}"></audio></label>
  </div>
  <div class="choices">
    {''.join(f'<label><input type="radio" name="choice-{order}" value="{value}"> {label}</label>' for value, label in [('source','Source'),('variation','Variation'),('both','Both'),('neither','Neither'),('unsure','Unsure')])}
  </div>
  <input class="comment" placeholder="Catatan opsional">
</section>""")
    review_audio = np.concatenate(chunks) if chunks else np.zeros(sample_rate)
    output_path = Path(output_path).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(output_path, review_audio, sample_rate, subtype="PCM_16")
    plan_path = output_path.with_suffix(".csv")
    plan.to_csv(plan_path, index=False)
    if html_path:
        html_path.write_text(f"""<!doctype html>
<html lang="id"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width">
<title>Chord A/B Review</title><style>
body{{font-family:-apple-system,BlinkMacSystemFont,sans-serif;max-width:900px;margin:32px auto;padding:0 16px;background:#f5f5f7;color:#1d1d1f}}
.card{{background:white;border-radius:14px;padding:18px;margin:16px 0;box-shadow:0 2px 10px #0001}}
h1{{margin-bottom:4px}} h2{{margin-top:0}} small{{font-size:14px;color:#666;font-weight:400}}
.players{{display:grid;grid-template-columns:1fr 1fr;gap:16px}} audio{{display:block;width:100%;margin-top:6px}}
.choices{{display:flex;flex-wrap:wrap;gap:14px;margin:16px 0}} .comment{{width:100%;box-sizing:border-box;padding:10px}}
button{{font-size:16px;padding:12px 18px;border:0;border-radius:10px;background:#147ce5;color:white;cursor:pointer}}
@media(max-width:600px){{.players{{grid-template-columns:1fr}}}}
</style></head><body>
<h1>Chord A/B Review</h1><p>Dengarkan Source dan Variation, lalu pilih berdasarkan pendengaran Anda.</p>
{''.join(html_cards)}
<button onclick="downloadCSV()">Download feedback CSV</button>
<script>
function csvCell(value){{return '"'+String(value??'').replaceAll('"','""')+'"'}}
function downloadCSV(){{
 const rows=[['review_order','start','end','source_chord','candidate_chord','choice','comment']];
 document.querySelectorAll('.card').forEach(card=>{{
  const checked=card.querySelector('input[type=radio]:checked');
  rows.push([card.dataset.order,card.dataset.start,card.dataset.end,card.dataset.source,
    card.dataset.candidate,checked?checked.value:'',card.querySelector('.comment').value]);
 }});
 const csv=rows.map(row=>row.map(csvCell).join(',')).join('\\n');
 const link=document.createElement('a');link.href=URL.createObjectURL(new Blob([csv],{{type:'text/csv'}}));
 link.download='chord-feedback.csv';link.click();URL.revokeObjectURL(link.href);
}}
</script></body></html>""")
    return output_path, plan_path


def _manifest_audio_path(manifest_path: Path, track_id: str) -> Path:
    manifest = pd.read_csv(manifest_path)
    matching = manifest[manifest.track_id.astype(str) == track_id]
    if len(matching) != 1:
        raise ValueError(f"Expected one manifest row for {track_id!r}, found {len(matching)}")
    path = Path(str(matching.iloc[0].audio_path)).expanduser()
    if path.is_absolute():
        return path
    project_candidate = Path.cwd() / path
    if project_candidate.exists():
        return project_candidate.resolve()
    return (manifest_path.parent / path).resolve()


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate top-k masked-chord alternatives")
    parser.add_argument("--dataset-root", type=Path, default=Path("data/pretraining"))
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--track-id", required=True)
    parser.add_argument("--output", type=Path, default=Path("artifacts/chord-suggestions.csv"))
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--include-source", action="store_true")
    parser.add_argument("--include-no-chord", action="store_true")
    parser.add_argument("--midi-output", type=Path)
    parser.add_argument("--source-midi-output", type=Path)
    parser.add_argument("--change-rate", type=float, default=0.1)
    parser.add_argument("--manifest", type=Path, default=Path("data/manifest.csv"))
    parser.add_argument("--source-preview-output", type=Path)
    parser.add_argument("--preview-output", type=Path)
    parser.add_argument("--review-output", type=Path)
    parser.add_argument("--review-html-output", type=Path)
    parser.add_argument("--review-count", type=int, default=5)
    parser.add_argument("--review-stem", choices=["vocals", "original"], default="vocals")
    parser.add_argument("--processed-root", type=Path, default=Path("data/processed"))
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "mps", "cuda"])
    args = parser.parse_args()
    summary = generate_chord_suggestions(
        args.dataset_root, args.checkpoint, args.output, track_id=args.track_id,
        top_k=args.top_k, include_source=args.include_source,
        include_no_chord=args.include_no_chord, device=args.device,
    )
    if args.midi_output:
        summary["midi"] = str(render_suggestion_midi(
            args.output, args.midi_output, change_rate=args.change_rate
        ))
        summary["change_rate"] = args.change_rate
    if args.source_midi_output:
        summary["source_midi"] = str(render_suggestion_midi(
            args.output, args.source_midi_output, source_only=True
        ))
    if (
        args.source_preview_output or args.preview_output
        or args.review_output or args.review_html_output
    ):
        audio_path = _manifest_audio_path(args.manifest.expanduser().resolve(), args.track_id)
        if args.source_preview_output:
            if not args.source_midi_output:
                raise ValueError("--source-preview-output requires --source-midi-output")
            summary["source_preview"] = str(render_mixed_audio_preview(
                audio_path, args.source_midi_output, args.source_preview_output
            ))
        if args.preview_output:
            if not args.midi_output:
                raise ValueError("--preview-output requires --midi-output")
            summary["preview"] = str(render_mixed_audio_preview(
                audio_path, args.midi_output, args.preview_output
            ))
        if args.review_output or args.review_html_output:
            if not args.source_midi_output or not args.midi_output:
                raise ValueError(
                    "--review-output requires --source-midi-output and --midi-output"
                )
            review_audio = audio_path
            if args.review_stem == "vocals":
                review_audio = (
                    args.processed_root.expanduser().resolve() / args.track_id / "vocals.wav"
                )
                if not review_audio.is_file():
                    raise FileNotFoundError(
                        f"Vocal stem not found: {review_audio}. Use --review-stem original "
                        "or check --processed-root."
                    )
            review_output = args.review_output or Path("artifacts/chord-ab-review.wav")
            review_path, plan_path = render_ab_review(
                review_audio, args.source_midi_output, args.midi_output, args.output,
                review_output, change_rate=args.change_rate,
                review_count=args.review_count,
                html_output=args.review_html_output,
            )
            summary["review"] = str(review_path)
            summary["review_plan"] = str(plan_path)
            summary["review_stem"] = args.review_stem
            if args.review_html_output:
                summary["review_html"] = str(args.review_html_output.expanduser().resolve())
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
