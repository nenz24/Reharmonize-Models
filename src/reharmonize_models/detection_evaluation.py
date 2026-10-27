from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import mir_eval.chord
import mir_eval.io
import pandas as pd


def initialize_annotation_drafts(
    processed_root: str | Path,
    annotations_root: str | Path,
    track_ids: list[str],
    *,
    overwrite: bool = False,
) -> dict[str, object]:
    """Copy BTC labels into explicitly unverified drafts for human correction."""
    processed_root = Path(processed_root).expanduser().resolve()
    annotations_root = Path(annotations_root).expanduser().resolve()
    drafts_root = annotations_root / "drafts"
    drafts_root.mkdir(parents=True, exist_ok=True)
    status_path = annotations_root / "annotation-status.csv"
    existing = (
        pd.read_csv(status_path, dtype=str).fillna("")
        if status_path.exists() else pd.DataFrame()
    )
    records = existing.to_dict("records")
    by_track = {str(row["track_id"]): row for row in records}
    created = skipped = 0
    for track_id in track_ids:
        source = processed_root / track_id / "chords.lab"
        if not source.is_file():
            raise FileNotFoundError(source)
        destination = drafts_root / f"{track_id}.lab"
        if destination.exists() and not overwrite:
            skipped += 1
        else:
            shutil.copyfile(source, destination)
            created += 1
        by_track[track_id] = {
            "track_id": track_id,
            "annotation_path": str(destination.relative_to(annotations_root)),
            "status": by_track.get(track_id, {}).get("status", "draft") or "draft",
            "annotator": by_track.get(track_id, {}).get("annotator", ""),
            "notes": by_track.get(track_id, {}).get("notes", ""),
        }
    status = pd.DataFrame(by_track.values()).sort_values("track_id")
    status.to_csv(status_path, index=False)
    (annotations_root / "README.txt").write_text(
        "Each .lab row is: start_seconds end_seconds chord_label.\n"
        "Drafts are copied from BTC and are not ground truth. Correct labels and boundaries, "
        "then set status=verified and record the annotator in annotation-status.csv.\n"
        "The evaluator ignores every row whose status is not verified.\n"
    )
    return {
        "requested": len(track_ids), "created": created, "skipped": skipped,
        "status_file": str(status_path), "drafts_root": str(drafts_root),
    }


def evaluate_verified_annotations(
    processed_root: str | Path,
    annotations_root: str | Path,
    output_path: str | Path,
) -> dict[str, object]:
    """Evaluate BTC estimates only against annotations explicitly marked verified."""
    processed_root = Path(processed_root).expanduser().resolve()
    annotations_root = Path(annotations_root).expanduser().resolve()
    output_path = Path(output_path).expanduser().resolve()
    status_path = annotations_root / "annotation-status.csv"
    if not status_path.is_file():
        raise FileNotFoundError(status_path)
    status = pd.read_csv(status_path, dtype=str).fillna("")
    verified = status[status.status.str.lower() == "verified"]
    if verified.empty:
        raise ValueError("No annotations are marked status=verified")
    rows: list[dict[str, object]] = []
    for item in verified.itertuples(index=False):
        reference_path = annotations_root / str(item.annotation_path)
        estimate_path = processed_root / str(item.track_id) / "chords.lab"
        reference_intervals, reference_labels = mir_eval.io.load_labeled_intervals(
            str(reference_path)
        )
        estimate_intervals, estimate_labels = mir_eval.io.load_labeled_intervals(
            str(estimate_path)
        )
        scores = mir_eval.chord.evaluate(
            reference_intervals, reference_labels, estimate_intervals, estimate_labels
        )
        rows.append({
            "track_id": item.track_id,
            "annotator": item.annotator,
            **{name: round(float(value), 4) for name, value in scores.items()},
        })
    report = pd.DataFrame(rows)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    report.to_csv(output_path, index=False)
    metric_columns = [column for column in report if column not in {"track_id", "annotator"}]
    macro = {column: round(float(report[column].mean()), 4) for column in metric_columns}
    summary: dict[str, object] = {
        "verified_tracks": len(report), "macro": macro, "report": str(output_path),
    }
    output_path.with_suffix(".summary.json").write_text(json.dumps(summary, indent=2))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare and evaluate verified chord annotations")
    subparsers = parser.add_subparsers(dest="command", required=True)
    initialize = subparsers.add_parser("init", help="Create unverified BTC-based annotation drafts")
    initialize.add_argument("--processed-root", type=Path, default=Path("data/processed"))
    initialize.add_argument("--annotations-root", type=Path, default=Path("data/annotations"))
    initialize.add_argument("--track-id", action="append", dest="track_ids", required=True)
    initialize.add_argument("--overwrite", action="store_true")
    evaluate = subparsers.add_parser("evaluate", help="Score BTC against verified annotations")
    evaluate.add_argument("--processed-root", type=Path, default=Path("data/processed"))
    evaluate.add_argument("--annotations-root", type=Path, default=Path("data/annotations"))
    evaluate.add_argument("--output", type=Path, default=Path("data/annotations/evaluation.csv"))
    args = parser.parse_args()
    result = (
        initialize_annotation_drafts(
            args.processed_root, args.annotations_root, args.track_ids, overwrite=args.overwrite
        ) if args.command == "init" else
        evaluate_verified_annotations(args.processed_root, args.annotations_root, args.output)
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
