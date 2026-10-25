from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

import librosa
import numpy as np
import pandas as pd

_AUDIO_EXTENSIONS = {".mp3", ".wav", ".flac", ".ogg"}
_MANUAL_COLUMNS = [
    "title", "version", "performer", "arranger", "primary_genre", "style_tags",
    "genre_source", "genre_confidence",
]
_COLUMNS = [
    "track_id", "work_id", *_MANUAL_COLUMNS, "audio_path", "work_group", "sha256",
    "duration_seconds", "status",
]


def _file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _display_name(value: str) -> str:
    value = re.sub(r"^(?:work|track)-\d+__", "", value, flags=re.IGNORECASE)
    return re.sub(r"[-_]+", " ", value).strip().title()


def _next_id(prefix: str, existing: set[str]) -> str:
    numbers = [int(match.group(1)) for value in existing if (match := re.fullmatch(fr"{prefix}-(\d+)", value))]
    candidate = max(numbers, default=0) + 1
    identifier = f"{prefix}-{candidate:04d}"
    existing.add(identifier)
    return identifier


def _portable_path(path: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(Path.cwd().resolve()).as_posix()
    except ValueError:
        return resolved.as_posix()


def build_manifest(audio_root: str | Path, manifest_path: str | Path) -> dict[str, int]:
    """Scan audio folders and create or refresh a metadata manifest."""
    audio_root = Path(audio_root).expanduser().resolve()
    manifest_path = Path(manifest_path).expanduser().resolve()
    if not audio_root.is_dir():
        raise FileNotFoundError(audio_root)

    existing = pd.read_csv(manifest_path, dtype=str).fillna("") if manifest_path.exists() else pd.DataFrame()
    existing_by_path = {
        str(row.audio_path): row._asdict() for row in existing.itertuples(index=False)
    } if "audio_path" in existing else {}
    existing_by_hash = {
        str(row.sha256): row._asdict() for row in existing.itertuples(index=False) if str(row.sha256)
    } if "sha256" in existing else {}
    work_by_group = {
        str(row.work_group): str(row.work_id) for row in existing.itertuples(index=False)
    } if "work_group" in existing else {}
    track_ids = set(existing.get("track_id", pd.Series(dtype=str)).astype(str))
    work_ids = set(existing.get("work_id", pd.Series(dtype=str)).astype(str))

    audio_files = sorted(
        path for path in audio_root.rglob("*")
        if path.is_file() and path.suffix.lower() in _AUDIO_EXTENSIONS
    )
    rows: list[dict[str, object]] = []
    seen_hashes: set[str] = set()
    duplicate_count = 0
    invalid_count = 0

    for path in audio_files:
        digest = _file_hash(path)
        if digest in seen_hashes:
            duplicate_count += 1
            continue
        seen_hashes.add(digest)
        portable_path = _portable_path(path)
        relative = path.relative_to(audio_root)
        group = relative.parts[0] if len(relative.parts) > 1 else f"__file__/{path.stem}"
        old = existing_by_path.get(portable_path) or existing_by_hash.get(digest, {})

        work_id = str(old.get("work_id", "")) or work_by_group.get(group, "")
        if not work_id:
            work_id = _next_id("work", work_ids)
        work_by_group[group] = work_id
        track_id = str(old.get("track_id", "")) or _next_id("track", track_ids)

        try:
            duration = round(float(librosa.get_duration(path=path)), 3)
            status = "ready"
        except Exception:
            duration = np.nan
            status = "unreadable"
            invalid_count += 1

        default_title = _display_name(group.split("/", 1)[-1])
        default_version = _display_name(path.stem)
        row: dict[str, object] = {
            "track_id": track_id,
            "work_id": work_id,
            "title": old.get("title") or default_title,
            "version": old.get("version") or default_version,
            "performer": old.get("performer", ""),
            "arranger": old.get("arranger", ""),
            "primary_genre": old.get("primary_genre") or "unknown",
            "style_tags": old.get("style_tags", ""),
            "genre_source": old.get("genre_source") or "unlabeled",
            "genre_confidence": old.get("genre_confidence") or "0.0",
            "audio_path": portable_path,
            "work_group": group,
            "sha256": digest,
            "duration_seconds": duration,
            "status": status,
        }
        rows.append(row)

    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest = pd.DataFrame(rows, columns=_COLUMNS)
    if not manifest.empty:
        manifest = manifest.sort_values(["work_id", "track_id"])
    manifest.to_csv(manifest_path, index=False)
    return {
        "audio_files": len(audio_files),
        "manifest_tracks": len(manifest),
        "works": int(manifest.work_id.nunique()) if not manifest.empty else 0,
        "duplicates_skipped": duplicate_count,
        "unreadable": invalid_count,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate the Reharmonize audio manifest")
    parser.add_argument("audio_root", type=Path, nargs="?", default=Path("data/audio"))
    parser.add_argument("--manifest", type=Path, default=Path("data/manifest.csv"))
    args = parser.parse_args()
    print(json.dumps(build_manifest(args.audio_root, args.manifest), indent=2))


if __name__ == "__main__":
    main()
