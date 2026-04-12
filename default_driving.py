"""Persisted default square driving video (set by admin) for photo-only user flows."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent / "data"
FILE_PATH = DATA_DIR / "default_driving.mp4"
META_PATH = DATA_DIR / "default_driving_meta.json"


def has_default() -> bool:
    return FILE_PATH.is_file() and FILE_PATH.stat().st_size > 0


def path_str() -> str:
    return str(FILE_PATH.resolve())


def write_meta(set_by_chat_id: int, width: int, height: int) -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    META_PATH.write_text(
        json.dumps(
            {
                "updated_at": datetime.now(timezone.utc).isoformat(),
                "set_by_chat_id": set_by_chat_id,
                "width": width,
                "height": height,
            },
            indent=2,
        ),
        encoding="utf-8",
    )


def read_meta() -> dict | None:
    if not META_PATH.is_file():
        return None
    try:
        return json.loads(META_PATH.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None


def clear_files() -> None:
    for p in (FILE_PATH, META_PATH):
        try:
            if p.is_file():
                p.unlink()
        except OSError:
            pass
