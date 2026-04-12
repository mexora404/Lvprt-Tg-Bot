"""Persisted allow-list of Telegram chat IDs (private chats: same as user id)."""

from __future__ import annotations

import json
import os
from pathlib import Path

ALLOWED_FILE = Path(__file__).resolve().parent / "data" / "allowed_chats.json"


def _ensure_dir() -> None:
    ALLOWED_FILE.parent.mkdir(parents=True, exist_ok=True)


def load_allowed_ids() -> set[int]:
    ids: set[int] = set()
    raw = os.environ.get("ALLOWED_CHAT_IDS", "").strip()
    if raw:
        for part in raw.replace(" ", "").split(","):
            if part.isdigit() or (part.startswith("-") and part[1:].isdigit()):
                ids.add(int(part))
    _ensure_dir()
    if ALLOWED_FILE.is_file():
        try:
            data = json.loads(ALLOWED_FILE.read_text(encoding="utf-8"))
            if isinstance(data, list):
                for x in data:
                    ids.add(int(x))
        except (json.JSONDecodeError, OSError, ValueError):
            pass
    return ids


def save_allowed_ids(ids: set[int]) -> None:
    _ensure_dir()
    ALLOWED_FILE.write_text(json.dumps(sorted(ids), indent=2), encoding="utf-8")


def is_chat_allowed(chat_id: int) -> bool:
    return chat_id in load_allowed_ids()
