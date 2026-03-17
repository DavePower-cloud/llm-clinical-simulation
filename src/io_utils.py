from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List

from .models import Conversation, Turn


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def generate_conversation_id(prefix: str = "conv") -> str:
    return f"{prefix}_{uuid.uuid4().hex[:8]}"


def ensure_parent_dir(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def append_jsonl(path: Path, item: Dict[str, Any]) -> None:
    ensure_parent_dir(path)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(item, ensure_ascii=False) + "\n")


def save_json(path: Path, payload: Dict[str, Any]) -> None:
    ensure_parent_dir(path)
    with path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False, indent=2)


def turn_to_dict(turn: Turn) -> Dict[str, Any]:
    return {
        "conversation_id": turn.conversation_id,
        "turn": turn.turn,
        "speaker": turn.speaker,
        "text": turn.text,
        "timestamp": turn.timestamp,
        "meta": turn.meta,
    }


def conversation_to_payload(
    meta: Dict[str, Any],
    conversation: Conversation,
) -> Dict[str, Any]:
    return {
        **meta,
        "ended_at": now_iso(),
        "num_turns": len(conversation),
        "turns": [turn_to_dict(t) for t in conversation],
    }


def transcript_text(conversation: Conversation) -> str:
    return "\n".join(f"{turn.speaker}: {turn.text}" for turn in conversation)
