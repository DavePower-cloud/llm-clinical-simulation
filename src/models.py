from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class AgentSpec:
    agent_id: str
    provider: str
    model: str
    system_prompt: str


@dataclass
class Turn:
    conversation_id: str
    turn: int
    speaker: str
    text: str
    timestamp: str
    meta: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ConversationMeta:
    conversation_id: str
    started_at: str
    agents: Dict[str, str]
    role_guard_failures: int = 0


Conversation = List[Turn]


ROLE_RULES: Dict[str, List[str]] = {
    "doctor": ["as the patient", "as the nurse", "my chest pain"],
    "patient": ["ecg", "blood pressure is", "st elevation", "diagnosis"],
    "nurse": ["diagnosis", "heart attack confirmed", "treatment plan"],
}


DEFAULT_ORDER = ["doctor", "patient", "nurse"]
