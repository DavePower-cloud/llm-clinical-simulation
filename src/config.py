from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass(frozen=True)
class ProviderConfig:
    provider: str
    model: str
    api_key_env: Optional[str] = None


@dataclass(frozen=True)
class AppConfig:
    context_window: int
    deterministic: bool
    default_temperature: float
    max_retries: int
    max_cycles: int
    log_file: Path
    conversations_dir: Path

    doctor: ProviderConfig
    patient: ProviderConfig
    nurse: ProviderConfig


def _get_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


def _get_int(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None:
        return default
    return int(value)


def _get_float(name: str, default: float) -> float:
    value = os.getenv(name)
    if value is None:
        return default
    return float(value)


def load_config() -> AppConfig:
    output_root = Path(os.getenv("SIM_OUTPUT_DIR", "data"))
    conversations_dir = output_root / "conversations"
    log_file = output_root / "simulations.jsonl"

    return AppConfig(
        context_window=_get_int("SIM_CONTEXT_WINDOW", 6),
        deterministic=_get_bool("SIM_DETERMINISTIC", False),
        default_temperature=_get_float("SIM_TEMPERATURE", 0.3),
        max_retries=_get_int("SIM_MAX_RETRIES", 2),
        max_cycles=_get_int("SIM_MAX_CYCLES", 5),
        log_file=log_file,
        conversations_dir=conversations_dir,
        doctor=ProviderConfig(
            provider="openai",
            model=os.getenv("DOCTOR_MODEL", "gpt-4o-mini"),
            api_key_env="OPENAI_API_KEY",
        ),
        patient=ProviderConfig(
            provider="anthropic",
            model=os.getenv("PATIENT_MODEL", "claude-haiku-4-5-20251001"),
            api_key_env="ANTHROPIC_API_KEY",
        ),
        nurse=ProviderConfig(
            provider="google",
            model=os.getenv("NURSE_MODEL", "gemini-2.0-flash-exp"),
            api_key_env="GOOGLE_API_KEY",
        ),
    )
