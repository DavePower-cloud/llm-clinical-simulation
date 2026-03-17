from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import anthropic
import google.generativeai as google_genai
from openai import OpenAI


@dataclass
class ClientBundle:
    openai_client: Optional[OpenAI]
    anthropic_client: Optional[anthropic.Anthropic]
    google_client: Optional[OpenAI]


def _require_env(name: str) -> str:
    value = os.getenv(name)
    if not value:
        raise EnvironmentError(f"Missing required environment variable: {name}")
    return value


def create_clients(
    require_openai: bool = True,
    require_anthropic: bool = True,
    require_google: bool = True,
) -> ClientBundle:
    openai_client: Optional[OpenAI] = None
    anthropic_client: Optional[anthropic.Anthropic] = None
    google_client: Optional[OpenAI] = None

    if require_openai:
        _require_env("OPENAI_API_KEY")
        openai_client = OpenAI()

    if require_anthropic:
        anthropic_api_key = _require_env("ANTHROPIC_API_KEY")
        anthropic_client = anthropic.Anthropic(api_key=anthropic_api_key)

    if require_google:
        google_api_key = _require_env("GOOGLE_API_KEY")
        google_genai.configure(api_key=google_api_key)
        google_client = OpenAI(
            api_key=google_api_key,
            base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
        )

    return ClientBundle(
        openai_client=openai_client,
        anthropic_client=anthropic_client,
        google_client=google_client,
    )


def call_openai_chat(
    client: OpenAI,
    messages: List[Dict[str, str]],
    model: str,
    temperature: float,
) -> Tuple[str, Dict[str, str]]:
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
    )
    text = response.choices[0].message.content or ""
    return text, {"provider": "openai", "model": model}


def call_anthropic_chat(
    client: anthropic.Anthropic,
    messages: List[Dict[str, str]],
    system_prompt: str,
    model: str,
    temperature: float,
    max_tokens: int = 400,
) -> Tuple[str, Dict[str, str]]:
    cleaned_messages = [m for m in messages if m["role"] != "system"]

    response = client.messages.create(
        model=model,
        system=system_prompt,
        messages=cleaned_messages,
        max_tokens=max_tokens,
        temperature=temperature,
    )

    text_parts = []
    for block in response.content:
        if getattr(block, "type", None) == "text":
            text_parts.append(block.text)

    text = "\n".join(text_parts).strip()
    return text, {"provider": "anthropic", "model": model}


def call_google_chat(
    client: OpenAI,
    messages: List[Dict[str, str]],
    model: str,
    temperature: float,
) -> Tuple[str, Dict[str, str]]:
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
    )
    text = response.choices[0].message.content or ""
    return text, {"provider": "google", "model": model}
