from __future__ import annotations

import json
import os
import re
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from dotenv import load_dotenv
from openai import OpenAI


JUDGE_SYSTEM_PROMPT = """You are an expert clinical educator evaluating simulated emergency department conversations.

You MUST respond with valid JSON only.
Do not include markdown, explanations, or extra text.
Do not wrap the JSON in code fences.

Evaluate realism, role fidelity, and educational suitability.
Be conservative in your ratings.
"""

JUDGE_USER_PROMPT = """
Conversation:
----------------
{FULL_CONVERSATION_TEXT}
----------------

Please rate the conversation using the following scale:

1 = Strongly disagree
2 = Disagree
3 = Neutral
4 = Agree
5 = Strongly agree

Return your answer in JSON with EXACTLY this structure:

{{
  "role_fidelity": 1-5,
  "turn_coherence": 1-5,
  "communication_realism": 1-5,
  "educational_usable": true/false,
  "comments": "string"
}}
""".strip()


@dataclass(frozen=True)
class JudgeConfig:
    model: str = "gpt-4o"
    temperature: float = 0.0
    max_retries: int = 3
    retry_sleep_seconds: float = 1.5


REQUIRED_KEYS = {
    "role_fidelity",
    "turn_coherence",
    "communication_realism",
    "educational_usable",
    "comments",
}


def load_openai_client() -> OpenAI:
    """
    Load environment variables and create an OpenAI client.

    Raises
    ------
    EnvironmentError
        If OPENAI_API_KEY is missing.
    """
    load_dotenv(override=True)
    if not os.getenv("OPENAI_API_KEY"):
        raise EnvironmentError("Missing OPENAI_API_KEY in environment or .env file.")
    return OpenAI()


def build_judge_messages(conversation_text: str) -> List[Dict[str, str]]:
    """
    Build chat messages for the judge model.
    """
    return [
        {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": JUDGE_USER_PROMPT.format(
                FULL_CONVERSATION_TEXT=conversation_text
            ),
        },
    ]


def _extract_json_object(text: str) -> str:
    """
    Extract the first JSON object from a string.

    Handles cases where the model accidentally adds prose before/after JSON.
    """
    text = text.strip()

    # Direct success case
    if text.startswith("{") and text.endswith("}"):
        return text

    # Remove code fences if present
    text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s*```$", "", text)

    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError("No JSON object found in model response.")

    return text[start : end + 1]


def _coerce_bool(value: Any) -> bool:
    """
    Coerce common truthy/falsy values into bool.
    """
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"true", "yes", "y", "1"}:
            return True
        if lowered in {"false", "no", "n", "0"}:
            return False
    if isinstance(value, (int, float)):
        return bool(value)
    raise ValueError(f"Cannot coerce to bool: {value!r}")


def _coerce_int_1_to_5(value: Any, field_name: str) -> int:
    """
    Coerce a field to int and validate that it lies in [1, 5].
    """
    try:
        num = int(value)
    except Exception as exc:
        raise ValueError(f"{field_name} is not an integer: {value!r}") from exc

    if num < 1 or num > 5:
        raise ValueError(f"{field_name} must be between 1 and 5, got {num}.")
    return num


def validate_judge_result(payload: Dict[str, Any]) -> Dict[str, Any]:
    """
    Validate and normalise the judge JSON payload.

    Returns
    -------
    dict
        Normalised payload with correct types.

    Raises
    ------
    ValueError
        If required keys are missing or fields are invalid.
    """
    missing = REQUIRED_KEYS - set(payload.keys())
    if missing:
        raise ValueError(f"Judge result missing required keys: {sorted(missing)}")

    normalised = {
        "role_fidelity": _coerce_int_1_to_5(payload["role_fidelity"], "role_fidelity"),
        "turn_coherence": _coerce_int_1_to_5(payload["turn_coherence"], "turn_coherence"),
        "communication_realism": _coerce_int_1_to_5(
            payload["communication_realism"], "communication_realism"
        ),
        "educational_usable": _coerce_bool(payload["educational_usable"]),
        "comments": str(payload["comments"]).strip(),
    }
    return normalised


def parse_judge_response(raw_text: str) -> Dict[str, Any]:
    """
    Parse and validate a raw judge model response.
    """
    json_text = _extract_json_object(raw_text)
    payload = json.loads(json_text)
    return validate_judge_result(payload)


def evaluate_conversation_automatically(
    conversation_text: str,
    client: Optional[OpenAI] = None,
    config: Optional[JudgeConfig] = None,
) -> Dict[str, Any]:
    """
    Evaluate a conversation transcript using the OpenAI judge model.

    Parameters
    ----------
    conversation_text : str
        Transcript text to evaluate.
    client : OpenAI, optional
        Existing OpenAI client. If omitted, one is created from environment.
    config : JudgeConfig, optional
        Judge configuration.

    Returns
    -------
    dict
        Normalised judge result.

    Raises
    ------
    RuntimeError
        If all retries fail.
    """
    if not conversation_text or not conversation_text.strip():
        raise ValueError("conversation_text is empty.")

    client = client or load_openai_client()
    config = config or JudgeConfig()
    messages = build_judge_messages(conversation_text)

    last_error: Optional[Exception] = None

    for attempt in range(1, config.max_retries + 1):
        try:
            response = client.chat.completions.create(
                model=config.model,
                messages=messages,
                temperature=config.temperature,
            )

            raw_text = response.choices[0].message.content or ""
            result = parse_judge_response(raw_text)
            return result

        except Exception as exc:
            last_error = exc
            if attempt < config.max_retries:
                time.sleep(config.retry_sleep_seconds)

    raise RuntimeError(
        f"Judge evaluation failed after {config.max_retries} attempts: {last_error}"
    ) from last_error
