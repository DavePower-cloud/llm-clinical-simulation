from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from openai import OpenAI

from .judge import JudgeConfig, evaluate_conversation_automatically, load_openai_client


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def conversation_json_to_text(convo_json: Dict[str, Any]) -> str:
    """
    Convert a saved conversation JSON payload into readable transcript text.

    Expected structure:
    {
      "conversation_id": "...",
      "num_turns": 10,
      "role_guard_failures": 0,
      "turns": [
        {"speaker": "doctor", "text": "..."},
        ...
      ]
    }
    """
    if "turns" not in convo_json or not isinstance(convo_json["turns"], list):
        raise ValueError("Conversation JSON missing 'turns' list.")

    lines: List[str] = []

    for idx, turn in enumerate(convo_json["turns"], start=1):
        if not isinstance(turn, dict):
            raise ValueError(f"Turn {idx} is not a dict.")
        speaker = str(turn.get("speaker", "UNKNOWN")).upper().strip()
        text = str(turn.get("text", "")).strip()
        lines.append(f"{speaker}: {text}")

    return "\n\n".join(lines)


def load_conversation_json(file_path: Path) -> Dict[str, Any]:
    """
    Load a conversation JSON file.
    """
    with file_path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def build_evaluation_record(
    convo: Dict[str, Any],
    file_name: str,
    judge_result: Dict[str, Any],
    model: str,
) -> Dict[str, Any]:
    """
    Build a structured evaluation record.
    """
    return {
        "conversation_id": convo.get("conversation_id"),
        "file": file_name,
        "num_turns": convo.get("num_turns"),
        "role_guard_failures": convo.get("role_guard_failures"),
        "judge_model": model,
        "judge": judge_result,
        "evaluated_at": now_iso(),
    }


def append_jsonl(path: Path, record: Dict[str, Any]) -> None:
    """
    Append one JSON record to a JSONL file.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as out:
        out.write(json.dumps(record, ensure_ascii=False) + "\n")


def save_json(path: Path, payload: Dict[str, Any]) -> None:
    """
    Save a single JSON payload.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as out:
        json.dump(payload, out, ensure_ascii=False, indent=2)


def evaluate_conversation_file(
    json_file: Path,
    client: Optional[OpenAI] = None,
    config: Optional[JudgeConfig] = None,
    output_jsonl: Optional[Path] = None,
    per_file_output_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    """
    Evaluate one saved conversation JSON file.

    Parameters
    ----------
    json_file : Path
        Path to a conversation JSON file.
    client : OpenAI, optional
        Existing OpenAI client.
    config : JudgeConfig, optional
        Judge configuration.
    output_jsonl : Path, optional
        JSONL file to append results to.
    per_file_output_dir : Path, optional
        If provided, also saves one JSON result file per conversation.

    Returns
    -------
    dict
        Structured evaluation record.
    """
    if not json_file.exists():
        raise FileNotFoundError(f"Conversation file not found: {json_file}")

    config = config or JudgeConfig()
    client = client or load_openai_client()

    convo = load_conversation_json(json_file)
    conversation_text = conversation_json_to_text(convo)

    judge_result = evaluate_conversation_automatically(
        conversation_text=conversation_text,
        client=client,
        config=config,
    )

    record = build_evaluation_record(
        convo=convo,
        file_name=json_file.name,
        judge_result=judge_result,
        model=config.model,
    )

    if output_jsonl is not None:
        append_jsonl(output_jsonl, record)

    if per_file_output_dir is not None:
        per_file_output_dir.mkdir(parents=True, exist_ok=True)
        convo_id = record.get("conversation_id") or json_file.stem
        save_json(per_file_output_dir / f"{convo_id}.judge.json", record)

    return record


def evaluate_conversation_batch(
    conversations_dir: Path | str,
    output_path: Path | str = "data/judge_results.jsonl",
    model: str = "gpt-4o",
    temperature: float = 0.0,
    max_retries: int = 3,
    per_file_output_dir: Optional[Path | str] = None,
    fail_fast: bool = False,
) -> List[Dict[str, Any]]:
    """
    Evaluate all conversation JSON files in a directory using the LLM judge.

    Parameters
    ----------
    conversations_dir : str or Path
        Folder containing conversation JSON files.
    output_path : str or Path
        JSONL file to append judge results to.
    model : str
        Judge model name.
    temperature : float
        Judge model temperature, usually 0.0.
    max_retries : int
        Number of retries per file.
    per_file_output_dir : str or Path, optional
        Directory to save one JSON result per conversation.
    fail_fast : bool
        If True, stop on first error. If False, continue through batch.

    Returns
    -------
    list of dict
        All successful evaluation records.
    """
    conversations_dir = Path(conversations_dir)
    output_path = Path(output_path)
    per_file_output = Path(per_file_output_dir) if per_file_output_dir else None

    if not conversations_dir.exists():
        raise FileNotFoundError(f"Directory not found: {conversations_dir}")

    client = load_openai_client()
    config = JudgeConfig(
        model=model,
        temperature=temperature,
        max_retries=max_retries,
    )

    results: List[Dict[str, Any]] = []
    files = sorted(conversations_dir.glob("*.json"))

    if not files:
        raise FileNotFoundError(f"No .json conversation files found in: {conversations_dir}")

    for json_file in files:
        try:
            record = evaluate_conversation_file(
                json_file=json_file,
                client=client,
                config=config,
                output_jsonl=output_path,
                per_file_output_dir=per_file_output,
            )
            results.append(record)
            print(f"✓ Evaluated {json_file.name}")

        except Exception as exc:
            print(f"✗ Failed on {json_file.name}: {exc}")
            if fail_fast:
                raise

    return results


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate saved clinical conversation JSON files using an LLM judge."
    )
    parser.add_argument(
        "--input-dir",
        default="data/conversations",
        help="Directory containing conversation JSON files.",
    )
    parser.add_argument(
        "--output-jsonl",
        default="data/judge_results.jsonl",
        help="Output JSONL file for batch results.",
    )
    parser.add_argument(
        "--per-file-output-dir",
        default="data/judge_results",
        help="Optional directory for one JSON result per conversation.",
    )
    parser.add_argument(
        "--model",
        default="gpt-4o",
        help="Judge model name.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Judge temperature.",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=3,
        help="Max retries per evaluation.",
    )
    parser.add_argument(
        "--fail-fast",
        action="store_true",
        help="Stop on first failed file.",
    )
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()

    results = evaluate_conversation_batch(
        conversations_dir=args.input_dir,
        output_path=args.output_jsonl,
        model=args.model,
        temperature=args.temperature,
        max_retries=args.max_retries,
        per_file_output_dir=args.per_file_output_dir,
        fail_fast=args.fail_fast,
    )

    print(f"\nCompleted {len(results)} evaluations.")


if __name__ == "__main__":
    main()
