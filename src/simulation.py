from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from dotenv import load_dotenv

from .clients import (
    ClientBundle,
    call_anthropic_chat,
    call_google_chat,
    call_openai_chat,
    create_clients,
)
from .config import AppConfig, load_config
from .io_utils import (
    append_jsonl,
    conversation_to_payload,
    generate_conversation_id,
    now_iso,
    save_json,
)
from .models import (
    DEFAULT_ORDER,
    ROLE_RULES,
    AgentSpec,
    Conversation,
    ConversationMeta,
    Turn,
)
from .prompts import load_default_prompts


class SimulationRunner:
    def __init__(self, config: Optional[AppConfig] = None) -> None:
        load_dotenv(override=True)

        self.config = config or load_config()
        self.prompts = load_default_prompts()
        self.clients: ClientBundle = create_clients()

        self.agents: Dict[str, AgentSpec] = {
            "doctor": AgentSpec(
                agent_id="doctor",
                provider=self.config.doctor.provider,
                model=self.config.doctor.model,
                system_prompt=self.prompts.doctor,
            ),
            "patient": AgentSpec(
                agent_id="patient",
                provider=self.config.patient.provider,
                model=self.config.patient.model,
                system_prompt=self.prompts.patient,
            ),
            "nurse": AgentSpec(
                agent_id="nurse",
                provider=self.config.nurse.provider,
                model=self.prompts and self.config.nurse.model,
                system_prompt=self.prompts.nurse,
            ),
        }

        self.conversation_id = generate_conversation_id()
        self.conversation: Conversation = []
        self.meta = ConversationMeta(
            conversation_id=self.conversation_id,
            started_at=now_iso(),
            agents={
                "doctor": self.agents["doctor"].model,
                "patient": self.agents["patient"].model,
                "nurse": self.agents["nurse"].model,
            },
            role_guard_failures=0,
        )

    def reset(self) -> None:
        self.conversation_id = generate_conversation_id()
        self.conversation = []
        self.meta = ConversationMeta(
            conversation_id=self.conversation_id,
            started_at=now_iso(),
            agents={
                "doctor": self.agents["doctor"].model,
                "patient": self.agents["patient"].model,
                "nurse": self.agents["nurse"].model,
            },
            role_guard_failures=0,
        )

    def append_utterance(
        self,
        speaker: str,
        text: str,
        meta: Optional[Dict[str, Any]] = None,
        log_to_jsonl: bool = True,
    ) -> Turn:
        turn = Turn(
            conversation_id=self.conversation_id,
            turn=len(self.conversation) + 1,
            speaker=speaker,
            text=text,
            timestamp=now_iso(),
            meta=meta or {},
        )
        self.conversation.append(turn)

        if log_to_jsonl:
            append_jsonl(self.config.log_file, {
                "conversation_id": turn.conversation_id,
                "turn": turn.turn,
                "speaker": turn.speaker,
                "text": turn.text,
                "timestamp": turn.timestamp,
                "meta": turn.meta,
            })

        return turn

    def get_recent_context(self) -> Conversation:
        return deepcopy(self.conversation[-self.config.context_window :])

    def build_view_for(self, agent_id: str) -> List[Dict[str, str]]:
        agent = self.agents[agent_id]
        messages: List[Dict[str, str]] = []

        if agent.provider in {"openai", "google"}:
            messages.append({"role": "system", "content": agent.system_prompt})

        for item in self.get_recent_context():
            role = "assistant" if item.speaker == agent_id else "user"
            messages.append({"role": role, "content": item.text})

        return messages

    def detect_role_leakage(self, agent_id: str, text: Any) -> Tuple[bool, List[str]]:
        if not isinstance(text, str):
            return False, ["non_string_response"]

        lower = text.lower()
        violations = [
            phrase for phrase in ROLE_RULES.get(agent_id, [])
            if phrase in lower
        ]
        return len(violations) == 0, violations

    def _temperature(self, explicit_temperature: Optional[float]) -> float:
        if self.config.deterministic:
            return 0.0
        if explicit_temperature is not None:
            return explicit_temperature
        return self.config.default_temperature

    def call_model_for_agent(
        self,
        agent_id: str,
        temperature: Optional[float] = None,
        max_retries: Optional[int] = None,
    ) -> Tuple[str, Dict[str, Any]]:
        agent = self.agents[agent_id]
        temp = self._temperature(temperature)
        retries = max_retries if max_retries is not None else self.config.max_retries

        for attempt in range(1, retries + 1):
            messages = self.build_view_for(agent_id)

            if agent.provider == "openai":
                if not self.clients.openai_client:
                    raise RuntimeError("OpenAI client is not initialised.")
                text, meta = call_openai_chat(
                    client=self.clients.openai_client,
                    messages=messages,
                    model=agent.model,
                    temperature=temp,
                )
            elif agent.provider == "anthropic":
                if not self.clients.anthropic_client:
                    raise RuntimeError("Anthropic client is not initialised.")
                text, meta = call_anthropic_chat(
                    client=self.clients.anthropic_client,
                    messages=messages,
                    system_prompt=agent.system_prompt,
                    model=agent.model,
                    temperature=temp,
                )
            elif agent.provider == "google":
                if not self.clients.google_client:
                    raise RuntimeError("Google client is not initialised.")
                text, meta = call_google_chat(
                    client=self.clients.google_client,
                    messages=messages,
                    model=agent.model,
                    temperature=temp,
                )
            else:
                raise ValueError(f"Unsupported provider: {agent.provider}")

            if not isinstance(text, str) or not text.strip():
                self.meta.role_guard_failures += 1
                self.append_utterance(
                    agent_id,
                    "[EMPTY OR NULL MODEL RESPONSE – rejected]",
                    meta={"guard": True, "attempt": attempt},
                )
                continue

            valid, violations = self.detect_role_leakage(agent_id, text)

            if valid:
                meta.update({"role_guard": "passed", "attempt": attempt})
                return text.strip(), meta

            self.meta.role_guard_failures += 1
            self.append_utterance(
                agent_id,
                f"[ROLE VIOLATION – rejected: {violations}]",
                meta={"guard": True, "attempt": attempt},
            )

        fallback = {
            "doctor": "I will continue assessing and managing your condition.",
            "patient": "I'm feeling very unwell and frightened.",
            "nurse": "I'll continue close monitoring and support.",
        }
        return fallback[agent_id], {"role_guard": "fallback"}

    def seed_conversation(self) -> None:
        self.append_utterance(
            "doctor",
            "John, I'm the doctor looking after you today. Can you tell me what's happening?",
        )
        self.append_utterance(
            "patient",
            "I… I just have this awful chest pain.",
        )
        self.append_utterance(
            "nurse",
            "I'll start getting vitals now, doctor.",
        )

    def run_one_cycle(
        self,
        order: Optional[List[str]] = None,
        max_cycles: Optional[int] = None,
        verbose: bool = False,
    ) -> Conversation:
        turn_order = order or DEFAULT_ORDER
        cycles = max_cycles if max_cycles is not None else self.config.max_cycles

        for cycle in range(cycles):
            for agent_id in turn_order:
                text, meta = self.call_model_for_agent(agent_id)
                self.append_utterance(
                    agent_id,
                    text,
                    meta={"cycle": cycle + 1, **meta},
                )
                if verbose:
                    print(f"\n[{agent_id.upper()}]\n{text}\n")

        return self.conversation

    def save_conversation(self) -> Path:
        payload = conversation_to_payload(asdict(self.meta), self.conversation)
        self.config.conversations_dir.mkdir(parents=True, exist_ok=True)
        output_path = self.config.conversations_dir / f"{self.meta.conversation_id}.json"
        save_json(output_path, payload)
        return output_path

    def run_and_save(
        self,
        verbose: bool = False,
        order: Optional[List[str]] = None,
        max_cycles: Optional[int] = None,
    ) -> Path:
        self.seed_conversation()
        self.run_one_cycle(order=order, max_cycles=max_cycles, verbose=verbose)
        return self.save_conversation()

    def run_batch(
        self,
        num_runs: int,
        verbose: bool = False,
        order: Optional[List[str]] = None,
        max_cycles: Optional[int] = None,
    ) -> List[Path]:
        saved_paths: List[Path] = []

        for i in range(num_runs):
            self.reset()
            self.seed_conversation()
            self.run_one_cycle(order=order, max_cycles=max_cycles, verbose=verbose)
            path = self.save_conversation()
            saved_paths.append(path)
            print(f"✓ Completed simulation {i + 1}/{num_runs}: {self.meta.conversation_id}")

        return saved_paths
