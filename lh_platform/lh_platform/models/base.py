"""Base class for all models."""

from __future__ import annotations

import copy
import hashlib
import json
import uuid
from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

MODEL_GENERATION_EVENT_SCHEMA = "lh_model_generation_event/v1"


def sha256_text(value: str) -> str:
    """Return the persisted digest format used by generation evidence."""

    return "sha256:" + hashlib.sha256(value.encode("utf-8")).hexdigest()


def generation_event_sha256(event: Mapping[str, Any]) -> str:
    """Hash one event excluding its self-referential digest field.

    The runner persists this digest alongside an action-to-response binding.
    It is not a signature against malicious in-process code; it is an
    integrity boundary between the trusted model adapter and ordinary method
    adapters, and catches later artifact mutation.
    """

    payload = {str(key): value for key, value in event.items() if key != "event_sha256"}
    encoded = json.dumps(
        payload,
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":"),
        default=str,
    ).encode("utf-8")
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


@dataclass
class Message:
    role: str
    content: str


@dataclass
class ModelOutput:
    text: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cost_usd: float = 0.0
    raw: object = None
    # Optional: token-level logprobs / top-logprobs.
    # Shape (matches OpenAI's `logprobs` field):
    #   {"tokens": [str, ...],
    #    "token_logprobs": [float, ...],
    #    "top_logprobs": [{"<tok>": float, ...}, ...]}
    # May be None for models that don't support it (e.g. Dummy).
    logprobs: dict[str, Any] | None = None
    # Optional method-side annotations, e.g. 'jitter:on' for JitRL.
    info: dict[str, Any] = field(default_factory=dict)
    # Runner-bound model provenance.  Methods may read these fields for
    # diagnostics, but official evaluation never trusts them directly: it
    # reads the immutable event ledger from the model instance.
    generation_id: str | None = None
    generation_event_sha256: str | None = None


class ModelBase(ABC):
    """Abstract LLM interface."""

    name: str = "base"

    @abstractmethod
    def generate(self, messages: list[Message], **kwargs: object) -> ModelOutput: ...

    def chat(self, user: str, system: str | None = None, **kwargs: object) -> ModelOutput:
        msgs: list[Message] = []
        if system:
            msgs.append(Message(role="system", content=system))
        msgs.append(Message(role="user", content=user))
        return self.generate(msgs, **kwargs)

    def _record_generation_event(
        self,
        messages: Sequence[Message],
        output: ModelOutput,
        *,
        outcome: str,
        backend: str | None = None,
        requested_model: str | None = None,
    ) -> ModelOutput:
        """Append one model-owned, content-addressed generation event.

        A remote-call counter only proves that *some* request occurred.  The
        event ledger additionally records the response digest, so the Runner
        can prove that a persisted action/patch was exactly a model response
        observed during this ``predict`` call.  We deliberately keep only
        hashes of prompts/responses in the ledger to avoid copying potentially
        sensitive task content into generic provenance artifacts.
        """

        ledger = self._generation_ledger()
        sequence = len(ledger) + 1
        instance_id = self._generation_instance_id()
        request_payload = [
            {"role": str(message.role), "content": str(message.content)} for message in messages
        ]
        response_text = str(output.text or "")
        event: dict[str, Any] = {
            "schema": MODEL_GENERATION_EVENT_SCHEMA,
            "generation_id": f"{self.name}:{instance_id}:{sequence}",
            "sequence": sequence,
            "requested_model": requested_model
            or str(getattr(self, "model_name", None) or self.name),
            "backend": backend or str(output.info.get("backend") or self.name),
            "outcome": str(outcome),
            "request_sha256": sha256_text(
                json.dumps(
                    request_payload, sort_keys=True, ensure_ascii=False, separators=(",", ":")
                )
            ),
            "response_sha256": sha256_text(response_text),
            "response_chars": len(response_text),
        }
        event["event_sha256"] = generation_event_sha256(event)
        ledger.append(copy.deepcopy(event))

        # This is convenient for method debugging, but is intentionally not
        # used as the source of truth by the Runner (methods can edit output
        # annotations before returning an Action).
        output.generation_id = str(event["generation_id"])
        output.generation_event_sha256 = str(event["event_sha256"])
        output.info = dict(output.info or {})
        output.info.setdefault("generation_id", output.generation_id)
        output.info.setdefault("generation_event_sha256", output.generation_event_sha256)
        return output

    def generation_cursor(self) -> int:
        """Return the current model-owned event count for Runner snapshots."""

        return len(self._generation_ledger())

    def generation_events_since(self, cursor: int) -> tuple[dict[str, Any], ...]:
        """Return defensive copies of events created after ``cursor``.

        The method layer receives no mutable reference to the persisted
        ledger.  The Runner records selected event facts before a method's
        ``update`` hook executes.
        """

        if isinstance(cursor, bool) or not isinstance(cursor, int) or cursor < 0:
            return ()
        return tuple(copy.deepcopy(self._generation_ledger()[cursor:]))

    def _generation_ledger(self) -> list[dict[str, Any]]:
        ledger = getattr(self, "_lh_generation_ledger", None)
        if not isinstance(ledger, list):
            ledger = []
            self._lh_generation_ledger = ledger
        return ledger

    def _generation_instance_id(self) -> str:
        instance_id = getattr(self, "_lh_generation_instance_id", None)
        if not isinstance(instance_id, str) or not instance_id:
            instance_id = uuid.uuid4().hex
            self._lh_generation_instance_id = instance_id
        return instance_id

    def runtime_evidence(self) -> dict[str, Any]:
        """Return model-owned execution facts for the run evidence contract.

        Implementations should expose whether a real backend or a fallback was
        actually used.  The default is deliberately conservative: an adapter
        that does not implement this method cannot be mistaken for a verified
        local-model execution.
        """

        return {
            "backend": self.name,
            "model_instantiated": True,
            "dummy_model_used": False,
            "dummy_fallback_used": False,
        }
