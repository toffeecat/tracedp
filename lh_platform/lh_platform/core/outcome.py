"""Outcome Evidence primitives."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

EVIDENCE_STRONG = "strong"
EVIDENCE_WEAK = "weak"
EVIDENCE_ABSENT = "absent"
EVIDENCE_QUALITIES = frozenset({EVIDENCE_STRONG, EVIDENCE_WEAK, EVIDENCE_ABSENT})


@dataclass
class OutcomeEvidence:
    """Benchmark-owned evidence explaining task success or failure."""

    success: bool
    reason: str
    evidence_quality: str
    source: str
    final_answer: str | None = None
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    relevant_apis: list[str] = field(default_factory=list)
    test_results: dict[str, Any] | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def validate(self) -> OutcomeEvidence:
        if type(self.success) is not bool:
            raise ValueError("outcome success must be boolean")
        if not isinstance(self.reason, str) or not self.reason.strip():
            raise ValueError("outcome evidence requires a reason")
        if self.evidence_quality not in EVIDENCE_QUALITIES:
            raise ValueError("outcome evidence quality is unsupported")
        if not isinstance(self.source, str) or not self.source.strip():
            raise ValueError("outcome evidence requires a source")
        if self.final_answer is not None and not isinstance(self.final_answer, str):
            raise ValueError("outcome final_answer must be text or null")
        if not isinstance(self.tool_calls, list) or not all(
            isinstance(item, dict) for item in self.tool_calls
        ):
            raise ValueError("outcome tool_calls must be a list of objects")
        if not isinstance(self.relevant_apis, list) or not all(
            isinstance(item, str) for item in self.relevant_apis
        ):
            raise ValueError("outcome relevant_apis must be a list of strings")
        if self.test_results is not None and not isinstance(self.test_results, dict):
            raise ValueError("outcome test_results must be an object or null")
        if not isinstance(self.extra, dict):
            raise ValueError("outcome extra must be an object")
        return self

    @classmethod
    def producer_error(cls, detail: str) -> OutcomeEvidence:
        """The benchmark's outcome producer failed; never inherit the actor's self-report."""

        return cls(
            success=False,
            reason=f"benchmark outcome producer failed: {detail}",
            evidence_quality=EVIDENCE_ABSENT,
            source="runner_outcome_producer_error",
            extra={"outcome_producer_error": detail},
        ).validate()

    @classmethod
    def fallback(cls, success: bool = False) -> OutcomeEvidence:
        return cls(
            success=bool(success),
            reason="no benchmark outcome evidence produced",
            evidence_quality=EVIDENCE_ABSENT,
            source="runner_fallback",
        ).validate()


__all__ = [
    "EVIDENCE_ABSENT",
    "EVIDENCE_QUALITIES",
    "EVIDENCE_STRONG",
    "EVIDENCE_WEAK",
    "OutcomeEvidence",
]
