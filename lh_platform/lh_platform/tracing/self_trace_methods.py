"""Capability inventory for tracing methods admitted to self-trace experiments."""

from __future__ import annotations

from dataclasses import dataclass

from lh_platform.tracing.registry import TRACERS
from lh_platform.tracing.types import EvidenceLevel


@dataclass(frozen=True)
class SelfTraceMethodV1:
    trace_method_id: str
    capabilities: tuple[str, ...]
    evidence_ceiling: EvidenceLevel
    literature_status: str
    execution_path: str

    @property
    def supports_counterfactual_replay(self) -> bool:
        return "counterfactual_replay" in self.capabilities


def available_self_trace_methods() -> list[str]:
    available = TRACERS.available()
    for trace_method_id in available:
        self_trace_method(trace_method_id)
    return available


def self_trace_method(trace_method_id: str) -> SelfTraceMethodV1:
    try:
        adapter = TRACERS.registered(trace_method_id)
    except KeyError as error:
        raise ValueError(
            f"unknown self-trace method {trace_method_id!r}; available={TRACERS.available()}"
        ) from error
    capabilities = getattr(adapter, "tracer_capabilities", None)
    evidence_ceiling = getattr(adapter, "evidence_ceiling", None)
    literature_status = getattr(adapter, "literature_status", None)
    execution_path = getattr(adapter, "execution_path", None)
    if (
        not isinstance(capabilities, tuple)
        or not capabilities
        or not all(isinstance(value, str) and value for value in capabilities)
        or not isinstance(evidence_ceiling, EvidenceLevel)
        or not isinstance(literature_status, str)
        or not literature_status
        or not isinstance(execution_path, str)
        or not execution_path
    ):
        raise RuntimeError(f"self-trace method descriptor is incomplete: {trace_method_id}")
    resolved = TRACERS.resolve(trace_method_id)
    return SelfTraceMethodV1(
        trace_method_id=resolved,
        capabilities=capabilities,
        evidence_ceiling=evidence_ceiling,
        literature_status=literature_status,
        execution_path=execution_path,
    )


__all__ = [
    "SelfTraceMethodV1",
    "available_self_trace_methods",
    "self_trace_method",
]
