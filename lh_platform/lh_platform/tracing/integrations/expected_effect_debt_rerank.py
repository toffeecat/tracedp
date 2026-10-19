"""Candidate-constrained LLM reranking for expected-effect-debt traces."""

from __future__ import annotations

import hashlib
import json
import subprocess
import tempfile
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol
from urllib.parse import unquote, urlparse

from lh_platform.tracing.integrations._provenance import isolated_worker_environment
from lh_platform.tracing.integrations._runtime import (
    NativeTraceRuntimeV1,
    require_python_modules,
)
from lh_platform.tracing.types import EvidenceLevel, TraceInputV1
from lh_platform.tracing.unified import (
    TraceFindingKind,
    TraceFindingV1,
    TraceInputArtifactV1,
    TraceMethodBlocked,
    TraceMethodDescriptorV1,
    TraceMethodOutputV1,
    TraceMethodRequestV1,
)

EXPECTED_EFFECT_DEBT_PROPOSAL_SCHEMA = (
    "lh_alfworld_terminal_expected_effect_debt_slice/v2"
)
EXPECTED_EFFECT_DEBT_RERANK_METHOD_ID = "expected_effect_debt.llm_rerank"


def expected_effect_debt_rerank_catalog_descriptor() -> TraceMethodDescriptorV1:
    """Expose the platform-native method even when runtime admission is blocked."""

    return TraceMethodDescriptorV1(
        method_id=EXPECTED_EFFECT_DEBT_RERANK_METHOD_ID,
        display_name="Expected-Effect-Debt LLM Reranker",
        family="failure_attribution",
        capabilities=("step_attribution",),
        input_modalities=("text_trajectory", "candidate_event_set"),
        output_kinds=(TraceFindingKind.STEP_ATTRIBUTION,),
        integration_level="catalog_only",
        execution_path="expected_effect_debt_candidates_then_llm_rerank",
        evidence_ceiling=EvidenceLevel.L3_LOCALIZED,
        source_relpath=(
            "lh_platform/tracing/integrations/expected_effect_debt_rerank.py"
        ),
        source_revision=None,
        unavailable_reason="platform-native reranker runtime has not passed admission",
        corpus_method_id="expected_effect_debt",
    )


@dataclass(frozen=True)
class ExpectedEffectDebtCandidateSetV1:
    """The only localizable events visible to the reranker."""

    event_ids: tuple[str, ...]
    reasons: tuple[tuple[str, str, str], ...]
    artifact_sha256: str


@dataclass(frozen=True)
class ExpectedEffectDebtRerankPredictionV1:
    event_id: str
    explanation: str
    model_calls: int
    raw_output_sha256: str
    response_model_id: str


class ExpectedEffectDebtRerankRunner(Protocol):
    def run(
        self,
        *,
        case: TraceInputV1,
        candidate_set: ExpectedEffectDebtCandidateSetV1,
    ) -> ExpectedEffectDebtRerankPredictionV1: ...


def _candidate_artifact_file(artifact: TraceInputArtifactV1) -> Path:
    parsed = urlparse(artifact.uri)
    if parsed.scheme != "file" or parsed.netloc not in ("", "localhost"):
        raise ValueError("candidate_event_set artifact must use a local file URI")
    path = Path(unquote(parsed.path)).resolve()
    if not path.is_file():
        raise ValueError(f"candidate_event_set artifact is missing: {path}")
    digest = "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != artifact.content_sha256:
        raise ValueError("candidate_event_set artifact hash mismatch")
    return path


def _optional_text(value: object, *, field: str) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise ValueError(f"candidate reason {field} must be string or null")
    return value.strip() or None


def load_expected_effect_debt_candidate_set(
    *,
    artifact: TraceInputArtifactV1,
    case: TraceInputV1,
) -> ExpectedEffectDebtCandidateSetV1:
    """Validate one proposal and project only gold-blind model inputs."""

    path = _candidate_artifact_file(artifact)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError("candidate_event_set artifact is not valid JSON") from error
    if not isinstance(payload, Mapping):
        raise ValueError("candidate_event_set artifact must be an object")
    if payload.get("schema") != EXPECTED_EFFECT_DEBT_PROPOSAL_SCHEMA:
        raise ValueError("candidate_event_set artifact schema mismatch")
    candidate_source_trace_sha256 = payload.get("source_trace_sha256")
    metadata = dict(artifact.metadata)
    if candidate_source_trace_sha256 != case.sha256 and (
        metadata
        != {
            "candidate_source_condition": "per_step_symbolic_state_v1",
            "candidate_source_trace_sha256": candidate_source_trace_sha256,
            "model_input_condition": "observation_rich_static_v1",
            "model_trace_sha256": case.sha256,
            "projection_event_identity_validated": "true",
        }
    ):
        raise ValueError("candidate_event_set source trace mismatch")
    if payload.get("privileged_causal_labels_used") is not False:
        raise ValueError("candidate_event_set is not causal-gold blind")

    raw_ids = payload.get("candidate_event_ids")
    if not isinstance(raw_ids, list) or not all(
        isinstance(event_id, str) and event_id for event_id in raw_ids
    ):
        raise ValueError("candidate_event_set ids must be a non-empty-string list")
    event_ids = tuple(raw_ids)
    if len(event_ids) != len(set(event_ids)):
        raise ValueError("candidate_event_set ids must be unique")
    unknown = sorted(set(event_ids) - case.event_ids)
    if unknown:
        raise ValueError(f"candidate event ids are absent from trace: {unknown}")

    status = payload.get("candidate_generation_status")
    valid_statuses = (
        {"candidates_proposed"}
        if event_ids
        else {"no_candidates", "unsupported_only"}
    )
    if status not in valid_statuses:
        raise ValueError(
            "candidate_event_set status/count mismatch: "
            f"status={status!r} count={len(event_ids)}"
        )

    raw_reasons = payload.get("candidate_reasons")
    if not isinstance(raw_reasons, list) or not all(
        isinstance(reason, Mapping) for reason in raw_reasons
    ):
        raise ValueError("candidate_event_set reasons must be an object list")
    reasons: list[tuple[str, str, str]] = []
    for reason in raw_reasons:
        event_id = reason.get("event_id")
        reason_code = reason.get("reason_code")
        if not isinstance(event_id, str) or event_id not in set(event_ids):
            raise ValueError("candidate reason must reference a candidate event")
        if not isinstance(reason_code, str) or not reason_code.strip():
            raise ValueError("candidate reason code must be non-empty")
        matched_action = _optional_text(
            reason.get("matched_action"), field="matched_action"
        )
        matched_fact = _optional_text(reason.get("matched_fact"), field="matched_fact")
        evidence = matched_action or matched_fact or "no matched action/fact"
        reasons.append((event_id, reason_code.strip(), evidence))
    return ExpectedEffectDebtCandidateSetV1(
        event_ids=event_ids,
        reasons=tuple(reasons),
        artifact_sha256=artifact.content_sha256,
    )


def build_expected_effect_debt_rerank_prompt(
    case: TraceInputV1,
    candidate_set: ExpectedEffectDebtCandidateSetV1,
) -> str:
    """Build the same full-context, candidate-restricted prompt for every case."""

    candidate_ids = set(candidate_set.event_ids)
    trajectory = "\n".join(
        f"[{event.event_id}] {event.kind}: {event.content}"
        + ("  <CANDIDATE>" if event.event_id in candidate_ids else "")
        for event in case.events
    )
    reasons = "\n".join(
        f"- {event_id}: reason={reason_code}; observed_match={evidence}"
        for event_id, reason_code, evidence in candidate_set.reasons
    )
    return (
        "You are tracing the earliest action that caused the observed task failure.\n"
        "Select exactly one event from CANDIDATE_EVENT_IDS. Do not select any other "
        "trajectory event. Judge delayed state consequences globally, not merely the "
        "last visible error. Return only a JSON object with exactly two string fields: "
        '"event_id" and "explanation".\n\n'
        f"CONTEXT\n{case.context_text}\n\n"
        f"OBSERVED_TRAJECTORY\n{trajectory}\n\n"
        f"CANDIDATE_EVENT_IDS\n{json.dumps(candidate_set.event_ids)}\n\n"
        f"CANDIDATE_EVIDENCE\n{reasons or '<none>'}"
    )


class ExpectedEffectDebtRerankTraceAdapter:
    def __init__(
        self,
        *,
        source_revision: str,
        runner: ExpectedEffectDebtRerankRunner,
    ) -> None:
        if not source_revision:
            raise ValueError("expected-effect-debt reranker revision must be non-empty")
        self._runner = runner
        self.descriptor = TraceMethodDescriptorV1(
            method_id=EXPECTED_EFFECT_DEBT_RERANK_METHOD_ID,
            display_name="Expected-Effect-Debt LLM Reranker",
            family="failure_attribution",
            capabilities=("step_attribution",),
            input_modalities=("text_trajectory", "candidate_event_set"),
            output_kinds=(TraceFindingKind.STEP_ATTRIBUTION,),
            integration_level="platform_native",
            execution_path="expected_effect_debt_candidates_then_llm_rerank",
            evidence_ceiling=EvidenceLevel.L3_LOCALIZED,
            source_relpath=(
                "lh_platform/tracing/integrations/expected_effect_debt_rerank.py"
            ),
            source_revision=source_revision,
            corpus_method_id="expected_effect_debt",
        )

    def execute(self, request: TraceMethodRequestV1) -> TraceMethodOutputV1:
        artifacts = tuple(
            artifact
            for artifact in request.artifacts
            if artifact.modality == "candidate_event_set"
        )
        if len(artifacts) != 1:
            raise ValueError(
                "expected-effect-debt reranker requires exactly one candidate_event_set"
            )
        candidate_set = load_expected_effect_debt_candidate_set(
            artifact=artifacts[0], case=request.case
        )
        if not candidate_set.event_ids:
            raise TraceMethodBlocked("candidate_event_set contains no candidates")
        prediction = self._runner.run(case=request.case, candidate_set=candidate_set)
        if prediction.event_id not in set(candidate_set.event_ids):
            raise ValueError(
                "reranker selected an event outside candidate_event_set: "
                f"{prediction.event_id}"
            )
        if prediction.model_calls != 1:
            raise ValueError("reranker must make exactly one model call per non-empty case")
        if not prediction.explanation.strip():
            raise ValueError("reranker explanation is empty")
        if not prediction.response_model_id:
            raise ValueError("reranker response model attestation is empty")
        if not prediction.raw_output_sha256.startswith("sha256:"):
            raise ValueError("reranker raw output digest is invalid")
        return TraceMethodOutputV1(
            findings=(
                TraceFindingV1(
                    kind=TraceFindingKind.STEP_ATTRIBUTION,
                    event_ids=(prediction.event_id,),
                    label="expected_effect_debt_reranked_root_cause",
                    score=None,
                    explanation=prediction.explanation,
                    attributes=(
                        ("candidate_artifact_sha256", candidate_set.artifact_sha256),
                        ("candidate_count", str(len(candidate_set.event_ids))),
                        ("model_calls", str(prediction.model_calls)),
                        ("raw_output_sha256", prediction.raw_output_sha256),
                        ("response_model_id", prediction.response_model_id),
                    ),
                ),
            ),
            evidence_level=EvidenceLevel.L3_LOCALIZED,
        )


class ExpectedEffectDebtRerankSubprocessRunner:
    def __init__(self, *, runtime: NativeTraceRuntimeV1) -> None:
        self._runtime = runtime
        require_python_modules(
            runtime,
            method_name="Expected-Effect-Debt LLM Reranker",
            modules=("openai",),
        )

    def run(
        self,
        *,
        case: TraceInputV1,
        candidate_set: ExpectedEffectDebtCandidateSetV1,
    ) -> ExpectedEffectDebtRerankPredictionV1:
        with tempfile.TemporaryDirectory(prefix="lh-expected-debt-rerank-") as temporary:
            request_path = Path(temporary) / "request.json"
            candidates_path = Path(temporary) / "candidates.json"
            request_path.write_text(
                json.dumps(case.to_dict(), ensure_ascii=False, sort_keys=True),
                encoding="utf-8",
            )
            candidates_path.write_text(
                json.dumps(
                    {
                        "event_ids": list(candidate_set.event_ids),
                        "reasons": [list(reason) for reason in candidate_set.reasons],
                        "artifact_sha256": candidate_set.artifact_sha256,
                    },
                    ensure_ascii=False,
                    sort_keys=True,
                ),
                encoding="utf-8",
            )
            completed = subprocess.run(
                [
                    self._runtime.python_executable,
                    str(Path(__file__).with_name("expected_effect_debt_rerank_worker.py")),
                    "--request",
                    str(request_path),
                    "--candidates",
                    str(candidates_path),
                    *self._runtime.worker_arguments(),
                ],
                check=False,
                capture_output=True,
                text=True,
                timeout=self._runtime.timeout_seconds,
                cwd=temporary,
                env=isolated_worker_environment(
                    "LH_EXPECTED_EFFECT_DEBT_RERANK_API_KEY", self._runtime.api_key
                ),
            )
        if completed.returncode != 0:
            error = completed.stderr.strip() or completed.stdout.strip()
            raise RuntimeError(f"expected-effect-debt reranker failed: {error[-2000:]}")
        try:
            payload = json.loads(completed.stdout)
        except json.JSONDecodeError as error:
            raise RuntimeError("expected-effect-debt worker returned invalid JSON") from error
        response_model_id = str(payload["response_model_id"])
        if response_model_id != self._runtime.model_id:
            raise RuntimeError("reranker worker response model attestation mismatch")
        return ExpectedEffectDebtRerankPredictionV1(
            event_id=str(payload["event_id"]),
            explanation=str(payload["explanation"]),
            model_calls=int(payload["model_calls"]),
            raw_output_sha256=str(payload["raw_output_sha256"]),
            response_model_id=response_model_id,
        )


def native_expected_effect_debt_rerank_adapter(
    *, runtime: NativeTraceRuntimeV1
) -> ExpectedEffectDebtRerankTraceAdapter:
    source_paths = (
        Path(__file__),
        Path(__file__).with_name("expected_effect_debt_rerank_worker.py"),
    )
    digest = hashlib.sha256()
    for path in source_paths:
        digest.update(path.name.encode("utf-8"))
        digest.update(path.read_bytes())
    return ExpectedEffectDebtRerankTraceAdapter(
        source_revision="sha256:" + digest.hexdigest(),
        runner=ExpectedEffectDebtRerankSubprocessRunner(runtime=runtime),
    )


__all__ = [
    "EXPECTED_EFFECT_DEBT_RERANK_METHOD_ID",
    "ExpectedEffectDebtCandidateSetV1",
    "ExpectedEffectDebtRerankPredictionV1",
    "ExpectedEffectDebtRerankRunner",
    "ExpectedEffectDebtRerankSubprocessRunner",
    "ExpectedEffectDebtRerankTraceAdapter",
    "build_expected_effect_debt_rerank_prompt",
    "expected_effect_debt_rerank_catalog_descriptor",
    "load_expected_effect_debt_candidate_set",
    "native_expected_effect_debt_rerank_adapter",
]
