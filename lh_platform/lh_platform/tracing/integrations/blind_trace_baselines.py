"""Canonical-fair static baselines over one immutable public trace request."""

from __future__ import annotations

import hashlib
import json
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol
from urllib.parse import quote

from lh_platform.test_time_adaptation.blind_full_trace_localization import BlindTraceRanking
from lh_platform.test_time_adaptation.blind_trace_method_matrix import (
    build_deterministic_baseline_ranking,
)
from lh_platform.tracing.integrations._provenance import isolated_worker_environment
from lh_platform.tracing.integrations._runtime import (
    NativeTraceRuntimeV1,
    require_python_modules,
    require_python_worker_entrypoint,
)
from lh_platform.tracing.types import EvidenceLevel, TraceCaseV1
from lh_platform.tracing.unified import (
    TraceDiagnosticArtifactV1,
    TraceDiagnosticEventV1,
    TraceDiagnosticStage,
    TraceDiagnosticStatus,
    TraceFindingKind,
    TraceFindingV1,
    TraceMethodDescriptorV1,
    TraceMethodOutputV1,
    TraceMethodRequestV1,
)

DETERMINISTIC_BASELINE_METHOD_IDS = (
    "baseline.recent_step",
    "baseline.uniform_positions",
)
MODEL_STYLE_BASELINE_METHOD_IDS = (
    "baseline.generic_causal",
    "baseline.who_when_style",
    "baseline.traceelephant_style",
)
BLIND_TRACE_BASELINE_METHOD_IDS = (
    *DETERMINISTIC_BASELINE_METHOD_IDS,
    *MODEL_STYLE_BASELINE_METHOD_IDS,
)
_DETERMINISTIC_KIND = {
    "baseline.recent_step": "recent_step",
    "baseline.uniform_positions": "uniform_positions",
}


def _descriptor(method_id: str) -> TraceMethodDescriptorV1:
    model_backed = method_id in MODEL_STYLE_BASELINE_METHOD_IDS
    return TraceMethodDescriptorV1(
        method_id=method_id,
        display_name=method_id.removeprefix("baseline.").replace("_", " ").title(),
        family="failure_attribution",
        capabilities=("step_attribution", "ranked_candidates"),
        input_modalities=("text_trajectory",),
        output_kinds=(TraceFindingKind.STEP_ATTRIBUTION,),
        integration_level=("canonical_fair_style" if model_backed else "deterministic_control"),
        execution_path=("isolated_platform_subprocess" if model_backed else "in_process"),
        evidence_ceiling=EvidenceLevel.L3_LOCALIZED,
        source_relpath=(
            "lh_platform/test_time_adaptation/blind_full_trace_localization.py"
            if model_backed
            else "lh_platform/test_time_adaptation/blind_trace_method_matrix.py"
        ),
        source_revision="v1",
        corpus_method_id=("longh_style_baselines" if model_backed else "controls"),
    )


def blind_trace_baseline_catalog_descriptors() -> tuple[TraceMethodDescriptorV1, ...]:
    """Declare controls and style baselines without overstating provenance."""

    return tuple(_descriptor(method_id) for method_id in BLIND_TRACE_BASELINE_METHOD_IDS)


def _output(
    ranking: BlindTraceRanking,
    *,
    attributes: tuple[tuple[str, str], ...] = (),
    diagnostics: tuple[TraceDiagnosticEventV1, ...] = (),
) -> TraceMethodOutputV1:
    return TraceMethodOutputV1(
        findings=(
            TraceFindingV1(
                kind=TraceFindingKind.STEP_ATTRIBUTION,
                event_ids=ranking.ranked_event_ids,
                label="ranked_root_candidates",
                score=None,
                explanation=ranking.explanation,
                attributes=attributes,
            ),
        ),
        evidence_level=EvidenceLevel.L3_LOCALIZED,
        diagnostics=diagnostics,
    )


class DeterministicBlindTraceAdapter:
    """Expose position-only controls through the unified tracing contract."""

    def __init__(self, method_id: str) -> None:
        if method_id not in DETERMINISTIC_BASELINE_METHOD_IDS:
            raise ValueError(f"unknown deterministic trace baseline: {method_id}")
        self.descriptor = _descriptor(method_id)

    def execute(self, request: TraceMethodRequestV1) -> TraceMethodOutputV1:
        """Rank only visible event positions without model or evaluator access."""

        ranking = build_deterministic_baseline_ranking(
            tuple(event.event_id for event in request.case.events),
            method_id=_DETERMINISTIC_KIND[self.descriptor.method_id],
        )
        return _output(
            ranking,
            attributes=(
                ("model_calls", "0"),
                ("reasoning_tokens", "0"),
                ("reasoning_tokens_reported", "true"),
            ),
        )


@dataclass(frozen=True)
class BlindStylePredictionV1:
    """One parsed ranking plus the model audit facts needed for admission."""

    ranking: BlindTraceRanking
    model_calls: int
    response_model_id: str
    reasoning_tokens: int
    reasoning_tokens_reported: bool
    raw_response_reused: bool
    raw_artifact_path: Path
    raw_artifact_uri: str
    input_tokens: int | None = None
    output_tokens: int | None = None
    cache_read_tokens: int | None = None
    cache_write_tokens: int | None = None
    latency_seconds: float | None = None


class BlindStyleRunner(Protocol):
    def run(
        self,
        *,
        method_id: str,
        request: TraceMethodRequestV1,
    ) -> BlindStylePredictionV1: ...


class BlindStyleTraceAdapter:
    """Validate one canonical-fair model call and publish its raw response."""

    def __init__(self, *, method_id: str, runner: BlindStyleRunner) -> None:
        if method_id not in MODEL_STYLE_BASELINE_METHOD_IDS:
            raise ValueError(f"unknown model trace baseline: {method_id}")
        self._runner = runner
        self.descriptor = _descriptor(method_id)

    def execute(self, request: TraceMethodRequestV1) -> TraceMethodOutputV1:
        """Run one style baseline and fail closed on any identity drift."""

        if not isinstance(request.case, TraceCaseV1):
            raise ValueError("blind style baseline requires TraceCaseV1")
        prediction = self._runner.run(
            method_id=self.descriptor.method_id,
            request=request,
        )
        if (
            prediction.model_calls != 1
            or prediction.response_model_id == ""
            or prediction.reasoning_tokens != 0
            or type(prediction.reasoning_tokens_reported) is not bool
            or (prediction.input_tokens is not None and prediction.input_tokens < 0)
            or (prediction.output_tokens is not None and prediction.output_tokens < 0)
            or (prediction.cache_read_tokens is not None and prediction.cache_read_tokens < 0)
            or (prediction.cache_write_tokens is not None and prediction.cache_write_tokens < 0)
            or (prediction.latency_seconds is not None and prediction.latency_seconds < 0)
        ):
            raise ValueError("blind style baseline model audit failed")
        raw = prediction.raw_artifact_path.read_bytes()
        artifact = TraceDiagnosticArtifactV1(
            artifact_id="raw_model_response",
            uri=prediction.raw_artifact_uri,
            media_type="application/json",
            byte_count=len(raw),
            content_sha256="sha256:" + hashlib.sha256(raw).hexdigest(),
        )
        cost_attributes = tuple(
            (key, value)
            for key, raw in (
                ("input_tokens", prediction.input_tokens),
                ("output_tokens", prediction.output_tokens),
                ("cache_read_tokens", prediction.cache_read_tokens),
                ("cache_write_tokens", prediction.cache_write_tokens),
                ("latency_seconds", prediction.latency_seconds),
            )
            if raw is not None
            for value in (f"{raw:.6f}" if key == "latency_seconds" else str(raw),)
        )
        return _output(
            prediction.ranking,
            attributes=(
                ("model_calls", str(prediction.model_calls)),
                ("response_model_id", prediction.response_model_id),
                ("reasoning_tokens", str(prediction.reasoning_tokens)),
                (
                    "reasoning_tokens_reported",
                    str(prediction.reasoning_tokens_reported).lower(),
                ),
                ("raw_response_reused", str(prediction.raw_response_reused).lower()),
                *cost_attributes,
            ),
            diagnostics=(
                TraceDiagnosticEventV1(
                    stage=TraceDiagnosticStage.ADAPTER_EXECUTION,
                    status=TraceDiagnosticStatus.COMPLETED,
                    artifacts=(artifact,),
                ),
            ),
        )


class BlindStyleSubprocessRunner:
    """Isolate model calls and reuse a durable response after interruption."""

    def __init__(self, *, runtime: NativeTraceRuntimeV1) -> None:
        if runtime.diagnostic_artifact_root is None:
            raise ValueError("blind style baselines require a diagnostic artifact root")
        self._runtime = runtime
        self._artifact_root = Path(runtime.diagnostic_artifact_root).resolve()
        require_python_modules(
            runtime,
            method_name="canonical blind trace baselines",
            modules=("openai",),
        )
        require_python_worker_entrypoint(
            runtime,
            method_name="canonical blind trace baselines",
            worker_path=Path(__file__).with_name("canonical_blind_worker.py"),
            environment=isolated_worker_environment("LH_CANONICAL_BLIND_API_KEY", runtime.api_key),
            arguments=("--import-check",),
        )

    def run(
        self,
        *,
        method_id: str,
        request: TraceMethodRequestV1,
    ) -> BlindStylePredictionV1:
        """Run or recover exactly one response for a case-method cell."""

        if method_id not in MODEL_STYLE_BASELINE_METHOD_IDS:
            raise ValueError("unknown canonical blind style method")
        case = request.case
        if not isinstance(case, TraceCaseV1):
            raise ValueError("canonical blind style requires TraceCaseV1")
        # Percent encoding preserves distinct case identities while preventing
        # path separators and parent segments from escaping the artifact root.
        case_stem = "case-" + quote(case.case_id, safe="")
        artifact_path = (
            self._artifact_root
            / "canonical_blind"
            / case_stem
            / method_id.replace(".", "_")
            / "raw_response.json"
        )
        artifact_path.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix=".worker-", dir=self._artifact_root) as temporary:
            request_path = Path(temporary) / "request.json"
            request_path.write_text(
                json.dumps(case.to_dict(), ensure_ascii=False, sort_keys=True),
                encoding="utf-8",
            )
            completed = subprocess.run(
                [
                    self._runtime.python_executable,
                    str(Path(__file__).with_name("canonical_blind_worker.py")),
                    "--request",
                    str(request_path),
                    "--source-request-sha256",
                    request.sha256,
                    "--method-id",
                    method_id,
                    "--raw-output",
                    str(artifact_path),
                    *self._runtime.worker_arguments(),
                ],
                check=False,
                capture_output=True,
                text=True,
                timeout=self._runtime.timeout_seconds,
                env=isolated_worker_environment(
                    "LH_CANONICAL_BLIND_API_KEY", self._runtime.api_key
                ),
                cwd=temporary,
            )
        if completed.returncode != 0:
            error = completed.stderr.strip() or completed.stdout.strip()
            raise RuntimeError(f"canonical blind worker failed: {error[-2000:]}")
        try:
            payload = json.loads(completed.stdout)
        except json.JSONDecodeError as error:
            raise RuntimeError("canonical blind worker returned invalid JSON") from error
        return BlindStylePredictionV1(
            ranking=BlindTraceRanking(
                ranked_event_ids=tuple(payload["ranked_event_ids"]),
                explanation=str(payload["explanation"]),
            ),
            model_calls=int(payload["model_calls"]),
            response_model_id=str(payload["response_model_id"]),
            reasoning_tokens=int(payload["reasoning_tokens"]),
            reasoning_tokens_reported=bool(payload["reasoning_tokens_reported"]),
            raw_response_reused=bool(payload["raw_response_reused"]),
            raw_artifact_path=artifact_path,
            raw_artifact_uri=str(artifact_path.relative_to(self._artifact_root)),
        )


def deterministic_blind_trace_adapters() -> tuple[DeterministicBlindTraceAdapter, ...]:
    """Build the complete deterministic control family."""

    return tuple(
        DeterministicBlindTraceAdapter(method_id) for method_id in DETERMINISTIC_BASELINE_METHOD_IDS
    )


def model_blind_trace_adapters(
    *, runtime: NativeTraceRuntimeV1
) -> tuple[BlindStyleTraceAdapter, ...]:
    """Build model-backed style baselines sharing one audited runtime."""

    runner = BlindStyleSubprocessRunner(runtime=runtime)
    return tuple(
        BlindStyleTraceAdapter(method_id=method_id, runner=runner)
        for method_id in MODEL_STYLE_BASELINE_METHOD_IDS
    )


__all__ = [
    "BLIND_TRACE_BASELINE_METHOD_IDS",
    "DETERMINISTIC_BASELINE_METHOD_IDS",
    "MODEL_STYLE_BASELINE_METHOD_IDS",
    "BlindStylePredictionV1",
    "BlindStyleTraceAdapter",
    "blind_trace_baseline_catalog_descriptors",
    "deterministic_blind_trace_adapters",
    "model_blind_trace_adapters",
]
