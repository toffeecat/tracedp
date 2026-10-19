"""Native TRACER uncertainty-risk aggregation adapter."""

from __future__ import annotations

import hashlib
import json
import math
import subprocess
import sys
import tempfile
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol
from urllib.parse import unquote, urlparse

from lh_platform.tracing.integrations._provenance import (
    isolated_worker_environment,
    require_clean_source_revision,
)
from lh_platform.tracing.types import EvidenceLevel, TraceInputV1
from lh_platform.tracing.unified import (
    TraceDiagnosticEventV1,
    TraceDiagnosticStage,
    TraceDiagnosticStatus,
    TraceFindingKind,
    TraceFindingV1,
    TraceInputArtifactV1,
    TraceMethodDescriptorV1,
    TraceMethodExecutionError,
    TraceMethodOutputV1,
    TraceMethodRequestV1,
    TraceOutcomeScope,
)

TRACER_RISK_WORKER_SCHEMA = "lh_tracer_risk_worker/v2"
TRACER_SIGNAL_PROFILE_FULL = "tau2_full"
TRACER_SIGNAL_PROFILE_ENTROPY_LEXICAL = "tau2_entropy_lexical"
TRACER_SIGNAL_PROFILE_LEGACY = "legacy_unattested"
TRACER_SIGNAL_PROFILES = (
    TRACER_SIGNAL_PROFILE_FULL,
    TRACER_SIGNAL_PROFILE_ENTROPY_LEXICAL,
    TRACER_SIGNAL_PROFILE_LEGACY,
)


@dataclass(frozen=True)
class TracerRiskNativePredictionV1:
    tracer_score: float
    num_steps: int
    num_top_k: int
    mean_ui: float
    mean_penalty: float
    raw_output_sha256: str
    signal_profile: str
    num_logprob_steps: int
    num_da_steps: int
    num_do_agent_steps: int
    num_do_user_steps: int
    collector_model_id: str
    collector_dataset_id: str
    diagnostics: tuple[TraceDiagnosticEventV1, ...] = ()


class TracerRiskRunner(Protocol):
    def run(
        self,
        *,
        case: TraceInputV1,
        artifact: TraceInputArtifactV1,
    ) -> TracerRiskNativePredictionV1: ...


class TracerRiskTraceAdapter:
    """Run TRACER on pre-collected logprobs and semantic-distance signals."""

    def __init__(self, *, source_revision: str, runner: TracerRiskRunner) -> None:
        if not source_revision:
            raise ValueError("TRACER source revision must be non-empty")
        self._runner = runner
        self.descriptor = TraceMethodDescriptorV1(
            method_id="tracer.risk_aggregation",
            display_name="TRACER Risk Aggregation",
            family="risk_detection",
            capabilities=("trajectory_risk",),
            input_modalities=("text_trajectory", "tracer_signal_bundle"),
            output_kinds=(TraceFindingKind.TRAJECTORY_RISK,),
            integration_level="upstream_native",
            execution_path="isolated_upstream_subprocess",
            evidence_ceiling=EvidenceLevel.L2_TYPED,
            source_relpath="code/agent-tracer",
            source_revision=source_revision,
            corpus_method_id="tracer",
            outcome_scope=TraceOutcomeScope.ANY_TERMINAL_OUTCOME,
        )

    def execute(self, request: TraceMethodRequestV1) -> TraceMethodOutputV1:
        artifacts = tuple(
            artifact
            for artifact in request.artifacts
            if artifact.modality == "tracer_signal_bundle"
        )
        if len(artifacts) != 1:
            raise ValueError("TRACER requires exactly one tracer_signal_bundle artifact")
        prediction = self._runner.run(case=request.case, artifact=artifacts[0])

        def reject(message: str) -> None:
            raise TraceMethodExecutionError(
                message,
                error_type="ValueError",
                diagnostics=(
                    *prediction.diagnostics,
                    TraceDiagnosticEventV1(
                        stage=TraceDiagnosticStage.ADAPTER_VALIDATION,
                        status=TraceDiagnosticStatus.FAILED,
                        error_type="ValueError",
                    ),
                ),
            )

        numeric = (
            prediction.tracer_score,
            prediction.mean_ui,
            prediction.mean_penalty,
        )
        if any(isinstance(value, bool) or not math.isfinite(value) for value in numeric):
            reject("TRACER returned a non-finite score")
        if (
            prediction.num_steps < 1
            or prediction.num_top_k < 1
            or prediction.num_top_k > prediction.num_steps
        ):
            reject("TRACER returned an empty aggregation")
        if prediction.signal_profile not in TRACER_SIGNAL_PROFILES:
            reject("TRACER returned an unsupported signal profile")
        counts = (
            prediction.num_logprob_steps,
            prediction.num_da_steps,
            prediction.num_do_agent_steps,
            prediction.num_do_user_steps,
        )
        if any(count < 0 or count > prediction.num_steps for count in counts):
            reject("TRACER returned invalid signal coverage")
        if prediction.num_logprob_steps != prediction.num_steps:
            reject("TRACER did not receive token logprobs for every aggregated step")
        semantic_steps = prediction.num_do_agent_steps + prediction.num_do_user_steps
        if prediction.signal_profile == TRACER_SIGNAL_PROFILE_FULL and semantic_steps == 0:
            reject("TRACER full profile has no semantic-distance observations")
        if (
            prediction.signal_profile == TRACER_SIGNAL_PROFILE_ENTROPY_LEXICAL
            and semantic_steps != 0
        ):
            reject("TRACER entropy/lexical profile contains semantic-distance observations")
        if prediction.signal_profile != TRACER_SIGNAL_PROFILE_LEGACY and (
            not prediction.collector_model_id or not prediction.collector_dataset_id
        ):
            reject("TRACER signal collection provenance is incomplete")
        explanation = (
            "TRACER aggregated token uncertainty and lexical repetition; "
            "semantic-distance signals were unavailable"
            if prediction.signal_profile == TRACER_SIGNAL_PROFILE_ENTROPY_LEXICAL
            else "TRACER aggregated token uncertainty and supplied situational-distance signals"
        )
        return TraceMethodOutputV1(
            findings=(
                TraceFindingV1(
                    kind=TraceFindingKind.TRAJECTORY_RISK,
                    event_ids=(),
                    label="tracer_risk_score",
                    score=prediction.tracer_score,
                    explanation=explanation,
                    attributes=(
                        ("num_steps", str(prediction.num_steps)),
                        ("num_top_k", str(prediction.num_top_k)),
                        ("mean_ui", str(prediction.mean_ui)),
                        ("mean_penalty", str(prediction.mean_penalty)),
                        ("signal_profile", prediction.signal_profile),
                        ("num_logprob_steps", str(prediction.num_logprob_steps)),
                        ("num_da_steps", str(prediction.num_da_steps)),
                        ("num_do_agent_steps", str(prediction.num_do_agent_steps)),
                        ("num_do_user_steps", str(prediction.num_do_user_steps)),
                        ("collector_model_id", prediction.collector_model_id),
                        ("collector_dataset_id", prediction.collector_dataset_id),
                        ("raw_output_sha256", prediction.raw_output_sha256),
                        ("signal_artifact_sha256", artifacts[0].content_sha256),
                    ),
                ),
            ),
            evidence_level=EvidenceLevel.L2_TYPED,
            diagnostics=(
                *prediction.diagnostics,
                TraceDiagnosticEventV1(
                    stage=TraceDiagnosticStage.ADAPTER_VALIDATION,
                    status=TraceDiagnosticStatus.COMPLETED,
                ),
            ),
        )


def _artifact_file(artifact: TraceInputArtifactV1) -> Path:
    parsed = urlparse(artifact.uri)
    if parsed.scheme != "file" or parsed.netloc not in ("", "localhost"):
        raise ValueError("TRACER signal artifact must use a local file URI")
    path = Path(unquote(parsed.path)).resolve()
    if not path.is_file():
        raise ValueError(f"TRACER signal artifact is missing: {path}")
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if f"sha256:{digest}" != artifact.content_sha256:
        raise ValueError("TRACER signal artifact hash mismatch")
    return path


class TracerRiskSubprocessRunner:
    def __init__(
        self,
        *,
        corpus_root: str | Path,
        python_executable: str = sys.executable,
        timeout_seconds: int = 60,
    ) -> None:
        root = Path(corpus_root).resolve()
        self._source = (root / "code" / "agent-tracer").resolve()
        if not self._source.is_relative_to(root) or not self._source.is_dir():
            raise ValueError(f"TRACER source directory is missing: {self._source}")
        self._python = python_executable
        self._timeout_seconds = timeout_seconds

    @property
    def source_revision(self) -> str:
        return require_clean_source_revision(self._source, method_name="TRACER")

    def run(
        self,
        *,
        case: TraceInputV1,
        artifact: TraceInputArtifactV1,
    ) -> TracerRiskNativePredictionV1:
        artifact_path = _artifact_file(artifact)
        with tempfile.TemporaryDirectory(prefix="lh-tracer-risk-") as temporary:
            case_path = Path(temporary) / "case.json"
            case_path.write_text(
                json.dumps(case.to_dict(), ensure_ascii=False, sort_keys=True),
                encoding="utf-8",
            )
            try:
                completed = subprocess.run(
                    [
                        self._python,
                        str(Path(__file__).with_name("tracer_risk_worker.py")),
                        "--source",
                        str(self._source),
                        "--case",
                        str(case_path),
                        "--signals",
                        str(artifact_path),
                        "--source-revision",
                        self.source_revision,
                    ],
                    check=False,
                    capture_output=True,
                    text=True,
                    timeout=self._timeout_seconds,
                    env=isolated_worker_environment("LH_TRACER_UNUSED", "1"),
                    cwd=temporary,
                )
            except subprocess.TimeoutExpired as error:
                raise TraceMethodExecutionError(
                    "TRACER worker timed out",
                    error_type=type(error).__name__,
                    diagnostics=(
                        TraceDiagnosticEventV1(
                            stage=TraceDiagnosticStage.WORKER_LAUNCH,
                            status=TraceDiagnosticStatus.FAILED,
                            error_type=type(error).__name__,
                        ),
                    ),
                ) from error
        try:
            payload = json.loads(completed.stdout)
        except json.JSONDecodeError as error:
            detail = completed.stderr.strip()
            suffix = f": {detail[-1000:]}" if detail else ""
            raise RuntimeError(f"TRACER worker returned invalid JSON{suffix}") from error
        if not isinstance(payload, Mapping) or payload.get("schema") != TRACER_RISK_WORKER_SCHEMA:
            raise RuntimeError("TRACER worker returned an unsupported envelope")
        raw_diagnostics = payload.get("diagnostics")
        if not isinstance(raw_diagnostics, list) or not all(
            isinstance(event, Mapping) for event in raw_diagnostics
        ):
            raise RuntimeError("TRACER worker diagnostics are invalid")
        diagnostics = tuple(
            TraceDiagnosticEventV1.from_dict(event) for event in raw_diagnostics
        )
        if completed.returncode != 0 or payload.get("status") != "completed":
            error_type = str(payload.get("error_type") or "RuntimeError")
            error_message = str(payload.get("error_message") or "worker failed")
            raise TraceMethodExecutionError(
                f"TRACER upstream process failed: {error_message}",
                error_type=error_type,
                diagnostics=diagnostics,
            )
        required_keys = {
            "tracer_score",
            "num_steps",
            "num_top_k",
            "mean_ui",
            "mean_penalty",
            "raw_output_sha256",
            "signal_profile",
            "num_logprob_steps",
            "num_da_steps",
            "num_do_agent_steps",
            "num_do_user_steps",
            "collector_model_id",
            "collector_dataset_id",
        }
        missing_keys = sorted(required_keys - payload.keys())
        if missing_keys:
            raise TraceMethodExecutionError(
                f"TRACER worker response omitted fields: {missing_keys}",
                error_type="RuntimeError",
                diagnostics=(
                    *diagnostics,
                    TraceDiagnosticEventV1(
                        stage=TraceDiagnosticStage.ADAPTER_VALIDATION,
                        status=TraceDiagnosticStatus.FAILED,
                        error_type="RuntimeError",
                    ),
                ),
            )
        return TracerRiskNativePredictionV1(
            tracer_score=float(payload["tracer_score"]),
            num_steps=int(payload["num_steps"]),
            num_top_k=int(payload["num_top_k"]),
            mean_ui=float(payload["mean_ui"]),
            mean_penalty=float(payload["mean_penalty"]),
            raw_output_sha256=str(payload["raw_output_sha256"]),
            signal_profile=str(payload["signal_profile"]),
            num_logprob_steps=int(payload["num_logprob_steps"]),
            num_da_steps=int(payload["num_da_steps"]),
            num_do_agent_steps=int(payload["num_do_agent_steps"]),
            num_do_user_steps=int(payload["num_do_user_steps"]),
            collector_model_id=str(payload["collector_model_id"]),
            collector_dataset_id=str(payload["collector_dataset_id"]),
            diagnostics=diagnostics,
        )


def native_tracer_risk_adapter(
    *,
    corpus_root: str | Path,
    python_executable: str = sys.executable,
    timeout_seconds: int = 60,
) -> TracerRiskTraceAdapter:
    runner = TracerRiskSubprocessRunner(
        corpus_root=corpus_root,
        python_executable=python_executable,
        timeout_seconds=timeout_seconds,
    )
    return TracerRiskTraceAdapter(source_revision=runner.source_revision, runner=runner)


__all__ = [
    "TRACER_RISK_WORKER_SCHEMA",
    "TRACER_SIGNAL_PROFILE_ENTROPY_LEXICAL",
    "TRACER_SIGNAL_PROFILE_FULL",
    "TRACER_SIGNAL_PROFILE_LEGACY",
    "TracerRiskNativePredictionV1",
    "TracerRiskSubprocessRunner",
    "TracerRiskTraceAdapter",
    "native_tracer_risk_adapter",
]
