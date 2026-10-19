"""AEGIS error-taxonomy adapter for the unified tracing seam."""

from __future__ import annotations

import json
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from lh_platform.tracing.integrations._provenance import (
    isolated_worker_environment,
    require_clean_source_revision,
)
from lh_platform.tracing.integrations._runtime import (
    NativeTraceRuntimeV1,
    require_python_modules,
)
from lh_platform.tracing.types import EvidenceLevel, TraceInputV1
from lh_platform.tracing.unified import (
    TraceFindingKind,
    TraceFindingV1,
    TraceMethodDescriptorV1,
    TraceMethodOutputV1,
    TraceMethodRequestV1,
    TraceOutcomeScope,
)


@dataclass(frozen=True)
class AegisFaultV1:
    agent_name: str
    error_type: str


@dataclass(frozen=True)
class AegisNativePredictionV1:
    faults: tuple[AegisFaultV1, ...]
    model_calls: int
    raw_output_sha256: str
    response_model_id: str


class AegisRunner(Protocol):
    def run(self, *, case: TraceInputV1) -> AegisNativePredictionV1: ...


class AegisTraceAdapter:
    def __init__(self, *, source_revision: str, runner: AegisRunner) -> None:
        self._runner = runner
        self.descriptor = TraceMethodDescriptorV1(
            method_id="aegis.error_attribution",
            display_name="AEGIS Error Attribution",
            family="failure_attribution",
            capabilities=("component_attribution", "error_classification"),
            input_modalities=("text_trajectory",),
            output_kinds=(
                TraceFindingKind.COMPONENT_ATTRIBUTION,
                TraceFindingKind.ERROR_CLASSIFICATION,
            ),
            integration_level="upstream_native",
            execution_path="isolated_upstream_subprocess",
            evidence_ceiling=EvidenceLevel.L2_TYPED,
            source_relpath="code/AEGIS",
            source_revision=source_revision,
            corpus_method_id="aegis",
            outcome_scope=TraceOutcomeScope.ANY_TERMINAL_OUTCOME,
        )

    def execute(self, request: TraceMethodRequestV1) -> TraceMethodOutputV1:
        case = request.case
        prediction = self._runner.run(case=case)
        if prediction.model_calls < 1:
            raise ValueError("AEGIS upstream execution made no model calls")
        if not prediction.response_model_id:
            raise ValueError("AEGIS response model attestation is empty")
        findings: list[TraceFindingV1] = []
        for fault in prediction.faults:
            if not fault.agent_name or not fault.error_type:
                raise ValueError("AEGIS returned an incomplete fault")
            matching = tuple(
                event.event_id
                for event in case.events
                if event.kind.casefold() == fault.agent_name.casefold()
            )
            findings.append(
                TraceFindingV1(
                    kind=TraceFindingKind.COMPONENT_ATTRIBUTION,
                    event_ids=matching,
                    label=fault.error_type,
                    score=None,
                    explanation=(
                        f"AEGIS attributed {fault.error_type} to {fault.agent_name}"
                    ),
                    attributes=(
                        ("agent_name", fault.agent_name),
                        ("model_calls", str(prediction.model_calls)),
                        ("raw_output_sha256", prediction.raw_output_sha256),
                        ("response_model_id", prediction.response_model_id),
                        ("taxonomy", "AEGIS_FM_1_3"),
                    ),
                )
            )
        if not findings:
            findings.append(
                TraceFindingV1(
                    kind=TraceFindingKind.ERROR_CLASSIFICATION,
                    event_ids=(),
                    label="no_fault_detected",
                    score=None,
                    explanation="AEGIS returned an empty faulty-agent set",
                    attributes=(
                        ("model_calls", str(prediction.model_calls)),
                        ("raw_output_sha256", prediction.raw_output_sha256),
                        ("response_model_id", prediction.response_model_id),
                        ("taxonomy", "AEGIS_FM_1_3"),
                    ),
                )
            )
        return TraceMethodOutputV1(
            findings=tuple(findings),
            evidence_level=EvidenceLevel.L2_TYPED,
        )


class AegisSubprocessRunner:
    def __init__(
        self,
        *,
        corpus_root: str | Path,
        runtime: NativeTraceRuntimeV1,
    ) -> None:
        root = Path(corpus_root).resolve()
        self._source = (root / "code" / "AEGIS").resolve()
        if not self._source.is_relative_to(root) or not self._source.is_dir():
            raise ValueError(f"AEGIS source directory is missing: {self._source}")
        self._runtime = runtime
        require_python_modules(
            runtime,
            method_name="AEGIS",
            modules=("openai", "yaml"),
        )

    @property
    def source_revision(self) -> str:
        return require_clean_source_revision(self._source, method_name="AEGIS")

    def run(self, *, case: TraceInputV1) -> AegisNativePredictionV1:
        with tempfile.TemporaryDirectory(prefix="lh-aegis-") as temporary:
            request_path = Path(temporary) / "request.json"
            request_path.write_text(
                json.dumps(case.to_dict(), ensure_ascii=False, sort_keys=True),
                encoding="utf-8",
            )
            environment = isolated_worker_environment(
                "LH_AEGIS_API_KEY", self._runtime.api_key
            )
            completed = subprocess.run(
                [
                    self._runtime.python_executable,
                    str(Path(__file__).with_name("aegis_worker.py")),
                    "--source",
                    str(self._source),
                    "--request",
                    str(request_path),
                    *self._runtime.worker_arguments(),
                ],
                check=False,
                capture_output=True,
                text=True,
                timeout=self._runtime.timeout_seconds,
                env=environment,
                cwd=temporary,
            )
        if completed.returncode != 0:
            error = completed.stderr.strip() or completed.stdout.strip()
            raise RuntimeError(f"AEGIS upstream process failed: {error[-2000:]}")
        payload = json.loads(completed.stdout)
        response_model_id = str(payload["response_model_id"])
        if response_model_id != self._runtime.model_id:
            raise RuntimeError("AEGIS worker response model attestation mismatch")
        return AegisNativePredictionV1(
            faults=tuple(
                AegisFaultV1(
                    agent_name=str(item["agent_name"]),
                    error_type=str(item["error_type"]),
                )
                for item in payload["faults"]
            ),
            model_calls=int(payload["model_calls"]),
            raw_output_sha256=str(payload["raw_output_sha256"]),
            response_model_id=response_model_id,
        )


def native_aegis_adapter(
    *,
    corpus_root: str | Path,
    runtime: NativeTraceRuntimeV1,
) -> AegisTraceAdapter:
    runner = AegisSubprocessRunner(
        corpus_root=corpus_root,
        runtime=runtime,
    )
    return AegisTraceAdapter(source_revision=runner.source_revision, runner=runner)


__all__ = [
    "AegisFaultV1",
    "AegisNativePredictionV1",
    "AegisSubprocessRunner",
    "AegisTraceAdapter",
    "native_aegis_adapter",
]
