"""EvoTool module-blame adapter for the unified tracing seam."""

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
)

EVOTOOL_MODULES = ("planner", "selector", "caller", "synthesizer")


@dataclass(frozen=True)
class EvoToolNativePredictionV1:
    primary_module: str
    module_scores: tuple[tuple[str, float], ...]
    diagnosis: str
    resolved_by: str
    model_calls: int
    total_tokens: int
    raw_output_sha256: str
    response_model_id: str


class EvoToolRunner(Protocol):
    def run(self, *, case: TraceInputV1) -> EvoToolNativePredictionV1: ...


class EvoToolTraceAdapter:
    """Run upstream trajectory-grounded blame and retain its module semantics."""

    def __init__(self, *, source_revision: str, runner: EvoToolRunner) -> None:
        if not source_revision:
            raise ValueError("EvoTool source revision must be non-empty")
        self._runner = runner
        self.descriptor = TraceMethodDescriptorV1(
            method_id="evotool.blame",
            display_name="EvoTool Trajectory-Grounded Blame",
            family="failure_attribution",
            capabilities=("component_attribution",),
            input_modalities=("text_trajectory",),
            output_kinds=(TraceFindingKind.COMPONENT_ATTRIBUTION,),
            integration_level="upstream_native_port",
            execution_path="isolated_upstream_subprocess",
            evidence_ceiling=EvidenceLevel.L2_TYPED,
            source_relpath="code/ACL_2026_EvoTool",
            source_revision=source_revision,
            corpus_method_id="evotool",
        )

    def execute(self, request: TraceMethodRequestV1) -> TraceMethodOutputV1:
        prediction = self._runner.run(case=request.case)
        if prediction.primary_module not in EVOTOOL_MODULES:
            raise ValueError(
                f"EvoTool returned unknown module: {prediction.primary_module}"
            )
        scores = dict(prediction.module_scores)
        if set(scores) != set(EVOTOOL_MODULES):
            raise ValueError("EvoTool did not return all four module scores")
        if any(
            isinstance(score, bool) or not 0.0 <= score <= 1.0
            for score in scores.values()
        ):
            raise ValueError("EvoTool module scores must be in [0, 1]")
        if not prediction.diagnosis:
            raise ValueError("EvoTool diagnosis is empty")
        if prediction.model_calls != 1:
            raise ValueError("EvoTool blame must make exactly one model call")
        if not prediction.response_model_id:
            raise ValueError("EvoTool response model attestation is empty")
        attributes = tuple(
            (f"score_{module}", str(scores[module])) for module in EVOTOOL_MODULES
        ) + (
            ("resolved_by", prediction.resolved_by),
            ("model_calls", str(prediction.model_calls)),
            ("total_tokens", str(prediction.total_tokens)),
            ("raw_output_sha256", prediction.raw_output_sha256),
            ("response_model_id", prediction.response_model_id),
            ("port", "gold_free_trace_case"),
        )
        return TraceMethodOutputV1(
            findings=(
                TraceFindingV1(
                    kind=TraceFindingKind.COMPONENT_ATTRIBUTION,
                    event_ids=(),
                    label=prediction.primary_module,
                    score=scores[prediction.primary_module],
                    explanation=prediction.diagnosis,
                    attributes=attributes,
                ),
            ),
            evidence_level=EvidenceLevel.L2_TYPED,
        )


class EvoToolSubprocessRunner:
    """Invoke the pinned EvoTool blamer without importing it into the platform."""

    def __init__(
        self,
        *,
        corpus_root: str | Path,
        runtime: NativeTraceRuntimeV1,
    ) -> None:
        root = Path(corpus_root).resolve()
        self._source = (root / "code" / "ACL_2026_EvoTool").resolve()
        if not self._source.is_relative_to(root) or not self._source.is_dir():
            raise ValueError(f"EvoTool source directory is missing: {self._source}")
        self._runtime = runtime
        require_python_modules(runtime, method_name="EvoTool", modules=("openai",))

    @property
    def source_revision(self) -> str:
        return require_clean_source_revision(self._source, method_name="EvoTool")

    def run(self, *, case: TraceInputV1) -> EvoToolNativePredictionV1:
        with tempfile.TemporaryDirectory(prefix="lh-evotool-") as temporary:
            request_path = Path(temporary) / "request.json"
            request_path.write_text(
                json.dumps(case.to_dict(), ensure_ascii=False, sort_keys=True),
                encoding="utf-8",
            )
            environment = isolated_worker_environment(
                "LH_EVOTOOL_API_KEY", self._runtime.api_key
            )
            completed = subprocess.run(
                [
                    self._runtime.python_executable,
                    str(Path(__file__).with_name("evotool_worker.py")),
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
            raise RuntimeError(f"EvoTool upstream process failed: {error[-2000:]}")
        try:
            payload = json.loads(completed.stdout)
        except json.JSONDecodeError as error:
            raise RuntimeError("EvoTool worker returned invalid JSON") from error
        response_model_id = str(payload["response_model_id"])
        if response_model_id != self._runtime.model_id:
            raise RuntimeError("EvoTool worker response model attestation mismatch")
        return EvoToolNativePredictionV1(
            primary_module=str(payload["primary_module"]),
            module_scores=tuple(
                (module, float(payload["module_scores"][module]))
                for module in EVOTOOL_MODULES
            ),
            diagnosis=str(payload["diagnosis"]),
            resolved_by=str(payload["resolved_by"]),
            model_calls=int(payload["model_calls"]),
            total_tokens=int(payload["total_tokens"]),
            raw_output_sha256=str(payload["raw_output_sha256"]),
            response_model_id=response_model_id,
        )


def native_evotool_adapter(
    *,
    corpus_root: str | Path,
    runtime: NativeTraceRuntimeV1,
) -> EvoToolTraceAdapter:
    runner = EvoToolSubprocessRunner(
        corpus_root=corpus_root,
        runtime=runtime,
    )
    return EvoToolTraceAdapter(source_revision=runner.source_revision, runner=runner)


__all__ = [
    "EVOTOOL_MODULES",
    "EvoToolNativePredictionV1",
    "EvoToolSubprocessRunner",
    "EvoToolTraceAdapter",
    "native_evotool_adapter",
]
