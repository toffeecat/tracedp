"""AgentDiagnose reasoning-quality adapter."""

from __future__ import annotations

import json
import math
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

AGENTDIAGNOSE_DIMENSIONS = (
    "strategic_backtracking",
    "task_decomposition",
    "observation_reading",
    "self_verification",
)


@dataclass(frozen=True)
class AgentDiagnoseNativePredictionV1:
    overall_score: float
    dimension_scores: tuple[tuple[str, float | None], ...]
    justifications: tuple[tuple[str, str], ...]
    model_calls: int
    raw_output_sha256: str
    response_model_id: str


class AgentDiagnoseRunner(Protocol):
    def run(self, *, case: TraceInputV1) -> AgentDiagnoseNativePredictionV1: ...


class AgentDiagnoseTraceAdapter:
    def __init__(self, *, source_revision: str, runner: AgentDiagnoseRunner) -> None:
        if not source_revision:
            raise ValueError("AgentDiagnose source revision must be non-empty")
        self._runner = runner
        self.descriptor = TraceMethodDescriptorV1(
            method_id="agentdiagnose.reasoning_quality",
            display_name="AgentDiagnose Reasoning Quality",
            family="credit_assignment",
            capabilities=("process_scoring",),
            input_modalities=("text_trajectory",),
            output_kinds=(TraceFindingKind.PROCESS_SCORE,),
            integration_level="upstream_native_port",
            execution_path="isolated_upstream_subprocess",
            evidence_ceiling=EvidenceLevel.L2_TYPED,
            source_relpath="code/AgentDiagnose",
            source_revision=source_revision,
            corpus_method_id="agentdiagnose",
            outcome_scope=TraceOutcomeScope.ANY_TERMINAL_OUTCOME,
        )

    def execute(self, request: TraceMethodRequestV1) -> TraceMethodOutputV1:
        prediction = self._runner.run(case=request.case)
        scores = dict(prediction.dimension_scores)
        if set(scores) != set(AGENTDIAGNOSE_DIMENSIONS):
            raise ValueError("AgentDiagnose did not return all reasoning dimensions")
        present = [prediction.overall_score] + [
            score for score in scores.values() if score is not None
        ]
        if any(
            isinstance(score, bool) or not math.isfinite(score) or not 0.0 <= score <= 1.0
            for score in present
        ):
            raise ValueError("AgentDiagnose scores must be finite values in [0, 1]")
        if prediction.model_calls != 1:
            raise ValueError("AgentDiagnose must make exactly one model call")
        if not prediction.response_model_id:
            raise ValueError("AgentDiagnose response model attestation is empty")
        justifications = dict(prediction.justifications)
        attributes = tuple(
            (dimension, "N/A" if scores[dimension] is None else str(scores[dimension]))
            for dimension in AGENTDIAGNOSE_DIMENSIONS
        ) + tuple(
            (f"justification_{key}", value)
            for key, value in sorted(justifications.items())
        ) + (
            ("model_calls", str(prediction.model_calls)),
            ("raw_output_sha256", prediction.raw_output_sha256),
            ("response_model_id", prediction.response_model_id),
            ("port", "generic_text_trace"),
        )
        return TraceMethodOutputV1(
            findings=(
                TraceFindingV1(
                    kind=TraceFindingKind.PROCESS_SCORE,
                    event_ids=(),
                    label="reasoning_quality",
                    score=prediction.overall_score,
                    explanation=(
                        "AgentDiagnose scored backtracking, decomposition, observation "
                        "reading and self-verification"
                    ),
                    attributes=attributes,
                ),
            ),
            evidence_level=EvidenceLevel.L2_TYPED,
        )


class AgentDiagnoseSubprocessRunner:
    def __init__(
        self,
        *,
        corpus_root: str | Path,
        runtime: NativeTraceRuntimeV1,
    ) -> None:
        root = Path(corpus_root).resolve()
        self._source = (root / "code" / "AgentDiagnose").resolve()
        if not self._source.is_relative_to(root) or not self._source.is_dir():
            raise ValueError(f"AgentDiagnose source directory is missing: {self._source}")
        self._runtime = runtime
        require_python_modules(
            runtime,
            method_name="AgentDiagnose",
            modules=("dotenv", "litellm", "openai", "pydantic", "spacy", "tiktoken", "tqdm"),
        )

    @property
    def source_revision(self) -> str:
        return require_clean_source_revision(self._source, method_name="AgentDiagnose")

    def run(self, *, case: TraceInputV1) -> AgentDiagnoseNativePredictionV1:
        with tempfile.TemporaryDirectory(prefix="lh-agentdiagnose-") as temporary:
            request_path = Path(temporary) / "request.json"
            request_path.write_text(
                json.dumps(case.to_dict(), ensure_ascii=False, sort_keys=True),
                encoding="utf-8",
            )
            environment = isolated_worker_environment(
                "LH_AGENTDIAGNOSE_API_KEY", self._runtime.api_key
            )
            completed = subprocess.run(
                [
                    self._runtime.python_executable,
                    str(Path(__file__).with_name("agentdiagnose_worker.py")),
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
            raise RuntimeError(f"AgentDiagnose upstream process failed: {error[-2000:]}")
        try:
            payload = json.loads(completed.stdout)
        except json.JSONDecodeError as error:
            raise RuntimeError("AgentDiagnose worker returned invalid JSON") from error
        response_model_id = str(payload["response_model_id"])
        if response_model_id != self._runtime.model_id:
            raise RuntimeError("AgentDiagnose worker response model attestation mismatch")
        return AgentDiagnoseNativePredictionV1(
            overall_score=float(payload["overall_score"]),
            dimension_scores=tuple(
                (
                    dimension,
                    None
                    if payload["dimension_scores"][dimension] is None
                    else float(payload["dimension_scores"][dimension]),
                )
                for dimension in AGENTDIAGNOSE_DIMENSIONS
            ),
            justifications=tuple(sorted(payload["justifications"].items())),
            model_calls=int(payload["model_calls"]),
            raw_output_sha256=str(payload["raw_output_sha256"]),
            response_model_id=response_model_id,
        )


def native_agentdiagnose_adapter(
    *,
    corpus_root: str | Path,
    runtime: NativeTraceRuntimeV1,
) -> AgentDiagnoseTraceAdapter:
    runner = AgentDiagnoseSubprocessRunner(
        corpus_root=corpus_root,
        runtime=runtime,
    )
    return AgentDiagnoseTraceAdapter(source_revision=runner.source_revision, runner=runner)


__all__ = [
    "AGENTDIAGNOSE_DIMENSIONS",
    "AgentDiagnoseNativePredictionV1",
    "AgentDiagnoseSubprocessRunner",
    "AgentDiagnoseTraceAdapter",
    "native_agentdiagnose_adapter",
]
