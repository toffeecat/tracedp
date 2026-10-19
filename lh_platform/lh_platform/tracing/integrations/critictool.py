"""CriticTool self-critique adapter for tool-use failures."""

from __future__ import annotations

import hashlib
import json
import subprocess
import tempfile
from collections.abc import Mapping
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Protocol

from lh_platform.core.atomic_io import atomic_write_bytes
from lh_platform.tracing.integrations._provenance import (
    isolated_worker_environment,
    require_clean_source_revision,
)
from lh_platform.tracing.integrations._runtime import (
    NativeTraceRuntimeV1,
    require_python_modules,
)
from lh_platform.tracing.types import (
    EvidenceLevel,
    TraceInputV1,
    trace_terminal_event_id,
)
from lh_platform.tracing.unified import (
    TraceDiagnosticArtifactV1,
    TraceDiagnosticEventV1,
    TraceDiagnosticStage,
    TraceDiagnosticStatus,
    TraceFindingKind,
    TraceFindingV1,
    TraceMethodDescriptorV1,
    TraceMethodExecutionError,
    TraceMethodOutputV1,
    TraceMethodRequestV1,
)

CRITICTOOL_WORKER_SCHEMA = "lh_critictool_worker/v3"
CRITICTOOL_PROMPT_PROFILE_NO_COT = "llm_no_cot"
CRITICTOOL_PROMPT_PROFILE_COT = "llm_cot"
CRITICTOOL_PROMPT_PROFILES = (
    CRITICTOOL_PROMPT_PROFILE_NO_COT,
    CRITICTOOL_PROMPT_PROFILE_COT,
)

CRITICTOOL_ERROR_TYPES = (
    "tool_select_error",
    "tool_hallucination_error",
    "parameters_value_error",
    "parameters_key_error",
)


@dataclass(frozen=True)
class CriticToolNativePredictionV1:
    error_type: str
    recovery_tool: str
    recovery_arguments_json: str
    explanation: str
    model_calls: int
    raw_output_sha256: str
    response_model_id: str
    prompt_profile: str = CRITICTOOL_PROMPT_PROFILE_NO_COT
    diagnostics: tuple[TraceDiagnosticEventV1, ...] = ()


class CriticToolRunner(Protocol):
    def run(self, *, case: TraceInputV1) -> CriticToolNativePredictionV1: ...


class CriticToolTraceAdapter:
    """Run upstream CriticTool prompting and parsing on one failed trace."""

    def __init__(
        self,
        *,
        source_revision: str,
        runner: CriticToolRunner,
        prompt_profile: str = CRITICTOOL_PROMPT_PROFILE_NO_COT,
    ) -> None:
        if not source_revision:
            raise ValueError("CriticTool source revision must be non-empty")
        if prompt_profile not in CRITICTOOL_PROMPT_PROFILES:
            raise ValueError("CriticTool prompt profile is unsupported")
        self._runner = runner
        self._prompt_profile = prompt_profile
        self.descriptor = TraceMethodDescriptorV1(
            method_id="critictool",
            display_name="CriticTool Self-Critique",
            family="reflection",
            capabilities=("error_classification", "recovery_proposal"),
            input_modalities=("text_trajectory",),
            output_kinds=(
                TraceFindingKind.ERROR_CLASSIFICATION,
                TraceFindingKind.RECOVERY_PROPOSAL,
            ),
            integration_level="upstream_native_port",
            execution_path=(
                "isolated_upstream_subprocess"
                if prompt_profile == CRITICTOOL_PROMPT_PROFILE_COT
                else f"isolated_upstream_subprocess:{prompt_profile}"
            ),
            evidence_ceiling=EvidenceLevel.L2_TYPED,
            source_relpath="code/CriticTool",
            source_revision=source_revision,
            corpus_method_id="critictool",
        )

    def execute(self, request: TraceMethodRequestV1) -> TraceMethodOutputV1:
        prediction = self._runner.run(case=request.case)

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

        if prediction.prompt_profile != self._prompt_profile:
            reject("CriticTool worker prompt profile does not match the adapter")
        if prediction.error_type not in CRITICTOOL_ERROR_TYPES:
            reject(f"CriticTool returned unknown error type: {prediction.error_type}")
        if prediction.model_calls != 1:
            reject("CriticTool must make exactly one model call")
        if not prediction.response_model_id:
            reject("CriticTool response model attestation is empty")
        if (
            self._prompt_profile == CRITICTOOL_PROMPT_PROFILE_COT
            and not prediction.explanation
        ):
            reject("CriticTool explanation is empty")
        if not prediction.recovery_tool:
            reject("CriticTool recovery tool is empty")
        try:
            recovery_arguments = json.loads(prediction.recovery_arguments_json)
        except json.JSONDecodeError as error:
            reject("CriticTool recovery arguments are not JSON")
            raise AssertionError("unreachable") from error
        if not isinstance(recovery_arguments, Mapping):
            reject("CriticTool recovery arguments must be an object")
        canonical_arguments = json.dumps(
            recovery_arguments,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        if canonical_arguments != prediction.recovery_arguments_json:
            reject("CriticTool recovery arguments are not canonical JSON")

        explanation_source = (
            "upstream_thought" if prediction.explanation else "adapter_summary"
        )
        explanation = prediction.explanation or (
            "CriticTool non-CoT output classified the error as "
            f"{prediction.error_type} and proposed {prediction.recovery_tool}."
        )
        common_attributes = (
            ("model_calls", str(prediction.model_calls)),
            ("prompt_profile", prediction.prompt_profile),
            ("explanation_source", explanation_source),
            ("raw_output_sha256", prediction.raw_output_sha256),
            ("response_model_id", prediction.response_model_id),
            ("port", "generic_text_trace"),
        )
        event_ids = (trace_terminal_event_id(request.case),)
        return TraceMethodOutputV1(
            findings=(
                TraceFindingV1(
                    kind=TraceFindingKind.ERROR_CLASSIFICATION,
                    event_ids=event_ids,
                    label=prediction.error_type,
                    score=None,
                    explanation=explanation,
                    attributes=common_attributes,
                ),
                TraceFindingV1(
                    kind=TraceFindingKind.RECOVERY_PROPOSAL,
                    event_ids=event_ids,
                    label=prediction.recovery_tool,
                    score=None,
                    explanation=explanation,
                    attributes=(
                        ("recovery_arguments_json", canonical_arguments),
                        *common_attributes,
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


class CriticToolSubprocessRunner:
    def __init__(
        self,
        *,
        corpus_root: str | Path,
        runtime: NativeTraceRuntimeV1,
        prompt_profile: str = CRITICTOOL_PROMPT_PROFILE_NO_COT,
    ) -> None:
        root = Path(corpus_root).resolve()
        self._source = (root / "code" / "CriticTool").resolve()
        if not self._source.is_relative_to(root) or not self._source.is_dir():
            raise ValueError(f"CriticTool source directory is missing: {self._source}")
        self._runtime = runtime
        if prompt_profile not in CRITICTOOL_PROMPT_PROFILES:
            raise ValueError("CriticTool prompt profile is unsupported")
        self._prompt_profile = prompt_profile
        require_python_modules(runtime, method_name="CriticTool", modules=("openai",))

    @property
    def source_revision(self) -> str:
        return require_clean_source_revision(self._source, method_name="CriticTool")

    def run(self, *, case: TraceInputV1) -> CriticToolNativePredictionV1:
        with tempfile.TemporaryDirectory(prefix="lh-critictool-") as temporary:
            request_path = Path(temporary) / "request.json"
            raw_output_path = Path(temporary) / "model-response.txt"
            request_path.write_text(
                json.dumps(case.to_dict(), ensure_ascii=False, sort_keys=True),
                encoding="utf-8",
            )
            environment = isolated_worker_environment(
                "LH_CRITICTOOL_API_KEY", self._runtime.api_key
            )
            try:
                completed = subprocess.run(
                    [
                        self._runtime.python_executable,
                        str(Path(__file__).with_name("critictool_worker.py")),
                        "--source",
                        str(self._source),
                        "--request",
                        str(request_path),
                        "--diagnostic-output",
                        str(raw_output_path),
                        "--prompt-profile",
                        self._prompt_profile,
                        *self._runtime.worker_arguments(),
                    ],
                    check=False,
                    capture_output=True,
                    text=True,
                    timeout=self._runtime.timeout_seconds,
                    env=environment,
                    cwd=temporary,
                )
            except subprocess.TimeoutExpired as error:
                raise TraceMethodExecutionError(
                    "CriticTool worker timed out",
                    error_type=type(error).__name__,
                    diagnostics=(
                        TraceDiagnosticEventV1(
                            stage=TraceDiagnosticStage.WORKER_LAUNCH,
                            status=TraceDiagnosticStatus.COMPLETED,
                        ),
                        TraceDiagnosticEventV1(
                            stage=TraceDiagnosticStage.MODEL_TRANSPORT,
                            status=TraceDiagnosticStatus.FAILED,
                            error_type=type(error).__name__,
                        ),
                    ),
                ) from error
            try:
                payload = json.loads(completed.stdout)
            except json.JSONDecodeError as error:
                stderr = completed.stderr.strip()
                detail = f": {stderr[-1000:]}" if stderr else ""
                raise RuntimeError(f"CriticTool worker returned invalid JSON{detail}") from error
            if not isinstance(payload, Mapping) or payload.get("schema") != CRITICTOOL_WORKER_SCHEMA:
                raise RuntimeError("CriticTool worker returned an unsupported envelope")
            raw_diagnostics = payload.get("diagnostics")
            if not isinstance(raw_diagnostics, list) or not all(
                isinstance(event, Mapping) for event in raw_diagnostics
            ):
                raise RuntimeError("CriticTool worker diagnostics are invalid")
            diagnostics = tuple(
                TraceDiagnosticEventV1.from_dict(event) for event in raw_diagnostics
            )
            try:
                diagnostics = self._publish_raw_output(
                    case=case,
                    raw_output_path=raw_output_path,
                    diagnostics=diagnostics,
                )
            except (FileExistsError, OSError, RuntimeError, ValueError) as error:
                raise TraceMethodExecutionError(
                    "CriticTool diagnostic artifact publication failed",
                    error_type=type(error).__name__,
                    diagnostics=(
                        *diagnostics,
                        TraceDiagnosticEventV1(
                            stage=TraceDiagnosticStage.ARTIFACT_PUBLICATION,
                            status=TraceDiagnosticStatus.FAILED,
                            error_type=type(error).__name__,
                        ),
                    ),
                ) from error
        if completed.returncode != 0 or payload.get("status") != "completed":
            error_type = str(payload.get("error_type") or "RuntimeError")
            error_message = str(payload.get("error_message") or "worker failed")
            if not any(
                event.status is TraceDiagnosticStatus.FAILED for event in diagnostics
            ):
                diagnostics = (
                    *diagnostics,
                    TraceDiagnosticEventV1(
                        stage=TraceDiagnosticStage.WORKER_LAUNCH,
                        status=TraceDiagnosticStatus.FAILED,
                        error_type=error_type,
                    ),
                )
            raise TraceMethodExecutionError(
                f"CriticTool upstream process failed: {error_message}",
                error_type=error_type,
                diagnostics=diagnostics,
            )
        required_keys = {
            "error_type",
            "recovery_tool",
            "recovery_arguments_json",
            "explanation",
            "model_calls",
            "raw_output_sha256",
            "response_model_id",
            "prompt_profile",
        }
        missing_keys = sorted(required_keys - payload.keys())
        if missing_keys:
            raise TraceMethodExecutionError(
                f"CriticTool worker response omitted fields: {missing_keys}",
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
        response_model_id = str(payload["response_model_id"])
        if response_model_id != self._runtime.model_id:
            raise TraceMethodExecutionError(
                "CriticTool worker response model attestation mismatch",
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
        prompt_profile = str(payload["prompt_profile"])
        if prompt_profile != self._prompt_profile:
            raise TraceMethodExecutionError(
                "CriticTool worker prompt profile attestation mismatch",
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
        return CriticToolNativePredictionV1(
            error_type=str(payload["error_type"]),
            recovery_tool=str(payload["recovery_tool"]),
            recovery_arguments_json=str(payload["recovery_arguments_json"]),
            explanation=str(payload["explanation"]),
            model_calls=int(payload["model_calls"]),
            raw_output_sha256=str(payload["raw_output_sha256"]),
            response_model_id=response_model_id,
            prompt_profile=prompt_profile,
            diagnostics=diagnostics,
        )

    def _publish_raw_output(
        self,
        *,
        case: TraceInputV1,
        raw_output_path: Path,
        diagnostics: tuple[TraceDiagnosticEventV1, ...],
    ) -> tuple[TraceDiagnosticEventV1, ...]:
        if not raw_output_path.is_file():
            return diagnostics
        payload = raw_output_path.read_bytes()
        content_sha256 = "sha256:" + hashlib.sha256(payload).hexdigest()
        response_indices = tuple(
            index
            for index, event in enumerate(diagnostics)
            if event.stage is TraceDiagnosticStage.MODEL_RESPONSE
        )
        if len(response_indices) != 1:
            raise RuntimeError("CriticTool diagnostics require one model response event")
        response_index = response_indices[0]
        response_event = diagnostics[response_index]
        reported_sha256 = dict(response_event.attributes).get("raw_output_sha256")
        if reported_sha256 != content_sha256:
            raise RuntimeError("CriticTool diagnostic raw output hash mismatch")
        root_value = self._runtime.diagnostic_artifact_root
        if root_value is None:
            return diagnostics
        root = Path(root_value)
        root.mkdir(parents=True, exist_ok=True)
        if not root.is_dir() or root.is_symlink():
            raise ValueError("CriticTool diagnostic artifact root is unsafe")
        case_digest = case.sha256.removeprefix("sha256:")
        output_digest = content_sha256.removeprefix("sha256:")
        target = root / f"{case_digest}-{output_digest}.model-response.txt"
        atomic_write_bytes(target, payload, overwrite=False)
        artifact = TraceDiagnosticArtifactV1(
            artifact_id="model_response",
            uri=str(Path(root.name) / target.name),
            media_type="text/plain; charset=utf-8",
            byte_count=len(payload),
            content_sha256=content_sha256,
            visibility="private",
        )
        updated = list(diagnostics)
        updated[response_index] = replace(
            response_event,
            artifacts=(*response_event.artifacts, artifact),
        )
        return (
            *updated,
            TraceDiagnosticEventV1(
                stage=TraceDiagnosticStage.ARTIFACT_PUBLICATION,
                status=TraceDiagnosticStatus.COMPLETED,
                attributes=(("artifact_count", "1"),),
            ),
        )


def native_critictool_adapter(
    *,
    corpus_root: str | Path,
    runtime: NativeTraceRuntimeV1,
    prompt_profile: str = CRITICTOOL_PROMPT_PROFILE_NO_COT,
) -> CriticToolTraceAdapter:
    runner = CriticToolSubprocessRunner(
        corpus_root=corpus_root,
        runtime=runtime,
        prompt_profile=prompt_profile,
    )
    return CriticToolTraceAdapter(
        source_revision=runner.source_revision,
        runner=runner,
        prompt_profile=prompt_profile,
    )


__all__ = [
    "CRITICTOOL_ERROR_TYPES",
    "CRITICTOOL_PROMPT_PROFILE_COT",
    "CRITICTOOL_PROMPT_PROFILE_NO_COT",
    "CRITICTOOL_PROMPT_PROFILES",
    "CRITICTOOL_WORKER_SCHEMA",
    "CriticToolNativePredictionV1",
    "CriticToolRunner",
    "CriticToolSubprocessRunner",
    "CriticToolTraceAdapter",
    "native_critictool_adapter",
]
