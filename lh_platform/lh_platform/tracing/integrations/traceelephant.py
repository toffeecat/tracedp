"""Native TraceElephant localization adapters."""

from __future__ import annotations

import json
import subprocess
import tempfile
from pathlib import Path

from lh_platform.tracing.integrations._provenance import (
    isolated_worker_environment,
    require_content_clean_source_checkout,
    require_source_checkout_identity,
)
from lh_platform.tracing.integrations._runtime import (
    NativeTraceRuntimeV1,
    require_python_modules,
    require_python_worker_entrypoint,
)
from lh_platform.tracing.integrations.whowhen import (
    WHO_WHEN_VARIANTS,
    WhoWhenNativePredictionV1,
    WhoWhenRunner,
    _strict_model_call_count,
    _strict_reasoning_audit,
    _validate_prediction_audit,
)
from lh_platform.tracing.types import EvidenceLevel, TraceInputV1
from lh_platform.tracing.unified import (
    TraceFindingKind,
    TraceFindingV1,
    TraceMethodDescriptorV1,
    TraceMethodOutputV1,
    TraceMethodRequestV1,
)


class TraceElephantTraceAdapter:
    def __init__(
        self,
        *,
        variant: str,
        source_revision: str,
        runner: WhoWhenRunner,
    ) -> None:
        if variant not in WHO_WHEN_VARIANTS:
            raise ValueError(f"unknown TraceElephant variant: {variant}")
        self._variant = variant
        self._runner = runner
        self.descriptor = TraceMethodDescriptorV1(
            method_id=f"traceelephant.{variant}",
            display_name=f"TraceElephant {variant.replace('_', ' ').title()}",
            family="failure_attribution",
            capabilities=("step_attribution",),
            input_modalities=("text_trajectory",),
            output_kinds=(TraceFindingKind.STEP_ATTRIBUTION,),
            integration_level="native_core_adapted_public_contract",
            execution_path="isolated_upstream_subprocess",
            evidence_ceiling=EvidenceLevel.L3_LOCALIZED,
            source_relpath="code/TraceElephant",
            source_revision=source_revision,
            corpus_method_id="traceelephant",
        )

    def execute(self, request: TraceMethodRequestV1) -> TraceMethodOutputV1:
        case = request.case
        prediction = self._runner.run(variant=self._variant, case=case)
        _validate_prediction_audit(prediction, method_name="TraceElephant")
        if type(prediction.abstained) is not bool:
            raise ValueError("TraceElephant upstream abstention audit is malformed")
        if not prediction.explanation:
            raise ValueError("TraceElephant upstream explanation is empty")
        if prediction.abstained:
            if self._variant != "step_by_step":
                raise ValueError("TraceElephant only defines abstention for step-by-step")
            if prediction.step_index is not None or prediction.agent_name:
                raise ValueError("TraceElephant abstention payload is inconsistent")
            return TraceMethodOutputV1(
                findings=(
                    TraceFindingV1(
                        kind=TraceFindingKind.STEP_ATTRIBUTION,
                        event_ids=(),
                        label="no_decisive_error_attributed",
                        score=None,
                        explanation=prediction.explanation,
                        attributes=(
                            ("abstained", "true"),
                            ("agent_name", ""),
                            ("model_calls", str(prediction.model_calls)),
                            ("response_model_id", prediction.response_model_id),
                            ("reasoning_tokens", str(prediction.reasoning_tokens)),
                            (
                                "reasoning_tokens_reported",
                                str(prediction.reasoning_tokens_reported).lower(),
                            ),
                            ("upstream_variant", self._variant),
                        ),
                    ),
                ),
                evidence_level=EvidenceLevel.L1_EXPLANATION,
            )
        if isinstance(prediction.step_index, bool) or not isinstance(prediction.step_index, int):
            raise ValueError("TraceElephant upstream step index is not an integer")
        if not 0 <= prediction.step_index < len(case.events):
            raise ValueError(
                f"TraceElephant upstream step index out of range: {prediction.step_index}"
            )
        if prediction.explanation.casefold().startswith("api call failed"):
            raise ValueError("TraceElephant upstream reported an API failure")
        event = case.events[prediction.step_index]
        return TraceMethodOutputV1(
            findings=(
                TraceFindingV1(
                    kind=TraceFindingKind.STEP_ATTRIBUTION,
                    event_ids=(event.event_id,),
                    label="decisive_error_step",
                    score=None,
                    explanation=prediction.explanation,
                    attributes=(
                        ("abstained", "false"),
                        ("agent_name", prediction.agent_name),
                        ("model_calls", str(prediction.model_calls)),
                        ("response_model_id", prediction.response_model_id),
                        ("reasoning_tokens", str(prediction.reasoning_tokens)),
                        (
                            "reasoning_tokens_reported",
                            str(prediction.reasoning_tokens_reported).lower(),
                        ),
                        ("upstream_variant", self._variant),
                    ),
                ),
            ),
            evidence_level=EvidenceLevel.L3_LOCALIZED,
        )


class TraceElephantSubprocessRunner:
    def __init__(
        self,
        *,
        corpus_root: str | Path,
        runtime: NativeTraceRuntimeV1,
    ) -> None:
        root = Path(corpus_root).resolve()
        self._source = (root / "code" / "TraceElephant").resolve()
        if not self._source.is_relative_to(root) or not self._source.is_dir():
            raise ValueError(f"TraceElephant source directory is missing: {self._source}")
        require_source_checkout_identity(
            self._source,
            method_name="TraceElephant",
            accepted_origin_urls=(
                "git@github.com:TraceElephant/TraceElephant.git",
                "https://github.com/TraceElephant/TraceElephant.git",
            ),
            required_paths=("code/trace_locate/lib/utils_parallel.py",),
        )
        self._source_attestation = require_content_clean_source_checkout(
            self._source, method_name="TraceElephant"
        )
        self._runtime = runtime
        require_python_modules(
            runtime,
            method_name="TraceElephant",
            modules=("openai", "tqdm", "yaml"),
        )
        require_python_worker_entrypoint(
            runtime,
            method_name="TraceElephant",
            worker_path=Path(__file__).with_name("traceelephant_worker.py"),
            environment=isolated_worker_environment("LH_TRACEELEPHANT_API_KEY", runtime.api_key),
            arguments=("--import-check", "--source", str(self._source)),
        )

    @property
    def source_revision(self) -> str:
        return self._source_attestation.version_label

    def run(
        self,
        *,
        variant: str,
        case: TraceInputV1,
    ) -> WhoWhenNativePredictionV1:
        with tempfile.TemporaryDirectory(prefix="lh-traceelephant-") as temporary:
            request_path = Path(temporary) / "request.json"
            request_path.write_text(
                json.dumps(case.to_dict(), ensure_ascii=False, sort_keys=True),
                encoding="utf-8",
            )
            environment = isolated_worker_environment(
                "LH_TRACEELEPHANT_API_KEY", self._runtime.api_key
            )
            completed = subprocess.run(
                [
                    self._runtime.python_executable,
                    str(Path(__file__).with_name("traceelephant_worker.py")),
                    "--source",
                    str(self._source),
                    "--request",
                    str(request_path),
                    "--variant",
                    variant,
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
            raise RuntimeError(f"TraceElephant upstream process failed: {error[-2000:]}")
        payload = json.loads(completed.stdout)
        response_model_id = str(payload["response_model_id"])
        if response_model_id != self._runtime.model_id:
            raise RuntimeError("TraceElephant worker response model attestation mismatch")
        reasoning_tokens, reasoning_reported = _strict_reasoning_audit(payload)
        model_calls = _strict_model_call_count(payload)
        abstained = payload.get("abstained")
        if type(abstained) is not bool:
            raise RuntimeError("TraceElephant worker abstention audit is malformed")
        return WhoWhenNativePredictionV1(
            step_index=(None if payload.get("step_index") is None else int(payload["step_index"])),
            agent_name=str(payload["agent_name"]),
            explanation=str(payload["explanation"]),
            model_calls=model_calls,
            response_model_id=response_model_id,
            reasoning_tokens=reasoning_tokens,
            reasoning_tokens_reported=reasoning_reported,
            abstained=abstained,
        )


def native_traceelephant_adapters(
    *,
    corpus_root: str | Path,
    runtime: NativeTraceRuntimeV1,
) -> tuple[TraceElephantTraceAdapter, ...]:
    runner = TraceElephantSubprocessRunner(
        corpus_root=corpus_root,
        runtime=runtime,
    )
    revision = runner.source_revision
    return tuple(
        TraceElephantTraceAdapter(variant=variant, source_revision=revision, runner=runner)
        for variant in WHO_WHEN_VARIANTS
    )


__all__ = [
    "TraceElephantSubprocessRunner",
    "TraceElephantTraceAdapter",
    "native_traceelephant_adapters",
]
