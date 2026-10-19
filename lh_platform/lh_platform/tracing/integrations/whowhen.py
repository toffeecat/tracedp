"""Native Who&When adapter normalized to the unified tracing seam."""

from __future__ import annotations

import json
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

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
from lh_platform.tracing.types import EvidenceLevel, TraceInputV1
from lh_platform.tracing.unified import (
    TraceFindingKind,
    TraceFindingV1,
    TraceMethodDescriptorV1,
    TraceMethodOutputV1,
    TraceMethodRequestV1,
)

WHO_WHEN_VARIANTS = ("all_at_once", "step_by_step", "binary_search")


@dataclass(frozen=True)
class WhoWhenNativePredictionV1:
    step_index: int | None
    agent_name: str
    explanation: str
    model_calls: int
    response_model_id: str | None
    reasoning_tokens: int
    reasoning_tokens_reported: bool
    abstained: bool = False


def _strict_model_call_count(payload: dict[str, object]) -> int:
    """Require an explicit positive integer call count from the worker."""

    model_calls = payload.get("model_calls")
    if isinstance(model_calls, bool) or not isinstance(model_calls, int) or model_calls < 1:
        raise RuntimeError("native worker violated the model call audit contract")
    return model_calls


def _strict_reasoning_audit(payload: dict[str, object]) -> tuple[int, bool]:
    """Require an explicit zero-reasoning receipt from the upstream worker."""

    reasoning_tokens = payload.get("reasoning_tokens")
    reasoning_reported = payload.get("reasoning_tokens_reported")
    if (
        isinstance(reasoning_tokens, bool)
        or not isinstance(reasoning_tokens, int)
        or reasoning_tokens != 0
        or reasoning_reported is not True
    ):
        raise RuntimeError("native worker violated the reasoning token audit contract")
    return reasoning_tokens, reasoning_reported


def _validate_prediction_audit(
    prediction: WhoWhenNativePredictionV1, *, method_name: str,
    allow_no_model_call: bool = False,
) -> None:
    """Attest model use, or the author's explicitly bound no-call singleton leaf."""

    if allow_no_model_call:
        if (
            type(prediction.model_calls) is not int or prediction.model_calls != 0
            or prediction.response_model_id is not None
            or type(prediction.reasoning_tokens) is not int or prediction.reasoning_tokens != 0
            or prediction.reasoning_tokens_reported is not False
        ):
            raise ValueError(f"{method_name} singleton leaf has an invalid no-call audit")
        return

    if (
        isinstance(prediction.model_calls, bool)
        or not isinstance(prediction.model_calls, int)
        or prediction.model_calls < 1
    ):
        raise ValueError(f"{method_name} upstream execution made no model calls")
    if not prediction.response_model_id:
        raise ValueError(f"{method_name} response model attestation is empty")
    if (
        isinstance(prediction.reasoning_tokens, bool)
        or not isinstance(prediction.reasoning_tokens, int)
        or prediction.reasoning_tokens != 0
        or prediction.reasoning_tokens_reported is not True
    ):
        raise ValueError(f"{method_name} violated the reasoning audit contract")


class WhoWhenRunner(Protocol):
    def run(
        self,
        *,
        variant: str,
        case: TraceInputV1,
    ) -> WhoWhenNativePredictionV1: ...


class WhoWhenTraceAdapter:
    """Run one upstream Who&When search variant and normalize its prediction."""

    def __init__(
        self,
        *,
        variant: str,
        source_revision: str,
        runner: WhoWhenRunner,
    ) -> None:
        if variant not in WHO_WHEN_VARIANTS:
            raise ValueError(f"unknown Who&When variant: {variant}")
        if not source_revision:
            raise ValueError("Who&When source revision must be non-empty")
        self._variant = variant
        self._runner = runner
        self.descriptor = TraceMethodDescriptorV1(
            method_id=f"who_when.{variant}",
            display_name=f"Who&When {variant.replace('_', ' ').title()}",
            family="failure_attribution",
            capabilities=("step_attribution",),
            input_modalities=("text_trajectory",),
            output_kinds=(TraceFindingKind.STEP_ATTRIBUTION,),
            integration_level="native_code_adapted_input",
            execution_path="isolated_upstream_subprocess",
            evidence_ceiling=EvidenceLevel.L3_LOCALIZED,
            source_relpath="code/Agents_Failure_Attribution",
            source_revision=source_revision,
            corpus_method_id="who_when",
        )

    def execute(self, request: TraceMethodRequestV1) -> TraceMethodOutputV1:
        case = request.case
        prediction = self._runner.run(variant=self._variant, case=case)
        _validate_prediction_audit(
            prediction, method_name="Who&When",
            allow_no_model_call=self._variant == "binary_search" and len(case.events) == 1,
        )
        if type(prediction.abstained) is not bool:
            raise ValueError("Who&When upstream abstention audit is malformed")
        if not prediction.explanation:
            raise ValueError("Who&When upstream explanation is empty")
        if prediction.abstained:
            if self._variant != "step_by_step":
                raise ValueError("Who&When only defines abstention for step-by-step")
            if prediction.step_index is not None or prediction.agent_name:
                raise ValueError("Who&When abstention payload is inconsistent")
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
                            ("response_model_id", prediction.response_model_id or ""),
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
            raise ValueError("Who&When upstream step index is not an integer")
        if not prediction.agent_name:
            raise ValueError("Who&When upstream agent name is empty")
        if not 0 <= prediction.step_index < len(case.events):
            raise ValueError(f"Who&When upstream step index out of range: {prediction.step_index}")
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
                        ("response_model_id", prediction.response_model_id or ""),
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


class WhoWhenSubprocessRunner:
    """Execute pinned upstream Who&When code outside the platform process."""

    def __init__(
        self,
        *,
        corpus_root: str | Path,
        runtime: NativeTraceRuntimeV1,
    ) -> None:
        root = Path(corpus_root).resolve()
        source = (root / "code" / "Agents_Failure_Attribution").resolve()
        if not source.is_relative_to(root):
            raise ValueError("Who&When source escaped the supplied corpus root")
        if not source.is_dir():
            raise ValueError(f"Who&When source directory is missing: {source}")
        require_source_checkout_identity(
            source,
            method_name="Who&When",
            accepted_origin_urls=(
                "git@github.com:mingyin1/Agents_Failure_Attribution",
                "https://github.com/mingyin1/Agents_Failure_Attribution.git",
            ),
            required_paths=("Automated_FA/Lib/utils.py",),
        )
        self._source = source
        self._source_attestation = require_content_clean_source_checkout(
            source, method_name="Who&When"
        )
        self._runtime = runtime
        require_python_modules(
            runtime,
            method_name="Who&When",
            modules=("openai", "tqdm", "yaml"),
        )
        require_python_worker_entrypoint(
            runtime,
            method_name="Who&When",
            worker_path=Path(__file__).with_name("whowhen_worker.py"),
            environment=isolated_worker_environment("LH_WHOWHEN_API_KEY", runtime.api_key),
            arguments=("--import-check", "--source", str(source)),
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
        if variant not in WHO_WHEN_VARIANTS:
            raise ValueError(f"unknown Who&When variant: {variant}")
        with tempfile.TemporaryDirectory(prefix="lh-whowhen-") as temporary:
            request_path = Path(temporary) / "request.json"
            request_path.write_text(
                json.dumps(case.to_dict(), ensure_ascii=False, sort_keys=True),
                encoding="utf-8",
            )
            environment = isolated_worker_environment("LH_WHOWHEN_API_KEY", self._runtime.api_key)
            completed = subprocess.run(
                [
                    self._runtime.python_executable,
                    str(Path(__file__).with_name("whowhen_worker.py")),
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
            raise RuntimeError(f"Who&When upstream process failed: {error[-2000:]}")
        try:
            payload = json.loads(completed.stdout)
        except json.JSONDecodeError as error:
            raise RuntimeError("Who&When worker returned invalid JSON") from error
        no_call_leaf = variant == "binary_search" and len(case.events) == 1
        if no_call_leaf:
            response_model_id = payload["response_model_id"]
            reasoning_tokens = payload["reasoning_tokens"]
            reasoning_reported = payload["reasoning_tokens_reported"]
            model_calls = payload["model_calls"]
        else:
            response_model_id = str(payload["response_model_id"])
            if response_model_id != self._runtime.model_id:
                raise RuntimeError("Who&When worker response model attestation mismatch")
            reasoning_tokens, reasoning_reported = _strict_reasoning_audit(payload)
            model_calls = _strict_model_call_count(payload)
        abstained = payload.get("abstained")
        if type(abstained) is not bool:
            raise RuntimeError("Who&When worker abstention audit is malformed")
        prediction = WhoWhenNativePredictionV1(
            step_index=(None if payload.get("step_index") is None else int(payload["step_index"])),
            agent_name=str(payload["agent_name"]),
            explanation=str(payload["explanation"]),
            model_calls=model_calls,
            response_model_id=response_model_id,
            reasoning_tokens=reasoning_tokens,
            reasoning_tokens_reported=reasoning_reported,
            abstained=abstained,
        )
        _validate_prediction_audit(
            prediction, method_name="Who&When", allow_no_model_call=no_call_leaf,
        )
        return prediction


def native_whowhen_adapters(
    *,
    corpus_root: str | Path,
    runtime: NativeTraceRuntimeV1,
) -> tuple[WhoWhenTraceAdapter, ...]:
    runner = WhoWhenSubprocessRunner(
        corpus_root=corpus_root,
        runtime=runtime,
    )
    revision = runner.source_revision
    return tuple(
        WhoWhenTraceAdapter(variant=variant, source_revision=revision, runner=runner)
        for variant in WHO_WHEN_VARIANTS
    )


__all__ = [
    "WHO_WHEN_VARIANTS",
    "WhoWhenNativePredictionV1",
    "WhoWhenRunner",
    "WhoWhenSubprocessRunner",
    "WhoWhenTraceAdapter",
    "native_whowhen_adapters",
]
