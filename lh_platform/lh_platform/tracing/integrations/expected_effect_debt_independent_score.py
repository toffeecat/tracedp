"""Order-invariant candidate-wise scoring for expected-effect-debt traces."""

from __future__ import annotations

import hashlib
import json
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from lh_platform.tracing.integrations._provenance import isolated_worker_environment
from lh_platform.tracing.integrations._runtime import (
    NativeTraceRuntimeV1,
    require_python_modules,
)
from lh_platform.tracing.integrations.expected_effect_debt_rerank import (
    ExpectedEffectDebtCandidateSetV1,
    load_expected_effect_debt_candidate_set,
)
from lh_platform.tracing.types import EvidenceLevel, TraceInputV1
from lh_platform.tracing.unified import (
    TraceFindingKind,
    TraceFindingV1,
    TraceMethodBlocked,
    TraceMethodDescriptorV1,
    TraceMethodOutputV1,
    TraceMethodRequestV1,
)

EXPECTED_EFFECT_DEBT_INDEPENDENT_SCORE_METHOD_ID = "expected_effect_debt.independent_score"


def expected_effect_debt_independent_score_catalog_descriptor() -> TraceMethodDescriptorV1:
    return TraceMethodDescriptorV1(
        method_id=EXPECTED_EFFECT_DEBT_INDEPENDENT_SCORE_METHOD_ID,
        display_name="Expected-Effect-Debt Independent Candidate Scorer",
        family="failure_attribution",
        capabilities=("step_attribution", "candidate_wise_scoring"),
        input_modalities=("text_trajectory", "candidate_event_set"),
        output_kinds=(TraceFindingKind.STEP_ATTRIBUTION,),
        integration_level="catalog_only",
        execution_path="expected_effect_debt_candidates_then_independent_score",
        evidence_ceiling=EvidenceLevel.L3_LOCALIZED,
        source_relpath=(
            "lh_platform/tracing/integrations/expected_effect_debt_independent_score.py"
        ),
        source_revision=None,
        unavailable_reason="independent scorer runtime has not passed admission",
        corpus_method_id="expected_effect_debt",
    )


@dataclass(frozen=True)
class ExpectedEffectDebtCandidateScoreV1:
    event_id: str
    causal_score: int
    explanation: str
    raw_output_sha256: str


@dataclass(frozen=True)
class ExpectedEffectDebtIndependentScorePredictionV1:
    selected_event_id: str
    candidate_scores: tuple[ExpectedEffectDebtCandidateScoreV1, ...]
    model_calls: int
    response_model_id: str


class ExpectedEffectDebtIndependentScoreRunner(Protocol):
    def run(
        self,
        *,
        case: TraceInputV1,
        candidate_set: ExpectedEffectDebtCandidateSetV1,
    ) -> ExpectedEffectDebtIndependentScorePredictionV1: ...


def _event_sequence(case: TraceInputV1) -> dict[str, int]:
    return {event.event_id: event.sequence for event in case.events}


def _stable_selected_event(
    case: TraceInputV1,
    scores: tuple[ExpectedEffectDebtCandidateScoreV1, ...],
) -> str:
    sequence = _event_sequence(case)
    return min(scores, key=lambda item: (-item.causal_score, sequence[item.event_id])).event_id


def build_expected_effect_debt_independent_score_prompt(
    case: TraceInputV1,
    candidate_set: ExpectedEffectDebtCandidateSetV1,
    *,
    event_id: str,
) -> str:
    """Build one prompt whose candidate membership contains exactly one event."""

    if event_id not in set(candidate_set.event_ids):
        raise ValueError("independent-score target is outside candidate set")
    trajectory = "\n".join(
        f"[{event.event_id}] {event.kind}: {event.content}" for event in case.events
    )
    reasons = "\n".join(
        f"- reason={reason_code}; observed_match={evidence}"
        for reason_event_id, reason_code, evidence in candidate_set.reasons
        if reason_event_id == event_id
    )
    return (
        "Assess one candidate action as the earliest cause of the observed task "
        "failure. No other event is identified as a candidate. Distinguish an "
        "action that first created an unresolved state/precondition debt from a "
        "later symptom that merely exposed it. Score causal responsibility from "
        "0 to 100. Return only a JSON object with exactly two fields: integer "
        '"causal_score" and string "explanation".\n\n'
        f"CONTEXT\n{case.context_text}\n\n"
        f"OBSERVED_TRAJECTORY\n{trajectory}\n\n"
        f"TARGET_CANDIDATE\n{event_id}\n\n"
        f"TARGET_EVIDENCE\n{reasons or '<none>'}"
    )


class ExpectedEffectDebtIndependentScoreTraceAdapter:
    def __init__(
        self,
        *,
        source_revision: str,
        runner: ExpectedEffectDebtIndependentScoreRunner,
    ) -> None:
        if not source_revision:
            raise ValueError("independent scorer revision must be non-empty")
        self._runner = runner
        self.descriptor = TraceMethodDescriptorV1(
            method_id=EXPECTED_EFFECT_DEBT_INDEPENDENT_SCORE_METHOD_ID,
            display_name="Expected-Effect-Debt Independent Candidate Scorer",
            family="failure_attribution",
            capabilities=("step_attribution", "candidate_wise_scoring"),
            input_modalities=("text_trajectory", "candidate_event_set"),
            output_kinds=(TraceFindingKind.STEP_ATTRIBUTION,),
            integration_level="platform_native",
            execution_path="expected_effect_debt_candidates_then_independent_score",
            evidence_ceiling=EvidenceLevel.L3_LOCALIZED,
            source_relpath=(
                "lh_platform/tracing/integrations/expected_effect_debt_independent_score.py"
            ),
            source_revision=source_revision,
            corpus_method_id="expected_effect_debt",
        )

    def execute(self, request: TraceMethodRequestV1) -> TraceMethodOutputV1:
        artifacts = tuple(
            artifact for artifact in request.artifacts if artifact.modality == "candidate_event_set"
        )
        if len(artifacts) != 1:
            raise ValueError("independent scorer requires exactly one candidate_event_set")
        candidate_set = load_expected_effect_debt_candidate_set(
            artifact=artifacts[0], case=request.case
        )
        if not candidate_set.event_ids:
            raise TraceMethodBlocked("candidate_event_set contains no candidates")
        prediction = self._runner.run(case=request.case, candidate_set=candidate_set)
        score_ids = tuple(item.event_id for item in prediction.candidate_scores)
        if len(score_ids) != len(set(score_ids)) or set(score_ids) != set(candidate_set.event_ids):
            raise ValueError("independent scorer must return one score per candidate")
        for item in prediction.candidate_scores:
            if (
                isinstance(item.causal_score, bool)
                or not isinstance(item.causal_score, int)
                or not 0 <= item.causal_score <= 100
            ):
                raise ValueError("independent causal score must be an integer in [0, 100]")
            if not item.explanation.strip():
                raise ValueError("independent score explanation is empty")
            if not item.raw_output_sha256.startswith("sha256:"):
                raise ValueError("independent score raw output digest is invalid")
        expected_selected = _stable_selected_event(request.case, prediction.candidate_scores)
        if prediction.selected_event_id != expected_selected:
            raise ValueError("independent scorer selected event does not match stable argmax")
        if prediction.model_calls != len(candidate_set.event_ids):
            raise ValueError("independent scorer must make exactly one call per candidate")
        if not prediction.response_model_id:
            raise ValueError("independent scorer model attestation is empty")
        scores_by_event = {item.event_id: item.causal_score for item in prediction.candidate_scores}
        selected = next(
            item
            for item in prediction.candidate_scores
            if item.event_id == prediction.selected_event_id
        )
        return TraceMethodOutputV1(
            findings=(
                TraceFindingV1(
                    kind=TraceFindingKind.STEP_ATTRIBUTION,
                    event_ids=(selected.event_id,),
                    label="expected_effect_debt_independent_root_cause",
                    score=selected.causal_score / 100.0,
                    explanation=selected.explanation,
                    attributes=(
                        ("candidate_artifact_sha256", candidate_set.artifact_sha256),
                        ("candidate_count", str(len(candidate_set.event_ids))),
                        ("model_calls", str(prediction.model_calls)),
                        (
                            "candidate_scores",
                            json.dumps(scores_by_event, sort_keys=True),
                        ),
                        ("response_model_id", prediction.response_model_id),
                        ("selection_rule", "max_score_then_earliest_sequence"),
                    ),
                ),
            ),
            evidence_level=EvidenceLevel.L3_LOCALIZED,
        )


class ExpectedEffectDebtIndependentScoreSubprocessRunner:
    def __init__(self, *, runtime: NativeTraceRuntimeV1) -> None:
        self._runtime = runtime
        require_python_modules(
            runtime,
            method_name="Expected-Effect-Debt Independent Candidate Scorer",
            modules=("openai",),
        )

    def run(
        self,
        *,
        case: TraceInputV1,
        candidate_set: ExpectedEffectDebtCandidateSetV1,
    ) -> ExpectedEffectDebtIndependentScorePredictionV1:
        with tempfile.TemporaryDirectory(prefix="lh-expected-debt-independent-") as temp:
            request_path = Path(temp) / "request.json"
            candidates_path = Path(temp) / "candidates.json"
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
                    str(
                        Path(__file__).with_name("expected_effect_debt_independent_score_worker.py")
                    ),
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
                cwd=temp,
                env=isolated_worker_environment(
                    "LH_EXPECTED_EFFECT_DEBT_RERANK_API_KEY", self._runtime.api_key
                ),
            )
        if completed.returncode != 0:
            error = completed.stderr.strip() or completed.stdout.strip()
            raise RuntimeError(f"independent scorer failed: {error[-2000:]}")
        try:
            payload = json.loads(completed.stdout)
        except json.JSONDecodeError as error:
            raise RuntimeError("independent scorer worker returned invalid JSON") from error
        response_model_id = str(payload["response_model_id"])
        if response_model_id != self._runtime.model_id:
            raise RuntimeError("independent scorer model attestation mismatch")
        scores = tuple(
            ExpectedEffectDebtCandidateScoreV1(
                event_id=str(item["event_id"]),
                causal_score=int(item["causal_score"]),
                explanation=str(item["explanation"]),
                raw_output_sha256=str(item["raw_output_sha256"]),
            )
            for item in payload["candidate_scores"]
        )
        return ExpectedEffectDebtIndependentScorePredictionV1(
            selected_event_id=str(payload["selected_event_id"]),
            candidate_scores=scores,
            model_calls=int(payload["model_calls"]),
            response_model_id=response_model_id,
        )


def native_expected_effect_debt_independent_score_adapter(
    *, runtime: NativeTraceRuntimeV1
) -> ExpectedEffectDebtIndependentScoreTraceAdapter:
    source_paths = (
        Path(__file__),
        Path(__file__).with_name("expected_effect_debt_independent_score_worker.py"),
    )
    digest = hashlib.sha256()
    for path in source_paths:
        digest.update(path.name.encode())
        digest.update(path.read_bytes())
    return ExpectedEffectDebtIndependentScoreTraceAdapter(
        source_revision="sha256:" + digest.hexdigest(),
        runner=ExpectedEffectDebtIndependentScoreSubprocessRunner(runtime=runtime),
    )


__all__ = [
    "EXPECTED_EFFECT_DEBT_INDEPENDENT_SCORE_METHOD_ID",
    "ExpectedEffectDebtCandidateScoreV1",
    "ExpectedEffectDebtIndependentScorePredictionV1",
    "ExpectedEffectDebtIndependentScoreRunner",
    "ExpectedEffectDebtIndependentScoreSubprocessRunner",
    "ExpectedEffectDebtIndependentScoreTraceAdapter",
    "build_expected_effect_debt_independent_score_prompt",
    "expected_effect_debt_independent_score_catalog_descriptor",
    "native_expected_effect_debt_independent_score_adapter",
]
