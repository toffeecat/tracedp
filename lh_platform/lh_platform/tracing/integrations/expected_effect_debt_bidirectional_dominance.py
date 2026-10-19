"""Bidirectional pairwise causal dominance over expected-effect-debt candidates."""

from __future__ import annotations

import hashlib
import json
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from lh_platform.tracing.integrations._provenance import strict_worker_environment
from lh_platform.tracing.integrations._runtime import NativeTraceRuntimeV1, require_python_modules
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

EXPECTED_EFFECT_DEBT_BIDIRECTIONAL_DOMINANCE_METHOD_ID = (
    "expected_effect_debt.bidirectional_dominance"
)


@dataclass(frozen=True)
class BidirectionalDominanceOrientationV1:
    presented_first_event_id: str
    presented_second_event_id: str
    winner_event_id: str | None
    confidence: int | None
    explanation: str
    raw_output: str
    raw_output_sha256: str
    finish_reason: str
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    parse_valid: bool
    parse_error: str | None

    def __post_init__(self) -> None:
        pair = {self.presented_first_event_id, self.presented_second_event_id}
        if len(pair) != 2 or not all(pair):
            raise ValueError("bidirectional orientation requires two distinct events")
        expected_digest = "sha256:" + hashlib.sha256(self.raw_output.encode()).hexdigest()
        if self.raw_output_sha256 != expected_digest:
            raise ValueError("bidirectional orientation raw output digest changed")
        token_values = (self.prompt_tokens, self.completion_tokens, self.total_tokens)
        if any(isinstance(value, bool) or value < 0 for value in token_values):
            raise ValueError("bidirectional orientation token usage is invalid")
        if self.total_tokens != self.prompt_tokens + self.completion_tokens:
            raise ValueError("bidirectional orientation token usage does not add up")
        if not self.finish_reason:
            raise ValueError("bidirectional orientation finish reason is empty")
        if self.parse_valid:
            if self.parse_error is not None:
                raise ValueError("valid bidirectional orientation cannot carry parse error")
            if self.winner_event_id is not None and self.winner_event_id not in pair:
                raise ValueError("bidirectional winner is outside the compared pair")
            if (
                isinstance(self.confidence, bool)
                or not isinstance(self.confidence, int)
                or not 0 <= self.confidence <= 100
                or not self.explanation.strip()
            ):
                raise ValueError("valid bidirectional orientation payload is invalid")
        elif self.parse_error is None:
            raise ValueError("invalid bidirectional orientation requires parse error")

    def to_dict(self) -> dict[str, object]:
        return {
            "presented_first_event_id": self.presented_first_event_id,
            "presented_second_event_id": self.presented_second_event_id,
            "winner_event_id": self.winner_event_id,
            "confidence": self.confidence,
            "explanation": self.explanation,
            "raw_output": self.raw_output,
            "raw_output_sha256": self.raw_output_sha256,
            "finish_reason": self.finish_reason,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
            "parse_valid": self.parse_valid,
            "parse_error": self.parse_error,
        }


@dataclass(frozen=True)
class BidirectionalDominanceComparisonV1:
    left_event_id: str
    right_event_id: str
    orientations: tuple[
        BidirectionalDominanceOrientationV1,
        BidirectionalDominanceOrientationV1,
    ]
    orientation_consistent: bool
    consistent_winner_event_id: str | None

    def __post_init__(self) -> None:
        if (
            not self.left_event_id
            or not self.right_event_id
            or self.left_event_id == self.right_event_id
        ):
            raise ValueError("bidirectional comparison pair is invalid")
        if len(self.orientations) != 2:
            raise ValueError("bidirectional comparison requires two orientations")
        expected_orders = (
            (self.left_event_id, self.right_event_id),
            (self.right_event_id, self.left_event_id),
        )
        observed_orders = tuple(
            (item.presented_first_event_id, item.presented_second_event_id)
            for item in self.orientations
        )
        if observed_orders != expected_orders:
            raise ValueError("bidirectional comparison orientation order changed")
        observed_consistency = all(item.parse_valid for item in self.orientations) and (
            self.orientations[0].winner_event_id == self.orientations[1].winner_event_id
        )
        if self.orientation_consistent != observed_consistency:
            raise ValueError("bidirectional comparison consistency flag changed")
        expected_winner = self.orientations[0].winner_event_id if observed_consistency else None
        if self.consistent_winner_event_id != expected_winner:
            raise ValueError("bidirectional comparison consistent winner changed")

    def to_dict(self) -> dict[str, object]:
        return {
            "left_event_id": self.left_event_id,
            "right_event_id": self.right_event_id,
            "orientations": [item.to_dict() for item in self.orientations],
            "orientation_consistent": self.orientation_consistent,
            "consistent_winner_event_id": self.consistent_winner_event_id,
        }


@dataclass(frozen=True)
class BidirectionalDominancePredictionV1:
    selected_event_id: str | None
    comparisons: tuple[BidirectionalDominanceComparisonV1, ...]
    model_calls: int
    response_model_id: str


def parse_bidirectional_orientation_response(
    *,
    raw_output: str,
    presented_first_event_id: str,
    presented_second_event_id: str,
    finish_reason: str,
    prompt_tokens: int,
    completion_tokens: int,
    total_tokens: int,
) -> BidirectionalDominanceOrientationV1:
    """Parse only after binding the exact raw transport evidence."""

    digest = "sha256:" + hashlib.sha256(raw_output.encode()).hexdigest()
    winner: str | None = None
    confidence: int | None = None
    explanation = ""
    parse_valid = False
    parse_error: str | None = None
    try:
        payload = json.loads(raw_output)
        if not isinstance(payload, dict) or set(payload) != {
            "winner_event_id",
            "confidence",
            "explanation",
        }:
            raise ValueError("orientation response must contain exactly three fields")
        raw_winner = payload["winner_event_id"]
        allowed = {presented_first_event_id, presented_second_event_id, "abstain"}
        if not isinstance(raw_winner, str) or raw_winner not in allowed:
            raise ValueError("orientation winner is outside the dynamic pair enum")
        raw_confidence = payload["confidence"]
        raw_explanation = payload["explanation"]
        if (
            isinstance(raw_confidence, bool)
            or not isinstance(raw_confidence, int)
            or not 0 <= raw_confidence <= 100
        ):
            raise ValueError("orientation confidence must be an integer in [0, 100]")
        if not isinstance(raw_explanation, str) or not raw_explanation.strip():
            raise ValueError("orientation explanation is empty")
        winner = None if raw_winner == "abstain" else raw_winner
        confidence = raw_confidence
        explanation = raw_explanation
        parse_valid = True
    except (json.JSONDecodeError, TypeError, ValueError) as error:
        parse_error = f"{type(error).__name__}: {error}"
    return BidirectionalDominanceOrientationV1(
        presented_first_event_id=presented_first_event_id,
        presented_second_event_id=presented_second_event_id,
        winner_event_id=winner,
        confidence=confidence,
        explanation=explanation,
        raw_output=raw_output,
        raw_output_sha256=digest,
        finish_reason=finish_reason,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=total_tokens,
        parse_valid=parse_valid,
        parse_error=parse_error,
    )


class ExpectedEffectDebtBidirectionalDominanceRunner(Protocol):
    def run(
        self,
        *,
        case: TraceInputV1,
        candidate_set: ExpectedEffectDebtCandidateSetV1,
    ) -> BidirectionalDominancePredictionV1: ...


def _orientation_from_dict(payload: dict[str, object]) -> BidirectionalDominanceOrientationV1:
    return BidirectionalDominanceOrientationV1(
        presented_first_event_id=str(payload["presented_first_event_id"]),
        presented_second_event_id=str(payload["presented_second_event_id"]),
        winner_event_id=(
            None if payload.get("winner_event_id") is None else str(payload["winner_event_id"])
        ),
        confidence=(None if payload.get("confidence") is None else int(payload["confidence"])),
        explanation=str(payload["explanation"]),
        raw_output=str(payload["raw_output"]),
        raw_output_sha256=str(payload["raw_output_sha256"]),
        finish_reason=str(payload["finish_reason"]),
        prompt_tokens=int(payload["prompt_tokens"]),
        completion_tokens=int(payload["completion_tokens"]),
        total_tokens=int(payload["total_tokens"]),
        parse_valid=bool(payload["parse_valid"]),
        parse_error=(None if payload.get("parse_error") is None else str(payload["parse_error"])),
    )


def _comparison_from_dict(payload: dict[str, object]) -> BidirectionalDominanceComparisonV1:
    raw_orientations = payload.get("orientations")
    if (
        not isinstance(raw_orientations, list)
        or len(raw_orientations) != 2
        or not all(isinstance(item, dict) for item in raw_orientations)
    ):
        raise ValueError("bidirectional worker comparison orientations are invalid")
    return BidirectionalDominanceComparisonV1(
        left_event_id=str(payload["left_event_id"]),
        right_event_id=str(payload["right_event_id"]),
        orientations=(
            _orientation_from_dict(raw_orientations[0]),
            _orientation_from_dict(raw_orientations[1]),
        ),
        orientation_consistent=bool(payload["orientation_consistent"]),
        consistent_winner_event_id=(
            None
            if payload.get("consistent_winner_event_id") is None
            else str(payload["consistent_winner_event_id"])
        ),
    )


class ExpectedEffectDebtBidirectionalDominanceSubprocessRunner:
    def __init__(self, *, runtime: NativeTraceRuntimeV1) -> None:
        self._runtime = runtime
        require_python_modules(
            runtime,
            method_name="Expected-Effect-Debt Bidirectional Causal Dominance",
            modules=("openai",),
        )

    def run(
        self,
        *,
        case: TraceInputV1,
        candidate_set: ExpectedEffectDebtCandidateSetV1,
    ) -> BidirectionalDominancePredictionV1:
        pair_count = len(candidate_set.event_ids) * (len(candidate_set.event_ids) - 1) // 2
        call_budget = 2 * pair_count
        with tempfile.TemporaryDirectory(prefix="lh-expected-debt-dominance-") as temp:
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
                        Path(__file__).with_name(
                            "expected_effect_debt_bidirectional_dominance_worker.py"
                        )
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
                timeout=self._runtime.timeout_seconds * call_budget + 30,
                cwd=temp,
                env=strict_worker_environment(
                    "LH_EXPECTED_EFFECT_DEBT_RERANK_API_KEY", self._runtime.api_key
                ),
            )
        if completed.returncode != 0:
            error = completed.stderr.strip() or completed.stdout.strip()
            raise RuntimeError(f"bidirectional dominance worker failed: {error[-4000:]}")
        try:
            payload = json.loads(completed.stdout)
        except json.JSONDecodeError as error:
            raise RuntimeError("bidirectional dominance worker returned invalid JSON") from error
        if not isinstance(payload, dict):
            raise RuntimeError("bidirectional dominance worker payload is not an object")
        response_model_id = str(payload["response_model_id"])
        if response_model_id != self._runtime.model_id:
            raise RuntimeError("bidirectional dominance model attestation mismatch")
        raw_comparisons = payload.get("comparisons")
        if not isinstance(raw_comparisons, list) or not all(
            isinstance(item, dict) for item in raw_comparisons
        ):
            raise RuntimeError("bidirectional dominance comparisons are invalid")
        return BidirectionalDominancePredictionV1(
            selected_event_id=(
                None
                if payload.get("selected_event_id") is None
                else str(payload["selected_event_id"])
            ),
            comparisons=tuple(_comparison_from_dict(item) for item in raw_comparisons),
            model_calls=int(payload["model_calls"]),
            response_model_id=response_model_id,
        )


def expected_effect_debt_bidirectional_dominance_catalog_descriptor() -> TraceMethodDescriptorV1:
    return TraceMethodDescriptorV1(
        method_id=EXPECTED_EFFECT_DEBT_BIDIRECTIONAL_DOMINANCE_METHOD_ID,
        display_name="Expected-Effect-Debt Bidirectional Causal Dominance",
        family="failure_attribution",
        capabilities=("step_attribution", "bidirectional_pairwise_comparison"),
        input_modalities=("text_trajectory", "candidate_event_set"),
        output_kinds=(TraceFindingKind.STEP_ATTRIBUTION,),
        integration_level="catalog_only",
        execution_path="expected_effect_debt_candidates_then_bidirectional_dominance",
        evidence_ceiling=EvidenceLevel.L3_LOCALIZED,
        source_relpath=(
            "lh_platform/tracing/integrations/expected_effect_debt_bidirectional_dominance.py"
        ),
        source_revision=None,
        unavailable_reason="bidirectional dominance runtime has not passed admission",
        corpus_method_id="expected_effect_debt",
    )


def native_expected_effect_debt_bidirectional_dominance_adapter(
    *, runtime: NativeTraceRuntimeV1
) -> ExpectedEffectDebtBidirectionalDominanceTraceAdapter:
    return ExpectedEffectDebtBidirectionalDominanceTraceAdapter(
        source_revision=expected_effect_debt_bidirectional_dominance_source_revision(),
        runner=ExpectedEffectDebtBidirectionalDominanceSubprocessRunner(runtime=runtime),
    )


def expected_effect_debt_bidirectional_dominance_source_revision() -> str:
    """Bind both adapter and worker bytes into one portable runtime identity."""

    source_paths = (
        Path(__file__),
        Path(__file__).with_name("expected_effect_debt_bidirectional_dominance_worker.py"),
    )
    digest = hashlib.sha256()
    for path in source_paths:
        digest.update(path.name.encode())
        digest.update(path.read_bytes())
    return "sha256:" + digest.hexdigest()


def _sequence_ordered_candidates(
    case: TraceInputV1, candidate_set: ExpectedEffectDebtCandidateSetV1
) -> tuple[str, ...]:
    sequence = {event.event_id: event.sequence for event in case.events}
    return tuple(sorted(candidate_set.event_ids, key=sequence.__getitem__))


def _expected_pairs(candidate_ids: tuple[str, ...]) -> tuple[tuple[str, str], ...]:
    return tuple(
        (candidate_ids[left], candidate_ids[right])
        for left in range(len(candidate_ids))
        for right in range(left + 1, len(candidate_ids))
    )


def _condorcet_winner(
    candidate_ids: tuple[str, ...],
    comparisons: tuple[BidirectionalDominanceComparisonV1, ...],
) -> str | None:
    victories = {event_id: set() for event_id in candidate_ids}
    for comparison in comparisons:
        winner = comparison.consistent_winner_event_id
        if not comparison.orientation_consistent or winner is None:
            continue
        loser = (
            comparison.right_event_id
            if winner == comparison.left_event_id
            else comparison.left_event_id
        )
        victories[winner].add(loser)
    winners = [
        event_id
        for event_id, defeated in victories.items()
        if len(defeated) == len(candidate_ids) - 1
    ]
    return winners[0] if len(winners) == 1 else None


def build_expected_effect_debt_bidirectional_dominance_prompt(
    case: TraceInputV1,
    candidate_set: ExpectedEffectDebtCandidateSetV1,
    *,
    presented_first_event_id: str,
    presented_second_event_id: str,
) -> str:
    pair = (presented_first_event_id, presented_second_event_id)
    if len(set(pair)) != 2 or not set(pair).issubset(candidate_set.event_ids):
        raise ValueError("bidirectional prompt pair is outside the candidate set")
    trajectory = "\n".join(
        f"[{event.event_id}] {event.kind}: {event.content}" for event in case.events
    )
    evidence_by_event = {
        event_id: (reason, evidence)
        for event_id, reason, evidence in candidate_set.reasons
        if event_id in pair
    }
    evidence_lines = []
    for label, event_id in zip(("A", "B"), pair, strict=True):
        reason, evidence = evidence_by_event.get(event_id, ("<none>", "<none>"))
        evidence_lines.append(
            f"CANDIDATE_{label}={event_id}\nreason={reason}; observed_match={evidence}"
        )
    allowed = json.dumps([*pair, "abstain"], ensure_ascii=False)
    return (
        "Compare exactly two candidate actions. Decide which action first created "
        "the unresolved state/precondition debt that later caused the evaluator "
        "failure. Do not select a later symptom merely because it is closer to the "
        "failure. If neither candidate can be preferred from the evidence, abstain. "
        'Return only a JSON object with exactly three fields: "winner_event_id", '
        '"confidence", "explanation". "winner_event_id" must be one of '
        f"{allowed}; confidence must be an integer from 0 to 100.\n\n"
        f"CONTEXT\n{case.context_text}\n\n"
        f"OBSERVED_TRAJECTORY\n{trajectory}\n\n"
        "PAIR_TO_COMPARE\n" + "\n".join(evidence_lines)
    )


class ExpectedEffectDebtBidirectionalDominanceTraceAdapter:
    def __init__(
        self,
        *,
        source_revision: str,
        runner: ExpectedEffectDebtBidirectionalDominanceRunner,
    ) -> None:
        if not source_revision:
            raise ValueError("bidirectional dominance revision must be non-empty")
        self._runner = runner
        self.descriptor = TraceMethodDescriptorV1(
            method_id=EXPECTED_EFFECT_DEBT_BIDIRECTIONAL_DOMINANCE_METHOD_ID,
            display_name="Expected-Effect-Debt Bidirectional Causal Dominance",
            family="failure_attribution",
            capabilities=("step_attribution", "bidirectional_pairwise_comparison"),
            input_modalities=("text_trajectory", "candidate_event_set"),
            output_kinds=(TraceFindingKind.STEP_ATTRIBUTION,),
            integration_level="platform_native",
            execution_path="expected_effect_debt_candidates_then_bidirectional_dominance",
            evidence_ceiling=EvidenceLevel.L3_LOCALIZED,
            source_relpath=(
                "lh_platform/tracing/integrations/expected_effect_debt_bidirectional_dominance.py"
            ),
            source_revision=source_revision,
            corpus_method_id="expected_effect_debt",
        )

    def execute(self, request: TraceMethodRequestV1) -> TraceMethodOutputV1:
        artifacts = tuple(
            artifact for artifact in request.artifacts if artifact.modality == "candidate_event_set"
        )
        if len(artifacts) != 1:
            raise ValueError("bidirectional dominance requires exactly one candidate_event_set")
        candidate_set = load_expected_effect_debt_candidate_set(
            artifact=artifacts[0], case=request.case
        )
        if len(candidate_set.event_ids) < 2:
            raise TraceMethodBlocked("bidirectional dominance requires at least two candidates")
        candidate_ids = _sequence_ordered_candidates(request.case, candidate_set)
        prediction = self._runner.run(case=request.case, candidate_set=candidate_set)
        expected_pairs = _expected_pairs(candidate_ids)
        observed_pairs = tuple(
            (item.left_event_id, item.right_event_id) for item in prediction.comparisons
        )
        if observed_pairs != expected_pairs:
            raise ValueError("bidirectional dominance comparison coverage changed")
        expected_calls = 2 * len(expected_pairs)
        if prediction.model_calls != expected_calls:
            raise ValueError("bidirectional dominance model-call count changed")
        if not prediction.response_model_id:
            raise ValueError("bidirectional dominance model attestation is empty")
        expected_selected = _condorcet_winner(candidate_ids, prediction.comparisons)
        if prediction.selected_event_id != expected_selected:
            raise ValueError("bidirectional dominance selected event is not the Condorcet winner")
        consistent_count = sum(item.orientation_consistent for item in prediction.comparisons)
        parse_valid_count = sum(
            orientation.parse_valid
            for comparison in prediction.comparisons
            for orientation in comparison.orientations
        )
        pair_payload = json.dumps(
            [item.to_dict() for item in prediction.comparisons],
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        selected = prediction.selected_event_id
        explanation = (
            "Unique orientation-consistent Condorcet winner across all candidates."
            if selected is not None
            else "No unique orientation-consistent Condorcet winner; abstained."
        )
        return TraceMethodOutputV1(
            findings=(
                TraceFindingV1(
                    kind=TraceFindingKind.STEP_ATTRIBUTION,
                    event_ids=((selected,) if selected is not None else ()),
                    label=(
                        "expected_effect_debt_bidirectional_root_cause"
                        if selected is not None
                        else "expected_effect_debt_bidirectional_abstain"
                    ),
                    score=(1.0 if selected is not None else 0.0),
                    explanation=explanation,
                    attributes=(
                        ("candidate_artifact_sha256", candidate_set.artifact_sha256),
                        ("candidate_count", str(len(candidate_ids))),
                        ("pair_count", str(len(expected_pairs))),
                        ("model_calls", str(prediction.model_calls)),
                        ("parse_valid_orientation_count", str(parse_valid_count)),
                        ("orientation_consistent_pair_count", str(consistent_count)),
                        ("unique_condorcet_winner", str(selected is not None).lower()),
                        ("response_model_id", prediction.response_model_id),
                        ("selection_rule", "unique_orientation_consistent_condorcet_winner"),
                        ("pairwise_comparisons", pair_payload),
                    ),
                ),
            ),
            evidence_level=EvidenceLevel.L3_LOCALIZED,
        )


__all__ = [
    "EXPECTED_EFFECT_DEBT_BIDIRECTIONAL_DOMINANCE_METHOD_ID",
    "BidirectionalDominanceComparisonV1",
    "BidirectionalDominanceOrientationV1",
    "BidirectionalDominancePredictionV1",
    "ExpectedEffectDebtBidirectionalDominanceRunner",
    "ExpectedEffectDebtBidirectionalDominanceSubprocessRunner",
    "ExpectedEffectDebtBidirectionalDominanceTraceAdapter",
    "build_expected_effect_debt_bidirectional_dominance_prompt",
    "expected_effect_debt_bidirectional_dominance_catalog_descriptor",
    "expected_effect_debt_bidirectional_dominance_source_revision",
    "native_expected_effect_debt_bidirectional_dominance_adapter",
    "parse_bidirectional_orientation_response",
]
