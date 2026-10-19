"""Audit the claim boundary of controlled-fault ALFWorld causal gold."""

from __future__ import annotations

import hashlib
import json
import subprocess
from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from lh_platform.core.atomic_io import atomic_write_json
from lh_platform.core.evidence_contract import canonical_sha256
from lh_platform.tracing.alfworld_controlled_fault_injection import (
    CONTROLLED_FAULT_INJECTION_CAUSAL_SCHEMA,
    SOURCE_SUFFIX_CONTROLLER_SCHEMA,
)

CAUSAL_SCOPE_AUDIT_SCHEMA = "lh_alfworld_controlled_fault_causal_scope_audit/v1"
CAUSAL_SCOPE_AUDIT_RUN_SCHEMA = "lh_alfworld_controlled_fault_causal_scope_audit_run/v1"
CAUSAL_EVALUATION_MANIFEST_SCHEMA = "lh_alfworld_controlled_fault_evaluation_dataset/v1"
FROZEN_VALID_UNSEEN_CAUSAL_MANIFEST_SHA256 = (
    "346c95c404eafc320b9f21d0fab75b4658dc562ccd42f9f522190b7e22d4ea7d"
)
FROZEN_VALID_UNSEEN_CAUSAL_CASE_COUNT = 57
FROZEN_VALID_UNSEEN_CAUSAL_TASK_COUNT = 10


class AlfWorldCausalScopeAuditError(ValueError):
    """Raised when causal-gold receipts cannot support a scope audit."""


def _mapping(label: str, value: object) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise AlfWorldCausalScopeAuditError(f"{label} must be an object")
    return value


def _string(label: str, value: object) -> str:
    if not isinstance(value, str) or not value:
        raise AlfWorldCausalScopeAuditError(f"{label} must be a non-empty string")
    return value


def _raw_sha256(label: str, value: object) -> str:
    raw = _string(label, value)
    if len(raw) != 64 or any(character not in "0123456789abcdef" for character in raw):
        raise AlfWorldCausalScopeAuditError(f"{label} must be a lowercase SHA-256")
    return raw


def _prefixed_sha256(label: str, value: object) -> str:
    prefixed = _string(label, value)
    if not prefixed.startswith("sha256:"):
        raise AlfWorldCausalScopeAuditError(f"{label} must use the sha256: prefix")
    _raw_sha256(label, prefixed.removeprefix("sha256:"))
    return prefixed


def _nonnegative_int(label: str, value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise AlfWorldCausalScopeAuditError(f"{label} must be a non-negative integer")
    return value


@dataclass(frozen=True)
class CausalScopeCaseBindingV1:
    candidate_id: str
    task_id: str
    gold_step_idx: int
    trace_case_sha256: str
    trace_observability_class: str
    post_injection_action_count: int
    causal_gold_path: str
    causal_gold_file_sha256: str

    @classmethod
    def from_mapping(cls, raw: object) -> CausalScopeCaseBindingV1:
        case = _mapping("causal evaluation case", raw)
        candidate_id = _string("candidate ID", case.get("candidate_id"))
        task_id = _string("task ID", case.get("task_id"))
        if not candidate_id.startswith(f"{task_id}:step-"):
            raise AlfWorldCausalScopeAuditError("candidate/task binding changed")
        return cls(
            candidate_id=candidate_id,
            task_id=task_id,
            gold_step_idx=_nonnegative_int("gold step index", case.get("gold_step_idx")),
            trace_case_sha256=_prefixed_sha256("trace case SHA-256", case.get("trace_case_sha256")),
            trace_observability_class=_string(
                "trace observability class", case.get("trace_observability_class")
            ),
            post_injection_action_count=_nonnegative_int(
                "post-injection action count",
                case.get("post_injection_action_count"),
            ),
            causal_gold_path=_string("causal gold path", case.get("causal_gold_path")),
            causal_gold_file_sha256=_raw_sha256(
                "causal gold file SHA-256", case.get("causal_gold_file_sha256")
            ),
        )


def _manifest_case_bindings(
    manifest: Mapping[str, Any],
) -> tuple[CausalScopeCaseBindingV1, ...]:
    if manifest.get("schema") != CAUSAL_EVALUATION_MANIFEST_SCHEMA:
        raise AlfWorldCausalScopeAuditError("causal evaluation manifest schema changed")
    raw_cases = manifest.get("cases")
    if not isinstance(raw_cases, list) or not raw_cases:
        raise AlfWorldCausalScopeAuditError("causal evaluation cases are missing")
    cases = tuple(CausalScopeCaseBindingV1.from_mapping(raw) for raw in raw_cases)
    candidate_ids = [case.candidate_id for case in cases]
    if len(candidate_ids) != len(set(candidate_ids)):
        raise AlfWorldCausalScopeAuditError("candidate IDs are duplicated")
    return cases


def _sorted_counts(values: list[str]) -> dict[str, int]:
    return dict(sorted(Counter(values).items()))


def build_alfworld_causal_scope_audit(
    *,
    causal_evaluation_manifest: Mapping[str, Any],
    causal_gold_by_candidate: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    """Summarize what frozen-source-suffix causal receipts can and cannot claim."""

    cases = _manifest_case_bindings(causal_evaluation_manifest)

    candidate_ids = [case.candidate_id for case in cases]
    task_ids: set[str] = set()
    injected_actions: list[str] = []
    terminal_reasons: list[str] = []
    claim_scopes: list[str] = []
    observability_classes: list[str] = []
    post_injection_counts: list[str] = []
    root_cause_verified_count = 0

    for case in cases:
        candidate_id = case.candidate_id
        task_ids.add(case.task_id)
        observability_class = case.trace_observability_class

        receipt = causal_gold_by_candidate.get(candidate_id)
        if receipt is None:
            raise AlfWorldCausalScopeAuditError(
                f"causal gold is missing for candidate {candidate_id}"
            )
        if (
            receipt.get("schema") != CONTROLLED_FAULT_INJECTION_CAUSAL_SCHEMA
            or receipt.get("causal_gold_admitted") is not True
        ):
            raise AlfWorldCausalScopeAuditError(
                f"candidate {candidate_id} is not admitted causal gold"
            )
        binding = _mapping("causal binding", receipt.get("binding"))
        baseline = _mapping("baseline arm", receipt.get("baseline"))
        failure = _mapping("injected-failure arm", receipt.get("injected_failure"))
        matched = _mapping("matched-control arm", receipt.get("matched_control"))
        controller = _string("suffix controller SHA-256", binding.get("suffix_controller_sha256"))
        if any(
            arm.get("suffix_controller_sha256") != controller
            for arm in (baseline, failure, matched)
        ):
            raise AlfWorldCausalScopeAuditError(
                f"candidate {candidate_id} changed suffix-controller identity"
            )
        if (
            baseline.get("success") is not True
            or failure.get("success") is not False
            or matched.get("success") is not True
        ):
            raise AlfWorldCausalScopeAuditError(
                f"candidate {candidate_id} changed causal arm outcomes"
            )

        injection_step_idx = _nonnegative_int(
            "injection step index", binding.get("injection_step_idx")
        )
        if (
            injection_step_idx != case.gold_step_idx
            or binding.get("source_trace_sha256") != case.trace_case_sha256
        ):
            raise AlfWorldCausalScopeAuditError(f"candidate {candidate_id} trace binding changed")
        raw_steps = failure.get("steps")
        if not isinstance(raw_steps, list):
            raise AlfWorldCausalScopeAuditError("injected-failure steps are missing")
        steps = [_mapping("injected-failure step", step) for step in raw_steps]
        step_indices = [
            _nonnegative_int("injected-failure step index", step.get("step_idx")) for step in steps
        ]
        if len(step_indices) != len(set(step_indices)):
            raise AlfWorldCausalScopeAuditError(
                f"candidate {candidate_id} duplicated failure steps"
            )
        injected_action = _string("injected action", binding.get("injected_action"))
        injected_matches = [
            step
            for step in steps
            if step.get("step_idx") == injection_step_idx and step.get("action") == injected_action
        ]
        if len(injected_matches) != 1:
            raise AlfWorldCausalScopeAuditError(
                f"candidate {candidate_id} changed injected-action binding"
            )

        injected_actions.append(injected_action)
        terminal_reasons.append(_string("failure terminal reason", failure.get("terminal_reason")))
        claim_scopes.append(_string("claim scope", receipt.get("claim_scope")))
        post_injection_count = sum(step_idx > injection_step_idx for step_idx in step_indices)
        if post_injection_count != case.post_injection_action_count:
            raise AlfWorldCausalScopeAuditError(
                f"candidate {candidate_id} post-injection depth changed"
            )
        expected_observability_class = (
            "terminal_immediate"
            if post_injection_count == 0
            else "short_suffix_1_2"
            if post_injection_count <= 2
            else "long_suffix_ge_3"
        )
        if observability_class != expected_observability_class:
            raise AlfWorldCausalScopeAuditError(
                f"candidate {candidate_id} observability label changed"
            )
        observability_classes.append(observability_class)
        post_injection_counts.append(str(post_injection_count))
        root_cause_verified_count += int(receipt.get("root_cause_verified") is True)

    if set(causal_gold_by_candidate) != set(candidate_ids):
        raise AlfWorldCausalScopeAuditError(
            "causal-gold population differs from the evaluation manifest"
        )

    audit: dict[str, Any] = {
        "schema": CAUSAL_SCOPE_AUDIT_SCHEMA,
        "case_count": len(candidate_ids),
        "task_count": len(task_ids),
        "continuation_authority": "frozen_source_open_loop_suffix",
        "continuation_authority_schema": SOURCE_SUFFIX_CONTROLLER_SCHEMA,
        "claim_scope_counts": _sorted_counts(claim_scopes),
        "injected_action_counts": _sorted_counts(injected_actions),
        "terminal_reason_counts": _sorted_counts(terminal_reasons),
        "trace_observability_class_counts": _sorted_counts(observability_classes),
        "post_injection_executed_action_counts": _sorted_counts(post_injection_counts),
        "root_cause_verified_count": root_cause_verified_count,
        "on_policy_actor_evidence_case_count": 0,
        "on_policy_actor_claim_eligible": False,
        "claim_boundary": (
            "controlled single-action sufficient failure causes under the frozen "
            "source-suffix controller; not on-policy actor failures or unique natural "
            "root causes"
        ),
    }
    audit["content_sha256"] = "sha256:" + canonical_sha256(audit)
    return audit


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_object(path: Path, *, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise AlfWorldCausalScopeAuditError(f"cannot read {label}: {error}") from error
    if not isinstance(value, dict):
        raise AlfWorldCausalScopeAuditError(f"{label} must be a JSON object")
    return value


def _git_revision(repo_root: Path) -> str:
    revision = subprocess.run(
        ["git", "-C", str(repo_root), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    dirty = subprocess.run(
        ["git", "-C", str(repo_root), "status", "--porcelain"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if dirty:
        raise AlfWorldCausalScopeAuditError("scope-audit worktree is dirty")
    return revision


def run_frozen_valid_unseen_causal_scope_audit(
    *,
    causal_evaluation_manifest_path: Path,
    output_path: Path,
    repo_root: Path,
    expected_platform_revision: str,
) -> dict[str, Any]:
    """Verify and audit exactly the frozen 57-case valid-unseen population."""

    resolved_repo_root = repo_root.resolve()
    git_head = _git_revision(resolved_repo_root)
    if git_head != expected_platform_revision:
        raise AlfWorldCausalScopeAuditError("scope-audit platform revision changed")

    manifest_path = causal_evaluation_manifest_path.resolve()
    manifest_file_sha256 = _file_sha256(manifest_path)
    if manifest_file_sha256 != FROZEN_VALID_UNSEEN_CAUSAL_MANIFEST_SHA256:
        raise AlfWorldCausalScopeAuditError(
            "causal evaluation manifest is not the frozen valid-unseen population"
        )
    manifest = _load_object(manifest_path, label="causal evaluation manifest")
    cases = _manifest_case_bindings(manifest)
    if (
        len(cases) != FROZEN_VALID_UNSEEN_CAUSAL_CASE_COUNT
        or len({case.task_id for case in cases}) != FROZEN_VALID_UNSEEN_CAUSAL_TASK_COUNT
    ):
        raise AlfWorldCausalScopeAuditError(
            "frozen valid-unseen causal population coverage changed"
        )

    causal_gold_by_candidate: dict[str, Mapping[str, Any]] = {}
    verified_file_sha256: dict[str, str] = {}
    for case in cases:
        path = Path(case.causal_gold_path).resolve()
        actual_sha256 = _file_sha256(path)
        if actual_sha256 != case.causal_gold_file_sha256:
            raise AlfWorldCausalScopeAuditError(
                f"causal gold file SHA-256 changed for {case.candidate_id}"
            )
        causal_gold_by_candidate[case.candidate_id] = _load_object(
            path, label=f"causal gold {case.candidate_id}"
        )
        verified_file_sha256[case.candidate_id] = actual_sha256

    audit = build_alfworld_causal_scope_audit(
        causal_evaluation_manifest=manifest,
        causal_gold_by_candidate=causal_gold_by_candidate,
    )
    receipt: dict[str, Any] = {
        "schema": CAUSAL_SCOPE_AUDIT_RUN_SCHEMA,
        "status": "completed",
        "scientific_result": False,
        "git_head": git_head,
        "gpu_accessed": False,
        "model_accessed": False,
        "causal_evaluation_manifest_path": str(manifest_path),
        "causal_evaluation_manifest_file_sha256": manifest_file_sha256,
        "verified_causal_gold_file_count": len(verified_file_sha256),
        "verified_causal_gold_file_sha256": dict(sorted(verified_file_sha256.items())),
        "scope_audit": audit,
    }
    receipt["content_sha256"] = "sha256:" + canonical_sha256(receipt)
    atomic_write_json(output_path.resolve(), receipt, overwrite=False)
    return receipt


__all__ = [
    "CAUSAL_SCOPE_AUDIT_SCHEMA",
    "CAUSAL_SCOPE_AUDIT_RUN_SCHEMA",
    "FROZEN_VALID_UNSEEN_CAUSAL_MANIFEST_SHA256",
    "AlfWorldCausalScopeAuditError",
    "build_alfworld_causal_scope_audit",
    "run_frozen_valid_unseen_causal_scope_audit",
]
