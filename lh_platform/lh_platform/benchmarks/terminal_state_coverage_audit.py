"""Build and validate independent state-coverage evidence for Terminal recovery."""

from __future__ import annotations

import json
import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from lh_platform.benchmarks.terminal_forward_recovery_contract import (
    load_terminal_forward_recovery_source,
)
from lh_platform.benchmarks.terminal_replay_population import (
    compile_terminal_task_public_paths,
    is_terminal_database_path,
)
from lh_platform.benchmarks.terminal_task_contract import (
    resolve_terminal_source_task_contract,
    verify_terminal_task_contract,
)

PACKET_SCHEMA = "terminal_forward_state_coverage_audit_packet_v1"
DECISION_SCHEMA = "terminal_forward_state_coverage_audit_decision_v1"
RECEIPT_SCHEMA = "terminal_forward_state_coverage_receipt_v2"
PRODUCER_ID = "longh_terminal_state_coverage_packet_builder_v1"
PACKET_FILE = "audit_packet.json"
DECISION_FILE = "audit_decision.json"
RECEIPT_FILE = "state_coverage_receipt.json"

STATE_DOMAINS = (
    "declared_public_files",
    "unlisted_files",
    "processes",
    "services",
    "network",
    "database",
    "environment_variables",
    "permissions",
    "clock",
)


def _read_object(path: Path) -> dict[str, Any]:
    """Read one exact JSON object used by the independent audit bundle."""

    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"audit artifact must contain an object: {path}")
    return value


def _write_object(path: Path, value: object) -> None:
    """Publish one audit artifact without exposing a partial JSON document."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _relative_under(path: Path, root: Path, *, label: str) -> str:
    """Keep every audit source inside the portable LongH storage root."""

    resolved = path.resolve()
    root = root.resolve()
    if root not in resolved.parents:
        raise ValueError(f"{label} must remain under LONGH_STORAGE_ROOT")
    return resolved.relative_to(root).as_posix()


def _from_storage_relative(value: object, root: Path, *, label: str) -> Path:
    """Resolve one packet path without permitting an audit to escape storage."""

    if not isinstance(value, str) or not value or Path(value).is_absolute():
        raise ValueError(f"Terminal state coverage {label} path is invalid")
    root = root.resolve()
    resolved = (root / value).resolve()
    if root not in resolved.parents:
        raise ValueError(f"Terminal state coverage {label} escaped storage")
    return resolved


def _source_binding(
    source_map: Path,
    *,
    storage_root: Path,
    source_trial: Path,
) -> tuple[str, tuple[str, ...]]:
    """Resolve task and paths from the frozen source map, never launch prose."""

    value = _read_object(source_map)
    rows = value.get("sources")
    if (
        value.get("schema_version") != "terminal_source_map_v2"
        or value.get("benchmark_id") != "terminal-bench-2.1"
        or not isinstance(rows, list)
    ):
        raise ValueError("Terminal state audit source map schema differs")
    matches = []
    for row in rows:
        if not isinstance(row, Mapping) or not isinstance(row.get("run_relative_path"), str):
            continue
        run_root = (storage_root / str(row["run_relative_path"])).resolve()
        if source_trial.resolve().parent == run_root:
            matches.append(row)
    if len(matches) != 1:
        raise ValueError("Terminal state audit source trial is not uniquely mapped")
    row = matches[0]
    public_paths = row.get("public_paths")
    if (
        row.get("case_status") != "terminal"
        or not isinstance(row.get("task"), str)
        or not row["task"]
        or not isinstance(public_paths, list)
        or not public_paths
        or any(not isinstance(path, str) or not path for path in public_paths)
        or len(set(public_paths)) != len(public_paths)
    ):
        raise ValueError("Terminal state audit source binding is invalid")
    return str(row["task"]), tuple(public_paths)


def _fact(
    evidence_id: str,
    domain: str,
    source: str,
    finding: str,
    *,
    risk: bool,
) -> dict[str, object]:
    """Create one source-backed fact that an independent decision must address."""

    return {
        "evidence_id": evidence_id,
        "domain": domain,
        "source": source,
        "finding": finding,
        "risk": risk,
    }


def build_terminal_state_coverage_audit_packet(
    *,
    storage_root: Path,
    terminal_bench_root: Path,
    source_trial: Path,
    source_map: Path,
) -> dict[str, object]:
    """Compile objective source and task facts without deciding state sufficiency."""

    storage_root = storage_root.resolve()
    task, public_paths = _source_binding(
        source_map,
        storage_root=storage_root,
        source_trial=source_trial,
    )
    task_contract = resolve_terminal_source_task_contract(
        source_map,
        storage_root=storage_root,
        source_trial=source_trial,
        expected_task=task,
        expected_public_paths=public_paths,
    )
    task_dir = terminal_bench_root.resolve() / "tasks" / task
    contract = verify_terminal_task_contract(
        task_dir,
        task_contract,
        expected_task=task,
    )
    source = load_terminal_forward_recovery_source(
        source_trial,
        public_paths=public_paths,
        expected_task=task,
    )
    snapshot = task_contract / "snapshot"
    compilation = compile_terminal_task_public_paths(
        instruction=(snapshot / "instruction.md").read_text(encoding="utf-8"),
        dockerfile=(snapshot / "environment" / "Dockerfile").read_text(
            encoding="utf-8"
        ),
        maximum_replay_paths=24,
    )
    source_trial_relative = _relative_under(
        source_trial, storage_root, label="source trial"
    )
    source_map_relative = _relative_under(
        source_map, storage_root, label="source map"
    )
    contract_relative = _relative_under(
        task_contract, storage_root, label="task contract"
    )
    history_source = f"{source_trial_relative}/agent/pi/sessions + terminal-public-state.jsonl"
    contract_source = f"{contract_relative}/snapshot"
    facts: list[dict[str, object]] = [
        _fact(
            "public-path-compilation",
            "declared_public_files",
            contract_source,
            (
                f"Outcome-blind compilation produced {list(compilation.replay_paths)}; "
                f"the frozen source map declared {list(public_paths)}; "
                f"truncated={compilation.truncated}."
            ),
            risk=(compilation.truncated or compilation.replay_paths != public_paths),
        ),
        _fact(
            "public-transition-chain",
            "declared_public_files",
            history_source,
            (
                f"The natural failure has {len(source.public_history)} sequential public "
                f"tool transitions and an exact final observation for all declared paths."
            ),
            risk=False,
        ),
        _fact(
            "frozen-task-verifier-contract",
            "all",
            contract_source,
            (
                "Independent review must use the frozen instruction, task configuration, "
                "initial environment, and private verifier files listed in contract.json; "
                f"the contract contains {len(contract['files'])} files."
            ),
            risk=False,
        ),
    ]
    executed = [row for row in source.public_history if row["environment_step_executed"]]
    bash_rows = [row for row in executed if row["tool_name"] == "bash"]
    direct_unlisted_writes = [
        row
        for row in executed
        if row["tool_name"] == "write"
        and str(row["tool_input"].get("path", "")) not in public_paths
    ]
    direct_unlisted_reads = [
        row
        for row in executed
        if row["tool_name"] == "read"
        and str(row["tool_input"].get("path", "")) not in public_paths
    ]
    facts.append(
        _fact(
            "native-write-scope",
            "unlisted_files",
            history_source,
            (
                f"Executed native writes outside declared paths: "
                f"{[row['tool_input'].get('path') for row in direct_unlisted_writes]}."
            ),
            risk=bool(direct_unlisted_writes),
        )
    )
    facts.append(
        _fact(
            "native-read-scope",
            "unlisted_files",
            history_source,
            (
                "Executed native reads outside declared paths: "
                f"{[row['tool_input'].get('path') for row in direct_unlisted_reads]}."
            ),
            risk=bool(direct_unlisted_reads),
        )
    )
    for index, row in enumerate(
        [
            item
            for item in (*direct_unlisted_reads, *direct_unlisted_writes)
            if is_terminal_database_path(str(item["tool_input"].get("path", "")))
        ],
        start=1,
    ):
        facts.append(
            _fact(
                f"native-database-path-{index:03d}",
                "database",
                history_source,
                (
                    "Executed a native tool against an unobserved database-like path: "
                    f"{row['tool_input'].get('path')}."
                ),
                risk=True,
            )
        )
    replay_paths = set(compilation.replay_paths)
    for index, candidate in enumerate(compilation.candidates, start=1):
        if is_terminal_database_path(candidate.path):
            facts.append(
                _fact(
                    f"public-database-path-{index:03d}",
                    "database",
                    contract_source,
                    (
                        f"Public database-like path {candidate.path} requires semantic "
                        f"review; selected_for_replay={candidate.path in replay_paths}; "
                        f"reason={candidate.exclusion_reason or 'none'}."
                    ),
                    risk=True,
                )
            )
        if candidate.path in replay_paths:
            continue
        facts.append(
            _fact(
                f"unobserved-public-path-{index:03d}",
                "unlisted_files",
                contract_source,
                (
                    f"Public task text names {candidate.path}, but it is absent from the "
                    "restored projection; "
                    f"reason={candidate.exclusion_reason or 'path_limit'}; "
                    f"provenance={list(candidate.provenance)}."
                ),
                risk=True,
            )
        )
    for index, row in enumerate(bash_rows, start=1):
        command = str(row["tool_input"].get("command", ""))
        compact = " ".join(command.split())
        for domain in STATE_DOMAINS[1:]:
            facts.append(
                _fact(
                    f"bash-{index:03d}-{domain}",
                    domain,
                    history_source,
                    (
                        "Executed bash is not confined by the public-file observer and "
                        f"must be reviewed for this domain: {compact}"
                    ),
                    risk=True,
                )
            )
    if not bash_rows:
        for domain in STATE_DOMAINS[1:]:
            facts.append(
                _fact(
                    f"no-bash-{domain}",
                    domain,
                    history_source,
                    (
                        "No executed arbitrary bash action exists; the source used only "
                        "the bounded native read/write surface."
                    ),
                    risk=False,
                )
            )
    return {
        "schema_version": PACKET_SCHEMA,
        "packet_status": "pending_independent_review",
        "producer_id": PRODUCER_ID,
        "benchmark_id": "terminal-bench-2.1",
        "task_name": task,
        "source_trial_name": source_trial.name,
        "source_trial_relative_path": source_trial_relative,
        "source_map_relative_path": source_map_relative,
        "task_contract_receipt_relative_path": contract_relative,
        "public_paths": list(public_paths),
        "path_compilation_truncated": compilation.truncated,
        "public_path_compilation_matches_source": compilation.replay_paths
        == public_paths,
        "public_path_candidates": [
            {
                "path": candidate.path,
                "provenance": list(candidate.provenance),
                "replay_observable": candidate.replay_observable,
                "exclusion_reason": candidate.exclusion_reason,
                "selected_for_replay": candidate.path in replay_paths,
            }
            for candidate in compilation.candidates
        ],
        "state_domains": list(STATE_DOMAINS),
        "facts": facts,
    }


def prepare_terminal_state_coverage_audit(
    *,
    output: Path,
    storage_root: Path,
    terminal_bench_root: Path,
    source_trial: Path,
    source_map: Path,
) -> dict[str, object]:
    """Create one immutable pending-review directory for a source failure."""

    if output.exists():
        raise FileExistsError("Terminal state coverage audit output already exists")
    packet = build_terminal_state_coverage_audit_packet(
        storage_root=storage_root,
        terminal_bench_root=terminal_bench_root,
        source_trial=source_trial,
        source_map=source_map,
    )
    output.mkdir(parents=True)
    _write_object(output / PACKET_FILE, packet)
    return packet


def _validate_packet(value: object) -> dict[str, Any]:
    """Validate the complete machine-produced packet before independent review."""

    if not isinstance(value, Mapping):
        raise ValueError("Terminal state coverage packet must be an object")
    packet = dict(value)
    facts = packet.get("facts")
    if (
        packet.get("schema_version") != PACKET_SCHEMA
        or packet.get("packet_status") != "pending_independent_review"
        or packet.get("producer_id") != PRODUCER_ID
        or packet.get("benchmark_id") != "terminal-bench-2.1"
        or not isinstance(packet.get("task_name"), str)
        or not packet["task_name"]
        or not isinstance(packet.get("source_trial_name"), str)
        or not packet["source_trial_name"]
        or type(packet.get("path_compilation_truncated")) is not bool
        or packet.get("public_path_compilation_matches_source") is not True
        or packet.get("state_domains") != list(STATE_DOMAINS)
        or not isinstance(packet.get("public_paths"), list)
        or not packet["public_paths"]
        or not isinstance(packet.get("public_path_candidates"), list)
        or not packet["public_path_candidates"]
        or not isinstance(facts, list)
        or not facts
    ):
        raise ValueError("Terminal state coverage packet is not admissible")
    for candidate in packet["public_path_candidates"]:
        if (
            not isinstance(candidate, Mapping)
            or set(candidate)
            != {
                "path",
                "provenance",
                "replay_observable",
                "exclusion_reason",
                "selected_for_replay",
            }
            or not isinstance(candidate["path"], str)
            or not candidate["path"]
            or not isinstance(candidate["provenance"], list)
            or not candidate["provenance"]
            or any(
                not isinstance(item, str) or not item
                for item in candidate["provenance"]
            )
            or type(candidate["replay_observable"]) is not bool
            or (
                candidate["exclusion_reason"] is not None
                and not isinstance(candidate["exclusion_reason"], str)
            )
            or type(candidate["selected_for_replay"]) is not bool
        ):
            raise ValueError("Terminal state coverage path candidate is invalid")
    evidence_ids: set[str] = set()
    for fact in facts:
        if (
            not isinstance(fact, Mapping)
            or set(fact) != {"evidence_id", "domain", "source", "finding", "risk"}
            or not isinstance(fact["evidence_id"], str)
            or not fact["evidence_id"]
            or fact["evidence_id"] in evidence_ids
            or fact["domain"] not in {*STATE_DOMAINS, "all"}
            or not isinstance(fact["source"], str)
            or not fact["source"]
            or not isinstance(fact["finding"], str)
            or not fact["finding"]
            or type(fact["risk"]) is not bool
        ):
            raise ValueError("Terminal state coverage packet fact is invalid")
        evidence_ids.add(str(fact["evidence_id"]))
    return packet


def _validate_decision(value: object, *, packet: Mapping[str, Any]) -> dict[str, Any]:
    """Require a separate reviewer to address every domain and risk fact."""

    if not isinstance(value, Mapping):
        raise ValueError("Terminal state coverage decision must be an object")
    decision = dict(value)
    domain_decisions = decision.get("domain_decisions")
    if (
        set(decision)
        != {
            "schema_version",
            "review_status",
            "audit_id",
            "auditor_id",
            "task_name",
            "source_trial_name",
            "source_trial_relative_path",
            "source_map_relative_path",
            "task_contract_receipt_relative_path",
            "domain_decisions",
        }
        or decision.get("schema_version") != DECISION_SCHEMA
        or decision.get("review_status") != "approved"
        or not isinstance(decision.get("audit_id"), str)
        or not decision["audit_id"]
        or not isinstance(decision.get("auditor_id"), str)
        or not decision["auditor_id"]
        or decision["auditor_id"] == packet["producer_id"]
        or decision.get("task_name") != packet["task_name"]
        or decision.get("source_trial_name") != packet["source_trial_name"]
        or decision.get("source_trial_relative_path")
        != packet["source_trial_relative_path"]
        or decision.get("source_map_relative_path") != packet["source_map_relative_path"]
        or decision.get("task_contract_receipt_relative_path")
        != packet["task_contract_receipt_relative_path"]
        or not isinstance(domain_decisions, Mapping)
        or set(domain_decisions) != set(STATE_DOMAINS)
    ):
        raise ValueError("Terminal state coverage decision is invalid")
    facts = {str(row["evidence_id"]): row for row in packet["facts"]}
    for domain in STATE_DOMAINS:
        row = domain_decisions[domain]
        expected_verdict = (
            "restored" if domain == "declared_public_files" else "not_outcome_determining"
        )
        if (
            not isinstance(row, Mapping)
            or set(row) != {"verdict", "evidence_ids", "rationale"}
            or row["verdict"] != expected_verdict
            or not isinstance(row["evidence_ids"], list)
            or not row["evidence_ids"]
            or len(set(row["evidence_ids"])) != len(row["evidence_ids"])
            or not isinstance(row["rationale"], str)
            or not row["rationale"].strip()
        ):
            raise ValueError(f"Terminal state coverage decision is incomplete for {domain}")
        selected = set(row["evidence_ids"])
        if any(
            evidence_id not in facts
            or facts[evidence_id]["domain"] not in {domain, "all"}
            for evidence_id in selected
        ):
            raise ValueError(f"Terminal state coverage evidence differs for {domain}")
        if not any(facts[evidence_id]["domain"] == domain for evidence_id in selected):
            raise ValueError(f"Terminal state coverage lacks domain evidence for {domain}")
        required_risks = {
            evidence_id
            for evidence_id, fact in facts.items()
            if fact["domain"] == domain and fact["risk"] is True
        }
        if not required_risks <= selected:
            raise ValueError(f"Terminal state coverage skipped a risk fact for {domain}")
    return decision


def render_terminal_state_coverage_receipt(
    *,
    packet: Mapping[str, Any],
    decision: Mapping[str, Any],
) -> dict[str, object]:
    """Render the only coverage receipt accepted by the live Terminal executor."""

    packet = _validate_packet(packet)
    decision = _validate_decision(decision, packet=packet)
    evidence = []
    for domain in STATE_DOMAINS:
        row = decision["domain_decisions"][domain]
        for evidence_id in row["evidence_ids"]:
            evidence.append(
                {
                    "evidence_id": f"{domain}:{evidence_id}",
                    "source": f"{PACKET_FILE}:{evidence_id}",
                    "finding": f"Independent review accepted evidence for {domain}.",
                }
            )
    return {
        "schema_version": RECEIPT_SCHEMA,
        "producer_id": PRODUCER_ID,
        "review_status": "approved",
        "audit_packet_file": PACKET_FILE,
        "audit_decision_file": DECISION_FILE,
        "audit_id": decision["audit_id"],
        "auditor_id": decision["auditor_id"],
        "benchmark_id": "terminal-bench-2.1",
        "task_name": packet["task_name"],
        "source_trial_name": packet["source_trial_name"],
        "source_trial_relative_path": packet["source_trial_relative_path"],
        "source_map_relative_path": packet["source_map_relative_path"],
        "task_contract_receipt_relative_path": packet[
            "task_contract_receipt_relative_path"
        ],
        "public_paths": list(packet["public_paths"]),
        "path_compilation_truncated": packet["path_compilation_truncated"],
        "reconstruction_scope": "declared_public_files",
        "unsupported_state_outcome_determining": False,
        "assessed_state_domains": {
            domain: (
                "restored"
                if domain == "declared_public_files"
                else "not_outcome_determining"
            )
            for domain in STATE_DOMAINS
        },
        "evidence": evidence,
    }


def finalize_terminal_state_coverage_audit(
    *,
    output: Path,
    decision_path: Path,
    storage_root: Path,
    terminal_bench_root: Path,
) -> dict[str, object]:
    """Combine a frozen packet and separate review into one durable bundle."""

    packet_path = output / PACKET_FILE
    persisted_decision_path = output / DECISION_FILE
    receipt_path = output / RECEIPT_FILE
    if receipt_path.exists():
        raise FileExistsError("Terminal state coverage audit is already finalized")
    packet = verify_terminal_state_coverage_packet_sources(
        _read_object(packet_path),
        storage_root=storage_root,
        terminal_bench_root=terminal_bench_root,
    )
    decision = _read_object(decision_path)
    receipt = render_terminal_state_coverage_receipt(
        packet=packet,
        decision=decision,
    )
    if persisted_decision_path.exists():
        # Resume only the exact review whose durable write already completed.
        if _read_object(persisted_decision_path) != decision:
            raise ValueError(
                "Independent decision differs from the persisted decision"
            )
    else:
        _write_object(persisted_decision_path, decision)
    _write_object(receipt_path, receipt)
    return receipt


def verify_terminal_state_coverage_bundle(
    receipt_path: Path,
    *,
    receipt: Mapping[str, Any],
    storage_root: Path,
    terminal_bench_root: Path,
) -> None:
    """Rebuild the receipt from its packet and decision before every live run."""

    if receipt.get("audit_packet_file") != PACKET_FILE or receipt.get(
        "audit_decision_file"
    ) != DECISION_FILE:
        raise ValueError("Terminal state coverage audit artifact names differ")
    packet = verify_terminal_state_coverage_packet_sources(
        _read_object(receipt_path.parent / PACKET_FILE),
        storage_root=storage_root,
        terminal_bench_root=terminal_bench_root,
    )
    decision = _read_object(receipt_path.parent / DECISION_FILE)
    expected = render_terminal_state_coverage_receipt(
        packet=packet,
        decision=decision,
    )
    if dict(receipt) != expected:
        raise ValueError("Terminal state coverage receipt differs from its evidence bundle")


def verify_terminal_state_coverage_packet_sources(
    packet: object,
    *,
    storage_root: Path,
    terminal_bench_root: Path,
) -> dict[str, Any]:
    """Recompile a packet from frozen source artifacts before trusting review."""

    validated = _validate_packet(packet)
    source_trial = _from_storage_relative(
        validated["source_trial_relative_path"],
        storage_root,
        label="source trial",
    )
    source_map = _from_storage_relative(
        validated["source_map_relative_path"],
        storage_root,
        label="source map",
    )
    expected = build_terminal_state_coverage_audit_packet(
        storage_root=storage_root,
        terminal_bench_root=terminal_bench_root,
        source_trial=source_trial,
        source_map=source_map,
    )
    if validated != expected:
        raise ValueError("Terminal state coverage packet differs from frozen sources")
    return validated


__all__ = [
    "DECISION_FILE",
    "DECISION_SCHEMA",
    "PACKET_FILE",
    "PACKET_SCHEMA",
    "PRODUCER_ID",
    "RECEIPT_FILE",
    "RECEIPT_SCHEMA",
    "STATE_DOMAINS",
    "build_terminal_state_coverage_audit_packet",
    "finalize_terminal_state_coverage_audit",
    "prepare_terminal_state_coverage_audit",
    "render_terminal_state_coverage_receipt",
    "verify_terminal_state_coverage_bundle",
    "verify_terminal_state_coverage_packet_sources",
]
