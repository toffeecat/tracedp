"""LongH static localization from an actor-visible causal ledger."""

from __future__ import annotations

import json
import posixpath
import re
import subprocess
import tempfile
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Protocol
from urllib.parse import quote

from lh_platform.tracing.integrations._provenance import isolated_worker_environment
from lh_platform.tracing.integrations._runtime import (
    NativeTraceRuntimeV1,
    require_python_modules,
    require_python_worker_entrypoint,
)
from lh_platform.tracing.types import EvidenceLevel, TraceCaseV1, TraceEventV1
from lh_platform.tracing.unified import (
    TraceFindingKind,
    TraceFindingV1,
    TraceMethodDescriptorV1,
    TraceMethodOutputV1,
    TraceMethodRequestV1,
)

LONGH_RESIDUAL_CAUSAL_METHOD_ID = "longh.residual_causal_static"
LONGH_PUBLIC_EFFECT_PROVENANCE_METHOD_ID = "longh.public_effect_provenance_static_v2"

_PATH = re.compile(
    r"(?<![A-Za-z0-9_])(?:/(?:[A-Za-z0-9_@%+=:,.-]+/?)+|"
    r"\.\.?/(?:[A-Za-z0-9_@%+=:,.-]+/?)+|"
    r"[A-Za-z0-9_.-]+\.(?:c|cc|cpp|csv|go|h|html|java|js|json|md|py|rb|rs|sh|sql|toml|txt|xml|yaml|yml))"
)
_FAILURE_SIGNAL = re.compile(
    r"\b(error|failed|failure|exception|traceback|timeout|timed out|not found|"
    r"no such file|invalid|mismatch|rejected)\b",
    flags=re.IGNORECASE,
)
_WRITE_SIGNAL = re.compile(
    r"(?:\b(?:cp|mv|rm|touch|mkdir|tee|chmod|chown|ln)\b|"
    r"\bsed\s+[^\n]*-[A-Za-z]*i[A-Za-z]*\b|"
    r"\b(?:write_text|write_bytes)\s*\(|"
    r"\bopen\s*\([^\n]*,[^\n]*['\"][wax+]|"
    r"(?:^|\s)(?:>|>>)\s*\S+|"
    r"\bgit\s+(?:apply|checkout|cherry-pick|clean|commit|config|merge|rebase|reset)\b|"
    r"\b(?:apt|apt-get|pip|pip3|npm|yarn|cargo)\s+install\b)",
    flags=re.IGNORECASE,
)
_READ_SIGNAL = re.compile(
    r"(?:\b(?:cat|file|find|grep|head|less|ls|rg|stat|tail)\b|"
    r"\bgit\s+(?:diff|log|show|status)\b|\bread_(?:text|bytes)\s*\()",
    flags=re.IGNORECASE,
)
_EXECUTE_SIGNAL = re.compile(
    r"(?:\b(?:bash|cmake|gcc|make|pytest|python|python3|ruby|cargo|go|node)\b|"
    r"\bnpm\s+(?:run|test)\b)",
    flags=re.IGNORECASE,
)
_REDIRECT_TARGET = re.compile(r"(?:^|\s)(?:>|>>)\s*([^\s;&|]+)")
_PATH_WRITE = re.compile(r"Path\(\s*['\"]([^'\"]+)['\"]\s*\)\.(?:write_text|write_bytes)\s*\(")
_OPEN_WRITE = re.compile(
    r"\bopen\s*\(\s*['\"]([^'\"]+)['\"]\s*,\s*['\"][^'\"]*[wax+][^'\"]*['\"]",
    flags=re.IGNORECASE,
)
_CD_WORKING_DIRECTORY = re.compile(
    r"(?:^|[;&|]\s*)cd\s+['\"]?(/[^'\"\s;&|]+)['\"]?\s*&&",
    flags=re.IGNORECASE,
)
_PATH_ARGUMENT_KEYS = frozenset(
    {
        "destination",
        "destination_path",
        "file",
        "file_path",
        "path",
        "source",
        "source_path",
        "target",
        "target_path",
    }
)
_WRITE_TOOL_NAMES = frozenset(
    {"apply_patch", "create_file", "edit", "edit_file", "write", "write_file"}
)
_READ_TOOL_NAMES = frozenset({"read", "read_file"})


def _compact(text: str, *, limit: int) -> str:
    """Keep prompt evidence bounded without changing its event identity."""

    normalized = " ".join(text.split())
    return normalized if len(normalized) <= limit else normalized[:limit] + "..."


def _section(content: str, start: str, end: str | None) -> str:
    offset = content.find(start)
    if offset < 0:
        return ""
    offset += len(start)
    if end is None:
        return content[offset:].strip()
    boundary = content.find(end, offset)
    return content[offset : boundary if boundary >= 0 else None].strip()


def _section_until(content: str, start: str, ends: tuple[str, ...]) -> str:
    """Read a section across both legacy and canonical Terminal event layouts."""

    offset = content.find(start)
    if offset < 0:
        return ""
    offset += len(start)
    boundaries = [boundary for end in ends if (boundary := content.find(end, offset)) >= 0]
    return content[offset : min(boundaries) if boundaries else None].strip()


def _line_value(content: str, label: str) -> str:
    for line in content.splitlines():
        if line.startswith(label):
            return line[len(label) :].strip()
    return ""


def _event_arguments(event: TraceEventV1) -> dict[str, Any]:
    raw = _section_until(
        event.content,
        "Arguments:",
        (
            "\nEnvironment step executed:",
            "\nTool result status:",
            "\nTool error:",
            "\nResult:",
            "\nPublic writes:",
        ),
    )
    if not raw:
        return {}
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        return {}
    return dict(parsed) if isinstance(parsed, dict) else {}


def _command_text(event: TraceEventV1, arguments: dict[str, Any]) -> str:
    for key in ("command", "cmd", "script"):
        value = arguments.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def _clean_resource(value: str) -> str:
    return value.strip("'\"`()[]{}<>,:;").rstrip(".")


def _resources(text: str) -> tuple[str, ...]:
    return tuple(
        dict.fromkeys(
            resource
            for match in _PATH.finditer(text)
            if (resource := _clean_resource(match.group(0))) not in {"/", "./", "../"}
        )
    )


def _command_working_directory(command: str) -> str | None:
    matches = _CD_WORKING_DIRECTORY.findall(command)
    return _clean_resource(matches[-1]) if matches else None


def _normalize_resource(resource: str, *, working_directory: str | None) -> str:
    cleaned = _clean_resource(resource)
    if (
        not working_directory
        or cleaned.startswith("/")
        or cleaned.startswith(("environment:", "workspace:"))
    ):
        return cleaned
    return posixpath.normpath(posixpath.join(working_directory, cleaned))


def _command_resources(command: str) -> tuple[str, ...]:
    working_directory = _command_working_directory(command)
    return tuple(
        dict.fromkeys(
            _normalize_resource(resource, working_directory=working_directory)
            for resource in _resources(command)
        )
    )


def _argument_resources(arguments: dict[str, Any]) -> tuple[str, ...]:
    values = (
        value
        for key, value in arguments.items()
        if key.casefold() in _PATH_ARGUMENT_KEYS and isinstance(value, str)
    )
    return tuple(dict.fromkeys(resource for value in values for resource in _resources(value)))


def _public_write_resources(event: TraceEventV1) -> tuple[str, ...]:
    section = _section(event.content, "Public writes:", None)
    if not section:
        return ()
    raw = section.splitlines()[0].strip()
    if not raw or raw.casefold() == "none":
        return ()
    return tuple(
        dict.fromkeys(
            resource for value in raw.split(",") for resource in _resources(value.strip())
        )
    )


def _write_resources(
    command: str,
    resources: tuple[str, ...],
    *,
    tool_name: str | None,
    argument_resources: tuple[str, ...],
    public_write_resources: tuple[str, ...],
) -> tuple[str, ...]:
    """Identify explicit mutation targets; unknown command effects stay unclaimed."""

    targets = [_clean_resource(match.group(1)) for match in _REDIRECT_TARGET.finditer(command)]
    targets.extend(_clean_resource(match.group(1)) for match in _PATH_WRITE.finditer(command))
    targets.extend(_clean_resource(match.group(1)) for match in _OPEN_WRITE.finditer(command))
    normalized_tool_name = (tool_name or "").casefold()
    if normalized_tool_name in _WRITE_TOOL_NAMES:
        targets.extend(argument_resources)
    targets.extend(public_write_resources)
    lowered = command.casefold()
    if re.search(r"\b(?:cp|mv|tee|touch|mkdir|rm|chmod|chown|ln)\b", lowered):
        if resources:
            targets.append(resources[-1])
    if re.search(r"\bsed\s+[^\n]*-[a-z]*i[a-z]*\b", lowered):
        targets.extend(resources)
    if re.search(
        r"\bgit\s+(?:apply|checkout|cherry-pick|clean|commit|config|merge|rebase|reset)\b", lowered
    ):
        targets.append("workspace:git")
    if re.search(r"\b(?:apt|apt-get|pip|pip3|npm|yarn|cargo)\s+install\b", lowered):
        targets.append("environment:packages")
    working_directory = _command_working_directory(command)
    return tuple(
        dict.fromkeys(
            _normalize_resource(target, working_directory=working_directory)
            for target in targets
            if target
        )
    )


@dataclass(frozen=True)
class PublicCausalLedgerEntryV1:
    """One public event annotated with mechanical effect and dependency roles."""

    event_id: str
    sequence: int
    actor_summary: str
    tool_name: str | None
    command_summary: str
    result_summary: str
    result_status: str
    operation_roles: tuple[str, ...]
    resources: tuple[str, ...]
    write_resources: tuple[str, ...]
    upstream_writer_event_ids: tuple[str, ...]


@dataclass(frozen=True)
class PublicCausalLedgerV1:
    """A gold-blind ledger compiled only from one public trace request."""

    entries: tuple[PublicCausalLedgerEntryV1, ...]
    private_inputs_used: bool = False

    @property
    def rankable_event_ids(self) -> tuple[str, ...]:
        return tuple(entry.event_id for entry in self.entries)


@dataclass(frozen=True)
class LongHResidualCausalRankingV1:
    """A local ranked-event value object that keeps tracing independent of TTA."""

    ranked_event_ids: tuple[str, ...]
    explanation: str


@dataclass(frozen=True)
class LongHResidualCausalPredictionV1:
    """One ranking with the runtime facts required for fair admission."""

    ranking: LongHResidualCausalRankingV1
    model_calls: int
    response_model_id: str
    reasoning_tokens: int
    reasoning_tokens_reported: bool
    raw_response_reused: bool
    prompt_character_count: int
    ledger_event_count: int
    input_tokens: int | None = None
    output_tokens: int | None = None
    cache_read_tokens: int | None = None
    cache_write_tokens: int | None = None
    latency_seconds: float | None = None


class LongHResidualCausalRunner(Protocol):
    def run(
        self,
        *,
        request: TraceMethodRequestV1,
    ) -> LongHResidualCausalPredictionV1: ...


def _entry_without_dependencies(event: TraceEventV1) -> PublicCausalLedgerEntryV1:
    metadata = dict(event.metadata)
    arguments = _event_arguments(event)
    command = _command_text(event, arguments)
    actor = _section(event.content, "Actor message:", "\nTool:")
    result = _section_until(
        event.content,
        "\nResult:",
        ("\nPublic state delta:",),
    )
    result_status = metadata.get("tool_result_status") or _line_value(
        event.content, "Tool result status:"
    )
    if not result_status:
        tool_error = _line_value(event.content, "Tool error:").casefold()
        if tool_error in {"true", "false"}:
            result_status = "error" if tool_error == "true" else "returned"
    result_status = result_status.strip().casefold() or "not_applicable"
    raw_tool_name = metadata.get("tool_name") or _line_value(event.content, "Tool:")
    tool_name: str | None = raw_tool_name.strip() or None

    argument_resources = _argument_resources(arguments)
    public_writes = _public_write_resources(event)
    resources = tuple(
        dict.fromkeys((*_command_resources(command), *argument_resources, *public_writes))
    )
    writes = _write_resources(
        command,
        resources,
        tool_name=tool_name,
        argument_resources=argument_resources,
        public_write_resources=public_writes,
    )
    roles: list[str] = []
    normalized_tool_name = (tool_name or "").casefold()
    if writes or normalized_tool_name in _WRITE_TOOL_NAMES or _WRITE_SIGNAL.search(command):
        roles.append("write")
    if normalized_tool_name in _READ_TOOL_NAMES or _READ_SIGNAL.search(command):
        roles.append("read")
    if _EXECUTE_SIGNAL.search(command):
        roles.append("execute")
    if result_status in {"error", "missing"} or _FAILURE_SIGNAL.search(result):
        roles.append("failure_signal")
    if not roles:
        roles.append("decision" if tool_name is None else "other_tool")

    return PublicCausalLedgerEntryV1(
        event_id=event.event_id,
        sequence=event.sequence,
        actor_summary=_compact(actor or event.content, limit=260),
        tool_name=tool_name,
        command_summary=_compact(command, limit=700),
        result_summary=_compact(result, limit=420),
        result_status=result_status,
        operation_roles=tuple(dict.fromkeys(roles)),
        resources=resources,
        write_resources=writes,
        upstream_writer_event_ids=(),
    )


def compile_public_causal_ledger(case: TraceCaseV1) -> PublicCausalLedgerV1:
    """Compile last-writer relations from public commands before any gold is loaded."""

    last_writer_by_resource: dict[str, str] = {}
    entries: list[PublicCausalLedgerEntryV1] = []
    for event in case.events:
        base = _entry_without_dependencies(event)
        read_resources = tuple(
            resource for resource in base.resources if resource not in base.write_resources
        )
        upstream = tuple(
            dict.fromkeys(
                writer
                for resource in read_resources
                if (writer := last_writer_by_resource.get(resource)) is not None
            )
        )
        entries.append(
            PublicCausalLedgerEntryV1(
                event_id=base.event_id,
                sequence=base.sequence,
                actor_summary=base.actor_summary,
                tool_name=base.tool_name,
                command_summary=base.command_summary,
                result_summary=base.result_summary,
                result_status=base.result_status,
                operation_roles=base.operation_roles,
                resources=base.resources,
                write_resources=base.write_resources,
                upstream_writer_event_ids=upstream,
            )
        )
        for resource in base.write_resources:
            last_writer_by_resource[resource] = base.event_id
    return PublicCausalLedgerV1(entries=tuple(entries))


def build_longh_residual_causal_prompt(
    case: TraceCaseV1,
    *,
    ranking_count: int = 3,
) -> str:
    """Render a compact causal view while keeping every public event selectable."""

    if ranking_count < 1 or ranking_count > len(case.events):
        raise ValueError("ranking count must fit the visible events")
    if case.context is None:
        raise ValueError("LongH residual localization requires public task context")
    ledger = compile_public_causal_ledger(case)
    lines: list[str] = []
    for entry in ledger.entries:
        lines.append(
            "\n".join(
                (
                    f'<event id="{entry.event_id}" sequence="{entry.sequence}">',
                    f"roles={','.join(entry.operation_roles)}; status={entry.result_status}; "
                    f"tool={entry.tool_name or 'none'}",
                    f"command={entry.command_summary or '<none>'}",
                    f"resources={','.join(entry.resources) or '<none>'}",
                    f"writes={','.join(entry.write_resources) or '<none>'}",
                    f"upstream_writers={','.join(entry.upstream_writer_event_ids) or '<none>'}",
                    f"actor={entry.actor_summary or '<none>'}",
                    f"result={entry.result_summary or '<empty>'}",
                    "</event>",
                )
            )
        )
    return (
        "You are the LongH residual-causal tracer for a failed long-horizon agent run.\n"
        "The ledger below is a mechanical re-expression of the same public actor and "
        "tool trace. It contains no answer key. Every listed event remains selectable.\n"
        "Find the earliest action or decision that created an unresolved condition "
        "needed for the task. Distinguish a state-changing writer from later commands "
        "that only observed, tested, or reported the failure. Treat command errors as "
        "evidence, not automatically as causes. Consider missing required actions and "
        "wrong ordering when no explicit harmful write explains the outcome.\n"
        f"Rank exactly {ranking_count} event IDs from most to least likely.\n\n"
        f"Task:\n{case.context.task_specification}\n\n"
        f"Observed failure:\n{case.context.evaluator_failure}\n\n"
        "Public residual-causal ledger:\n"
        + "\n\n".join(lines)
        + "\n\nReturn only one JSON object with exactly two fields: ranked_event_ids "
        f"must contain exactly {ranking_count} unique IDs copied from the event id "
        "attributes, and explanation must be one brief string. Do not use markdown."
    )


def longh_residual_causal_catalog_descriptor() -> TraceMethodDescriptorV1:
    """Describe the static LongH method without claiming active DP or replay."""

    return TraceMethodDescriptorV1(
        method_id=LONGH_RESIDUAL_CAUSAL_METHOD_ID,
        display_name="LongH Public Residual-Causal Static Tracer",
        family="failure_attribution",
        capabilities=(
            "step_attribution",
            "ranked_candidates",
            "public_causal_ledger",
        ),
        input_modalities=("text_trajectory",),
        output_kinds=(TraceFindingKind.STEP_ATTRIBUTION,),
        integration_level="catalog_only",
        execution_path="public_causal_ledger_then_one_call_ranker",
        evidence_ceiling=EvidenceLevel.L3_LOCALIZED,
        source_relpath=("lh_platform/tracing/integrations/longh_residual_causal.py"),
        source_revision=None,
        unavailable_reason="LongH residual-causal runtime has not passed admission",
        corpus_method_id="longh",
    )


class LongHResidualCausalTraceAdapter:
    """Run one audited ranker over a gold-blind public causal ledger."""

    def __init__(
        self, *, runner: LongHResidualCausalRunner, public_effect_provenance: bool = False
    ) -> None:
        self._runner = runner
        self._public_effect_provenance = public_effect_provenance
        self.descriptor = replace(
            longh_residual_causal_catalog_descriptor(),
            method_id=(
                LONGH_PUBLIC_EFFECT_PROVENANCE_METHOD_ID
                if public_effect_provenance
                else LONGH_RESIDUAL_CAUSAL_METHOD_ID
            ),
            display_name=(
                "LongH Public Effect Provenance Static Tracer v2"
                if public_effect_provenance
                else "LongH Public Residual-Causal Static Tracer"
            ),
            execution_path=(
                "structured_public_effect_provenance_then_one_call_ranker"
                if public_effect_provenance
                else "public_causal_ledger_then_one_call_ranker"
            ),
            capabilities=(
                ("step_attribution", "ranked_candidates", "public_effect_provenance")
                if public_effect_provenance
                else ("step_attribution", "ranked_candidates", "public_causal_ledger")
            ),
            integration_level="platform_native",
            source_revision="v2" if public_effect_provenance else "v1",
            unavailable_reason=None,
        )

    def execute(self, request: TraceMethodRequestV1) -> TraceMethodOutputV1:
        if not isinstance(request.case, TraceCaseV1):
            raise ValueError("LongH residual localization requires TraceCaseV1")
        prediction = self._runner.run(request=request)
        expected_count = min(3, len(request.case.events))
        if (
            prediction.model_calls != 1
            or not prediction.response_model_id
            or prediction.reasoning_tokens != 0
            or type(prediction.reasoning_tokens_reported) is not bool
            or len(prediction.ranking.ranked_event_ids) != expected_count
            or prediction.prompt_character_count < 1
            or prediction.ledger_event_count != len(request.case.events)
            or (prediction.input_tokens is not None and prediction.input_tokens < 0)
            or (prediction.output_tokens is not None and prediction.output_tokens < 0)
            or (prediction.cache_read_tokens is not None and prediction.cache_read_tokens < 0)
            or (prediction.cache_write_tokens is not None and prediction.cache_write_tokens < 0)
            or (prediction.latency_seconds is not None and prediction.latency_seconds < 0)
        ):
            raise ValueError("LongH residual-causal model audit failed")
        cost_attributes = tuple(
            (key, value)
            for key, raw in (
                ("input_tokens", prediction.input_tokens),
                ("output_tokens", prediction.output_tokens),
                ("cache_read_tokens", prediction.cache_read_tokens),
                ("cache_write_tokens", prediction.cache_write_tokens),
                ("latency_seconds", prediction.latency_seconds),
            )
            if raw is not None
            for value in (f"{raw:.6f}" if key == "latency_seconds" else str(raw),)
        )
        return TraceMethodOutputV1(
            findings=(
                TraceFindingV1(
                    kind=TraceFindingKind.STEP_ATTRIBUTION,
                    event_ids=prediction.ranking.ranked_event_ids,
                    label="longh_residual_causal_root_candidates",
                    score=None,
                    explanation=prediction.ranking.explanation,
                    attributes=(
                        (
                            "representation",
                            "public_effect_provenance_v2"
                            if self._public_effect_provenance
                            else "public_residual_causal_ledger_v1",
                        ),
                        ("model_calls", str(prediction.model_calls)),
                        ("response_model_id", prediction.response_model_id),
                        ("reasoning_tokens", str(prediction.reasoning_tokens)),
                        (
                            "reasoning_tokens_reported",
                            str(prediction.reasoning_tokens_reported).lower(),
                        ),
                        (
                            "raw_response_reused",
                            str(prediction.raw_response_reused).lower(),
                        ),
                        (
                            "prompt_character_count",
                            str(prediction.prompt_character_count),
                        ),
                        ("ledger_event_count", str(prediction.ledger_event_count)),
                        *cost_attributes,
                        ("private_inputs_used", "false"),
                    ),
                ),
            ),
            evidence_level=EvidenceLevel.L3_LOCALIZED,
        )


class LongHResidualCausalSubprocessRunner:
    """Isolate one model call and durably reuse only an identical public case."""

    def __init__(
        self, *, runtime: NativeTraceRuntimeV1, public_effect_provenance: bool = False
    ) -> None:
        if runtime.diagnostic_artifact_root is None:
            raise ValueError("LongH residual-causal tracer requires a diagnostic artifact root")
        self._runtime = runtime
        self._public_effect_provenance = public_effect_provenance
        self._artifact_root = Path(runtime.diagnostic_artifact_root).resolve()
        self._artifact_root.mkdir(parents=True, exist_ok=True)
        environment = isolated_worker_environment(
            "LH_LONGH_RESIDUAL_CAUSAL_API_KEY", runtime.api_key
        )
        require_python_modules(
            runtime,
            method_name="LongH public residual-causal tracer",
            modules=("openai",),
        )
        require_python_worker_entrypoint(
            runtime,
            method_name="LongH public residual-causal tracer",
            worker_path=Path(__file__).with_name("longh_residual_causal_worker.py"),
            environment=environment,
            arguments=("--import-check",),
        )

    def run(
        self,
        *,
        request: TraceMethodRequestV1,
    ) -> LongHResidualCausalPredictionV1:
        case = request.case
        if not isinstance(case, TraceCaseV1):
            raise ValueError("LongH residual-causal runner requires TraceCaseV1")
        case_stem = "case-" + quote(case.case_id, safe="")
        artifact_path = (
            self._artifact_root
            / (
                "longh_public_effect_provenance_v2"
                if self._public_effect_provenance
                else "longh_residual_causal"
            )
            / case_stem
            / "raw_response.json"
        )
        artifact_path.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix=".worker-", dir=self._artifact_root) as temp:
            request_path = Path(temp) / "request.json"
            request_path.write_text(
                json.dumps(case.to_dict(), ensure_ascii=False, sort_keys=True),
                encoding="utf-8",
            )
            completed = subprocess.run(
                [
                    self._runtime.python_executable,
                    str(Path(__file__).with_name("longh_residual_causal_worker.py")),
                    "--request",
                    str(request_path),
                    "--raw-output",
                    str(artifact_path),
                    *(["--public-effect-provenance"] if self._public_effect_provenance else []),
                    *self._runtime.worker_arguments(),
                ],
                check=False,
                capture_output=True,
                text=True,
                timeout=self._runtime.timeout_seconds,
                env=isolated_worker_environment(
                    "LH_LONGH_RESIDUAL_CAUSAL_API_KEY", self._runtime.api_key
                ),
                cwd=temp,
            )
        if completed.returncode != 0:
            error = completed.stderr.strip() or completed.stdout.strip()
            raise RuntimeError(f"LongH residual-causal worker failed: {error[-2000:]}")
        try:
            payload = json.loads(completed.stdout)
        except json.JSONDecodeError as error:
            raise RuntimeError("LongH residual-causal worker returned invalid JSON") from error
        return LongHResidualCausalPredictionV1(
            ranking=LongHResidualCausalRankingV1(
                ranked_event_ids=tuple(payload["ranked_event_ids"]),
                explanation=str(payload["explanation"]),
            ),
            model_calls=int(payload["model_calls"]),
            response_model_id=str(payload["response_model_id"]),
            reasoning_tokens=int(payload["reasoning_tokens"]),
            reasoning_tokens_reported=bool(payload["reasoning_tokens_reported"]),
            raw_response_reused=bool(payload["raw_response_reused"]),
            prompt_character_count=int(payload["prompt_character_count"]),
            ledger_event_count=int(payload["ledger_event_count"]),
        )


def native_longh_residual_causal_adapter(
    *,
    runtime: NativeTraceRuntimeV1,
    public_effect_provenance: bool = False,
) -> LongHResidualCausalTraceAdapter:
    """Build a static runtime adapter; scientific admission is a separate gate."""

    if type(public_effect_provenance) is not bool:
        raise ValueError("public_effect_provenance must be boolean")

    return LongHResidualCausalTraceAdapter(
        runner=LongHResidualCausalSubprocessRunner(
            runtime=runtime, public_effect_provenance=public_effect_provenance
        ),
        public_effect_provenance=public_effect_provenance,
    )


__all__ = [
    "LONGH_RESIDUAL_CAUSAL_METHOD_ID",
    "LONGH_PUBLIC_EFFECT_PROVENANCE_METHOD_ID",
    "LongHResidualCausalPredictionV1",
    "LongHResidualCausalRankingV1",
    "LongHResidualCausalRunner",
    "LongHResidualCausalSubprocessRunner",
    "LongHResidualCausalTraceAdapter",
    "PublicCausalLedgerEntryV1",
    "PublicCausalLedgerV1",
    "build_longh_residual_causal_prompt",
    "compile_public_causal_ledger",
    "longh_residual_causal_catalog_descriptor",
    "native_longh_residual_causal_adapter",
]
