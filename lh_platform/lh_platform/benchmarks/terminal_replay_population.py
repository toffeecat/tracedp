"""Outcome-blind public-path compilation for Terminal source replays."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path, PurePosixPath
from typing import Any

from lh_platform.benchmarks.terminal_public_state import validate_actor_visible_path

# Pi refuses these before any tool code runs, so the gateway (and its observer) never
# sees the call: the tool result itself is the receipt that nothing was dispatched.
_TOOL_LAYER_REFUSAL = re.compile(r"^(?:Tool \S+ not found|Validation failed for tool \"[^\"]+\":)")
_ABSOLUTE_APP_PATH = re.compile(r"(?<![A-Za-z0-9_.-])(/app(?:/[^\s`\"'<>|;,)}\]]+)+)")
_RELATIVE_FILE = re.compile(
    r"(?<![A-Za-z0-9_./-])"
    r"((?:[A-Za-z0-9_.+-]+/)*[A-Za-z0-9_][A-Za-z0-9_.+-]*\."
    r"(?:R|c|cc|cpp|h|hpp|py|rs|v|vo|js|ts|html|xml|txt|json|jsonl|csv|tsv|"
    r"toml|ya?ml|sql|sparql|ttl|ics|ini|cfg|conf|sh|pl|rb|java|go|pov|ppm|"
    r"gcode|red|dat))"
    r"(?![A-Za-z0-9_.-])",
    re.IGNORECASE,
)
_TRAILING_PUNCTUATION = ".!?:"
_UNBOUNDED_OR_BINARY_SUFFIXES = {
    ".7z",
    ".bin",
    ".bundle",
    ".gz",
    ".img",
    ".iso",
    ".jpeg",
    ".jpg",
    ".mp4",
    ".npy",
    ".npz",
    ".png",
    ".pt",
    ".pth",
    ".qcow2",
}
_DATABASE_SUFFIXES = {".db", ".db3", ".duckdb", ".sqlite", ".sqlite3"}
_DATABASE_SIDECAR_ENDINGS = ("-journal", "-shm", "-wal")


@dataclass(frozen=True)
class TerminalPublicPathCandidate:
    """One actor-visible path backed only by public task text."""

    path: str
    provenance: tuple[str, ...]
    replay_observable: bool
    exclusion_reason: str | None


@dataclass(frozen=True)
class TerminalPublicPathCompilation:
    """Freeze the paths available to a later zero-model replay."""

    candidates: tuple[TerminalPublicPathCandidate, ...]
    replay_paths: tuple[str, ...]
    truncated: bool


class ExecutionDisposition(StrEnum):
    """State what is actually known about one source call's dispatch and effect.

    A single `None` cannot carry this: "never dispatched" and "dispatched with an
    unobserved effect" are different facts with different replay obligations.  The
    repository already defines the third case in its own terms
    (`terminal_managed_runtime_result.py`: `error_stage == "tool_execution"` requires
    `environment_step_executed is None` and a partial-or-unknown-effect phase).
    """

    #: Reached the task environment and its result was observed.
    COMPLETED = "completed"
    #: Refused before environment access, with a public rejection receipt.
    NOT_EXECUTED = "not_executed"
    #: Dispatched, but the effect was never observed (the run was cut off).
    CENSORED_UNKNOWN = "censored_unknown"


@dataclass(frozen=True)
class FrozenTerminalReplayCall:
    """One source call plus what is known about its dispatch and its effect."""

    tool_call_id: str
    tool_name: str
    arguments: dict[str, Any]
    execution_disposition: ExecutionDisposition
    source_is_error: bool

    @property
    def dispatch_status(self) -> str:
        """`dispatched` unless the call was refused before environment access."""

        return (
            "not_dispatched"
            if self.execution_disposition is ExecutionDisposition.NOT_EXECUTED
            else "dispatched"
        )

    @property
    def tool_result_observed(self) -> bool:
        """False only for the censored case, where no result was ever recorded."""

        return self.execution_disposition is not ExecutionDisposition.CENSORED_UNKNOWN

    @property
    def environment_step_executed(self) -> bool | None:
        """Kept for existing consumers, now derived from the explicit disposition."""

        if self.execution_disposition is ExecutionDisposition.COMPLETED:
            return True
        if self.execution_disposition is ExecutionDisposition.NOT_EXECUTED:
            return False
        return None

    def replay_payload(self) -> dict[str, object]:
        return {
            "tool_name": self.tool_name,
            "arguments": self.arguments,
            "environment_step_executed": self.environment_step_executed,
        }


def canonical_replay_prefix(
    calls: tuple[FrozenTerminalReplayCall, ...],
) -> tuple[tuple[FrozenTerminalReplayCall, ...], int]:
    """Return the calls a canonical replay may execute, plus the censored tail count.

    A canonical replay reproduces only what the source actually observed.  The call
    whose effect was never observed must be neither executed nor silently skipped as
    if the trajectory continued, so the replay stops before it and the caller records
    the censoring instead.  Two indistinguishable choices would each assert an effect
    nobody observed, which is why the decision is a property of the disposition rather
    than of the replay code.
    """

    boundary = len(calls)
    for index, call in enumerate(calls):
        if call.execution_disposition is ExecutionDisposition.CENSORED_UNKNOWN:
            boundary = index
            break
    return calls[:boundary], len(calls) - boundary


def _paths_in_text(text: str) -> set[str]:
    paths: set[str] = set()
    for match in _ABSOLUTE_APP_PATH.finditer(text):
        if match.end() < len(text) and text[match.end()] == "<":
            continue
        raw = match.group(1).rstrip(_TRAILING_PUNCTUATION)
        try:
            normalized = validate_actor_visible_path(raw)
        except ValueError:
            continue
        if normalized != "/app":
            paths.add(normalized)
    for match in _RELATIVE_FILE.finditer(text):
        raw = match.group(1).rstrip(_TRAILING_PUNCTUATION)
        if raw.startswith("./"):
            raw = raw[2:]
        try:
            normalized = validate_actor_visible_path(str(PurePosixPath("/app") / raw))
        except ValueError:
            continue
        paths.add(normalized)
    return paths


def is_terminal_database_path(path: str) -> bool:
    """Recognize database files and standard SQLite sidecar artifacts."""

    pure_path = PurePosixPath(path)
    return (
        pure_path.suffix.lower() in _DATABASE_SUFFIXES
        or pure_path.name.lower().endswith(_DATABASE_SIDECAR_ENDINGS)
    )


def _replay_exclusion(path: str) -> str | None:
    suffix = PurePosixPath(path).suffix.lower()
    if is_terminal_database_path(path) or suffix in _UNBOUNDED_OR_BINARY_SUFFIXES:
        return "unbounded_or_binary_content"
    if any(part.startswith(".") for part in PurePosixPath(path).parts[2:]):
        return "hidden_workspace_path"
    return None


def compile_terminal_task_public_paths(
    *, instruction: str, dockerfile: str, maximum_replay_paths: int = 24
) -> TerminalPublicPathCompilation:
    """Compile public paths without reading a solution, verifier, run outcome, or trace.

    The compiler deliberately limits replay observation to bounded text/code-like
    paths. Large binary inputs remain recorded as public candidates, but they are
    not copied after every terminal action.
    """

    if maximum_replay_paths < 1:
        raise ValueError("maximum replay paths must be positive")
    by_source = {
        "instruction": _paths_in_text(instruction),
        "environment_dockerfile": _paths_in_text(dockerfile),
    }
    all_paths = sorted(set().union(*by_source.values()))
    candidates = tuple(
        TerminalPublicPathCandidate(
            path=path,
            provenance=tuple(source for source, paths in by_source.items() if path in paths),
            replay_observable=_replay_exclusion(path) is None,
            exclusion_reason=_replay_exclusion(path),
        )
        for path in all_paths
    )
    observable = tuple(candidate.path for candidate in candidates if candidate.replay_observable)
    return TerminalPublicPathCompilation(
        candidates=candidates,
        replay_paths=observable[:maximum_replay_paths],
        truncated=len(observable) > maximum_replay_paths,
    )


def _jsonl_objects(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as error:
            raise ValueError(f"invalid JSON at {path.name}:{line_number}") from error
        if not isinstance(row, dict):
            raise ValueError(f"expected JSON object at {path.name}:{line_number}")
        rows.append(row)
    return rows


def load_frozen_terminal_replay_calls(
    *, session_path: Path, observer_path: Path
) -> tuple[FrozenTerminalReplayCall, ...]:
    """Join actor calls to source execution receipts without using future state.

    The source census did not snapshot files, but its host gateway still recorded
    which calls executed and which were rejected before environment access. This
    join preserves that execution schedule for a fresh zero-model replay.
    """

    calls: list[tuple[str, str, dict[str, Any]]] = []
    result_errors: dict[str, bool] = {}
    result_texts: dict[str, str] = {}
    for row in _jsonl_objects(session_path):
        message = row.get("message")
        if row.get("type") != "message" or not isinstance(message, dict):
            continue
        if message.get("role") == "assistant":
            for block in message.get("content") or []:
                if not isinstance(block, dict) or block.get("type") != "toolCall":
                    continue
                arguments = block.get("arguments")
                if not isinstance(arguments, dict):
                    raise ValueError("Terminal source tool arguments must be an object")
                calls.append(
                    (str(block.get("id", "")), str(block.get("name", "")), arguments)
                )
        elif message.get("role") == "toolResult":
            call_id = str(message.get("toolCallId", ""))
            is_error = message.get("isError")
            if not call_id or type(is_error) is not bool or call_id in result_errors:
                raise ValueError("Terminal source tool result identity is invalid")
            result_errors[call_id] = is_error
            result_texts[call_id] = "".join(
                str(part.get("text", "")) for part in message.get("content") or [] if isinstance(part, dict)
            )
    call_ids = [call_id for call_id, _, _ in calls]
    if any(not call_id for call_id in call_ids) or len(call_ids) != len(set(call_ids)):
        raise ValueError("Terminal source tool-call identities are invalid")

    observer_rows = _jsonl_objects(observer_path)
    observer_by_id: dict[str, dict[str, Any]] = {}
    for row in observer_rows:
        call_id = str(row.get("tool_call_id", ""))
        if not call_id or call_id in observer_by_id:
            raise ValueError("Terminal source observer identities are invalid")
        observer_by_id[call_id] = row
    if not set(observer_by_id) <= set(call_ids):
        raise ValueError("Terminal source observer coverage differs")

    replay_calls: list[FrozenTerminalReplayCall] = []
    for call_id, tool_name, arguments in calls:
        observer = observer_by_id.get(call_id)
        if observer is None and result_errors.get(call_id) is True and _TOOL_LAYER_REFUSAL.match(
            result_texts.get(call_id, "")
        ):
            # Refused by the tool layer (unknown tool, schema validation) before the
            # gateway ran: the error result is the receipt that it was never dispatched.
            replay_calls.append(
                FrozenTerminalReplayCall(
                    tool_call_id=call_id,
                    tool_name=tool_name,
                    arguments=arguments,
                    execution_disposition=ExecutionDisposition.NOT_EXECUTED,
                    source_is_error=True,
                )
            )
            continue
        if observer is None:
            # No public receipt exists for this call, so neither its dispatch nor its
            # effect can be established.  That is an evidence gap, not an unknown
            # effect, and it must fail rather than be guessed.
            raise ValueError(f"Terminal source call lacks any observer receipt: {call_id}")
        if observer.get("tool_name") != tool_name or observer.get("input") != arguments:
            raise ValueError(f"Terminal source observer differs for {call_id}")
        executed = observer.get("environment_step_executed")
        stage = observer.get("error_stage")
        is_error = observer.get("is_error")
        if type(is_error) is not bool:
            raise ValueError(f"Terminal source observer error flag is invalid: {call_id}")
        if executed is True:
            disposition = ExecutionDisposition.COMPLETED
        elif executed is None:
            # Dispatched with an unobserved effect.  The repository's own validator
            # requires exactly this shape for a propagated tool exception.
            if stage != "tool_execution" or is_error is not True:
                raise ValueError(f"Terminal source censored call receipt is invalid: {call_id}")
            disposition = ExecutionDisposition.CENSORED_UNKNOWN
        elif executed is False:
            # A refusal must carry one of the reviewed justifications, otherwise the
            # false flag would assert something no receipt supports.
            justified = (
                stage == "parallel_rejection"
                and observer.get("overlapping_tool_calls") is True
            ) or stage in {"before_snapshot", "resource_before_snapshot"} or (
                stage is None and observer.get("observation_complete") is True
            )
            if not justified:
                raise ValueError(
                    f"non-executed source call lacks a public rejection receipt: {call_id}"
                )
            disposition = ExecutionDisposition.NOT_EXECUTED
        else:
            raise ValueError(f"Terminal source execution status is invalid: {call_id}")

        if disposition is ExecutionDisposition.CENSORED_UNKNOWN:
            # No result was ever recorded, so the error flag is known only from the
            # propagated-exception receipt.
            source_is_error = True
        elif call_id in result_errors:
            source_is_error = result_errors[call_id]
            if is_error != source_is_error:
                raise ValueError(f"Terminal source observer differs for {call_id}")
        else:
            # A propagated exception leaves no tool result.  Only a completed call may
            # claim one, and the observer must report the failure.
            if is_error is not True or disposition is ExecutionDisposition.COMPLETED:
                raise ValueError(f"Terminal source call lacks a tool result: {call_id}")
            source_is_error = True
        replay_calls.append(
            FrozenTerminalReplayCall(
                tool_call_id=call_id,
                tool_name=tool_name,
                arguments=arguments,
                execution_disposition=disposition,
                source_is_error=source_is_error,
            )
        )
    # A censored transition may occur anywhere: a tool can fail with an unobserved
    # effect and the actor can then continue.  The canonical replay prefix simply
    # stops before the first one, and the count is reported rather than asserted.
    return tuple(replay_calls)


def compare_terminal_replay_parity(
    source_calls: tuple[FrozenTerminalReplayCall, ...],
    replay_executed: tuple[bool, ...],
    replay_errors: tuple[bool, ...],
) -> dict[str, object]:
    """Compare a replay against the source on a three-valued footing.

    A field the source observed must match exactly, or the comparison fails.  A field
    the source never observed (`CENSORED_UNKNOWN`) is neither matched nor mismatched:
    the replay keeps its own observed value, and the field is reported as censored.

    Three outcomes are therefore possible, and the middle one must never be called a
    strict pass:

    * `fail` - an observed field mismatched;
    * `strict_pass` - every field matched and nothing was censored;
    * `observed_fields_pass_with_censoring` - every observed field matched while at
      least one field was censored.

    Only the last two may support a recovery experiment, and only when the terminal
    state is independently observed.  Neither authorizes a claim that the censored
    call caused the damage; that needs a separate execute-versus-skip probe.
    """

    if len(source_calls) != len(replay_executed) or len(source_calls) != len(
        replay_errors
    ):
        return {
            "prefix_parity_status": "fail",
            "reason": "replay_call_count_differs",
            "source_call_count": len(source_calls),
            "replay_call_count": len(replay_executed),
            "censored_fields": [],
            "censored_transition_count": 0,
        }
    censored_fields: list[str] = []
    for index, call in enumerate(source_calls):
        if call.execution_disposition is ExecutionDisposition.CENSORED_UNKNOWN:
            # Effect unobserved in the source: not comparable either way.
            censored_fields.append(f"environment_step_executed[{index}]")
            continue
        if replay_executed[index] != call.environment_step_executed:
            return {
                "prefix_parity_status": "fail",
                "reason": f"execution_mismatch_at_{index}",
                "censored_fields": censored_fields,
                "censored_transition_count": len(censored_fields),
            }
        if replay_errors[index] != call.source_is_error:
            return {
                "prefix_parity_status": "fail",
                "reason": f"error_mismatch_at_{index}",
                "censored_fields": censored_fields,
                "censored_transition_count": len(censored_fields),
            }
    return {
        "prefix_parity_status": (
            "observed_fields_pass_with_censoring" if censored_fields else "strict_pass"
        ),
        "reason": None,
        "censored_fields": censored_fields,
        "censored_transition_count": len(censored_fields),
        "strict_historical_replay_eligible": not censored_fields,
    }


__all__ = [
    "ExecutionDisposition",
    "TerminalPublicPathCandidate",
    "TerminalPublicPathCompilation",
    "FrozenTerminalReplayCall",
    "canonical_replay_prefix",
    "compare_terminal_replay_parity",
    "compile_terminal_task_public_paths",
    "is_terminal_database_path",
    "load_frozen_terminal_replay_calls",
]
