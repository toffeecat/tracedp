"""Controlled injection of silent faults into real successful Terminal trajectories.

Implements the construction rules for controlled fault injection that need no Docker or model: operator enumeration over a frozen prefix, seeded site
sampling, applying sites to the replay tool chain, and rewriting the Pi session and the
public-state observer log so the injected history matches what the environment actually
returned when the mutated chain was replayed.

Gold is by construction: every site names the source call it mutates or removes.
Whether a site is silent, breaks the official verifier, and is visible to a public-tier
check is decided later by real replays (qualification Q1-Q6), never here.
"""

from __future__ import annotations

import copy
import itertools
import json
import random
import re
import shlex
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass

SPEC_ID = "controlled_injection_spec_v1"

O1_VALUE = "O1_value_substitution"
O2_OMIT = "O2_omission"
O3_TARGET = "O3_target_offset"
O4_OPTION = "O4_option_removal"
O5_MODE = "O5_permission_change"
O6_TAIL = "O6_tail_omission"
# Destroy an input the task still needs, after the call that reads it. Forward-only
# continuation cannot recover it; rollback plus undo can.
O7_DESTROY = "O7_destroy_input"
# Value relay: the call's output carries a wrong value; the Actor reads it and passes it on.
O8B_RELAY = "O8b_value_relay"
OPERATORS = (O1_VALUE, O2_OMIT, O3_TARGET, O4_OPTION, O5_MODE, O6_TAIL, O7_DESTROY, O8B_RELAY)
OMISSION_OPERATORS = frozenset({O2_OMIT, O6_TAIL})

# A number ends at a word boundary; a sentence period may follow, a decimal part may not.
_NUMBER = re.compile(r"(?<![\w.])\d+(?!\w|\.\d)")
_BACKTICK = re.compile(r"`([^`\n]{2,80})`")
_QUOTED = re.compile(r"\"([^\"\n]{2,80})\"")
_PATH = re.compile(r"(/app/[A-Za-z0-9_.\-/]+)")
_SHORT_OPTION = re.compile(r"(?<=\s)-([A-Za-z])(?=\s)")
_CHMOD = re.compile(r"\bchmod\s+(?:-R\s+)?([0-7]{3,4})\b")
_REDIRECT = re.compile(r">>?\s*(/app/[A-Za-z0-9_.\-/]+)")
_TARGET_SUFFIX = ".orig"


@dataclass(frozen=True)
class InjectionSite:
    """One mutation of one source call; ``call_index`` is the prefix position."""

    call_index: int
    operator: str
    target: str
    replacement: str
    field: str

    def to_dict(self) -> dict[str, object]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> InjectionSite:
        return cls(**{key: payload[key] for key in ("call_index", "operator", "target", "replacement", "field")})


def requirement_literals(instruction: str) -> tuple[str, ...]:
    """Literals a public requirement names: numbers, backticked tokens, quoted strings, paths."""

    found: list[str] = []
    for pattern in (_BACKTICK, _QUOTED, _PATH):
        found.extend(match.group(1).strip() for match in pattern.finditer(instruction))
    found.extend(match.group(0) for match in _NUMBER.finditer(instruction))
    unique = []
    for item in found:
        if item and item not in unique:
            unique.append(item)
    return tuple(unique)


_MUTATING = re.compile(
    r"(?:^|[;&|(]\s*|\s)(?:git\s+(?:-C\s+\S+\s+)?(?:commit|merge|checkout|switch|reset|cherry-pick|"
    r"rebase|add|rm|mv|stash|branch|tag|filter-branch|filter-repo|reflog|gc|prune|apply|am|restore|"
    r"update-ref|replace)\b|cp\b|mv\b|rm\b|mkdir\b|touch\b|chmod\b|chown\b|ln\b|tee\b|"
    r"sed\s+-i|make\b|install\b|pip3?\s+install|apt(?:-get)?\s+install|tar\b|unzip\b|patch\b|"
    r"openssl\b|python3?\b|bash\s+\S+\.sh|\./\S+)"
)
_REDIRECT_ANY = re.compile(r"(?<![0-9&])>>?\s*[^\s&]")


def mutating_command(command: str) -> bool:
    """Heuristic: the shell command may change files (directories hide their content)."""

    return bool(_MUTATING.search(command) or _REDIRECT_ANY.search(command))


def effectful_indices(
    observer_rows: Sequence[Mapping[str, object]],
    prefix: Sequence[Mapping[str, object]] | None = None,
) -> tuple[int, ...]:
    """Positions whose observed public state changed, or whose call plainly mutates files.

    The observer records only existence and mode for directory resources, so a call that
    rewrites a repository under a public directory looks inert; the command heuristic only
    widens candidate enumeration, and replay qualification still decides every cell.
    """

    effectful = []
    for index, row in enumerate(observer_rows):
        if not row.get("environment_step_executed", True):
            continue
        call = prefix[index] if prefix is not None else {}
        arguments = call.get("arguments", {}) if isinstance(call, Mapping) else {}
        command = arguments.get("command") if isinstance(arguments, Mapping) else None
        if (
            row.get("before") != row.get("after")
            or (call.get("tool_name") in ("write", "edit"))
            or (isinstance(command, str) and mutating_command(command))
        ):
            effectful.append(index)
    return tuple(effectful)


def _text_fields(call: Mapping[str, object]) -> list[tuple[str, str]]:
    arguments = call["arguments"]
    if not isinstance(arguments, Mapping):
        return []
    tool = call["tool_name"]
    if tool == "bash" and isinstance(arguments.get("command"), str):
        return [("command", arguments["command"])]
    if tool == "write" and isinstance(arguments.get("content"), str):
        return [("content", arguments["content"])]
    if tool == "edit" and isinstance(arguments.get("newText"), str):
        return [("newText", arguments["newText"])]
    return []


def _perturb(literal: str) -> str:
    if literal.isdigit():
        return str(int(literal) + 1)
    return literal + "-mut"


def _single_call_messages(session_entries: Sequence[Mapping[str, object]]) -> set[str]:
    """Tool-call ids that are the only call of their assistant message (removable)."""

    single = set()
    for entry in session_entries:
        message = entry.get("message") if entry.get("type") == "message" else None
        if not isinstance(message, Mapping) or message.get("role") != "assistant":
            continue
        calls = [b for b in message.get("content", []) if isinstance(b, Mapping) and b.get("type") == "toolCall"]
        if len(calls) == 1:
            single.add(str(calls[0]["id"]))
    return single


def enumerate_sites(
    prefix: Sequence[Mapping[str, object]],
    *,
    instruction: str,
    observer_rows: Sequence[Mapping[str, object]],
    session_entries: Sequence[Mapping[str, object]],
) -> list[InjectionSite]:
    """Every operator application whose structural precondition holds."""

    if len(prefix) != len(observer_rows):
        raise ValueError("prefix and public-state observer rows differ in length")
    literals = requirement_literals(instruction)
    effectful = set(effectful_indices(observer_rows, prefix))
    removable = _single_call_messages(session_entries)
    sites: list[InjectionSite] = []
    for index, call in enumerate(prefix):
        if not call.get("environment_step_executed", True) or call.get("source_is_error"):
            continue
        for field, text in _text_fields(call):
            for literal in literals:
                pattern = (
                    re.compile(rf"(?<![\w.]){re.escape(literal)}(?!\w|\.\d)")
                    if literal.isdigit()
                    else re.compile(re.escape(literal))
                )
                if pattern.search(text) and not literal.startswith("/app/"):
                    sites.append(InjectionSite(index, O1_VALUE, literal, _perturb(literal), field))
            if field == "command" and index in effectful:
                for match in _SHORT_OPTION.finditer(text):
                    sites.append(InjectionSite(index, O4_OPTION, f"-{match.group(1)}", "", field))
                for match in _CHMOD.finditer(text):
                    mode = match.group(1)
                    sites.append(InjectionSite(index, O5_MODE, mode, "600" if mode == "644" else "644", field))
                for match in _REDIRECT.finditer(text):
                    path = match.group(1)
                    sites.append(InjectionSite(index, O3_TARGET, path, path + _TARGET_SUFFIX, field))
        if call["tool_name"] == "write" and index in effectful:
            path = str(call["arguments"].get("path", ""))
            if path.startswith("/app/"):
                sites.append(InjectionSite(index, O3_TARGET, path, path + _TARGET_SUFFIX, "path"))
        if index in effectful and str(call.get("source_tool_call_id")) in removable:
            sites.append(InjectionSite(index, O2_OMIT, str(call.get("source_tool_call_id")), "", ""))
    sites.extend(_destroy_input_sites(prefix, observer_rows, literals))
    tail = [i for i in sorted(effectful) if str(prefix[i].get("source_tool_call_id")) in removable]
    for start in tail[-3:]:
        sites.append(InjectionSite(start, O6_TAIL, str(prefix[start].get("source_tool_call_id")), "", ""))
    return _deduplicated(sites)


def _written_paths(prefix: Sequence[Mapping[str, object]]) -> set[str]:
    """Paths the task itself creates: write-tool targets and shell redirection targets."""

    written: set[str] = set()
    for call in prefix:
        if call.get("tool_name") == "write":
            written.add(str(call["arguments"].get("path", "")))
        written.update(match.group(1) for match in _REDIRECT.finditer(str(call["arguments"].get("command", ""))))
    return written


def _destroy_input_sites(
    prefix: Sequence[Mapping[str, object]],
    observer_rows: Sequence[Mapping[str, object]],
    literals: Sequence[str],
) -> list[InjectionSite]:
    """O7 sites: an input the task names, present before a call that has a later call to need it."""

    written = _written_paths(prefix)
    inputs = [lit for lit in literals if lit.startswith("/app/") and lit not in written]
    sites: list[InjectionSite] = []
    for index, call in enumerate(prefix[:-1]):
        if call.get("tool_name") != "bash" or not call.get("environment_step_executed", True):
            continue
        if call.get("source_is_error"):
            continue
        command = str(call["arguments"].get("command", ""))
        if "<<" in command or not command.strip():
            continue  # a heredoc would swallow the appended step
        present = {
            str(item.get("path")) for item in observer_rows[index].get("before", []) if item.get("exists")
        }
        for path in inputs:
            if path in present:
                sites.append(InjectionSite(index, O7_DESTROY, path, "", "command"))
    return sites


def _deduplicated(sites: list[InjectionSite]) -> list[InjectionSite]:
    seen, unique = set(), []
    for site in sites:
        key = (site.call_index, site.operator, site.target, site.field)
        if key not in seen:
            seen.add(key)
            unique.append(site)
    return unique


_MUTATIONS = frozenset({O1_VALUE, O3_TARGET, O4_OPTION, O5_MODE, O7_DESTROY})


def _compatible(combo: Sequence[InjectionSite], *, distinct_calls: bool) -> bool:
    """Whether a set of sites can be applied together as independent faults.

    Several mutations may share one call when they touch different literals; an omitted call
    (O2) takes no other fault; a tail omission (O6) removes every later call, so every other
    fault must precede it. ``distinct_calls=False`` skips the one-fault-per-call restriction
    only for mutations; omissions always keep their own call.
    """

    tails = [site.call_index for site in combo if site.operator == O6_TAIL]
    if len(tails) > 1 or any(
        site.call_index >= tails[0] for site in combo if tails and site.operator != O6_TAIL
    ):
        return False
    omitted = {site.call_index for site in combo if site.operator == O2_OMIT}
    by_call: dict[int, list[InjectionSite]] = {}
    for site in combo:
        by_call.setdefault(site.call_index, []).append(site)
    for call_index, group in by_call.items():
        if len(group) == 1:
            continue
        if call_index in omitted or not all(site.operator in _MUTATIONS for site in group):
            return False
        if not distinct_calls:
            continue
        # Two faults in one call must be distinct edits, never the same literal twice.
        keys = [(site.operator, site.field, site.target) for site in group]
        if len(set(keys)) != len(keys):
            return False
    return True


def sample_sites(
    sites: Sequence[InjectionSite], *, seed: int, count: int, distinct_calls: bool = True
) -> tuple[InjectionSite, ...]:
    """Seeded uniform choice among all compatible ``count``-site combinations."""

    pool = sorted(sites, key=lambda s: (s.call_index, s.operator, s.target, s.field))
    combos = [
        combo
        for combo in itertools.combinations(pool, count)
        if _compatible(combo, distinct_calls=distinct_calls)
    ]
    if not combos:
        raise ValueError("not enough applicable injection sites")
    return tuple(sorted(random.Random(seed).choice(combos), key=lambda s: s.call_index))


def omitted_indices(sites: Sequence[InjectionSite], call_count: int) -> set[int]:
    omitted: set[int] = set()
    for site in sites:
        if site.operator == O2_OMIT:
            omitted.add(site.call_index)
        elif site.operator == O6_TAIL:
            omitted.update(range(site.call_index, call_count))
    return omitted


def relay_command(command: str, token: str, replacement: str) -> str:
    """The command with ``token`` replaced in its standard output, exit status unchanged.

    The command runs as before; its stdout is passed through ``sed``, and ``pipefail`` keeps the
    command's own exit status (a failing command still fails). Requires bash, which the
    runtime uses.
    """

    if not token or not command.strip():
        raise ValueError("a relay needs a non-empty value and command")
    escaped_token = re.escape(token).replace("/", r"\/")
    escaped_replacement = replacement.replace("\\", "\\\\").replace("/", r"\/").replace("&", r"\&")
    return (
        f"set -o pipefail; {{ {command}\n}} | sed 's/{escaped_token}/{escaped_replacement}/g'"
    )


def relay_injection_sites(
    prefix: Sequence[Mapping[str, object]],
    observer_rows: Sequence[Mapping[str, object]],
    *,
    instruction: str,
    outputs: Sequence[str] | None = None,
    cwd: str = "/app",
) -> list[InjectionSite]:
    """O8b sites: a value first seen in a single-line command's output, consumed by a later write.

    ``outputs`` must be the full tool outputs from the session (one per call). The observer's
    ``tool_result.content`` is not the full output and misses relays.
    """

    from lh_platform.tracefix_bench.dependency_graph import call_accesses, output_relay_edges, relay_sites

    if outputs is None:
        outputs = [str((row.get("tool_result") or {}).get("content") or "") for row in observer_rows]
    if len(outputs) != len(prefix):
        raise ValueError("outputs and prefix differ in length")
    edges = output_relay_edges(prefix, outputs, instruction=instruction)
    accesses = call_accesses(prefix, observer_rows, cwd=cwd)
    sites: list[InjectionSite] = []
    for relay in relay_sites(accesses, edges):
        producer = int(relay["producer"])
        call = prefix[producer]
        command = str(call["arguments"].get("command", "")) if call.get("tool_name") == "bash" else ""
        if not command.strip():
            continue
        if not call.get("environment_step_executed", True) or call.get("source_is_error"):
            continue
        token = str(relay["token"])
        sites.append(InjectionSite(producer, O8B_RELAY, token, _perturb(token), "command"))
    return _deduplicated(sites)


def _replace_once(text: str, site: InjectionSite) -> str:
    if site.operator == O1_VALUE and site.target.isdigit():
        pattern = re.compile(rf"(?<![\w.]){re.escape(site.target)}(?!\w|\.\d)")
        mutated, count = pattern.subn(site.replacement, text, count=1)
    elif site.operator == O4_OPTION:
        mutated, count = re.subn(rf"(?<=\s){re.escape(site.target)}\s", "", text, count=1)
    elif site.operator == O5_MODE:
        mutated, count = re.subn(
            rf"(\bchmod\s+(?:-R\s+)?){re.escape(site.target)}\b", rf"\g<1>{site.replacement}", text, count=1
        )
    elif site.operator == O8B_RELAY:
        mutated = relay_command(text, site.target, site.replacement)
        count = 1
    elif site.operator == O7_DESTROY:
        # Appended after the call's own work: the call's output is unchanged, the input is gone after it.
        mutated = text.rstrip() + f" ; rm -f -- {shlex.quote(site.target)}"
        count = 1
    else:
        count = text.count(site.target)
        mutated = text.replace(site.target, site.replacement, 1)
    if not count:
        raise ValueError(f"injection target is absent from call {site.call_index}")
    return mutated


def apply_sites(
    prefix: Sequence[Mapping[str, object]], sites: Sequence[InjectionSite]
) -> list[dict[str, object]]:
    """Return the replay chain with mutated arguments and omitted calls removed."""

    chain = [copy.deepcopy(dict(call)) for call in prefix]
    for site in sites:
        if site.operator in OMISSION_OPERATORS:
            continue
        arguments = chain[site.call_index]["arguments"]
        arguments[site.field] = _replace_once(str(arguments[site.field]), site)
    dropped = omitted_indices(sites, len(chain))
    return [call for index, call in enumerate(chain) if index not in dropped]


def rewrite_session(
    session_entries: Sequence[Mapping[str, object]],
    *,
    prefix: Sequence[Mapping[str, object]],
    sites: Sequence[InjectionSite],
    replay_results: Sequence[Mapping[str, object]],
) -> list[dict[str, object]]:
    """Build the injected Pi session from the source session and the replayed chain.

    Mutated calls carry their new arguments; omitted calls lose their assistant message
    and tool result, and the next entry is relinked to the previous one; every remaining
    tool result is replaced by what the replay actually returned. Assistant text is kept
    verbatim (spec §12 decision 2).
    """

    chain = apply_sites(prefix, sites)
    if len(chain) != len(replay_results):
        raise ValueError("replay results do not cover the injected chain")
    results_by_id = {
        str(call["source_tool_call_id"]): result for call, result in zip(chain, replay_results, strict=True)
    }
    arguments_by_id = {str(call["source_tool_call_id"]): call["arguments"] for call in chain}
    dropped_ids = {
        str(prefix[index]["source_tool_call_id"]) for index in omitted_indices(sites, len(prefix))
    }
    rewritten: list[dict[str, object]] = []
    removed_entry_parent: dict[str, object] = {}
    for entry in session_entries:
        entry = copy.deepcopy(dict(entry))
        parent = entry.get("parentId")
        while parent in removed_entry_parent:
            parent = removed_entry_parent[parent]
        if "parentId" in entry:
            entry["parentId"] = parent
        message = entry.get("message") if entry.get("type") == "message" else None
        if isinstance(message, dict) and message.get("role") == "assistant":
            calls = [b for b in message.get("content", []) if isinstance(b, dict) and b.get("type") == "toolCall"]
            if calls and str(calls[0]["id"]) in dropped_ids:
                removed_entry_parent[entry["id"]] = entry.get("parentId")
                continue
            for block in calls:
                if str(block["id"]) in arguments_by_id:
                    block["arguments"] = copy.deepcopy(arguments_by_id[str(block["id"])])
        if isinstance(message, dict) and message.get("role") == "toolResult":
            call_id = str(message.get("toolCallId"))
            if call_id in dropped_ids:
                removed_entry_parent[entry["id"]] = entry.get("parentId")
                continue
            if call_id in results_by_id:
                result = results_by_id[call_id]
                message["isError"] = bool(result["is_error"])
                message["content"] = [{"type": "text", "text": str(result["content"])}]
        rewritten.append(entry)
    return rewritten


def rewrite_observer(
    replay_observer_rows: Sequence[Mapping[str, object]],
    *,
    injected_chain: Sequence[Mapping[str, object]],
) -> list[dict[str, object]]:
    """Rebind the replay's public-state rows to the source tool-call ids."""

    if len(replay_observer_rows) != len(injected_chain):
        raise ValueError("replay observer rows do not cover the injected chain")
    rows = []
    for row, call in zip(replay_observer_rows, injected_chain, strict=True):
        row = copy.deepcopy(dict(row))
        row["tool_call_id"] = str(call["source_tool_call_id"])
        rows.append(row)
    return rows


def replay_history(replay_session_rows: Sequence[Mapping[str, object]]) -> list[dict[str, object]]:
    """Read (tool_name, arguments, is_error, content) per call from a controlled replay."""

    calls: dict[str, dict[str, object]] = {}
    order: list[str] = []
    for row in replay_session_rows:
        message = row.get("message", {})
        if message.get("role") == "assistant":
            for block in message.get("content", []):
                if block.get("type") == "toolCall":
                    calls[block["id"]] = {"tool_name": block["name"], "arguments": block["arguments"]}
                    order.append(block["id"])
        elif message.get("role") == "toolResult":
            text = "".join(
                str(part.get("text", "")) for part in message.get("content", []) if isinstance(part, Mapping)
            )
            calls[message["toolCallId"]].update(is_error=bool(message.get("isError")), content=text)
    return [calls[call_id] for call_id in order]


CHECK_MARK = "__LH_MENU_CHECK__"
CHECK_COST_UNITS = 1
REPAIR_COST_UNITS = 2


def menu_check_calls(checks: Sequence[Mapping[str, object]]) -> list[dict[str, object]]:
    """Read-only menu checks appended to a replay chain; each prints its exit code and ms."""

    calls = []
    for check in checks:
        command = str(check["arguments"]["command"])
        wrapped = (
            "s=$(date +%s%N); ( " + command + " ) >/dev/null 2>&1; r=$?; e=$(date +%s%N); "
            f'echo "{CHECK_MARK} {check["check_id"]} $r $(( (e - s) / 1000000 ))"'
        )
        calls.append(
            {
                "source_tool_call_id": f"menu-check:{check['check_id']}",
                "tool_name": "bash",
                "arguments": {"command": wrapped},
                "environment_step_executed": True,
                "source_is_error": False,
            }
        )
    return calls


def parse_menu_verdicts(
    history: Sequence[Mapping[str, object]], checks: Sequence[Mapping[str, object]]
) -> dict[str, dict[str, object]]:
    """Map each appended check to supported (exit 0), refuted (exit 1) or unknown."""

    tail = history[len(history) - len(checks) :] if checks else []
    verdicts: dict[str, dict[str, object]] = {}
    for check, row in zip(checks, tail, strict=True):
        match = re.search(
            rf"{CHECK_MARK} {re.escape(str(check['check_id']))} (\d+) (-?\d+)", str(row.get("content", ""))
        )
        code = int(match.group(1)) if match else None
        verdicts[str(check["check_id"])] = {
            "exit_code": code,
            "elapsed_ms": int(match.group(2)) if match else None,
            "verdict": {0: "supported", 1: "refuted"}.get(code, "unknown"),
        }
    return verdicts


def refuted(verdicts: Mapping[str, Mapping[str, object]]) -> frozenset[str]:
    return frozenset(check_id for check_id, row in verdicts.items() if row["verdict"] == "refuted")


def classify_motif(
    both: frozenset[str], only_a: frozenset[str], only_b: frozenset[str]
) -> str:
    """Spec §5 motif of a two-fault cell from refuted-check sets (A+B, A alone, B alone)."""

    if not only_a or not only_b:
        return "rejected_fault_invisible_alone"
    if (only_a - both) or (only_b - both):
        return "D2_masked"
    shared = only_a & only_b
    if shared and only_a != only_b:
        return "D3_shared_symptom"
    if shared:
        return "rejected_indistinguishable"
    return "D1_independent" if both == only_a | only_b else "rejected_interacting"


def cell_budget(*, fault_count: int, check_count: int) -> dict[str, int]:
    """Optimal targeted cost plus one, strictly below checking and repairing everything."""

    optimal = fault_count * (CHECK_COST_UNITS + REPAIR_COST_UNITS)
    exhaustive = check_count * (CHECK_COST_UNITS + REPAIR_COST_UNITS)
    budget = optimal + 1
    if budget >= exhaustive:
        raise ValueError("menu too small: the budget would allow checking and repairing everything")
    return {"optimal_units": optimal, "budget_units": budget, "exhaustive_units": exhaustive}


def dumps_jsonl(rows: Sequence[Mapping[str, object]]) -> str:
    return "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows)


__all__ = [
    "OPERATORS",
    "SPEC_ID",
    "InjectionSite",
    "apply_sites",
    "dumps_jsonl",
    "effectful_indices",
    "mutating_command",
    "enumerate_sites",
    "omitted_indices",
    "replay_history",
    "cell_budget",
    "classify_motif",
    "menu_check_calls",
    "parse_menu_verdicts",
    "refuted",
    "requirement_literals",
    "rewrite_observer",
    "rewrite_session",
    "sample_sites",
]
