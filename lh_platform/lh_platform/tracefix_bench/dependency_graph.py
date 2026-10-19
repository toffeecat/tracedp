"""Write -> read dependencies between the calls of a frozen Terminal trajectory.

A long-horizon cell needs two numbers per candidate site that
the public-state observer alone cannot give:

* how far ahead a written resource is first read again (the def-use distance that makes
  an O8 "distant cause": written early, consumed late), and
* the causal depth ``H_c``: how many calls lie on the shortest write -> read -> write chain
  from a root call to the call that last wrote a refuted check's subject.

The observer sees only public paths, but sees them exactly; command text names every path
an Actor touches, but only heuristically. Both are combined, and every access records the
evidence that produced it, so an analysis can restrict itself to observed accesses.
"""

from __future__ import annotations

import re
import shlex
from collections import deque
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field

OBSERVER = "observer"
TOOL_ARGUMENT = "tool_argument"
COMMAND = "command"

_REDIRECT = re.compile(r"(?<![0-9&<])(>>?)\s*([^\s;&|<>()]+)")
_SEPARATORS = re.compile(r"\s*(?:&&|\|\||;|\|)\s*")
_HEREDOC = re.compile(r"<<-?\s*['\"]?(\w+)['\"]?.*?\n.*?\n\1\s*(?:\n|$)", re.DOTALL)
_PSEUDO_ROOTS = ("/dev/", "/proc/", "/sys/")
_PATHLIKE = re.compile(r"^(?:/|\./|\.\./)?[\w.+@-]+(?:/[\w.+@-]*)*$")

# Commands whose path operands are written; the value says which operands.
_WRITE_ALL = frozenset({"rm", "touch", "mkdir", "rmdir", "truncate", "shred"})
_WRITE_LAST = frozenset({"cp", "mv", "install", "ln", "rsync"})
_READ_AND_WRITE = frozenset({"sqlite3"})  # opening a database may checkpoint or drop its WAL
_SKIP_FIRST_OPERAND = frozenset({"chmod", "chown", "chgrp"})  # mode / owner precedes paths
# Options whose value is an output path (gcc -o, wget -O, curl -o, tar/make -C, unzip -d).
_OUTPUT_OPTIONS = {
    "-o": None, "--output": None, "-O": ("wget",), "-C": ("tar", "make"), "-d": ("unzip",),
    "--prefix": None, "--target-directory": None, "-t": ("cp", "mv", "install"),
}
# Programs that write into their working directory (or -C directory) when run.
_BUILDS_IN_CWD = frozenset({"make", "cmake", "ninja", "configure", "autoreconf", "meson", "cargo", "go", "mvn", "gradle"})


@dataclass(frozen=True)
class CallAccess:
    """The paths one call read and wrote, each with the evidence that names it."""

    index: int
    reads: Mapping[str, str] = field(default_factory=dict)
    writes: Mapping[str, str] = field(default_factory=dict)


def _normalize(path: str, cwd: str) -> str | None:
    path = path.strip().strip("'\"")
    if not path or path.startswith("-") or "$" in path or "*" in path or "`" in path:
        return None
    if not _PATHLIKE.match(path):
        return None
    if not path.startswith("/"):
        if "/" not in path and "." not in path:
            return None  # a bare word is more likely an argument than a file
        path = cwd.rstrip("/") + "/" + path
    parts: list[str] = []
    for part in path.split("/"):
        if part in ("", "."):
            continue
        if part == "..":
            if parts:
                parts.pop()
            continue
        parts.append(part)
    normalized = "/" + "/".join(parts)
    if normalized == "/" or normalized.startswith(_PSEUDO_ROOTS):
        return None
    return normalized


def _segments(command: str) -> list[str]:
    command = _HEREDOC.sub("\n", command)  # heredoc bodies are data, not commands
    return [segment for segment in _SEPARATORS.split(command) if segment.strip()]


def command_accesses(command: str, *, cwd: str = "/app") -> tuple[dict[str, str], dict[str, str]]:
    """Heuristic reads and writes of one shell command, tracking ``cd`` inside it."""

    reads: dict[str, str] = {}
    writes: dict[str, str] = {}
    for segment in _segments(command):
        for operator, target in _REDIRECT.findall(segment):
            path = _normalize(target, cwd)
            if path is not None:
                writes[path] = COMMAND
        segment = _REDIRECT.sub(" ", segment)
        try:
            words = shlex.split(segment, comments=True)
        except ValueError:
            words = segment.split()
        while words and "=" in words[0] and not words[0].startswith(("/", ".")):
            words = words[1:]  # leading VAR=value assignments
        if not words:
            continue
        program = words[0].rsplit("/", 1)[-1]
        if program in ("sudo", "env", "time", "nohup"):
            words = words[1:]
            if not words:
                continue
            program = words[0].rsplit("/", 1)[-1]
        option_outputs = []
        operands = []
        skip_next = False
        for position, word in enumerate(words[1:], start=1):
            if skip_next:
                skip_next = False
                continue
            name, _, inline = word.partition("=")
            programs = _OUTPUT_OPTIONS.get(name, False) if word.startswith("-") else False
            if programs is not False and (programs is None or program in programs):
                value = inline if inline else (words[position + 1] if position + 1 < len(words) else "")
                skip_next = not inline
                option_outputs.append(value)
                continue
            if not word.startswith("-"):
                operands.append(word)
        for value in option_outputs:
            path = _normalize(value, cwd)
            if path is not None:
                writes[path] = COMMAND
        if program in _BUILDS_IN_CWD or (program == "git" and operands[:1] == ["clone"]):
            target = cwd
            if program == "git" and len(operands) >= 3:
                target = _normalize(operands[2], cwd) or cwd
            writes[target] = COMMAND
            if program == "git":
                continue
        if program == "cd":
            target = _normalize(operands[0], cwd) if operands else None
            if target is not None:
                cwd = target
            continue
        if program in _SKIP_FIRST_OPERAND:
            operands = operands[1:]
        paths = [p for p in (_normalize(word, cwd) for word in operands) if p is not None]
        in_place = program == "sed" and any(word.startswith("-i") for word in words[1:])
        if program in _WRITE_ALL or program in _SKIP_FIRST_OPERAND:
            for path in paths:
                writes[path] = COMMAND
        elif program in _WRITE_LAST and len(paths) >= 2:
            for path in paths[:-1]:
                reads[path] = COMMAND
                if program == "mv":
                    writes[path] = COMMAND
            writes[paths[-1]] = COMMAND
        elif program == "tee":
            for path in paths:
                writes[path] = COMMAND
        elif in_place or program in _READ_AND_WRITE:
            for path in paths:
                reads[path] = COMMAND
                writes[path] = COMMAND
        else:
            for path in paths:
                reads[path] = COMMAND
    return reads, writes


def _observed_changes(row: Mapping[str, object]) -> set[str]:
    before = {item["path"]: item for item in row.get("before", []) if isinstance(item, Mapping)}
    after = {item["path"]: item for item in row.get("after", []) if isinstance(item, Mapping)}
    return {path for path in set(before) | set(after) if before.get(path) != after.get(path)}


def call_accesses(
    prefix: Sequence[Mapping[str, object]],
    observer_rows: Sequence[Mapping[str, object]] | None = None,
    *,
    cwd: str = "/app",
) -> tuple[CallAccess, ...]:
    """Reads and writes of every call; observed changes override nothing, they add writes."""

    if observer_rows is not None and len(observer_rows) != len(prefix):
        raise ValueError("prefix and public-state observer rows differ in length")
    accesses = []
    for index, call in enumerate(prefix):
        reads: dict[str, str] = {}
        writes: dict[str, str] = {}
        executed = call.get("environment_step_executed", True) is not False
        arguments = call.get("arguments", {})
        tool = call.get("tool_name")
        if executed and isinstance(arguments, Mapping):
            path = arguments.get("path")
            if tool in ("read", "write", "edit") and isinstance(path, str):
                normalized = _normalize(path, cwd)
                if normalized is not None:
                    if tool in ("read", "edit"):
                        reads[normalized] = TOOL_ARGUMENT
                    if tool in ("write", "edit"):
                        writes[normalized] = TOOL_ARGUMENT
            command = arguments.get("command")
            if tool == "bash" and isinstance(command, str):
                command_reads, command_writes = command_accesses(command, cwd=cwd)
                reads.update(command_reads)
                writes.update(command_writes)
        if executed and observer_rows is not None:
            for path in _observed_changes(observer_rows[index]):
                writes[path] = OBSERVER
        accesses.append(CallAccess(index=index, reads=reads, writes=writes))
    return tuple(accesses)


def _under(path: str, directory: str) -> bool:
    return path.startswith(directory.rstrip("/") + "/")


def _reaches(written: str, read: str) -> bool:
    """A write reaches a read of the same path, or of a file beneath a written directory.

    Reading an ancestor directory (``ls /app``) is not a read of the files beneath it.
    """

    return written == read or _under(read, written)


def _touches(written: str, subject: str) -> bool:
    """A write changes a subject when it is the subject, beneath it, or above it."""

    return written == subject or _under(written, subject) or _under(subject, written)


def last_writer(
    accesses: Sequence[CallAccess], path: str, before: int | None = None, *, read: bool = False
) -> int | None:
    """The last call before ``before`` that changed ``path``.

    ``read=True`` asks which write a read of ``path`` sees: only a write of the path itself
    or of a directory above it. Otherwise any write at, above or beneath ``path`` counts,
    which is what a check about a directory subject depends on.
    """

    end = len(accesses) if before is None else before
    relation = _reaches if read else _touches
    for access in reversed(accesses[:end]):
        if any(relation(written, path) for written in access.writes):
            return access.index
    return None


def def_use_edges(
    accesses: Sequence[CallAccess], *, evidence: Iterable[str] | None = None
) -> tuple[tuple[int, int, str], ...]:
    """Edges ``(writer, reader, path)``: ``reader`` reads what ``writer`` last wrote."""

    allowed = None if evidence is None else frozenset(evidence)
    edges = []
    for access in accesses:
        for path, source in access.reads.items():
            if allowed is not None and source not in allowed:
                continue
            writer = last_writer(accesses, path, before=access.index, read=True)
            if writer is not None:
                edges.append((writer, access.index, path))
    return tuple(sorted(set(edges)))


def first_read_distance(accesses: Sequence[CallAccess], index: int) -> dict[str, int | None]:
    """For each path ``index`` writes: calls until it is next read, before any rewrite."""

    distances: dict[str, int | None] = {}
    for path in accesses[index].writes:
        distance = None
        for later in accesses[index + 1 :]:
            if any(_reaches(path, read) for read in later.reads):
                distance = later.index - index
                break
            if any(_touches(written, path) for written in later.writes):
                break
        distances[path] = distance
    return distances


def causal_depth(
    accesses: Sequence[CallAccess],
    root: int,
    subject_paths: Iterable[str],
    *,
    edges: Sequence[tuple[int, int, str]] | None = None,
) -> int | None:
    """``H_c``: calls on the shortest chain from ``root`` to a subject's final writer.

    1 when the root itself last wrote a subject; ``None`` when no visible chain connects
    them (the dependency may still exist through state the evidence does not see).
    """

    graph: dict[int, set[int]] = {}
    for writer, reader, _ in edges if edges is not None else def_use_edges(accesses):
        graph.setdefault(writer, set()).add(reader)
    targets = {w for w in (last_writer(accesses, path) for path in subject_paths) if w is not None}
    if not targets:
        return None
    seen = {root: 1}
    queue = deque([root])
    while queue:
        node = queue.popleft()
        if node in targets:
            return seen[node]
        for successor in sorted(graph.get(node, ())):
            if successor not in seen:
                seen[successor] = seen[node] + 1
                queue.append(successor)
    return None


def downstream(accesses: Sequence[CallAccess], root: int) -> frozenset[int]:
    """Every call reachable from ``root`` along def-use edges (its fan-out)."""

    graph: dict[int, set[int]] = {}
    for writer, reader, _ in def_use_edges(accesses):
        graph.setdefault(writer, set()).add(reader)
    reached: set[int] = set()
    queue = deque([root])
    while queue:
        for successor in graph.get(queue.popleft(), ()):
            if successor not in reached:
                reached.add(successor)
                queue.append(successor)
    return frozenset(reached)


def site_horizon(
    accesses: Sequence[CallAccess], site: int, subject_paths: Iterable[str] = ()
) -> dict[str, object]:
    """Long-horizon numbers of one candidate site (plan section 7), all zero-model."""

    distances = first_read_distance(accesses, site)
    known = [value for value in distances.values() if value is not None]
    subjects = tuple(subject_paths)
    return {
        "site": site,
        "d_end": len(accesses) - 1 - site,
        "relative_position": site / (len(accesses) - 1) if len(accesses) > 1 else 0.0,
        "written_paths": sorted(distances),
        "max_first_read_distance": max(known) if known else None,
        # Written here and never read again by the Actor: only the final verifier sees it.
        "unread_paths": sorted(path for path, value in distances.items() if value is None),
        "fan_out": len(downstream(accesses, site)),
        "causal_depth": causal_depth(accesses, site, subjects) if subjects else None,
    }


# --- Value relay: a value first seen in a call's output that the
# Actor later writes or passes on, so it travels through memory before reaching a file.

_RELAY_TOKEN = re.compile(r"(?<![\w.])(\d{3,}(?:\.\d+)?|\d+\.\d+|[0-9a-f]{8,}|/[\w./+-]{4,})(?![\w])")


def _argument_text(call: Mapping[str, object]) -> str:
    arguments = call.get("arguments", {})
    if not isinstance(arguments, Mapping):
        return ""
    return "\n".join(str(value) for value in arguments.values() if isinstance(value, str))


def output_relay_edges(
    prefix: Sequence[Mapping[str, object]], outputs: Sequence[str], *, instruction: str = ""
) -> tuple[tuple[int, int, str], ...]:
    """Edges ``(producer, consumer, token)``: ``token`` first appears in ``producer``'s output
    (not in the instruction or any earlier argument) and later in ``consumer``'s arguments.
    """

    if len(outputs) != len(prefix):
        raise ValueError("prefix and outputs differ in length")
    seen_in_arguments = set(_RELAY_TOKEN.findall(instruction))
    argument_tokens = [set(_RELAY_TOKEN.findall(_argument_text(call))) for call in prefix]
    edges = []
    for producer, output in enumerate(outputs):
        seen_in_arguments |= argument_tokens[producer]
        fresh = set(_RELAY_TOKEN.findall(output or "")) - seen_in_arguments
        for token in sorted(fresh):
            for consumer in range(producer + 1, len(prefix)):
                if token in argument_tokens[consumer]:
                    edges.append((producer, consumer, token))
                    break
    return tuple(edges)


def relay_sites(
    accesses: Sequence[CallAccess], edges: Sequence[tuple[int, int, str]], *, min_distance: int = 5
) -> list[dict[str, object]]:
    """O8b candidates: a relayed value is consumed by a call that writes, far enough away
    that the written file's last writer is not the producer."""

    sites = []
    for producer, consumer, token in edges:
        if consumer - producer < min_distance or not accesses[consumer].writes:
            continue
        sites.append({
            "producer": producer,
            "consumer": consumer,
            "distance": consumer - producer,
            "token": token,
            "written_paths": sorted(accesses[consumer].writes),
        })
    return sites
