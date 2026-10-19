"""Zero-model-call evidence for the TraceDP v3 supervisor.

Every probe reads only public information: the task instruction, the public trace and
its state observer, and containers started from the initial task image or a verified
snapshot image (a throwaway container; the run's own environment is never touched).

* ``probe_inventory``: files under the directories the instruction names that exist in
  the initial environment and are missing at the failure state (a resource debt).
  Changed files are reported but are not evidence: many tasks edit their inputs.
* ``dependency_hypotheses``: the last step that wrote each refuted check's subject.
* ``probe_clean_context``: a script the instruction asks for runs under the failure
  state's ``python3`` but fails in a fresh interpreter without the packages the Actor
  installed (the grader may run it elsewhere).
"""

from __future__ import annotations

import re
import subprocess
from collections.abc import Mapping, Sequence
from pathlib import PurePosixPath

_ABS_PATH = re.compile(r"(?<![\w.~-])(/(?:[\w.-]+/?)+)")
DEFAULT_DIRECTORY = "/app"


def instruction_paths(instruction: str) -> list[str]:
    return sorted({match.rstrip(".,;:)") for match in _ABS_PATH.findall(instruction)})


def instruction_directories(instruction: str) -> list[str]:
    """Top-level directories the instruction refers to (``/app`` always included)."""

    directories = {DEFAULT_DIRECTORY}
    for path in instruction_paths(instruction):
        pure = PurePosixPath(path.rstrip("/"))
        directory = pure if path.endswith("/") or not pure.suffix else pure.parent
        if str(directory) not in ("/", ""):
            directories.add(str(directory))
    # Keep only the outermost directories; the inventory recurses three levels.
    ordered = sorted(directories, key=len)
    kept: list[str] = []
    for directory in ordered:
        if not any(directory == k or directory.startswith(k + "/") for k in kept):
            kept.append(directory)
    return kept


def required_scripts(instruction: str) -> list[str]:
    """Executable Python artifacts the instruction asks the Actor to create."""

    return [path for path in instruction_paths(instruction) if path.endswith(".py")]


def inventory_command(directories: Sequence[str]) -> str:
    quoted = " ".join(f"'{d}'" for d in directories)
    return (
        f"for d in {quoted}; do [ -d \"$d\" ] || continue; "
        "find \"$d\" -maxdepth 3 -type f -not -path '*/.git/*' 2>/dev/null; done | sort -u | "
        "while read -r f; do printf '%s %s\\n' \"$(sha256sum \"$f\" | cut -c1-16)\" \"$f\"; done"
    )


def parse_inventory(text: str) -> dict[str, str]:
    rows = {}
    for line in text.splitlines():
        digest, _, path = line.partition(" ")
        if digest and path.startswith("/"):
            rows[path] = digest
    return rows


def resource_debt(
    initial: Mapping[str, str], failure: Mapping[str, str], outputs: Sequence[str]
) -> dict[str, list[str]]:
    """Missing (evidence) and changed (reported only) input files, outputs excluded."""

    outputs = set(outputs)
    missing = sorted(p for p in initial if p not in failure and p not in outputs)
    changed = sorted(p for p in initial if p in failure and initial[p] != failure[p] and p not in outputs)
    return {"missing": missing, "changed": changed}


def dependency_hypotheses(
    observer_rows: Sequence[Mapping[str, object]], refuted_subjects: Sequence[str]
) -> dict[str, str]:
    """Refuted subject path -> the event id of the last step that changed it."""

    last: dict[str, int] = {}
    for index, row in enumerate(observer_rows, start=1):
        before = {x["path"]: (x.get("exists"), x.get("content_base64")) for x in row.get("before", [])}  # type: ignore[union-attr]
        after = {x["path"]: (x.get("exists"), x.get("content_base64")) for x in row.get("after", [])}  # type: ignore[union-attr]
        for path in set(before) | set(after):
            if before.get(path) != after.get(path):
                last[path] = index
    return {path: f"step-{last[path]:04d}" for path in refuted_subjects if path in last}


def clean_context_command(script: str) -> str:
    """Run ``script`` under python3, then in a fresh venv (or ``-I -S`` without venv)."""

    directory = str(PurePosixPath(script).parent)
    return (
        f"cd '{directory}' 2>/dev/null; [ -f '{script}' ] || {{ echo 'LH_ABSENT'; exit 0; }}; "
        f"timeout 60 python3 '{script}' >/dev/null 2>/tmp/lh_sys.err; s=$?; "
        "rm -rf /tmp/lh_clean_venv; "
        "if python3 -m venv --without-pip /tmp/lh_clean_venv >/dev/null 2>&1; then "
        f"timeout 60 /tmp/lh_clean_venv/bin/python '{script}' >/dev/null 2>/tmp/lh_clean.err; c=$?; m=venv; "
        f"else timeout 60 python3 -I -S '{script}' >/dev/null 2>/tmp/lh_clean.err; c=$?; m=isolated; fi; "
        "echo \"LH_SYS=$s LH_CLEAN=$c LH_MODE=$m\"; tail -c 600 /tmp/lh_clean.err"
    )


def clean_context_verdict(output: str) -> dict[str, object]:
    """``fail`` when the script succeeds under python3 and fails in the clean context."""

    if "LH_ABSENT" in output:
        return {"verdict": "absent"}
    match = re.search(r"LH_SYS=(\d+) LH_CLEAN=(\d+) LH_MODE=(\w+)", output)
    if not match:
        return {"verdict": "unknown", "output": output[-600:]}
    system, clean, mode = int(match.group(1)), int(match.group(2)), match.group(3)
    stderr = output[match.end():].strip()
    verdict = "fail" if system == 0 and clean != 0 else "pass"
    return {"verdict": verdict, "system_exit": system, "clean_exit": clean, "mode": mode, "clean_stderr": stderr[-600:]}


def run_in_image(image: str, command: str, *, docker_context: str, timeout: float = 300.0) -> tuple[int, str]:
    """Run a shell command in a throwaway container of ``image``."""

    process = subprocess.run(
        ["docker", "--context", docker_context, "run", "--rm", "--network", "none", "--entrypoint", "sh",
         image, "-c", command],
        capture_output=True, text=True, timeout=timeout,
    )
    return process.returncode, process.stdout


__all__ = [
    "clean_context_command", "clean_context_verdict", "dependency_hypotheses", "instruction_directories",
    "inventory_command", "parse_inventory", "required_scripts", "resource_debt", "run_in_image",
]
