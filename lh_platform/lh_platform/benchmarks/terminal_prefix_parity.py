"""Historical prefix replay parity policy (no heavy runtime dependencies).

Frozen source sessions were produced by the census-era gateway, whose semantics differ
from the hardened managed gateway in two reproducible ways:

- bash exit status was evaluated without `pipefail`, so `cmd 2>&1 | tail -5` recorded
  success even when the piped command failed (the replay, under `pipefail v5`, reports
  an error);
- source-side timing conditions are not reproducible: a slow source container hit the
  gateway's 120 s command timeout or `timeout`'s exit code 124 while the same command
  completes instantly in a fresh replay container;
- a replay container that cannot fetch packages leaves a source-session package missing,
  so commands that used the installed binary fail with a missing-executable error even
  though the install command itself was already exempted as pipeline semantics.

Both classes are recorded as explicit exemptions in the replay receipt. Every exemption
is narrow (direction, tool, and — for timing — a verbatim source timeout marker); any
other parity difference still fails the replay.
"""

from __future__ import annotations

PREFIX_PARITY_POLICY_EXACT = "exact_v1"
PREFIX_PARITY_POLICY_EXEMPTIONS = "documented_semantics_timing_and_install_cascade_v3"

PIPELINE_EXEMPTION_REASON = (
    "source_gateway_without_pipefail_replay_gateway_pipefail_v5"
)
TIMEOUT_EXEMPTION_REASON = "source_timing_timeout_not_reproducible_v1"
INSTALL_CASCADE_EXEMPTION_REASON = "downstream_of_exempted_install_replay_environment_v1"
SOURCE_TIMEOUT_MARKERS = ("Command exited with code 124", "timed out after")
MISSING_EXECUTABLE_MARKERS = (
    "command not found",
    "No such file or directory",
    "cannot access",
    "Unable to locate package",
    "No module named",
    "ModuleNotFoundError",
)
# Downstream failures caused by a package that an already exempted install could not
# fetch (the openssl replay could not pip-install cryptography, so check_cert.py exited
# with a custom 'required but not installed' message).
UNAVAILABLE_PACKAGE_MARKERS = (
    "not installed",
    "not found",
    "No module named",
    "No matching distribution",
)
_INSTALL_COMMAND_PREFIXES = ("apt-get install", "apt install", "pip install", "pip3 install")


def _bash_command(row: dict[str, object]) -> str:
    if row.get("tool_name") != "bash":
        return ""
    arguments = row.get("arguments")
    if isinstance(arguments, dict):
        return str(arguments.get("command") or "")
    return ""


def _is_install_command(command: str) -> bool:
    return any(prefix in command for prefix in _INSTALL_COMMAND_PREFIXES)


def _missing_executable_observed(row: dict[str, object]) -> bool:
    text = str(row.get("result") or "")
    return any(marker in text for marker in MISSING_EXECUTABLE_MARKERS)


def _install_packages(command: str) -> list[str]:
    """Package names requested by an install command (best effort; flags skipped)."""

    names: list[str] = []
    seen_install = False
    for token in command.replace("2>&1", " ").split():
        if token in ("apt-get", "apt", "pip", "pip3"):
            continue
        if token == "install":
            seen_install = True
            continue
        if not seen_install:
            continue
        if token.startswith("-") or "|" in token or "/" in token or token.endswith(".txt"):
            break
        names.append(token)
    return names


def _unavailable_exempted_package(row: dict[str, object], packages: list[str]) -> str | None:
    text = str(row.get("result") or "")
    for package in packages:
        if package and package in text and any(
            marker in text for marker in UNAVAILABLE_PACKAGE_MARKERS
        ):
            return package
    return None


def classify_prefix_parity(
    expected_calls: list[dict[str, object]],
    history: list[dict[str, object]],
) -> tuple[list[dict[str, object]], dict[str, object] | None]:
    """Classify prefix replay parity, returning (exemptions, first real mismatch).

    `expected_calls` are the frozen prefix records (each carries `source_is_error`
    and the verbatim `source_error_text`); `history` rows carry the replay outcome.
    """

    exemptions: list[dict[str, object]] = []
    exempted_install_indices: list[int] = []
    exempted_install_packages: list[str] = []
    if len(expected_calls) != len(history):
        return exemptions, {
            "index": None,
            "reason": "replayed_call_count_differs",
            "expected": len(expected_calls),
            "observed": len(history),
        }
    for index, (call, row) in enumerate(zip(expected_calls, history, strict=True)):
        expected = bool(call.get("source_is_error"))
        observed = bool(row.get("is_error"))
        if expected == observed:
            continue
        command = _bash_command(row)
        source_error_text = str(call.get("source_error_text") or "")
        if (not expected) and observed and "|" in command:
            exemptions.append(
                {
                    "index": index,
                    "tool_name": "bash",
                    "source_tool_call_id": row.get("source_tool_call_id"),
                    "command": command,
                    "source_is_error": False,
                    "replay_is_error": True,
                    "reason": PIPELINE_EXEMPTION_REASON,
                }
            )
            if _is_install_command(command):
                exempted_install_indices.append(index)
                exempted_install_packages.extend(_install_packages(command))
            continue
        unavailable_package = (
            _unavailable_exempted_package(row, exempted_install_packages)
            if exempted_install_packages
            else None
        )
        if (
            (not expected)
            and observed
            and exempted_install_indices
            and (_missing_executable_observed(row) or unavailable_package)
        ):
            exemptions.append(
                {
                    "index": index,
                    "tool_name": str(row.get("tool_name")),
                    "source_tool_call_id": row.get("source_tool_call_id"),
                    "command": command,
                    "source_is_error": False,
                    "replay_is_error": True,
                    "source_install_index": exempted_install_indices[-1],
                    "unavailable_package": unavailable_package,
                    "replay_error_tail": str(row.get("result"))[-120:],
                    "reason": INSTALL_CASCADE_EXEMPTION_REASON,
                }
            )
            continue
        if expected and (not observed) and any(
            marker in source_error_text for marker in SOURCE_TIMEOUT_MARKERS
        ):
            exemptions.append(
                {
                    "index": index,
                    "tool_name": str(row.get("tool_name")),
                    "source_tool_call_id": row.get("source_tool_call_id"),
                    "command": command,
                    "source_is_error": True,
                    "replay_is_error": False,
                    "source_error_text_tail": source_error_text[-120:],
                    "reason": TIMEOUT_EXEMPTION_REASON,
                }
            )
            continue
        return exemptions, {
            "index": index,
            "reason": "error_status_differs",
            "expected": expected,
            "observed": observed,
            "tool_name": row.get("tool_name"),
            "source_error_text_tail": source_error_text[-120:],
        }
    return exemptions, None
