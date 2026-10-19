"""Build and feed the host Pi process without exposing task text as CLI flags."""

from __future__ import annotations
import os

import asyncio
import hashlib
import json
import tempfile
from pathlib import Path

from lh_platform.benchmarks.terminal_runtime_identity import (
    CUSTOM_HOST_PI_MODELS,
    HOST_PI_RUNTIME,
    TerminalRuntimeIdentity,
)


def write_host_pi_models_config(
    *,
    config_dir: Path,
    max_output_tokens: int,
    runtime: TerminalRuntimeIdentity = HOST_PI_RUNTIME,
    base_url: str | None = None,
) -> Path:
    """Write the per-session Pi override that bounds one model response.

    ``base_url`` reroutes the provider, e.g. through a shared-budget proxy.
    """

    if max_output_tokens < 1:
        raise ValueError("host Pi max output tokens must be positive")
    config_dir.mkdir(parents=True, exist_ok=True)
    config_path = config_dir / "models.json"
    custom = CUSTOM_HOST_PI_MODELS.get((runtime.provider, runtime.model))
    if custom is None:
        provider_config: dict[str, object] = {
            **({"baseUrl": base_url} if base_url is not None else {}),
            "modelOverrides": {runtime.model: {"maxTokens": max_output_tokens}},
        }
    else:
        # Merged into the built-in provider by id; the key comes from the cell's own env.
        model = {k: v for k, v in custom.items() if k != "baseUrl"}
        provider_config = {
            "baseUrl": base_url if base_url is not None else custom["baseUrl"],
            "api": custom["api"],
            "apiKey": f"${runtime.credential_env}",
            "models": [{**model, "maxTokens": max_output_tokens}],
        }
    config_path.write_text(
        json.dumps({"providers": {runtime.provider: provider_config}}, indent=2) + "\n"
    )
    return config_path


def host_pi_command(
    *,
    executable: Path,
    extension: Path,
    session_dir: Path,
    runtime: TerminalRuntimeIdentity = HOST_PI_RUNTIME,
    session_id: str | None = None,
    session_file: Path | None = None,
    allow_check_registration: bool = False,
    require_check_registration_phase: bool = False,
) -> list[str]:
    """Build the frozen Terminal Pi command without embedding the task instruction."""

    if require_check_registration_phase and not allow_check_registration:
        raise ValueError("required check registration phase needs Actor registration")
    if session_id is not None and session_file is not None:
        raise ValueError("host Pi accepts either a session ID or a session file")
    allowed_tools = ["bash", "read", "write"]
    if allow_check_registration:
        allowed_tools.append("register_runtime_check")
    if require_check_registration_phase:
        allowed_tools.append("complete_runtime_check_registration")

    command = [
        str(executable),
        "--print",
        "--mode",
        "json",
        "--provider",
        runtime.provider,
        "--model",
        runtime.model,
        "--thinking",
        runtime.thinking,
        "--session-dir",
        str(session_dir),
        "--no-builtin-tools",
        "--tools",
        ",".join(allowed_tools),
        "--no-extensions",
        "--extension",
        str(extension),
        "--no-skills",
        "--no-context-files",
        "--no-prompt-templates",
        "--no-approve",
    ]
    if session_id is not None:
        command.extend(("--session-id", session_id))
    if session_file is not None:
        command.extend(("--session", str(session_file)))
    return command


def host_pi_rpc_command(
    *,
    executable: Path,
    extension: Path,
    session_dir: Path,
    runtime: TerminalRuntimeIdentity = HOST_PI_RUNTIME,
    session_id: str | None = None,
    session_file: Path | None = None,
    allow_check_registration: bool = False,
    require_check_registration_phase: bool = False,
) -> list[str]:
    """Build the locked Pi command for a host-controlled JSONL RPC session."""

    command = host_pi_command(
        executable=executable,
        extension=extension,
        session_dir=session_dir,
        runtime=runtime,
        session_id=session_id,
        session_file=session_file,
        allow_check_registration=allow_check_registration,
        require_check_registration_phase=require_check_registration_phase,
    )
    command.remove("--print")
    command[command.index("--mode") + 1] = "rpc"
    return command


async def send_host_pi_instruction(
    stdin: asyncio.StreamWriter,
    instruction: str,
) -> None:
    """Send the opaque task instruction through stdin and close it to start Pi."""

    stdin.write(instruction.encode("utf-8"))
    await stdin.drain()
    stdin.close()
    await stdin.wait_closed()


async def send_host_pi_rpc_command(
    stdin: asyncio.StreamWriter,
    *,
    request_id: str,
    command_type: str,
    message: str | None = None,
) -> None:
    """Write one correlated Pi RPC command while leaving the session open."""

    if command_type not in {"prompt", "abort", "get_state"}:
        raise ValueError("unsupported host Pi RPC command")
    payload: dict[str, str] = {"id": request_id, "type": command_type}
    if command_type == "prompt":
        if message is None:
            raise ValueError("host Pi prompt RPC requires a message")
        payload["message"] = message
    elif message is not None:
        raise ValueError("host Pi control RPC cannot carry a message")
    stdin.write((json.dumps(payload, ensure_ascii=False) + "\n").encode("utf-8"))
    await stdin.drain()



NEUTRAL_ACTOR_CWD_DIRNAME = "longh-actor-cwd"


def neutral_actor_cwd_digest(logs_dir: Path) -> str:
    return hashlib.sha256(str(logs_dir.resolve()).encode("utf-8")).hexdigest()[:16]


def _storage_spellings(path: Path) -> set[str]:
    # Some hosts reach the storage volume under more than one spelling (for example a case
    # variant seen by the VM). LONGH_STORAGE_ALIASES lists them as "old=new;old=new".
    text = str(path)
    variants = {text, str(path.resolve())}
    for pair in os.environ.get("LONGH_STORAGE_ALIASES", "").split(";"):
        if "=" in pair:
            old, new = pair.split("=", 1)
            variants |= {item.replace(old, new) for item in list(variants)}
    return variants


def is_neutral_actor_cwd_of(recorded_cwd: Path, trial_agent_dir: Path) -> bool:
    """True when a session's recorded cwd is the neutral directory of this trial.

    The neutral directory name is a hash of the trial log directory, so it can be
    recomputed from where the session file lives without trusting the session.
    """

    if recorded_cwd.parent.name != NEUTRAL_ACTOR_CWD_DIRNAME:
        return False
    return recorded_cwd.name in {
        neutral_actor_cwd_digest(Path(spelling)) for spelling in _storage_spellings(trial_agent_dir)
    }


def neutral_actor_cwd(logs_dir: Path, *, root: Path | None = None) -> Path:
    """Return a per-trial working directory whose path carries no run or task identity.

    Pi reports its working directory to the actor.  The directory is derived from a
    hash of the trial log directory, so it stays stable across resumes of one trial.
    """

    digest = neutral_actor_cwd_digest(logs_dir)
    base = root if root is not None else Path(tempfile.gettempdir()) / NEUTRAL_ACTOR_CWD_DIRNAME
    neutral = base / digest
    neutral.mkdir(parents=True, exist_ok=True)
    return neutral

__all__ = [
    "is_neutral_actor_cwd_of",
    "neutral_actor_cwd",
    "host_pi_command",
    "host_pi_rpc_command",
    "send_host_pi_instruction",
    "send_host_pi_rpc_command",
    "write_host_pi_models_config",
]
