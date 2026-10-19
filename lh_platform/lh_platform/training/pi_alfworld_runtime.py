"""Pi-backed ALFWorld runtime-fidelity calibration.

This module is intentionally inference-only. It records the observable chain
from the prompt sent to Pi through the parsed action and the real environment
transition. It does not manufacture policy token log-probabilities and cannot
enter an RL update.
"""

from __future__ import annotations

import json
import os
import re
import select
import shutil
import subprocess
import tempfile
import time
import urllib.parse
import urllib.request
import uuid
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from lh_platform.models.controlled_runtime import (
    ControlledRuntimeIdentity,
    RuntimeModelPurpose,
    require_runtime_model_identity,
)
from lh_platform.studies.runtime import AlfworldTask
from lh_platform.test_time_adaptation.public_residual_sidecar import (
    PublicTransitionEvidence,
    PublicTransitionSidecar,
)
from lh_platform.training.alfworld_agentic_suite import (
    AlfWorldBranch,
)
from lh_platform.training.g2po.actor_protocol import (
    BARE_MENU_INDEX_NON_THINKING_ALFWORLD_ACTOR_PROTOCOL_ID,
    MODEL_COMPATIBLE_NON_THINKING_ALFWORLD_ACTOR_PROTOCOL_ID,
    NATIVE_MENU_CHOICE_NON_THINKING_ALFWORLD_ACTOR_PROTOCOL_ID,
    UNIQUE_ACTION_TAG_NON_THINKING_ALFWORLD_ACTOR_PROTOCOL_ID,
    G2POActionProjection,
    G2POActorHistoryStep,
    build_alfworld_actor_protocol,
    extract_alfworld_task_description,
    unique_action_tag_parser_sha256,
)


class PiCliActorError(RuntimeError):
    """Pi failed to return the configured provider/model response."""


class PiTransportError(PiCliActorError):
    """Credential loading or provider transport failed before a usable response."""


class PiActorCompatibilityError(PiCliActorError):
    """The provider response violated the frozen actor/runtime contract."""


def _strip_pi_json_relaxations(content: str) -> str:
    """Mirror Pi 0.84.1 line-comment and trailing-comma JSON semantics."""

    string_or_comment = re.compile(r'"(?:\\.|[^"\\])*"|//[^\n]*')
    without_comments = string_or_comment.sub(
        lambda match: match.group(0) if match.group(0).startswith('"') else "",
        content,
    )
    string_or_trailing_comma = re.compile(r'"(?:\\.|[^"\\])*"|,(\s*[}\]])')
    return string_or_trailing_comma.sub(
        lambda match: match.group(1) or match.group(0),
        without_comments,
    )


def _prepare_isolated_pi_config(
    *,
    source_directory: Path,
    isolated_directory: Path,
    provider: str,
    model: str,
    max_output_tokens: int,
) -> None:
    """Snapshot Pi runtime state and merge one process-local output-token limit."""

    models_path = source_directory / "models.json"
    if models_path.is_file():
        try:
            models_config = json.loads(
                _strip_pi_json_relaxations(models_path.read_text(encoding="utf-8"))
            )
        except (OSError, json.JSONDecodeError) as error:
            raise ValueError("Pi models.json could not be loaded for budget isolation") from error
    else:
        models_config = {"providers": {}}
    if not isinstance(models_config, dict) or not isinstance(models_config.get("providers"), dict):
        raise ValueError("Pi models.json must contain a providers object")
    providers = models_config["providers"]
    provider_config = providers.setdefault(provider, {})
    if not isinstance(provider_config, dict):
        raise ValueError(f"Pi provider configuration for {provider!r} must be an object")
    model_overrides = provider_config.setdefault("modelOverrides", {})
    if not isinstance(model_overrides, dict):
        raise ValueError(f"Pi model overrides for {provider!r} must be an object")
    model_override = model_overrides.setdefault(model, {})
    if not isinstance(model_override, dict):
        raise ValueError(f"Pi model override for {provider}/{model} must be an object")
    model_override["maxTokens"] = max_output_tokens

    isolated_models_path = isolated_directory / "models.json"
    isolated_models_path.write_text(
        json.dumps(models_config, indent=2) + "\n",
        encoding="utf-8",
    )
    isolated_models_path.chmod(0o600)

    # Pi may refresh both files. Copies keep those writes inside this process snapshot.
    for filename in ("auth.json", "models-store.json"):
        source_path = source_directory / filename
        if source_path.is_file():
            destination_path = isolated_directory / filename
            shutil.copy2(source_path, destination_path)
            destination_path.chmod(0o600)


@dataclass(frozen=True)
class PiActorResponse:
    """Return model output plus explicit provenance for identity-sensitive experiments."""

    text: str
    provider: str
    model: str
    input_tokens: int
    output_tokens: int
    reasoning_tokens: int
    stop_reason: str
    latency_seconds: float
    provider_identity_evidence: str = "served_response_metadata"
    served_provider_reported: bool = True
    reasoning_tokens_reported: bool = True
    runtime_purpose: str = "unverified_external_actor"


def _pi_actor_response(
    assistant: Mapping[str, Any],
    *,
    provider: str,
    model: str,
    latency_seconds: float,
    runtime_purpose: RuntimeModelPurpose,
) -> PiActorResponse:
    served_provider = str(assistant.get("provider", ""))
    served_model = str(assistant.get("model", ""))
    if served_provider != provider:
        raise PiCliActorError(f"Pi served provider {served_provider!r}, expected {provider!r}")
    if served_model != model:
        raise PiCliActorError(f"Pi served model {served_model!r}, expected {model!r}")
    error_message = assistant.get("errorMessage")
    if assistant.get("stopReason") == "error" and isinstance(error_message, str):
        raise PiCliActorError(f"Pi provider error: {error_message.strip()}")
    content = assistant.get("content") or []
    text_parts = [
        str(item.get("text", ""))
        for item in content
        if isinstance(item, dict) and item.get("type") == "text"
    ]
    text_response = "".join(text_parts).strip()
    if not text_response:
        raise PiCliActorError("Pi returned an empty assistant response")
    usage = assistant.get("usage") or {}
    reasoning_value = usage.get("reasoning")
    reasoning_tokens_reported = type(reasoning_value) is int
    reasoning_tokens = reasoning_value if reasoning_tokens_reported else 0
    thinking_content_present = any(
        isinstance(item, Mapping) and item.get("type") == "thinking" for item in content
    )
    if reasoning_tokens != 0 or thinking_content_present:
        raise PiCliActorError("Pi returned reasoning tokens while thinking was disabled")
    return PiActorResponse(
        text=text_response,
        provider=served_provider,
        model=served_model,
        input_tokens=int(usage.get("input", 0)),
        output_tokens=int(usage.get("output", 0)),
        reasoning_tokens=reasoning_tokens,
        stop_reason=str(assistant.get("stopReason", "")),
        latency_seconds=latency_seconds,
        reasoning_tokens_reported=reasoning_tokens_reported,
        runtime_purpose=runtime_purpose,
    )


def decode_pi_assistant_response(
    assistant: Mapping[str, Any],
    *,
    provider: str,
    model: str,
    latency_seconds: float,
    runtime_purpose: RuntimeModelPurpose = "longh_controlled",
) -> PiActorResponse:
    """Decode durable Pi evidence with the same checks used by the live actor."""

    return _pi_actor_response(
        assistant,
        provider=provider,
        model=model,
        latency_seconds=latency_seconds,
        runtime_purpose=runtime_purpose,
    )


class Actor(Protocol):
    provider: str
    model: str

    def generate(self, prompt: str) -> PiActorResponse: ...


def _actor_response_record(
    *,
    prompt: str,
    observation: str,
    admissible_actions: Sequence[str],
    response: PiActorResponse,
    projection: G2POActionProjection,
) -> dict[str, Any]:
    in_menu = projection.is_action_valid and projection.environment_action in admissible_actions
    return {
        "prompt": prompt,
        "observation": observation,
        "admissible_actions": list(admissible_actions),
        "model_response": response.text,
        "environment_action": projection.environment_action,
        "parser_valid": projection.is_action_valid,
        "parser_failure_reasons": list(projection.invalid_reasons),
        "action_in_admissible_menu": in_menu,
        "provider": response.provider,
        "model": response.model,
        "input_tokens": response.input_tokens,
        "output_tokens": response.output_tokens,
        "reasoning_tokens": response.reasoning_tokens,
        "runtime_purpose": response.runtime_purpose,
        "stop_reason": response.stop_reason,
        "latency_seconds": response.latency_seconds,
    }


class SnapshotBranch(AlfWorldBranch, Protocol):
    def step_snapshot_only(self, action: str) -> Any: ...


BranchFactory = Callable[[AlfworldTask, int, Path], SnapshotBranch]


class PiCliActor:
    """Call one authenticated Pi model turn without tools or session state."""

    transport = "pi_cli_ephemeral_no_tools"

    def __init__(
        self,
        *,
        executable: str | Path,
        provider: str = "opencode-go",
        model: str = "mimo-v2.5",
        timeout_seconds: int = 60,
        working_directory: str | Path | None = None,
        runtime_purpose: RuntimeModelPurpose = "longh_controlled",
    ) -> None:
        require_runtime_model_identity(
            provider=provider,
            model=model,
            thinking="off",
            purpose=runtime_purpose,
        )
        self.executable = str(executable)
        self.provider = provider
        self.model = model
        self.runtime_purpose = runtime_purpose
        self.timeout_seconds = timeout_seconds
        self.working_directory = (
            None if working_directory is None else str(Path(working_directory).resolve())
        )

    def generate(self, prompt: str) -> PiActorResponse:
        """Issue one request only after revalidating the frozen model identity."""

        require_runtime_model_identity(
            provider=self.provider,
            model=self.model,
            thinking="off",
            purpose=self.runtime_purpose,
        )
        command = [
            self.executable,
            "--provider",
            self.provider,
            "--model",
            self.model,
            "--thinking",
            "off",
            "--mode",
            "json",
            "--print",
            "--no-session",
            "--no-tools",
            "--no-skills",
            "--no-extensions",
            "--no-context-files",
            "--no-prompt-templates",
            "--no-approve",
            "--system-prompt",
            (
                "You control the ALFWorld text environment. Follow the user's "
                "output contract exactly. Never invent observations or actions."
            ),
            prompt,
        ]
        started = time.monotonic()
        try:
            completed = subprocess.run(
                command,
                check=False,
                stdin=subprocess.DEVNULL,
                capture_output=True,
                text=True,
                timeout=self.timeout_seconds,
                cwd=self.working_directory,
            )
        except subprocess.TimeoutExpired as error:
            raise PiCliActorError(
                f"Pi request timed out after {self.timeout_seconds} seconds"
            ) from error
        if completed.returncode != 0:
            detail = completed.stderr.strip() or f"exit code {completed.returncode}"
            raise PiCliActorError(f"Pi request failed: {detail}")

        assistant: dict[str, Any] | None = None
        for line in completed.stdout.splitlines():
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue
            message = event.get("message")
            if (
                event.get("type") == "message_end"
                and isinstance(message, dict)
                and message.get("role") == "assistant"
            ):
                assistant = message
        if assistant is None:
            raise PiCliActorError("Pi response did not contain a completed assistant message")

        return _pi_actor_response(
            assistant,
            provider=self.provider,
            model=self.model,
            latency_seconds=time.monotonic() - started,
            runtime_purpose=self.runtime_purpose,
        )


class PiRpcActor:
    """Keep Pi isolated in one RPC process while preserving stateless policy turns."""

    transport = "pi_cli_rpc_fresh_session_no_tools"

    def __init__(
        self,
        *,
        executable: str | Path,
        provider: str = "opencode-go",
        model: str = "mimo-v2.5",
        timeout_seconds: int = 60,
        working_directory: str | Path | None = None,
        temporary_directory: str | Path | None = None,
        system_prompt: str = (
            "You control the ALFWorld text environment. Follow the user's "
            "output contract exactly. Never invent observations or actions."
        ),
        runtime_purpose: RuntimeModelPurpose = "longh_controlled",
        expected_runtime_identity: ControlledRuntimeIdentity | None = None,
        process_environment: Mapping[str, str] | None = None,
        raw_response_recorder: Callable[[dict[str, Any], float], None] | None = None,
        max_output_tokens: int | None = None,
    ) -> None:
        # Pi RPC starts model I/O during construction, so validate before Popen.
        require_runtime_model_identity(
            provider=provider,
            model=model,
            thinking="off",
            purpose=runtime_purpose,
            expected_identity=expected_runtime_identity,
        )
        self.provider = provider
        self.model = model
        self.runtime_purpose = runtime_purpose
        self.expected_runtime_identity = expected_runtime_identity
        self.timeout_seconds = timeout_seconds
        self.system_prompt = system_prompt
        if max_output_tokens is not None and (
            isinstance(max_output_tokens, bool) or max_output_tokens < 1
        ):
            raise ValueError("Pi RPC max output tokens must be a positive integer")
        self.max_output_tokens = max_output_tokens
        self._raw_response_recorder = raw_response_recorder
        self._generation_event_recorder: Callable[[dict[str, Any], float], None] | None = None
        self._request_index = 0
        self._stdout_buffer = bytearray()
        self._stderr = tempfile.TemporaryFile(
            mode="w+b",
            dir=(None if temporary_directory is None else str(Path(temporary_directory).resolve())),
        )
        command = [
            str(executable),
            "--provider",
            provider,
            "--model",
            model,
            "--thinking",
            "off",
            "--mode",
            "rpc",
            "--no-session",
            "--no-tools",
            "--no-skills",
            "--no-extensions",
            "--no-context-files",
            "--no-prompt-templates",
            "--no-approve",
            "--system-prompt",
            system_prompt,
        ]
        environment = None if process_environment is None else dict(process_environment)
        self._config_directory: tempfile.TemporaryDirectory[str] | None = None
        try:
            if max_output_tokens is not None:
                source_config_value = (environment or {}).get(
                    "PI_CODING_AGENT_DIR"
                ) or os.environ.get("PI_CODING_AGENT_DIR")
                source_config_directory = (
                    Path(source_config_value).expanduser().resolve()
                    if source_config_value
                    else Path.home() / ".pi" / "agent"
                )
                self._config_directory = tempfile.TemporaryDirectory(
                    prefix="longh-pi-rpc-config-",
                    dir=(
                        None
                        if temporary_directory is None
                        else str(Path(temporary_directory).resolve())
                    ),
                )
                _prepare_isolated_pi_config(
                    source_directory=source_config_directory,
                    isolated_directory=Path(self._config_directory.name),
                    provider=provider,
                    model=model,
                    max_output_tokens=max_output_tokens,
                )
                environment = dict(environment or {})
                environment["PI_CODING_AGENT_DIR"] = self._config_directory.name
            self._process = subprocess.Popen(
                command,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=self._stderr,
                text=False,
                bufsize=0,
                cwd=(None if working_directory is None else str(Path(working_directory).resolve())),
                env=(None if environment is None else {**os.environ, **environment}),
            )
        except Exception:
            self._stderr.close()
            if self._config_directory is not None:
                self._config_directory.cleanup()
                self._config_directory = None
            raise

    def _stderr_detail(self) -> str:
        size = os.fstat(self._stderr.fileno()).st_size
        return os.pread(self._stderr.fileno(), size, 0).decode("utf-8", errors="replace").strip()

    @property
    def process_alive(self) -> bool:
        """Expose transport liveness without reading or mutating the RPC stream."""

        return self._process.poll() is None

    def set_raw_response_recorder(
        self, recorder: Callable[[dict[str, Any], float], None] | None
    ) -> None:
        """Attach durable persistence before response validation and parsing."""

        self._raw_response_recorder = recorder

    def set_generation_event_recorder(
        self, recorder: Callable[[dict[str, Any], float], None] | None
    ) -> None:
        """Observe completed messages before settlement without changing acceptance."""

        self._generation_event_recorder = recorder

    def _next_event(self, deadline: float) -> dict[str, Any]:
        assert self._process.stdout is not None
        stdout_fd = self._process.stdout.fileno()
        while True:
            newline = self._stdout_buffer.find(b"\n")
            if newline >= 0:
                line = bytes(self._stdout_buffer[:newline])
                del self._stdout_buffer[: newline + 1]
                try:
                    event = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if isinstance(event, dict):
                    return event
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise PiCliActorError(f"Pi RPC request timed out: {self._stderr_detail()}")
            ready, _, _ = select.select([stdout_fd], [], [], remaining)
            if not ready:
                raise PiCliActorError(f"Pi RPC request timed out: {self._stderr_detail()}")
            chunk = os.read(stdout_fd, 65536)
            if not chunk:
                raise PiCliActorError(f"Pi RPC process exited: {self._stderr_detail()}")
            self._stdout_buffer.extend(chunk)

    def _send(self, payload: Mapping[str, Any]) -> str:
        if self._process.poll() is not None:
            raise PiCliActorError(f"Pi RPC process exited: {self._stderr_detail()}")
        self._request_index += 1
        request_id = f"request-{self._request_index}"
        message = dict(payload)
        message["id"] = request_id
        assert self._process.stdin is not None
        self._process.stdin.write((json.dumps(message, ensure_ascii=False) + "\n").encode("utf-8"))
        self._process.stdin.flush()
        return request_id

    @staticmethod
    def _require_success(event: Mapping[str, Any]) -> None:
        if event.get("success") is not True:
            raise PiCliActorError(f"Pi RPC request failed: {event.get('error', '')}")

    def _reset_session(self, deadline: float) -> None:
        request_id = self._send({"type": "new_session"})
        while True:
            event = self._next_event(deadline)
            if event.get("type") == "response" and event.get("id") == request_id:
                self._require_success(event)
                return

    def generate(self, prompt: str) -> PiActorResponse:
        """Issue one RPC request only under the constructor-validated identity."""

        require_runtime_model_identity(
            provider=self.provider,
            model=self.model,
            thinking="off",
            purpose=self.runtime_purpose,
            expected_identity=self.expected_runtime_identity,
        )
        started = time.monotonic()
        deadline = started + self.timeout_seconds
        self._reset_session(deadline)
        request_id = self._send({"type": "prompt", "message": prompt})
        assistant: dict[str, Any] | None = None
        acknowledged = False
        while True:
            event = self._next_event(deadline)
            if self._generation_event_recorder is not None and str(
                event.get("type", "")
            ).startswith(("compaction_", "auto_retry_", "summarization_retry_")):
                self._generation_event_recorder(dict(event), time.monotonic() - started)
            if event.get("type") == "response" and event.get("id") == request_id:
                self._require_success(event)
                acknowledged = True
            elif event.get("type") == "message_end":
                message = event.get("message")
                if isinstance(message, dict) and message.get("role") == "assistant":
                    assistant = message
                    if self._generation_event_recorder is not None:
                        self._generation_event_recorder(dict(event), time.monotonic() - started)
            elif event.get("type") == "agent_settled" and acknowledged:
                if self._generation_event_recorder is not None:
                    self._generation_event_recorder(dict(event), time.monotonic() - started)
                break
        if assistant is None:
            raise PiCliActorError("Pi RPC response did not contain an assistant message")
        latency_seconds = time.monotonic() - started
        if self._raw_response_recorder is not None:
            self._raw_response_recorder(dict(assistant), latency_seconds)
        return self.response_from_raw(assistant, latency_seconds=latency_seconds)

    def response_from_raw(
        self, response: Mapping[str, Any], *, latency_seconds: float
    ) -> PiActorResponse:
        """Validate one persisted Pi assistant message without another model call."""

        decoded = decode_pi_assistant_response(
            response,
            provider=self.provider,
            model=self.model,
            latency_seconds=latency_seconds,
            runtime_purpose=self.runtime_purpose,
        )
        if self.max_output_tokens is not None and decoded.output_tokens > self.max_output_tokens:
            raise PiCliActorError("Pi exceeded the frozen output-token ceiling")
        return decoded

    def close(self) -> None:
        if self._process.poll() is None:
            self._process.terminate()
            try:
                self._process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                self._process.kill()
                self._process.wait(timeout=2)
        if self._process.stdin is not None:
            self._process.stdin.close()
        if self._process.stdout is not None:
            self._process.stdout.close()
        self._stderr.close()
        if self._config_directory is not None:
            self._config_directory.cleanup()
            self._config_directory = None


class PiAuthenticatedDeepSeekActor:
    """Use Pi authentication with DeepSeek's inference endpoint directly."""

    transport = "pi_authenticated_deepseek_chat_completion"

    def __init__(
        self,
        *,
        executable: str | Path,
        provider: str = "deepseek",
        model: str = "deepseek-v4-flash",
        endpoint: str = "https://api.deepseek.com/chat/completions",
        timeout_seconds: int = 60,
        max_tokens: int = 64,
        runtime_purpose: RuntimeModelPurpose = "longh_controlled",
    ) -> None:
        require_runtime_model_identity(
            provider=provider,
            model=model,
            thinking="off",
            purpose=runtime_purpose,
        )
        self.executable = str(executable)
        self.provider = provider
        self.model = model
        self.runtime_purpose = runtime_purpose
        self.endpoint = endpoint
        self.timeout_seconds = timeout_seconds
        self.max_tokens = max_tokens
        self._api_key: str | None = None

    def _load_api_key(self) -> str:
        if self._api_key is None:
            completed = subprocess.run(
                [
                    self.executable,
                    "auth",
                    "print-api-key",
                    "--provider",
                    self.provider,
                    "--model",
                    self.model,
                ],
                check=True,
                capture_output=True,
                text=True,
            )
            self._api_key = completed.stdout.strip()
        return self._api_key

    def generate(self, prompt: str) -> PiActorResponse:
        """Send a legacy-compatible request only under an explicit purpose."""

        require_runtime_model_identity(
            provider=self.provider,
            model=self.model,
            thinking="off",
            purpose=self.runtime_purpose,
        )
        body = json.dumps(
            {
                "model": self.model,
                "messages": [
                    {
                        "role": "system",
                        "content": (
                            "You control the ALFWorld text environment. Follow the "
                            "user's output contract exactly. Never invent observations "
                            "or actions."
                        ),
                    },
                    {"role": "user", "content": prompt},
                ],
                "thinking": {"type": "disabled"},
                "max_tokens": self.max_tokens,
                "stream": False,
            }
        ).encode()
        request = urllib.request.Request(
            self.endpoint,
            data=body,
            headers={
                "Authorization": f"Bearer {self._load_api_key()}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        started = time.monotonic()
        with urllib.request.urlopen(request, timeout=self.timeout_seconds) as response:
            payload = json.loads(response.read())

        served_model = str(payload.get("model", ""))
        if served_model != self.model:
            raise PiCliActorError(
                f"DeepSeek served model {served_model!r}, expected {self.model!r}"
            )
        choice = payload["choices"][0]
        text_response = str(choice["message"].get("content", "")).strip()
        if not text_response:
            raise PiCliActorError("DeepSeek returned an empty assistant response")
        usage = payload.get("usage") or {}
        token_details = usage.get("completion_tokens_details") or {}
        return PiActorResponse(
            text=text_response,
            provider=self.provider,
            model=served_model,
            input_tokens=int(usage.get("prompt_tokens", 0)),
            output_tokens=int(usage.get("completion_tokens", 0)),
            reasoning_tokens=int(token_details.get("reasoning_tokens", 0)),
            stop_reason=str(choice.get("finish_reason", "")),
            latency_seconds=time.monotonic() - started,
            runtime_purpose=self.runtime_purpose,
        )


class PiAuthenticatedOpenAiActor:
    """Use Pi credentials for one bounded OpenAI-compatible chat completion."""

    transport = "pi_authenticated_openai_chat_completion"

    def __init__(
        self,
        *,
        executable: str | Path,
        provider: str,
        model: str,
        base_url: str,
        max_tokens: int,
        system_prompt: str,
        timeout_seconds: int = 180,
        session_id: str | None = None,
        user_agent: str = "longh-agent-runtime/1.0",
        json_object_output: bool = False,
        raw_response_recorder: Callable[[dict[str, Any], float], None] | None = None,
        runtime_purpose: RuntimeModelPurpose = "longh_controlled",
    ) -> None:
        if max_tokens < 1 or timeout_seconds < 1:
            raise ValueError("OpenAI-compatible actor limits must be positive")
        require_runtime_model_identity(
            provider=provider,
            model=model,
            thinking="off",
            purpose=runtime_purpose,
        )
        self.executable = str(executable)
        self.provider = provider
        self.model = model
        self.runtime_purpose = runtime_purpose
        self.base_url = base_url
        self.max_tokens = max_tokens
        self.system_prompt = system_prompt
        self.timeout_seconds = timeout_seconds
        is_opencode = (
            provider in {"opencode", "opencode-go"}
            or urllib.parse.urlparse(base_url).hostname == "opencode.ai"
        )
        if session_id is not None and session_id != session_id.strip():
            raise ValueError("OpenAI-compatible session identity is not normalized")
        if not user_agent.strip():
            raise ValueError("OpenAI-compatible user agent is empty")
        # OpenCode routes and caches each bounded conversation by this stable identity.
        self.session_id = session_id or (f"longh-{uuid.uuid4()}" if is_opencode else None)
        self.user_agent = user_agent.strip()
        self.client_id = "longh"
        self.json_object_output = json_object_output
        self._is_opencode = is_opencode
        self._raw_response_recorder = raw_response_recorder
        self._api_key: str | None = None

    def _load_api_key(self) -> str:
        if self._api_key is None:
            try:
                completed = subprocess.run(
                    [
                        self.executable,
                        "auth",
                        "print-api-key",
                        "--provider",
                        self.provider,
                        "--model",
                        self.model,
                    ],
                    check=True,
                    capture_output=True,
                    text=True,
                )
            except (OSError, subprocess.CalledProcessError) as error:
                raise PiTransportError("Pi credential loading failed") from error
            self._api_key = completed.stdout.strip()
        return self._api_key

    def set_raw_response_recorder(
        self, recorder: Callable[[dict[str, Any], float], None] | None
    ) -> None:
        """Attach a durable recorder before response-contract validation."""

        self._raw_response_recorder = recorder

    def generate(self, prompt: str) -> PiActorResponse:
        """Send the frozen generation budget in the provider request itself."""

        return self.generate_with_timeout(prompt, timeout_seconds=self.timeout_seconds)

    def generate_with_timeout(
        self, prompt: str, *, timeout_seconds: int
    ) -> PiActorResponse:
        """Run one request with a caller-bounded timeout below the configured ceiling."""

        if not 1 <= timeout_seconds <= self.timeout_seconds:
            raise ValueError("request timeout exceeds the configured actor limit")

        require_runtime_model_identity(
            provider=self.provider,
            model=self.model,
            thinking="off",
            purpose=self.runtime_purpose,
        )

        try:
            from openai import OpenAI
        except ImportError as error:  # pragma: no cover - dependency gate.
            raise PiCliActorError("OpenAI-compatible actor requires openai") from error
        client = OpenAI(
            api_key=self._load_api_key(),
            base_url=self.base_url,
            timeout=timeout_seconds,
            default_headers={
                "User-Agent": self.user_agent,
                **(
                    {
                        "x-opencode-session": self.session_id,
                        "x-opencode-client": self.client_id,
                    }
                    if self._is_opencode and self.session_id is not None
                    else {}
                ),
            },
        )
        started = time.monotonic()
        try:
            request_options: dict[str, Any] = {}
            if self.json_object_output:
                # MiMo JSON mode constrains syntax; semantic validation remains local.
                request_options["response_format"] = {"type": "json_object"}
            response = client.chat.completions.create(
                model=self.model,
                messages=(
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": prompt},
                ),
                max_tokens=self.max_tokens,
                reasoning_effort="none",
                stream=False,
                **request_options,
            )
        except Exception as error:
            raise PiTransportError(f"OpenAI-compatible provider error: {error}") from error
        finally:
            client.close()
            self._api_key = None

        latency_seconds = time.monotonic() - started
        raw_response = response.model_dump(mode="json")
        if self._raw_response_recorder is not None:
            self._raw_response_recorder(raw_response, latency_seconds)
        return self.response_from_raw(
            raw_response,
            latency_seconds=latency_seconds,
        )

    def response_from_raw(
        self, response: Mapping[str, Any], *, latency_seconds: float
    ) -> PiActorResponse:
        """Validate a persisted provider payload without issuing another API call."""

        served_model = str(response.get("model", ""))
        if served_model != self.model:
            raise PiActorCompatibilityError(
                f"OpenAI-compatible endpoint served {served_model!r}, expected {self.model!r}"
            )
        choices = response.get("choices")
        if not isinstance(choices, list) or not choices:
            raise PiActorCompatibilityError("OpenAI-compatible endpoint returned no choices")
        choice = choices[0]
        if not isinstance(choice, Mapping):
            raise PiActorCompatibilityError("OpenAI-compatible endpoint returned invalid choices")
        message = choice.get("message")
        if not isinstance(message, Mapping):
            raise PiActorCompatibilityError("OpenAI-compatible endpoint omitted the message")
        text_response = str(message.get("content") or "").strip()
        if not text_response:
            raise PiActorCompatibilityError("OpenAI-compatible endpoint returned an empty response")
        usage = response.get("usage")
        if not isinstance(usage, Mapping):
            raise PiActorCompatibilityError("OpenAI-compatible endpoint omitted token usage")
        prompt_tokens = usage.get("prompt_tokens")
        completion_tokens = usage.get("completion_tokens")
        if type(prompt_tokens) is not int or type(completion_tokens) is not int:
            raise PiActorCompatibilityError("OpenAI-compatible endpoint omitted token usage")
        output_tokens = completion_tokens
        if output_tokens > self.max_tokens:
            raise PiActorCompatibilityError(
                "OpenAI-compatible endpoint exceeded the requested token limit"
            )
        token_details = usage.get("completion_tokens_details")
        if not isinstance(token_details, Mapping):
            raise PiActorCompatibilityError(
                "OpenAI-compatible endpoint omitted reasoning token usage"
            )
        reasoning_tokens = token_details.get("reasoning_tokens")
        if type(reasoning_tokens) is not int:
            raise PiActorCompatibilityError(
                "OpenAI-compatible endpoint omitted reasoning token usage"
            )
        if reasoning_tokens != 0:
            raise PiActorCompatibilityError(
                "OpenAI-compatible endpoint used reasoning while disabled"
            )
        return PiActorResponse(
            text=text_response,
            provider=self.provider,
            model=served_model,
            input_tokens=prompt_tokens,
            output_tokens=output_tokens,
            reasoning_tokens=reasoning_tokens,
            stop_reason=str(choice.get("finish_reason") or ""),
            latency_seconds=latency_seconds,
            provider_identity_evidence=(
                "requested_pi_credential_and_frozen_endpoint; provider_not_echoed"
            ),
            served_provider_reported=False,
            reasoning_tokens_reported=True,
            runtime_purpose=self.runtime_purpose,
        )

    def close(self) -> None:
        """Match the shared actor lifecycle without retaining network state."""

        self._api_key = None


def run_pi_rpc_environment_compatibility_replay(
    *,
    source_rows: Sequence[Mapping[str, Any]],
    partial_rows: Sequence[Mapping[str, Any]],
    actor: Actor,
    branch: AlfWorldBranch,
    output_dir: str | Path,
) -> dict[str, Any]:
    """Reproduce Pi/TextWorld coexistence without creating scientific evidence."""

    if not partial_rows:
        raise ValueError("compatibility replay requires saved partial rows")
    ordered_partial = sorted(partial_rows, key=lambda row: int(row["step_index"]))
    first_partial_step = int(ordered_partial[0]["step_index"])
    ordered_prefix = sorted(
        (row for row in source_rows if int(row["step_index"]) < first_partial_step),
        key=lambda row: int(row["step_index"]),
    )
    rows = (*ordered_prefix, *ordered_partial)
    current = branch.reset()
    model_call_count = 0
    for row in rows:
        step_index = int(row["step_index"])
        if current.text != str(row["observation"]):
            raise ValueError(f"saved observation diverged before step {step_index}")
        if row.get("phase") == "continuation":
            actor.generate(str(row["prompt"]))
            model_call_count += 1
        transition = branch.step_snapshot_only(str(row["environment_action"]))
        if transition.observation.text != str(row["next_observation"]):
            raise ValueError(f"saved observation diverged after step {step_index}")
        current = transition.observation

    result = {
        "status": "completed",
        "diagnostic_only": True,
        "provider": actor.provider,
        "model": actor.model,
        "model_call_count": model_call_count,
        "environment_step_count": len(rows),
        "last_step_index": int(rows[-1]["step_index"]),
    }
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    (output / "result.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return result


def select_runtime_fidelity_tasks(
    tasks: Sequence[AlfworldTask],
    *,
    limit: int,
) -> tuple[AlfworldTask, ...]:
    """Select task-type coverage before filling remaining calibration slots."""

    if limit < 1:
        raise ValueError("task limit must be positive")
    selected: list[AlfworldTask] = []
    seen_types: set[str] = set()
    for task in tasks:
        if task.task_type in seen_types:
            continue
        selected.append(task)
        seen_types.add(task.task_type)
        if len(selected) == limit:
            return tuple(selected)
    for task in tasks:
        if task in selected:
            continue
        selected.append(task)
        if len(selected) == limit:
            break
    return tuple(selected)


def run_runtime_fidelity(
    *,
    tasks: Sequence[AlfworldTask],
    actor: Actor,
    branch_factory: BranchFactory,
    output_dir: str | Path,
    max_steps: int,
    history_length: int,
    environment_seed: int = 1701,
    actor_protocol_id: str = MODEL_COMPATIBLE_NON_THINKING_ALFWORLD_ACTOR_PROTOCOL_ID,
    public_sidecar_builder: Callable[[PublicTransitionEvidence], PublicTransitionSidecar]
    | None = None,
) -> dict[str, Any]:
    """Run Pi in the real policy-parser-environment loop and persist every step."""

    if not tasks:
        raise ValueError("runtime-fidelity calibration requires at least one task")
    if max_steps < 1:
        raise ValueError("max steps must be positive")
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=False)
    states = output / "states"
    states.mkdir()
    rows_path = output / "trajectories.jsonl"
    protocol = build_alfworld_actor_protocol(
        protocol_id=actor_protocol_id,
        history_length=history_length,
        show_admissible_actions=True,
    )

    task_results: list[dict[str, Any]] = []
    with rows_path.open("w", encoding="utf-8") as rows:
        for task_index, task in enumerate(tasks):
            branch = branch_factory(
                task,
                environment_seed + task_index,
                states / task.task_id,
            )
            history: list[G2POActorHistoryStep] = []
            parser_failures = 0
            out_of_menu_actions = 0
            input_tokens = 0
            output_tokens = 0
            try:
                observation = branch.reset()
                task_description = extract_alfworld_task_description(observation.text)
                won = False
                terminal_reason = "max_episode_steps"
                steps = 0
                for step_index in range(max_steps):
                    prompt = protocol.render_prompt(
                        task_description=task_description,
                        current_observation=observation.text,
                        admissible_actions=observation.admissible_actions,
                        history=history,
                    )
                    response = actor.generate(prompt)
                    projection = protocol.project_response(
                        response.text,
                        admissible_actions=observation.admissible_actions,
                    )
                    response_record = _actor_response_record(
                        prompt=prompt,
                        observation=observation.text,
                        admissible_actions=observation.admissible_actions,
                        response=response,
                        projection=projection,
                    )
                    in_menu = bool(response_record["action_in_admissible_menu"])
                    parser_failures += int(not projection.is_action_valid)
                    out_of_menu_actions += int(projection.is_action_valid and not in_menu)
                    input_tokens += response.input_tokens
                    output_tokens += response.output_tokens
                    steps = step_index + 1
                    row: dict[str, Any] = {
                        "task_id": task.task_id,
                        "task_type": task.task_type,
                        "horizon_bucket": task.horizon_bucket,
                        "step_index": step_index,
                        **response_record,
                    }
                    if not projection.is_action_valid or not in_menu:
                        row.update(
                            {
                                "environment_step_executed": False,
                                "next_observation": None,
                                "environment_done": False,
                                "won": False,
                            }
                        )
                        rows.write(json.dumps(row, ensure_ascii=False) + "\n")
                        rows.flush()
                        terminal_reason = "actor_incompatible"
                        break

                    transition = branch.step_snapshot_only(projection.environment_action)
                    row.update(
                        {
                            "environment_step_executed": True,
                            "next_observation": transition.observation.text,
                            "next_admissible_actions": list(
                                transition.observation.admissible_actions
                            ),
                            "environment_done": transition.done,
                            "won": transition.won,
                        }
                    )
                    if public_sidecar_builder is not None:
                        sidecar = public_sidecar_builder(
                            PublicTransitionEvidence(
                                task_id=task.task_id,
                                event_id=f"{task.task_id}:step-{step_index:04d}",
                                sequence=step_index,
                                action=projection.environment_action,
                                observation=observation.text,
                                next_observation=transition.observation.text,
                                admissible_actions=tuple(observation.admissible_actions),
                                next_admissible_actions=tuple(
                                    transition.observation.admissible_actions
                                ),
                            )
                        )
                        if not isinstance(sidecar, PublicTransitionSidecar):
                            raise TypeError(
                                "live sidecar builder must return PublicTransitionSidecar"
                            )
                        if sidecar.task_id != task.task_id or sidecar.sequence != step_index:
                            raise ValueError("public sidecar identity does not match runtime step")
                        row["public_transition_sidecar"] = sidecar.to_record()
                    rows.write(json.dumps(row, ensure_ascii=False) + "\n")
                    rows.flush()
                    history.append(
                        G2POActorHistoryStep(
                            observation=observation.text,
                            environment_action=projection.environment_action,
                        )
                    )
                    observation = transition.observation
                    if transition.done or transition.won:
                        won = bool(transition.won)
                        terminal_reason = "success" if won else "environment_done"
                        break
                task_results.append(
                    {
                        "task_id": task.task_id,
                        "task_type": task.task_type,
                        "horizon_bucket": task.horizon_bucket,
                        "steps": steps,
                        "won": won,
                        "terminal_reason": terminal_reason,
                        "parser_failures": parser_failures,
                        "out_of_menu_actions": out_of_menu_actions,
                        "input_tokens": input_tokens,
                        "output_tokens": output_tokens,
                    }
                )
            except PiCliActorError as error:
                # A single actor timeout or transport failure must not kill the
                # whole planning cohort: record the task as an infra failure and
                # continue with the next task. Fidelity analysis then separates
                # actor_incompatible infra failures from natural failures.
                infra_reason = str(error)
                rows.write(
                    json.dumps(
                        {
                            "task_id": task.task_id,
                            "task_type": task.task_type,
                            "horizon_bucket": task.horizon_bucket,
                            "infra_failure": True,
                            "infra_reason": infra_reason,
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
                rows.flush()
                task_results.append(
                    {
                        "task_id": task.task_id,
                        "task_type": task.task_type,
                        "horizon_bucket": task.horizon_bucket,
                        "steps": steps,
                        "won": False,
                        "terminal_reason": "infra_failure",
                        "parser_failures": 0,
                        "out_of_menu_actions": 0,
                        "input_tokens": 0,
                        "output_tokens": 0,
                        "infra_reason": infra_reason,
                    }
                )
            finally:
                branch.close()

    result = {
        "status": "completed",
        "purpose": "agent_runtime_fidelity_calibration",
        "scientific_scope": (
            "Verifies the declared Pi actor execution chain; does not evaluate tracers."
        ),
        "provider": actor.provider,
        "model": actor.model,
        "actor_transport": getattr(actor, "transport", "custom"),
        "actor_protocol_id": actor_protocol_id,
        "actor_contract": (
            f"action_only_non_thinking_history{history_length}_menu_visible"
            if actor_protocol_id == MODEL_COMPATIBLE_NON_THINKING_ALFWORLD_ACTOR_PROTOCOL_ID
            else (
                f"bare_menu_index_non_thinking_history{history_length}_menu_visible"
                if actor_protocol_id == BARE_MENU_INDEX_NON_THINKING_ALFWORLD_ACTOR_PROTOCOL_ID
                else (
                    f"native_menu_choice_non_thinking_history{history_length}_menu_visible"
                    if actor_protocol_id
                    == NATIVE_MENU_CHOICE_NON_THINKING_ALFWORLD_ACTOR_PROTOCOL_ID
                    else (
                        f"menu_index_non_thinking_history{history_length}_menu_visible"
                        if protocol.response_contract.action_representation == "menu_index"
                        else (
                            f"unique_action_tag_non_thinking_history{history_length}_menu_visible"
                            if actor_protocol_id
                            == UNIQUE_ACTION_TAG_NON_THINKING_ALFWORLD_ACTOR_PROTOCOL_ID
                            else actor_protocol_id
                        )
                    )
                )
            )
        ),
        "continuation": "closed_loop_model_replanning",
        "task_count": len(task_results),
        "success_count": sum(int(item["won"]) for item in task_results),
        "parser_failure_count": sum(item["parser_failures"] for item in task_results),
        "out_of_menu_action_count": sum(item["out_of_menu_actions"] for item in task_results),
        "infra_failure_count": sum(
            1 for item in task_results if item.get("terminal_reason") == "infra_failure"
        ),
        "task_results": task_results,
    }
    if actor_protocol_id == UNIQUE_ACTION_TAG_NON_THINKING_ALFWORLD_ACTOR_PROTOCOL_ID:
        # The v2 cell must carry its parser identity in every run artifact.
        result["actor_parser_id"] = "alfworld_unique_action_tag_parser_v1"
        result["actor_parser_source_sha256"] = unique_action_tag_parser_sha256()
    (output / "result.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return result


def _run_natural_failure_sequence_intervention(
    *,
    task: AlfworldTask,
    source_rows: Sequence[Mapping[str, Any]],
    cause_step: int,
    arm_action_sequences: Mapping[str, Sequence[str]],
    actor: Actor,
    branch_factory: BranchFactory,
    output_dir: str | Path,
    repeats: int,
    max_total_steps: int,
    history_length: int,
    actor_protocol_id: str = MODEL_COMPATIBLE_NON_THINKING_ALFWORLD_ACTOR_PROTOCOL_ID,
    environment_seed: int = 1701,
    purpose: str,
    result_metadata: Mapping[str, Any] | None = None,
    expected_post_intervention_observation: str | None = None,
) -> dict[str, Any]:
    """Execute equal-length interventions before closed-loop actor continuation."""

    if repeats < 1:
        raise ValueError("repeats must be positive")
    if cause_step < 0 or cause_step >= len(source_rows):
        raise ValueError("cause step is outside the source trajectory")
    if max_total_steps <= cause_step:
        raise ValueError("total step budget must extend beyond the cause step")
    for index, row in enumerate(source_rows[: cause_step + 1]):
        if int(row["step_index"]) != index:
            raise ValueError("source trajectory steps must be contiguous")

    source_step = source_rows[cause_step]
    arms = tuple(
        (str(name), tuple(str(action) for action in actions))
        for name, actions in arm_action_sequences.items()
    )
    if len(arms) != 3:
        raise ValueError("the intervention requires exactly three arms")
    sequence_lengths = {len(actions) for _, actions in arms}
    if sequence_lengths == {0} or len(sequence_lengths) != 1:
        raise ValueError("all intervention arms must have the same positive length")
    if len({name for name, _ in arms}) != 3 or len({actions for _, actions in arms}) != 3:
        raise ValueError("intervention arm names and action sequences must be distinct")
    intervention_length = next(iter(sequence_lengths))
    if max_total_steps <= cause_step + intervention_length:
        raise ValueError("total step budget must include closed-loop continuation")

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=False)
    states = output / "states"
    states.mkdir()
    rows_path = output / "trajectories.jsonl"
    protocol = build_alfworld_actor_protocol(
        protocol_id=actor_protocol_id,
        history_length=history_length,
        show_admissible_actions=True,
    )
    task_description = extract_alfworld_task_description(str(source_rows[0]["observation"]))
    arm_results: list[dict[str, Any]] = []

    with rows_path.open("w", encoding="utf-8") as output_rows:
        for repeat_index in range(repeats):
            offset = repeat_index % len(arms)
            ordered_arms = arms[offset:] + arms[:offset]
            for arm_name, arm_actions in ordered_arms:
                branch = branch_factory(
                    task,
                    environment_seed,
                    states / f"repeat_{repeat_index}" / arm_name,
                )
                history: list[G2POActorHistoryStep] = []
                parser_failures = 0
                out_of_menu_actions = 0
                try:
                    observation = branch.reset()
                    state_fact_history = [list(observation.facts)]
                    for prefix_row in source_rows[:cause_step]:
                        expected_actions = tuple(prefix_row["admissible_actions"])
                        if (
                            observation.text != str(prefix_row["observation"])
                            or observation.admissible_actions != expected_actions
                        ):
                            raise ValueError("source prefix did not reconstruct the same state")
                        prefix_action = str(prefix_row["environment_action"])
                        if prefix_action not in observation.admissible_actions:
                            raise ValueError("source prefix contains an inadmissible action")
                        history.append(
                            G2POActorHistoryStep(
                                observation=observation.text,
                                environment_action=prefix_action,
                            )
                        )
                        observation = branch.step_snapshot_only(prefix_action).observation
                        state_fact_history.append(list(observation.facts))

                    expected_cause_actions = tuple(source_step["admissible_actions"])
                    prefix_state_verified = (
                        observation.text == str(source_step["observation"])
                        and observation.admissible_actions == expected_cause_actions
                    )
                    if not prefix_state_verified:
                        raise ValueError("intervention pre-action state did not match the source")
                    transition = None
                    first_next_observation = ""
                    for intervention_offset, arm_action in enumerate(arm_actions):
                        if arm_action not in observation.admissible_actions:
                            raise ValueError(
                                f"{arm_name} action {intervention_offset} is not admissible"
                            )
                        before_text = observation.text
                        before_actions = observation.admissible_actions
                        transition = branch.step(arm_action)
                        if intervention_offset == 0:
                            first_next_observation = transition.observation.text
                        history.append(
                            G2POActorHistoryStep(
                                observation=before_text,
                                environment_action=arm_action,
                            )
                        )
                        output_rows.write(
                            json.dumps(
                                {
                                    "repeat_index": repeat_index,
                                    "arm": arm_name,
                                    "phase": "intervention",
                                    "intervention_offset": intervention_offset,
                                    "step_index": cause_step + intervention_offset,
                                    "observation": before_text,
                                    "admissible_actions": list(before_actions),
                                    "model_response": None,
                                    "environment_action": arm_action,
                                    "parser_valid": True,
                                    "parser_failure_reasons": [],
                                    "action_in_admissible_menu": True,
                                    "environment_step_executed": True,
                                    "next_observation": transition.observation.text,
                                    "next_facts": list(transition.observation.facts),
                                    "next_admissible_actions": list(
                                        transition.observation.admissible_actions
                                    ),
                                    "environment_done": transition.done,
                                    "won": transition.won,
                                },
                                ensure_ascii=False,
                            )
                            + "\n"
                        )
                        output_rows.flush()
                        observation = transition.observation
                        state_fact_history.append(list(observation.facts))
                        if (transition.done or transition.won) and (
                            intervention_offset + 1 < intervention_length
                        ):
                            raise ValueError(f"{arm_name} terminated before its sequence ended")
                    assert transition is not None
                    executed_steps = cause_step + intervention_length
                    post_intervention_observation = observation.text
                    post_intervention_actions = list(observation.admissible_actions)
                    post_intervention_facts = list(observation.facts)
                    checkpoint_verified = (
                        expected_post_intervention_observation is None
                        or post_intervention_observation == expected_post_intervention_observation
                    )
                    if not checkpoint_verified:
                        raise ValueError(f"{arm_name} did not reach the expected checkpoint")
                    actor_incompatible = False
                    actor_calls_after_intervention = 0

                    while (
                        executed_steps < max_total_steps
                        and not transition.done
                        and not transition.won
                    ):
                        prompt = protocol.render_prompt(
                            task_description=task_description,
                            current_observation=observation.text,
                            admissible_actions=observation.admissible_actions,
                            history=history,
                        )
                        response = actor.generate(prompt)
                        actor_calls_after_intervention += 1
                        projection = protocol.project_response(
                            response.text,
                            admissible_actions=observation.admissible_actions,
                        )
                        before_text = observation.text
                        row: dict[str, Any] = {
                            "repeat_index": repeat_index,
                            "arm": arm_name,
                            "phase": "continuation",
                            "step_index": executed_steps,
                            **_actor_response_record(
                                prompt=prompt,
                                observation=before_text,
                                admissible_actions=observation.admissible_actions,
                                response=response,
                                projection=projection,
                            ),
                        }
                        in_menu = bool(row["action_in_admissible_menu"])
                        parser_failures += int(not projection.is_action_valid)
                        out_of_menu_actions += int(projection.is_action_valid and not in_menu)
                        if not projection.is_action_valid or not in_menu:
                            row.update(
                                {
                                    "environment_step_executed": False,
                                    "next_observation": None,
                                    "next_facts": None,
                                    "next_admissible_actions": None,
                                    "environment_done": False,
                                    "won": False,
                                }
                            )
                            output_rows.write(json.dumps(row, ensure_ascii=False) + "\n")
                            output_rows.flush()
                            actor_incompatible = True
                            break

                        transition = branch.step_snapshot_only(projection.environment_action)
                        row.update(
                            {
                                "environment_step_executed": True,
                                "next_observation": transition.observation.text,
                                "next_facts": list(transition.observation.facts),
                                "next_admissible_actions": list(
                                    transition.observation.admissible_actions
                                ),
                                "environment_done": transition.done,
                                "won": transition.won,
                            }
                        )
                        output_rows.write(json.dumps(row, ensure_ascii=False) + "\n")
                        output_rows.flush()
                        history.append(
                            G2POActorHistoryStep(
                                observation=before_text,
                                environment_action=projection.environment_action,
                            )
                        )
                        observation = transition.observation
                        state_fact_history.append(list(observation.facts))
                        executed_steps += 1

                    arm_results.append(
                        {
                            "repeat_index": repeat_index,
                            "arm": arm_name,
                            "applied_action": arm_actions[0] if len(arm_actions) == 1 else None,
                            "applied_actions": list(arm_actions),
                            "prefix_state_verified": prefix_state_verified,
                            "immediate_next_observation": first_next_observation,
                            "post_intervention_observation": post_intervention_observation,
                            "post_intervention_admissible_actions": post_intervention_actions,
                            "post_intervention_facts": post_intervention_facts,
                            "checkpoint_verified": checkpoint_verified,
                            "actor_calls_after_checkpoint": (
                                actor_calls_after_intervention
                                if expected_post_intervention_observation is not None
                                else None
                            ),
                            "state_fact_history": state_fact_history,
                            "steps": executed_steps,
                            "won": bool(transition.won),
                            "terminal_reason": (
                                "actor_incompatible"
                                if actor_incompatible
                                else "success"
                                if transition.won
                                else "environment_done"
                                if transition.done
                                else "max_total_steps"
                            ),
                            "parser_failures": parser_failures,
                            "out_of_menu_actions": out_of_menu_actions,
                        }
                    )
                finally:
                    branch.close()

    result = {
        "status": "completed",
        "purpose": purpose,
        "task_id": task.task_id,
        "task_type": task.task_type,
        "cause_step": cause_step,
        "arm_action_sequences": {name: list(actions) for name, actions in arms},
        "continuation": "closed_loop_model_replanning",
        "provider": actor.provider,
        "model": actor.model,
        "actor_transport": getattr(actor, "transport", "custom"),
        "actor_protocol_id": actor_protocol_id,
        "repeats": repeats,
        "max_total_steps": max_total_steps,
        "arm_results": arm_results,
        "actor_incompatible_arm_count": sum(
            arm["terminal_reason"] == "actor_incompatible" for arm in arm_results
        ),
    }
    if result_metadata:
        result.update(result_metadata)
    (output / "result.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return result


def run_natural_failure_intervention(
    *,
    task: AlfworldTask,
    source_rows: Sequence[Mapping[str, Any]],
    cause_step: int,
    replacement_action: str,
    sham_action: str,
    actor: Actor,
    branch_factory: BranchFactory,
    output_dir: str | Path,
    repeats: int,
    max_total_steps: int,
    history_length: int,
    actor_protocol_id: str = MODEL_COMPATIBLE_NON_THINKING_ALFWORLD_ACTOR_PROTOCOL_ID,
    environment_seed: int = 1701,
) -> dict[str, Any]:
    """Compare original, replacement, and sham from one natural failure state."""

    if cause_step < 0 or cause_step >= len(source_rows):
        raise ValueError("cause step is outside the source trajectory")
    original_action = str(source_rows[cause_step]["environment_action"])
    return _run_natural_failure_sequence_intervention(
        task=task,
        source_rows=source_rows,
        cause_step=cause_step,
        arm_action_sequences={
            "original": (original_action,),
            "replacement": (replacement_action,),
            "matched_sham": (sham_action,),
        },
        actor=actor,
        branch_factory=branch_factory,
        output_dir=output_dir,
        repeats=repeats,
        max_total_steps=max_total_steps,
        history_length=history_length,
        actor_protocol_id=actor_protocol_id,
        environment_seed=environment_seed,
        purpose="natural_failure_three_arm_intervention",
        result_metadata={
            "source_action": original_action,
            "replacement_action": replacement_action,
            "matched_sham_action": sham_action,
        },
    )


def run_natural_failure_sequence_intervention(
    *,
    task: AlfworldTask,
    source_rows: Sequence[Mapping[str, Any]],
    cause_step: int,
    arm_action_sequences: Mapping[str, Sequence[str]],
    actor: Actor,
    branch_factory: BranchFactory,
    output_dir: str | Path,
    repeats: int,
    max_total_steps: int,
    history_length: int,
    actor_protocol_id: str = MODEL_COMPATIBLE_NON_THINKING_ALFWORLD_ACTOR_PROTOCOL_ID,
    environment_seed: int = 1701,
) -> dict[str, Any]:
    """Compare matched action sequences before the same actor replans."""

    return _run_natural_failure_sequence_intervention(
        task=task,
        source_rows=source_rows,
        cause_step=cause_step,
        arm_action_sequences=arm_action_sequences,
        actor=actor,
        branch_factory=branch_factory,
        output_dir=output_dir,
        repeats=repeats,
        max_total_steps=max_total_steps,
        history_length=history_length,
        actor_protocol_id=actor_protocol_id,
        environment_seed=environment_seed,
        purpose="natural_failure_matched_sequence_intervention",
    )


def run_checkpoint_matched_intervention(
    *,
    task: AlfworldTask,
    source_rows: Sequence[Mapping[str, Any]],
    cause_step: int,
    arm_actions: Mapping[str, str],
    checkpoint_actions: Sequence[str],
    expected_checkpoint_observation: str | None,
    target_action: str,
    debt_object: str,
    branch_factory: BranchFactory,
    output_dir: str | Path,
    environment_seed: int = 1701,
) -> dict[str, Any]:
    """Replay three interventions to one evaluator-owned downstream checkpoint."""

    expected_arms = ("original", "replacement", "matched_sham")
    if tuple(arm_actions) != expected_arms:
        raise ValueError("checkpoint replay requires original, replacement, matched_sham")
    if not checkpoint_actions:
        raise ValueError("checkpoint replay requires at least one checkpoint action")
    if cause_step < 0 or cause_step >= len(source_rows):
        raise ValueError("cause step is outside the source trajectory")

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=False)
    states = output / "states"
    states.mkdir()
    cause_row = source_rows[cause_step]
    arm_results: list[dict[str, Any]] = []

    def holds_debt_object(facts: Sequence[Any]) -> bool:
        target = debt_object.strip().casefold()
        return any(
            isinstance(fact, Mapping)
            and str(fact.get("name", "")).strip().casefold() == "holds"
            and target in {str(value).strip().casefold() for value in fact.get("args", ())}
            for fact in facts
        )

    for arm_name in expected_arms:
        branch = branch_factory(task, environment_seed, states / arm_name)
        try:
            observation = branch.reset()
            for index, source_row in enumerate(source_rows[:cause_step]):
                if int(source_row["step_index"]) != index:
                    raise ValueError("source trajectory steps must be contiguous")
                if observation.text != str(source_row["observation"]):
                    raise ValueError("source prefix did not reconstruct the same state")
                source_action = str(source_row["environment_action"])
                if source_action not in observation.admissible_actions:
                    raise ValueError("source prefix contains an inadmissible action")
                observation = branch.step_snapshot_only(source_action).observation

            prefix_state_verified = observation.text == str(
                cause_row["observation"]
            ) and observation.admissible_actions == tuple(cause_row["admissible_actions"])
            if not prefix_state_verified:
                raise ValueError("intervention pre-action state did not match the source")

            intervention_action = str(arm_actions[arm_name])
            if intervention_action not in observation.admissible_actions:
                raise ValueError(f"{arm_name} intervention action is not admissible")
            observation = branch.step_snapshot_only(intervention_action).observation
            for checkpoint_action in checkpoint_actions:
                if checkpoint_action not in observation.admissible_actions:
                    raise ValueError(
                        f"{arm_name} checkpoint action is not admissible: {checkpoint_action}"
                    )
                observation = branch.step_snapshot_only(str(checkpoint_action)).observation

            checkpoint_observation = observation.text
            checkpoint_facts = list(observation.facts)
            checkpoint_admissible_actions = list(observation.admissible_actions)
            target_action_available = target_action in observation.admissible_actions
            target_probe_observation = None
            target_probe_facts = None
            target_probe_effect_observed = False
            if target_action_available:
                target_transition = branch.step_snapshot_only(target_action)
                target_probe_observation = target_transition.observation.text
                target_probe_facts = list(target_transition.observation.facts)
                target_probe_effect_observed = (
                    target_probe_observation != checkpoint_observation
                    or target_probe_facts != checkpoint_facts
                )

            arm_results.append(
                {
                    "arm": arm_name,
                    "prefix_state_verified": prefix_state_verified,
                    "intervention_action": intervention_action,
                    "checkpoint_actions": list(checkpoint_actions),
                    "checkpoint_observation": checkpoint_observation,
                    "checkpoint_facts": checkpoint_facts,
                    "checkpoint_admissible_actions": checkpoint_admissible_actions,
                    "holds_debt_object": holds_debt_object(checkpoint_facts),
                    "target_action_available": target_action_available,
                    "target_probe_executed": target_action_available,
                    "target_probe_observation": target_probe_observation,
                    "target_probe_facts": target_probe_facts,
                    "target_probe_effect_observed": target_probe_effect_observed,
                }
            )
        finally:
            branch.close()

    checkpoint_matched = len({str(arm["checkpoint_observation"]) for arm in arm_results}) == 1
    checkpoint_identity_verified = checkpoint_matched and (
        expected_checkpoint_observation is None
        or arm_results[0]["checkpoint_observation"] == expected_checkpoint_observation
    )
    arms_by_name = {str(arm["arm"]): arm for arm in arm_results}
    mechanism_supported = checkpoint_identity_verified and all(
        (
            arms_by_name["original"]["holds_debt_object"],
            not arms_by_name["replacement"]["holds_debt_object"],
            not arms_by_name["matched_sham"]["holds_debt_object"],
            not arms_by_name["original"]["target_action_available"],
            arms_by_name["replacement"]["target_action_available"],
            arms_by_name["matched_sham"]["target_action_available"],
            arms_by_name["replacement"]["target_probe_effect_observed"],
            arms_by_name["matched_sham"]["target_probe_effect_observed"],
        )
    )
    result = {
        "status": "completed",
        "purpose": "checkpoint_matched_state_debt_mechanism_replay",
        "task_id": task.task_id,
        "task_type": task.task_type,
        "cause_step": cause_step,
        "arm_actions": dict(arm_actions),
        "checkpoint_actions": list(checkpoint_actions),
        "expected_checkpoint_observation": expected_checkpoint_observation,
        "checkpoint_contract": (
            "cross_arm_public_parity"
            if expected_checkpoint_observation is None
            else "frozen_expected_observation"
        ),
        "target_action": target_action,
        "debt_object": debt_object,
        "continuation": "none_evaluator_checkpoint_only",
        "design_valid": checkpoint_identity_verified,
        "classification": (
            "checkpoint_not_matched"
            if not checkpoint_matched
            else "checkpoint_identity_mismatch"
            if not checkpoint_identity_verified
            else "verified_inventory_affordance_debt"
            if mechanism_supported
            else "mechanism_not_supported"
        ),
        "mechanism_supported": mechanism_supported,
        "arm_results": arm_results,
    }
    (output / "result.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return result


def run_checkpoint_closed_loop_intervention(
    *,
    task: AlfworldTask,
    source_rows: Sequence[Mapping[str, Any]],
    cause_step: int,
    arm_actions: Mapping[str, str],
    checkpoint_actions: Sequence[str],
    expected_checkpoint_observation: str,
    actor: Actor,
    branch_factory: BranchFactory,
    output_dir: str | Path,
    repeats: int,
    max_actor_steps: int,
    history_length: int,
    actor_protocol_id: str = MODEL_COMPATIBLE_NON_THINKING_ALFWORLD_ACTOR_PROTOCOL_ID,
    environment_seed: int = 1701,
) -> dict[str, Any]:
    """Hand the same actor control after three arms reach one known checkpoint."""

    expected_arms = ("original", "replacement", "matched_sham")
    if tuple(arm_actions) != expected_arms:
        raise ValueError("checkpoint continuation requires original, replacement, matched_sham")
    if not checkpoint_actions:
        raise ValueError("checkpoint continuation requires at least one checkpoint action")
    if max_actor_steps < 10:
        raise ValueError("checkpoint continuation requires at least ten actor steps")

    arm_action_sequences = {
        arm_name: (str(arm_actions[arm_name]), *tuple(checkpoint_actions))
        for arm_name in expected_arms
    }
    intervention_length = 1 + len(checkpoint_actions)
    return _run_natural_failure_sequence_intervention(
        task=task,
        source_rows=source_rows,
        cause_step=cause_step,
        arm_action_sequences=arm_action_sequences,
        actor=actor,
        branch_factory=branch_factory,
        output_dir=output_dir,
        repeats=repeats,
        max_total_steps=cause_step + intervention_length + max_actor_steps,
        history_length=history_length,
        actor_protocol_id=actor_protocol_id,
        environment_seed=environment_seed,
        purpose="checkpoint_matched_closed_loop_repair_utility",
        result_metadata={
            "design_valid": True,
            "checkpoint_actions": list(checkpoint_actions),
            "expected_checkpoint_observation": expected_checkpoint_observation,
            "max_actor_steps": max_actor_steps,
        },
        expected_post_intervention_observation=expected_checkpoint_observation,
    )
