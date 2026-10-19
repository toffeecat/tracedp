"""Actor-visible filesystem state for Terminal-Bench causal replay gates."""

from __future__ import annotations

import base64
import binascii
import re
import tomllib
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Literal

_ARM_NAMES = ("original", "replacement", "sham")
_POSIX_MODE = re.compile(r"[0-7]{3,4}")


def validate_configured_resource_path(path: str) -> str:
    """Accept an absolute non-root path declared by trusted runtime configuration."""

    normalized = PurePosixPath(path)
    if not normalized.is_absolute() or normalized == PurePosixPath("/"):
        raise ValueError(f"terminal configured resource must use an absolute non-root path: {path}")
    if ".." in normalized.parts:
        raise ValueError(f"terminal configured resource path cannot traverse parents: {path}")
    return str(normalized)


def validate_actor_visible_path(path: str) -> str:
    """Accept only absolute task-workspace paths visible to the terminal actor."""
    normalized = PurePosixPath(validate_configured_resource_path(path))
    if normalized.parts[:2] != ("/", "app"):
        raise ValueError(f"terminal public state must use an actor-visible /app path: {path}")
    return str(normalized)


@dataclass(frozen=True)
class PublicFileObservation:
    """One exact actor-visible file observation.

    Bytes stay in memory for parity checks. They enter only explicit exact-state
    receipts used to prove replay and cross-arm equivalence.
    """

    path: str
    exists: bool
    file_type: str | None
    mode: str | None
    content: bytes | None

    def __post_init__(self) -> None:
        object.__setattr__(self, "path", validate_actor_visible_path(self.path))

    @property
    def size_bytes(self) -> int | None:
        return len(self.content) if self.content is not None else None


def is_exact_restorable_public_file(observation: PublicFileObservation) -> bool:
    """Return whether a bounded public path can be restored without inference."""

    if not observation.exists:
        return (
            observation.file_type is None
            and observation.mode is None
            and observation.content is None
        )
    return (
        observation.file_type == "regular file"
        and observation.mode is not None
        and _POSIX_MODE.fullmatch(observation.mode) is not None
        and observation.content is not None
    )


def public_file_observation_payload(
    observation: PublicFileObservation,
) -> dict[str, object]:
    """Serialize exact actor-visible file bytes for replay or observation."""

    return {
        "path": observation.path,
        "exists": observation.exists,
        "file_type": observation.file_type,
        "mode": observation.mode,
        "content_base64": (
            base64.b64encode(observation.content).decode("ascii")
            if observation.content is not None
            else None
        ),
    }


def decode_exact_public_file_state(rows: object) -> tuple[PublicFileObservation, ...]:
    """Decode one exact, regular-file-only verifier intervention state."""

    if not isinstance(rows, list) or not rows:
        raise ValueError("terminal intervention state must be a non-empty list")
    observations = []
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError("terminal intervention state rows must be objects")
        encoded = row.get("content_base64")
        try:
            content = (
                base64.b64decode(encoded, validate=True)
                if isinstance(encoded, str)
                else None
            )
        except (binascii.Error, ValueError) as error:
            raise ValueError("terminal intervention contains invalid file bytes") from error
        exists = row.get("exists")
        if type(exists) is not bool:
            raise ValueError("terminal intervention has invalid existence state")
        observation = PublicFileObservation(
            path=str(row.get("path", "")),
            exists=exists,
            file_type=(
                str(row["file_type"]) if row.get("file_type") is not None else None
            ),
            mode=str(row["mode"]) if row.get("mode") is not None else None,
            content=content,
        )
        if not is_exact_restorable_public_file(observation):
            raise ValueError("terminal intervention supports exact regular files only")
        observations.append(observation)
    paths = [observation.path for observation in observations]
    if len(paths) != len(set(paths)):
        raise ValueError("terminal intervention paths must be unique")
    return tuple(observations)


def compact_public_file_state(
    observations: tuple[PublicFileObservation, ...],
) -> list[dict[str, object]]:
    """Remove file bytes while preserving readback evidence for run receipts."""

    return [
        {
            "path": observation.path,
            "exists": observation.exists,
            "file_type": observation.file_type,
            "mode": observation.mode,
            "size_bytes": observation.size_bytes,
        }
        for observation in observations
    ]


@dataclass(frozen=True)
class FilesystemParityAssessment:
    checkpoint_verified: bool
    invariants_preserved: bool
    arm_outputs_distinct: bool
    classification: Literal[
        "filesystem_restore_pass",
        "inconclusive_checkpoint_mismatch",
        "inconclusive_invariant_regression",
        "inconclusive_arm_outputs_not_distinct",
    ]


@dataclass(frozen=True)
class TerminalFilesystemRestoreSpec:
    task: str
    task_dir: Path
    docker_image: str
    invariant_paths: tuple[str, ...]
    output_path: str


def load_terminal_filesystem_restore_specs(
    rows: list[Mapping[str, object]],
    *,
    terminal_bench_root: Path,
) -> tuple[TerminalFilesystemRestoreSpec, ...]:
    """Bind restore cases to official actor-visible task workspace paths."""
    specs = []
    for row in rows:
        task = str(row["task"])
        task_dir = terminal_bench_root / "tasks" / task
        instruction = (task_dir / "instruction.md").read_text()
        dockerfile_path = task_dir / "environment" / "Dockerfile"
        dockerfile = dockerfile_path.read_text() if dockerfile_path.exists() else ""
        task_config = tomllib.loads((task_dir / "task.toml").read_text())
        invariant_paths = tuple(
            validate_actor_visible_path(str(path)) for path in row["invariant_paths"]
        )
        output_path = validate_actor_visible_path(str(row["output_path"]))
        for path in invariant_paths:
            if path not in instruction and path not in dockerfile:
                raise ValueError(f"public input has no official task provenance: {path}")
        if output_path not in instruction:
            raise ValueError(f"public output is not declared in task instruction: {output_path}")
        specs.append(
            TerminalFilesystemRestoreSpec(
                task=task,
                task_dir=task_dir,
                docker_image=str(task_config["environment"]["docker_image"]),
                invariant_paths=invariant_paths,
                output_path=output_path,
            )
        )
    return tuple(specs)


def _by_path(state: tuple[PublicFileObservation, ...]) -> dict[str, PublicFileObservation]:
    return {observation.path: observation for observation in state}


def public_filesystem_equal(
    left: tuple[PublicFileObservation, ...],
    right: tuple[PublicFileObservation, ...],
) -> bool:
    """Compare actor-visible file metadata and bytes exactly."""
    return _by_path(left) == _by_path(right)


def assess_three_arm_filesystem_parity(
    *,
    checkpoint: tuple[PublicFileObservation, ...],
    arm_starts: Mapping[str, tuple[PublicFileObservation, ...]],
    arm_ends: Mapping[str, tuple[PublicFileObservation, ...]],
    invariant_paths: tuple[str, ...],
    output_path: str,
) -> FilesystemParityAssessment:
    """Assess whether three filesystem arms branch from one exact checkpoint."""
    output_path = validate_actor_visible_path(output_path)
    invariant_paths = tuple(validate_actor_visible_path(path) for path in invariant_paths)

    checkpoint_verified = all(
        public_filesystem_equal(checkpoint, arm_starts[name]) for name in _ARM_NAMES
    )
    checkpoint_by_path = _by_path(checkpoint)
    invariants_preserved = all(
        _by_path(arm_ends[name]).get(path) == checkpoint_by_path.get(path)
        for name in _ARM_NAMES
        for path in invariant_paths
    )
    output_observations = [_by_path(arm_ends[name]).get(output_path) for name in _ARM_NAMES]
    arm_outputs_distinct = (
        all(observation is not None and observation.exists for observation in output_observations)
        and len({observation.content for observation in output_observations if observation}) == 3
    )

    if not checkpoint_verified:
        classification = "inconclusive_checkpoint_mismatch"
    elif not invariants_preserved:
        classification = "inconclusive_invariant_regression"
    elif not arm_outputs_distinct:
        classification = "inconclusive_arm_outputs_not_distinct"
    else:
        classification = "filesystem_restore_pass"

    return FilesystemParityAssessment(
        checkpoint_verified=checkpoint_verified,
        invariants_preserved=invariants_preserved,
        arm_outputs_distinct=arm_outputs_distinct,
        classification=classification,
    )
