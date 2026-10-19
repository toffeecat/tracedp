"""Shared ALFWorld exact-state capability (snapshot / restore / verified step).

Owned outside GraphGPO study evidence so ``training`` and ``studies`` cross one
seam. GraphGPO-specific study APIs live in ``lh_platform.studies.graphgpo``;
native credit in ``training.graphgpo``; prompt/recipe adapter in
``methods.graphgpo``.

This module owns the method-neutral contract used by GRPO, G2PO, GraphGPO, and
verified-state experiments: snapshot a pre-action worker state, restore it,
replay exactly one semantic action, and compare the projected environment
outcome.
"""

from __future__ import annotations

import copy
import hashlib
import json
import pickle
from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any
from uuid import uuid4

ALFWORLD_STATE_CAPABILITY_SCHEMA = "lh_alfworld_exact_state_capability/v1"
ALFWORLD_RESTORE_RECEIPT_SCHEMA = "lh_alfworld_state_restore_receipt/v1"
ALFWORLD_STATE_ARTIFACT_SCHEMA = "lh_alfworld_exact_state_artifact/v1"
ALFWORLD_STATE_ARTIFACT_MANIFEST_SCHEMA = "lh_alfworld_state_artifact_manifest/v1"
ALFWORLD_PHYSICAL_CALL_COUNTER_SCHEMA = "lh_alfworld_physical_call_counters/v1"
ALFWORLD_REPLAY_BACKEND = "textworld_reset_replay/v1"
REQUIRED_RESTORE_COMPARISONS = ("observation", "reward", "done", "won", "wrapper_state")
DEFAULT_WRAPPER_FIELDS = (
    "history_items",
    "location",
    "holding",
    "item_location",
    "history_list",
    "done",
)


class AlfWorldStateError(RuntimeError):
    """Raised when an exact state cannot be snapshotted or restored."""


@dataclass(frozen=True)
class AlfWorldStateSnapshot:
    schema: str
    state_snapshot_sha256: str
    payload: bytes
    wrapper_fields: tuple[str, ...]


@dataclass(frozen=True)
class AlfWorldRestoreReceipt:
    schema: str
    state_snapshot_sha256: str
    action: str
    comparisons: Mapping[str, bool]
    restore_verified: bool
    first_outcome: Mapping[str, Any]
    replay_outcome: Mapping[str, Any]

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload.pop("first_outcome", None)
        payload.pop("replay_outcome", None)
        payload["first_outcome_sha256"] = _canonical_sha256(self.first_outcome)
        payload["replay_outcome_sha256"] = _canonical_sha256(self.replay_outcome)
        return payload


@dataclass(frozen=True)
class AlfWorldStateArtifact:
    """Content-addressed snapshot evidence kept outside the training JSONL."""

    schema: str
    state_snapshot_sha256: str
    payload_path: str
    payload_size_bytes: int
    wrapper_fields: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


_PHYSICAL_COUNTER_FIELDS = (
    "source_worker_reset_calls",
    "state_restore_calls",
    "batch_environment_close_calls",
    "batch_environment_load_calls",
    "batch_environment_reset_calls",
    "prefix_replay_environment_step_calls",
    "rollout_first_worker_step_calls",
    "rollout_verification_worker_step_calls",
    "snapshot_only_worker_step_calls",
    "unverified_worker_step_calls",
    "candidate_probe_first_worker_step_calls",
    "candidate_probe_replay_worker_step_calls",
    "logical_semantic_action_calls",
)


@dataclass(frozen=True)
class AlfWorldPhysicalCallCounters:
    """Cumulative logical-action and physical TextWorld call accounting.

    A source worker reset and a direct batch reset are separate physical reset
    calls. Worker actions and prefix-replay actions are likewise separate. The
    logical semantic-action count excludes candidate restore probes, so call
    efficiency cannot silently equate a verified transition with one physical
    environment step.
    """

    schema: str = ALFWORLD_PHYSICAL_CALL_COUNTER_SCHEMA
    source_worker_reset_calls: int = 0
    state_restore_calls: int = 0
    batch_environment_close_calls: int = 0
    batch_environment_load_calls: int = 0
    batch_environment_reset_calls: int = 0
    prefix_replay_environment_step_calls: int = 0
    rollout_first_worker_step_calls: int = 0
    rollout_verification_worker_step_calls: int = 0
    snapshot_only_worker_step_calls: int = 0
    unverified_worker_step_calls: int = 0
    candidate_probe_first_worker_step_calls: int = 0
    candidate_probe_replay_worker_step_calls: int = 0
    logical_semantic_action_calls: int = 0

    def __post_init__(self) -> None:
        if self.schema != ALFWORLD_PHYSICAL_CALL_COUNTER_SCHEMA:
            raise AlfWorldStateError("unsupported ALFWorld physical-call counter schema")
        for name in _PHYSICAL_COUNTER_FIELDS:
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise AlfWorldStateError(f"physical-call counter {name} must be non-negative")

    @property
    def worker_environment_step_calls(self) -> int:
        return sum(
            getattr(self, name)
            for name in (
                "rollout_first_worker_step_calls",
                "rollout_verification_worker_step_calls",
                "snapshot_only_worker_step_calls",
                "unverified_worker_step_calls",
                "candidate_probe_first_worker_step_calls",
                "candidate_probe_replay_worker_step_calls",
            )
        )

    @property
    def environment_step_calls(self) -> int:
        return self.worker_environment_step_calls + self.prefix_replay_environment_step_calls

    @property
    def environment_reset_calls(self) -> int:
        return self.source_worker_reset_calls + self.batch_environment_reset_calls

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> AlfWorldPhysicalCallCounters:
        if not isinstance(payload, Mapping):
            raise AlfWorldStateError("physical-call counter receipt must be a mapping")
        allowed = {
            "schema",
            *_PHYSICAL_COUNTER_FIELDS,
            "worker_environment_step_calls",
            "environment_step_calls",
            "environment_reset_calls",
        }
        unknown = sorted(set(payload) - allowed)
        if unknown:
            raise AlfWorldStateError(f"physical-call counter receipt has unknown fields: {unknown}")
        missing = sorted(set(_PHYSICAL_COUNTER_FIELDS) - set(payload))
        if missing:
            raise AlfWorldStateError(f"physical-call counter receipt is incomplete: {missing}")
        receipt = cls(
            schema=str(payload.get("schema", "")),
            **{name: payload[name] for name in _PHYSICAL_COUNTER_FIELDS},
        )
        derived = {
            "worker_environment_step_calls": receipt.worker_environment_step_calls,
            "environment_step_calls": receipt.environment_step_calls,
            "environment_reset_calls": receipt.environment_reset_calls,
        }
        for name, expected in derived.items():
            if name in payload and payload[name] != expected:
                raise AlfWorldStateError(f"physical-call derived total changed: {name}")
        return receipt

    def plus(
        self,
        other: AlfWorldPhysicalCallCounters,
    ) -> AlfWorldPhysicalCallCounters:
        if other.schema != self.schema:
            raise AlfWorldStateError("physical-call counter schemas differ")
        return AlfWorldPhysicalCallCounters(
            **{
                name: getattr(self, name) + getattr(other, name)
                for name in _PHYSICAL_COUNTER_FIELDS
            }
        )

    def delta_since(
        self,
        earlier: AlfWorldPhysicalCallCounters,
    ) -> AlfWorldPhysicalCallCounters:
        if earlier.schema != self.schema:
            raise AlfWorldStateError("physical-call counter schemas differ")
        values = {
            name: getattr(self, name) - getattr(earlier, name) for name in _PHYSICAL_COUNTER_FIELDS
        }
        if any(value < 0 for value in values.values()):
            raise AlfWorldStateError("physical-call counters are not monotonic")
        return AlfWorldPhysicalCallCounters(**values)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload.update(
            {
                "worker_environment_step_calls": self.worker_environment_step_calls,
                "environment_step_calls": self.environment_step_calls,
                "environment_reset_calls": self.environment_reset_calls,
            }
        )
        return payload

    @property
    def sha256(self) -> str:
        return _canonical_sha256(self.to_dict())


@dataclass(frozen=True)
class AlfWorldSnapshotOnlyStep:
    """One real transition with a pre-action snapshot but no replay proof.

    This lower-cost evidence is intended for frozen behavior screens.  It is
    deliberately distinguishable from :class:`AlfWorldVerifiedStep` so a
    caller cannot mistake snapshot availability for restore/replay parity.
    """

    outcome: Mapping[str, Any]
    raw_outcome: Any
    snapshot_artifact: AlfWorldStateArtifact
    post_snapshot_artifact: AlfWorldStateArtifact | None = None
    physical_call_delta: AlfWorldPhysicalCallCounters | None = None
    physical_call_cumulative: AlfWorldPhysicalCallCounters | None = None

    def trace_annotations(self) -> dict[str, Any]:
        return {
            "state_snapshot_sha256": self.snapshot_artifact.state_snapshot_sha256,
            "state_snapshot_artifact_sha256": (self.snapshot_artifact.state_snapshot_sha256),
            "state_snapshot_artifact_path": self.snapshot_artifact.payload_path,
            "post_state_snapshot_sha256": (
                self.post_snapshot_artifact.state_snapshot_sha256
                if self.post_snapshot_artifact is not None
                else ""
            ),
            "post_state_snapshot_artifact_path": (
                self.post_snapshot_artifact.payload_path
                if self.post_snapshot_artifact is not None
                else ""
            ),
            "physical_call_delta": (
                self.physical_call_delta.to_dict() if self.physical_call_delta is not None else {}
            ),
            "physical_call_cumulative": (
                self.physical_call_cumulative.to_dict()
                if self.physical_call_cumulative is not None
                else {}
            ),
            "transition_evidence_mode": "snapshot_only",
            "restore_verified": False,
            "restore_probe_receipt_sha256": "",
            "restore_probe_receipt_path": "",
        }


@dataclass(frozen=True)
class AlfWorldVerifiedStep:
    """One semantic environment action with recoverability evidence.

    The worker is left in the replayed post-action state.  The first action is
    executed only to establish a deterministic replay oracle; it is then
    rolled back before the action is replayed.  From the caller's perspective
    this is one logical environment transition, with no hidden post-terminal
    or duplicated training row.
    """

    outcome: Mapping[str, Any]
    raw_outcome: Any
    snapshot_artifact: AlfWorldStateArtifact
    restore_receipt: AlfWorldRestoreReceipt
    restore_receipt_path: str
    restore_receipt_sha256: str
    post_snapshot_artifact: AlfWorldStateArtifact | None = None
    physical_call_delta: AlfWorldPhysicalCallCounters | None = None
    physical_call_cumulative: AlfWorldPhysicalCallCounters | None = None

    def trace_annotations(self) -> dict[str, Any]:
        """Fields to copy into the observable transition trace."""

        return {
            "state_snapshot_sha256": self.snapshot_artifact.state_snapshot_sha256,
            "state_snapshot_artifact_sha256": self.snapshot_artifact.state_snapshot_sha256,
            "state_snapshot_artifact_path": self.snapshot_artifact.payload_path,
            "post_state_snapshot_sha256": (
                self.post_snapshot_artifact.state_snapshot_sha256
                if self.post_snapshot_artifact is not None
                else ""
            ),
            "post_state_snapshot_artifact_path": (
                self.post_snapshot_artifact.payload_path
                if self.post_snapshot_artifact is not None
                else ""
            ),
            "physical_call_delta": (
                self.physical_call_delta.to_dict() if self.physical_call_delta is not None else {}
            ),
            "physical_call_cumulative": (
                self.physical_call_cumulative.to_dict()
                if self.physical_call_cumulative is not None
                else {}
            ),
            "restore_verified": self.restore_receipt.restore_verified,
            "restore_probe_receipt_sha256": self.restore_receipt_sha256,
            "restore_probe_receipt_path": self.restore_receipt_path,
        }


class AlfWorldReplayStateController:
    """Exact-state controller for TextWorld environments that cannot pickle.

    TextWorld's ``TextworldBatchGymEnv`` stores a generator and cannot be
    serialized as an object.  This controller instead snapshots a *replay
    recipe*: fixed active gamefile, semantic action prefix, and every declared
    wrapper-state field.  Restore loads the same game directly into the live
    batch environment and replays the prefix.  The subsequent restore probe is
    still mandatory, so this backend never assumes reset/replay is exact.

    It supports a scalar ``batch_size == 1`` worker.  A vectorized runtime
    must create one controller per logical branch; sharing a batch-level
    controller would make the state identity depend on unrelated rollouts.
    """

    def __init__(
        self,
        worker: Any,
        *,
        output_dir: str | Path,
        wrapper_fields: tuple[str, ...] = (),
    ) -> None:
        self.worker = worker
        self.output_dir = Path(output_dir)
        self.wrapper_fields = tuple(wrapper_fields)
        self._gamefiles: tuple[str, ...] | None = None
        self._action_history: list[str] = []
        self._physical_counts = {name: 0 for name in _PHYSICAL_COUNTER_FIELDS}

    def _record_physical_call(self, name: str) -> None:
        if name not in self._physical_counts:
            raise AlfWorldStateError(f"unknown ALFWorld physical-call counter: {name}")
        self._physical_counts[name] += 1

    def physical_call_counters(self) -> AlfWorldPhysicalCallCounters:
        """Return an immutable cumulative call receipt for this controller."""

        return AlfWorldPhysicalCallCounters(**self._physical_counts)

    def reset(self) -> Any:
        """Reset the source worker and start a new recoverable trajectory."""

        self._record_physical_call("source_worker_reset_calls")
        result = self.worker.reset()
        self._gamefiles = _gamefiles_from_reset(result)
        self._action_history = []
        return result

    def snapshot(self) -> AlfWorldStateSnapshot:
        if self._gamefiles is None:
            raise AlfWorldStateError("replay state requires reset before snapshot")
        env = getattr(self.worker, "env", None)
        if env is None or not hasattr(env, "batch_env"):
            raise AlfWorldStateError("replay state requires a TextWorld-style worker.env.batch_env")
        if int(getattr(env, "batch_size", 1)) != 1:
            raise AlfWorldStateError("replay state supports exactly one environment branch")
        try:
            wrapper_state = {
                field: copy.deepcopy(getattr(self.worker, field))
                for field in self.wrapper_fields
                if hasattr(self.worker, field)
            }
            payload = pickle.dumps(
                {
                    "schema": ALFWORLD_STATE_CAPABILITY_SCHEMA,
                    "backend": ALFWORLD_REPLAY_BACKEND,
                    "gamefiles": self._gamefiles,
                    "action_history": tuple(self._action_history),
                    "wrapper_state": wrapper_state,
                },
                protocol=pickle.HIGHEST_PROTOCOL,
            )
            pickle.loads(payload)
        except Exception as error:  # pragma: no cover - wrapper determines serializability.
            raise AlfWorldStateError(
                f"ALFWorld replay-state wrapper is not serializable: {error!r}"
            ) from error
        return AlfWorldStateSnapshot(
            schema=ALFWORLD_STATE_CAPABILITY_SCHEMA,
            state_snapshot_sha256=hashlib.sha256(payload).hexdigest(),
            payload=payload,
            wrapper_fields=self.wrapper_fields,
        )

    def restore(self, snapshot: AlfWorldStateSnapshot) -> None:
        """Restore one scalar TextWorld branch from an auditable replay recipe."""

        if snapshot.schema != ALFWORLD_STATE_CAPABILITY_SCHEMA:
            raise AlfWorldStateError("unsupported replay state snapshot schema")
        if hashlib.sha256(snapshot.payload).hexdigest() != snapshot.state_snapshot_sha256:
            raise AlfWorldStateError("ALFWorld replay snapshot payload digest mismatch")
        try:
            state = pickle.loads(snapshot.payload)
        except Exception as error:  # pragma: no cover - corruption/foreign pickle.
            raise AlfWorldStateError(
                f"ALFWorld replay snapshot cannot be loaded: {error!r}"
            ) from error
        if (
            state.get("schema") != ALFWORLD_STATE_CAPABILITY_SCHEMA
            or state.get("backend") != ALFWORLD_REPLAY_BACKEND
        ):
            raise AlfWorldStateError("ALFWorld replay snapshot backend mismatch")
        gamefiles = tuple(str(value) for value in state.get("gamefiles") or ())
        history = tuple(str(value) for value in state.get("action_history") or ())
        if len(gamefiles) != 1:
            raise AlfWorldStateError("replay state requires exactly one current gamefile")
        env = getattr(self.worker, "env", None)
        batch_env = getattr(env, "batch_env", None)
        if batch_env is None or not callable(getattr(batch_env, "load", None)):
            raise AlfWorldStateError("TextWorld batch environment cannot load a fixed gamefile")
        try:
            self._record_physical_call("state_restore_calls")
            close = getattr(batch_env, "close", None)
            if callable(close):
                self._record_physical_call("batch_environment_close_calls")
                close()
            self._record_physical_call("batch_environment_load_calls")
            batch_env.load(list(gamefiles))
            env.last_commands = [None]
            self._record_physical_call("batch_environment_reset_calls")
            env.obs, _ = batch_env.reset()
            for action in history:
                self._record_physical_call("prefix_replay_environment_step_calls")
                env.step([action])
            for field, value in dict(state.get("wrapper_state") or {}).items():
                setattr(self.worker, field, value)
        except Exception as error:  # pragma: no cover - TextWorld runtime dependent.
            raise AlfWorldStateError(f"ALFWorld replay restore failed: {error!r}") from error
        self._gamefiles = gamefiles
        self._action_history = list(history)

    def step(
        self,
        action: str,
        *,
        project_outcome: Callable[[Any], Mapping[str, Any]],
    ) -> AlfWorldVerifiedStep:
        """Execute one semantic action with a reset/replay restore proof."""

        calls_before = self.physical_call_counters()
        snapshot = self.snapshot()
        artifact = write_alfworld_state_artifact(snapshot, output_dir=self.output_dir)
        self._record_physical_call("logical_semantic_action_calls")
        self._record_physical_call("rollout_first_worker_step_calls")
        first_raw = self.worker.step(action)
        self._action_history.append(action)
        first = dict(project_outcome(first_raw))
        first_wrapper = _wrapper_projection(self.worker, self.wrapper_fields)
        self.restore(snapshot)
        self._record_physical_call("rollout_verification_worker_step_calls")
        replay_raw = self.worker.step(action)
        self._action_history.append(action)
        replay = dict(project_outcome(replay_raw))
        replay_wrapper = _wrapper_projection(self.worker, self.wrapper_fields)
        post_snapshot = self.snapshot()
        post_artifact = write_alfworld_state_artifact(
            post_snapshot,
            output_dir=self.output_dir,
        )
        comparisons = {
            "observation": first.get("observation") == replay.get("observation"),
            "reward": first.get("reward") == replay.get("reward"),
            "done": first.get("done") == replay.get("done"),
            "won": first.get("won") == replay.get("won"),
            "wrapper_state": first_wrapper == replay_wrapper,
        }
        for field in ("facts", "admissible_actions"):
            if field in first or field in replay:
                comparisons[field] = (
                    field in first and field in replay and first[field] == replay[field]
                )
        receipt = AlfWorldRestoreReceipt(
            schema=ALFWORLD_RESTORE_RECEIPT_SCHEMA,
            state_snapshot_sha256=snapshot.state_snapshot_sha256,
            action=action,
            comparisons=comparisons,
            restore_verified=all(comparisons[name] for name in REQUIRED_RESTORE_COMPARISONS),
            first_outcome=first,
            replay_outcome=replay,
        )
        receipt_path = write_alfworld_restore_receipt(receipt, output_dir=self.output_dir)
        receipt_sha256 = _sha256_file(receipt_path)
        if not receipt.restore_verified:
            raise AlfWorldStateError(
                "ALFWorld reset/replay probe failed; refusing to emit a verified state transition"
            )
        return AlfWorldVerifiedStep(
            outcome=replay,
            raw_outcome=replay_raw,
            snapshot_artifact=artifact,
            restore_receipt=receipt,
            restore_receipt_path=str(receipt_path),
            restore_receipt_sha256=receipt_sha256,
            post_snapshot_artifact=post_artifact,
            physical_call_delta=self.physical_call_counters().delta_since(calls_before),
            physical_call_cumulative=self.physical_call_counters(),
        )

    def step_snapshot_only(
        self,
        action: str,
        *,
        project_outcome: Callable[[Any], Mapping[str, Any]],
    ) -> AlfWorldSnapshotOnlyStep:
        """Advance once while retaining only the pre-action snapshot.

        Unlike :meth:`step`, this does not restore and replay the growing
        action prefix.  The controller still owns the action history so every
        later snapshot remains a valid replay recipe.
        """

        calls_before = self.physical_call_counters()
        snapshot = self.snapshot()
        artifact = write_alfworld_state_artifact(
            snapshot,
            output_dir=self.output_dir,
        )
        self._record_physical_call("logical_semantic_action_calls")
        self._record_physical_call("snapshot_only_worker_step_calls")
        raw_outcome = self.worker.step(action)
        self._action_history.append(action)
        post_snapshot = self.snapshot()
        post_artifact = write_alfworld_state_artifact(
            post_snapshot,
            output_dir=self.output_dir,
        )
        return AlfWorldSnapshotOnlyStep(
            outcome=dict(project_outcome(raw_outcome)),
            raw_outcome=raw_outcome,
            snapshot_artifact=artifact,
            post_snapshot_artifact=post_artifact,
            physical_call_delta=self.physical_call_counters().delta_since(calls_before),
            physical_call_cumulative=self.physical_call_counters(),
        )

    def step_unverified(
        self,
        action: str,
        *,
        project_outcome: Callable[[Any], Mapping[str, Any]],
    ) -> tuple[Any, Mapping[str, Any], AlfWorldPhysicalCallCounters, AlfWorldPhysicalCallCounters]:
        """Advance one logical action without restore proof while counting calls."""

        calls_before = self.physical_call_counters()
        self._record_physical_call("logical_semantic_action_calls")
        self._record_physical_call("unverified_worker_step_calls")
        raw_outcome = self.worker.step(action)
        self._action_history.append(action)
        cumulative = self.physical_call_counters()
        return (
            raw_outcome,
            dict(project_outcome(raw_outcome)),
            cumulative.delta_since(calls_before),
            cumulative,
        )

    def probe_snapshot(
        self,
        snapshot: AlfWorldStateSnapshot,
        action: str,
        *,
        project_outcome: Callable[[Any], Mapping[str, Any]],
    ) -> AlfWorldVerifiedStep:
        """Prove one *candidate* state is restorable without advancing rollout.

        A full reset/load/replay costs far more than snapshotting the compact
        replay recipe.  The caller should first use
        ``select_restore_probe_candidates`` and invoke this only for a state
        that has multiple compatible visits.  The controller returns to the
        supplied pre-action snapshot before returning, so an audit probe never
        changes the live rollout branch.
        """

        calls_before = self.physical_call_counters()
        artifact = write_alfworld_state_artifact(snapshot, output_dir=self.output_dir)
        self.restore(snapshot)
        self._record_physical_call("candidate_probe_first_worker_step_calls")
        first_raw = self.worker.step(action)
        first = dict(project_outcome(first_raw))
        first_wrapper = _wrapper_projection(self.worker, self.wrapper_fields)
        self.restore(snapshot)
        self._record_physical_call("candidate_probe_replay_worker_step_calls")
        replay_raw = self.worker.step(action)
        self._action_history.append(action)
        replay = dict(project_outcome(replay_raw))
        replay_wrapper = _wrapper_projection(self.worker, self.wrapper_fields)
        post_snapshot = self.snapshot()
        post_artifact = write_alfworld_state_artifact(
            post_snapshot,
            output_dir=self.output_dir,
        )
        comparisons = {
            "observation": first.get("observation") == replay.get("observation"),
            "reward": first.get("reward") == replay.get("reward"),
            "done": first.get("done") == replay.get("done"),
            "won": first.get("won") == replay.get("won"),
            "wrapper_state": first_wrapper == replay_wrapper,
        }
        for field in ("facts", "admissible_actions"):
            if field in first or field in replay:
                comparisons[field] = (
                    field in first and field in replay and first[field] == replay[field]
                )
        receipt = AlfWorldRestoreReceipt(
            schema=ALFWORLD_RESTORE_RECEIPT_SCHEMA,
            state_snapshot_sha256=snapshot.state_snapshot_sha256,
            action=action,
            comparisons=comparisons,
            restore_verified=all(comparisons[name] for name in REQUIRED_RESTORE_COMPARISONS),
            first_outcome=first,
            replay_outcome=replay,
        )
        receipt_path = write_alfworld_restore_receipt(receipt, output_dir=self.output_dir)
        receipt_sha256 = _sha256_file(receipt_path)
        # Preserve the original rollout state even if the verifier rejects it.
        self.restore(snapshot)
        if not receipt.restore_verified:
            raise AlfWorldStateError(
                "ALFWorld candidate-state restore probe failed; state remains unverified"
            )
        return AlfWorldVerifiedStep(
            outcome=replay,
            raw_outcome=replay_raw,
            snapshot_artifact=artifact,
            restore_receipt=receipt,
            restore_receipt_path=str(receipt_path),
            restore_receipt_sha256=receipt_sha256,
            post_snapshot_artifact=post_artifact,
            physical_call_delta=self.physical_call_counters().delta_since(calls_before),
            physical_call_cumulative=self.physical_call_counters(),
        )


def project_scalar_textworld_outcome(raw: Any) -> dict[str, Any]:
    """Project one source-style TextWorld worker result into verifier fields.

    This deliberately accepts only a scalar ALFWorld worker result
    ``(observations, rewards, dones, infos)``.  A vector result has no
    unambiguous single snapshot identity and must be split into scalar worker
    branches before it can enter ``verified_state_dp``.
    """

    if not isinstance(raw, tuple) or len(raw) != 4:
        raise AlfWorldStateError(
            "TextWorld worker step must return (observations, rewards, dones, infos)"
        )
    observations, rewards, dones, infos = raw
    observation = _scalar_sequence_value(observations, "observations")
    reward = _scalar_sequence_value(rewards, "rewards")
    done = _scalar_sequence_value(dones, "dones")
    if not isinstance(infos, Mapping):
        raise AlfWorldStateError("TextWorld worker infos must be a mapping")
    won = _scalar_info_value(infos.get("won", False), "infos['won']")
    facts = _scalar_info_value(infos.get("facts"), "infos['facts']")
    admissible = _scalar_info_value(
        infos.get("admissible_commands"), "infos['admissible_commands']"
    )
    if not isinstance(facts, (list, tuple)):
        raise AlfWorldStateError("TextWorld worker infos must expose scalar facts")
    if not isinstance(admissible, (list, tuple)):
        raise AlfWorldStateError("TextWorld worker infos must expose scalar admissible commands")
    return {
        "observation": str(observation),
        "reward": float(reward),
        "done": bool(done),
        "won": bool(won),
        "facts": tuple(
            (
                str(getattr(fact, "name", "")),
                tuple(str(value) for value in getattr(fact, "names", ())),
            )
            if not isinstance(fact, Mapping)
            else (
                str(fact.get("name", "")),
                tuple(str(value) for value in fact.get("args") or ()),
            )
            for fact in facts
        ),
        "admissible_actions": tuple(str(action) for action in admissible),
    }


def build_verified_v2_transition_annotations(
    verified_step: AlfWorldVerifiedStep,
    *,
    goal: Any,
    remaining_budget: int,
    action_affordances: list[str] | tuple[str, ...],
    rollout_config: Mapping[str, Any],
    trajectory_length: int,
    terminal_reason: str,
    action_token_count: int,
    decision_role: str = "executor",
    planner_decision_kind: str | None = None,
    task_manifest_lock_sha256: str | None = None,
    verified_subproblem: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Produce the exact v2 trace fields from real environment evidence.

    The caller supplies observable task metadata only after the trajectory is
    complete, so every row has the same final ``trajectory_length``.  Hidden
    planner reasoning is not an accepted argument and therefore cannot enter
    this transition contract.
    """

    if remaining_budget < 0:
        raise AlfWorldStateError("remaining_budget must be non-negative")
    if trajectory_length < 1:
        raise AlfWorldStateError("trajectory_length must be positive")
    if action_token_count < 1:
        raise AlfWorldStateError("action_token_count must be positive")
    if not terminal_reason:
        raise AlfWorldStateError("terminal_reason must be observable and non-empty")
    if decision_role not in {"executor", "planner"}:
        raise AlfWorldStateError("decision_role must be executor or planner")
    if decision_role == "planner" and planner_decision_kind not in {
        "subgoal",
        "route",
        "rollback",
    }:
        raise AlfWorldStateError(
            "planner transition requires observable subgoal, route, or rollback"
        )
    if task_manifest_lock_sha256 is not None and not _is_sha256(task_manifest_lock_sha256):
        raise AlfWorldStateError("task_manifest_lock_sha256 must be a lowercase SHA-256")
    if not isinstance(rollout_config, Mapping):
        raise AlfWorldStateError("rollout_config must be a mapping")
    affordances = tuple(str(item) for item in action_affordances)
    annotations = verified_step.trace_annotations()
    annotations.update(
        {
            "goal_hash": _canonical_sha256(goal),
            "remaining_budget": remaining_budget,
            "remaining_budget_verified": True,
            "action_affordance_hash": _canonical_sha256(affordances),
            "action_affordance_verified": True,
            "rollout_config_hash": _canonical_sha256(dict(rollout_config)),
            "trajectory_length": trajectory_length,
            "terminal_reason": terminal_reason,
            "action_token_count": action_token_count,
            "action_token_count_verified": True,
            "decision_role": decision_role,
            "planner_decision_kind": planner_decision_kind,
            "v2_runtime_fields_verified": True,
        }
    )
    if task_manifest_lock_sha256 is not None:
        annotations["task_manifest_lock_sha256"] = task_manifest_lock_sha256
    if verified_subproblem is not None:
        annotations.update(build_verified_subproblem_annotation(**dict(verified_subproblem)))
    return annotations


def build_verified_subproblem_annotation(
    *,
    active_subproblem_id: str,
    contract: Mapping[str, Any],
    source: str,
    verifier_input: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Encode one P0 oracle subproblem annotation without plan leakage.

    The annotation identifies the *currently active* child contract.  It is
    intentionally not a plan string and never accepts reference actions or
    model reasoning as evidence.  The caller must derive ``contract`` from
    environment facts and expose the same actor prompt/harness as the baseline.
    """

    if not isinstance(active_subproblem_id, str) or not active_subproblem_id.strip():
        raise AlfWorldStateError("verified subproblem requires a non-empty active_subproblem_id")
    if source != "environment_contract":
        raise AlfWorldStateError(
            "verified subproblem annotations must come from environment_contract"
        )
    if not isinstance(contract, Mapping) or not contract:
        raise AlfWorldStateError("verified subproblem requires a non-empty environment contract")
    if not isinstance(verifier_input, Mapping):
        raise AlfWorldStateError("verified subproblem requires retained verifier input")
    from lh_platform.studies.alfworld_subproblems import (
        AlfWorldSubproblemError,
        verify_alfworld_subproblem_annotation,
    )

    try:
        verify_alfworld_subproblem_annotation(
            active_subproblem_id=active_subproblem_id.strip(),
            contract=contract,
            verifier_input=verifier_input,
        )
    except AlfWorldSubproblemError as error:
        raise AlfWorldStateError(f"invalid environment subproblem receipt: {error}") from error
    return {
        "active_subproblem_id": active_subproblem_id.strip(),
        "subproblem_contract": dict(contract),
        "subproblem_contract_sha256": _canonical_sha256(dict(contract)),
        "subproblem_contract_verified": True,
        "subproblem_source": source,
        "subproblem_verifier_input": dict(verifier_input),
    }


def snapshot_alfworld_worker(
    worker: Any, *, wrapper_fields: tuple[str, ...] = DEFAULT_WRAPPER_FIELDS
) -> AlfWorldStateSnapshot:
    """Serialize true environment plus wrapper state; never hash observations."""

    if not hasattr(worker, "env"):
        raise AlfWorldStateError("worker has no environment object to snapshot")
    try:
        state = {
            "schema": ALFWORLD_STATE_CAPABILITY_SCHEMA,
            "env": worker.env,
            "wrapper_state": {
                field: copy.deepcopy(getattr(worker, field))
                for field in wrapper_fields
                if hasattr(worker, field)
            },
        }
        payload = pickle.dumps(state, protocol=pickle.HIGHEST_PROTOCOL)
        # A serializable but non-round-trippable payload is not an exact state.
        pickle.loads(payload)
    except Exception as error:  # pragma: no cover - external env determines concrete errors.
        raise AlfWorldStateError(f"ALFWorld worker state is not serializable: {error!r}") from error
    return AlfWorldStateSnapshot(
        schema=ALFWORLD_STATE_CAPABILITY_SCHEMA,
        state_snapshot_sha256=hashlib.sha256(payload).hexdigest(),
        payload=payload,
        wrapper_fields=tuple(wrapper_fields),
    )


def restore_alfworld_worker(worker: Any, snapshot: AlfWorldStateSnapshot) -> None:
    """Restore exactly the environment and wrapper fields captured by a snapshot."""

    if snapshot.schema != ALFWORLD_STATE_CAPABILITY_SCHEMA:
        raise AlfWorldStateError("unsupported ALFWorld state snapshot schema")
    if hashlib.sha256(snapshot.payload).hexdigest() != snapshot.state_snapshot_sha256:
        raise AlfWorldStateError("ALFWorld snapshot payload digest mismatch")
    try:
        state = pickle.loads(snapshot.payload)
    except Exception as error:  # pragma: no cover - corruption/foreign pickle.
        raise AlfWorldStateError(f"ALFWorld snapshot cannot be loaded: {error!r}") from error
    if state.get("schema") != ALFWORLD_STATE_CAPABILITY_SCHEMA:
        raise AlfWorldStateError("ALFWorld snapshot schema mismatch")
    worker.env = state["env"]
    for field, value in dict(state.get("wrapper_state") or {}).items():
        setattr(worker, field, value)


def write_alfworld_state_artifact(
    snapshot: AlfWorldStateSnapshot, *, output_dir: str | Path
) -> AlfWorldStateArtifact:
    """Persist a verified pre-action snapshot by its content hash.

    Training records carry only the digest.  The serialized environment state
    is stored in an auditable sidecar so ``state_snapshot_sha256`` actually
    denotes a recoverable state rather than an observation-derived label.
    """

    if hashlib.sha256(snapshot.payload).hexdigest() != snapshot.state_snapshot_sha256:
        raise AlfWorldStateError("cannot archive a snapshot with a mismatched payload digest")
    root = Path(output_dir)
    payload_path = root / "snapshots" / f"{snapshot.state_snapshot_sha256}.pickle"
    payload_path.parent.mkdir(parents=True, exist_ok=True)
    if payload_path.exists():
        if payload_path.read_bytes() != snapshot.payload:
            raise AlfWorldStateError("snapshot digest collision with different payload bytes")
    else:
        staging = payload_path.with_name(f".{payload_path.name}.{uuid4().hex}.tmp")
        try:
            staging.write_bytes(snapshot.payload)
            staging.replace(payload_path)
        finally:
            staging.unlink(missing_ok=True)
    return AlfWorldStateArtifact(
        schema=ALFWORLD_STATE_ARTIFACT_SCHEMA,
        state_snapshot_sha256=snapshot.state_snapshot_sha256,
        payload_path=str(payload_path),
        payload_size_bytes=len(snapshot.payload),
        wrapper_fields=snapshot.wrapper_fields,
    )


def read_alfworld_state_artifact(artifact: AlfWorldStateArtifact) -> AlfWorldStateSnapshot:
    """Load a content-addressed snapshot only after digest and schema checks."""

    if artifact.schema != ALFWORLD_STATE_ARTIFACT_SCHEMA:
        raise AlfWorldStateError("unsupported ALFWorld state artifact schema")
    payload_path = Path(artifact.payload_path)
    try:
        payload = payload_path.read_bytes()
    except OSError as error:
        raise AlfWorldStateError(f"ALFWorld state artifact is unavailable: {error!r}") from error
    if len(payload) != artifact.payload_size_bytes:
        raise AlfWorldStateError("ALFWorld state artifact size mismatch")
    if hashlib.sha256(payload).hexdigest() != artifact.state_snapshot_sha256:
        raise AlfWorldStateError("ALFWorld state artifact digest mismatch")
    return AlfWorldStateSnapshot(
        schema=ALFWORLD_STATE_CAPABILITY_SCHEMA,
        state_snapshot_sha256=artifact.state_snapshot_sha256,
        payload=payload,
        wrapper_fields=artifact.wrapper_fields,
    )


def probe_alfworld_restore(
    worker: Any,
    action: str,
    *,
    step: Callable[[Any, str], Any],
    project_outcome: Callable[[Any], Mapping[str, Any]],
    wrapper_fields: tuple[str, ...] = DEFAULT_WRAPPER_FIELDS,
) -> AlfWorldRestoreReceipt:
    """Restore a pre-action state and prove replay equivalence for one action."""

    snapshot = snapshot_alfworld_worker(worker, wrapper_fields=wrapper_fields)
    first_raw = step(worker, action)
    first = dict(project_outcome(first_raw))
    first_wrapper = _wrapper_projection(worker, wrapper_fields)
    restore_alfworld_worker(worker, snapshot)
    replay_raw = step(worker, action)
    replay = dict(project_outcome(replay_raw))
    replay_wrapper = _wrapper_projection(worker, wrapper_fields)
    comparisons = {
        "observation": first.get("observation") == replay.get("observation"),
        "reward": first.get("reward") == replay.get("reward"),
        "done": first.get("done") == replay.get("done"),
        "won": first.get("won") == replay.get("won"),
        "wrapper_state": first_wrapper == replay_wrapper,
    }
    for field in ("facts", "admissible_actions"):
        if field in first or field in replay:
            comparisons[field] = (
                field in first and field in replay and first[field] == replay[field]
            )
    return AlfWorldRestoreReceipt(
        schema=ALFWORLD_RESTORE_RECEIPT_SCHEMA,
        state_snapshot_sha256=snapshot.state_snapshot_sha256,
        action=action,
        comparisons=comparisons,
        restore_verified=all(comparisons[name] for name in REQUIRED_RESTORE_COMPARISONS),
        first_outcome=first,
        replay_outcome=replay,
    )


def write_alfworld_restore_receipt(
    receipt: AlfWorldRestoreReceipt, *, output_dir: str | Path
) -> Path:
    """Store the replay proof without leaking full observations into the trace."""

    action_digest = hashlib.sha256(receipt.action.encode("utf-8")).hexdigest()[:16]
    target = (
        Path(output_dir)
        / "restore_receipts"
        / f"{receipt.state_snapshot_sha256}_{action_digest}.json"
    )
    target.parent.mkdir(parents=True, exist_ok=True)
    staging = target.with_name(f".{target.name}.{uuid4().hex}.tmp")
    try:
        staging.write_text(
            json.dumps(receipt.to_dict(), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        staging.replace(target)
    finally:
        staging.unlink(missing_ok=True)
    return target


def write_alfworld_state_artifact_manifest(
    artifacts: Mapping[str, AlfWorldStateArtifact]
    | tuple[AlfWorldStateArtifact, ...]
    | list[AlfWorldStateArtifact],
    *,
    output_dir: str | Path,
) -> Path:
    """Write a content-addressed snapshot index for a trace/run artifact set."""

    if isinstance(artifacts, Mapping):
        entries = dict(artifacts)
    else:
        entries = {item.state_snapshot_sha256: item for item in artifacts}
    if any(key != item.state_snapshot_sha256 for key, item in entries.items()):
        raise AlfWorldStateError("state artifact manifest key does not match snapshot digest")
    target = Path(output_dir) / "state_snapshot_manifest.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema": ALFWORLD_STATE_ARTIFACT_MANIFEST_SCHEMA,
        "artifacts": {digest: item.to_dict() for digest, item in sorted(entries.items())},
    }
    target.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return target


def read_alfworld_state_artifact_manifest(
    path: str | Path,
) -> dict[str, AlfWorldStateArtifact]:
    """Load a manifest but do not trust its entries until the payload is read."""

    source = Path(path)
    try:
        raw = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise AlfWorldStateError(f"state artifact manifest cannot be loaded: {error!r}") from error
    if raw.get("schema") != ALFWORLD_STATE_ARTIFACT_MANIFEST_SCHEMA:
        raise AlfWorldStateError("unsupported ALFWorld state artifact manifest schema")
    artifacts_raw = raw.get("artifacts")
    if not isinstance(artifacts_raw, Mapping):
        raise AlfWorldStateError("state artifact manifest artifacts must be a mapping")
    artifacts: dict[str, AlfWorldStateArtifact] = {}
    for digest, entry in artifacts_raw.items():
        if not isinstance(entry, Mapping):
            raise AlfWorldStateError("state artifact manifest contains a non-mapping entry")
        try:
            artifact = AlfWorldStateArtifact(
                schema=str(entry["schema"]),
                state_snapshot_sha256=str(entry["state_snapshot_sha256"]),
                payload_path=str(entry["payload_path"]),
                payload_size_bytes=int(entry["payload_size_bytes"]),
                wrapper_fields=tuple(str(field) for field in entry["wrapper_fields"]),
            )
        except (KeyError, TypeError, ValueError) as error:
            raise AlfWorldStateError(f"invalid state artifact manifest entry: {error!r}") from error
        if digest != artifact.state_snapshot_sha256:
            raise AlfWorldStateError("state artifact manifest digest does not match entry")
        artifacts[digest] = artifact
    return artifacts


def step_with_verified_state(
    worker: Any,
    action: str,
    *,
    step: Callable[[Any, str], Any],
    project_outcome: Callable[[Any], Mapping[str, Any]],
    output_dir: str | Path,
    wrapper_fields: tuple[str, ...] = DEFAULT_WRAPPER_FIELDS,
) -> AlfWorldVerifiedStep:
    """Execute one action only if exact restore/replay agrees.

    This is the integration boundary for an ALFWorld worker adapter.  A caller
    must use ``outcome`` as the actual post-action outcome and must persist
    ``trace_annotations`` on that same transition row.  A failed restore probe
    raises before any record is emitted, rather than silently falling back to
    observation grouping.
    """

    snapshot = snapshot_alfworld_worker(worker, wrapper_fields=wrapper_fields)
    artifact = write_alfworld_state_artifact(snapshot, output_dir=output_dir)
    first_raw = step(worker, action)
    first = dict(project_outcome(first_raw))
    first_wrapper = _wrapper_projection(worker, wrapper_fields)
    restore_alfworld_worker(worker, snapshot)
    replay_raw = step(worker, action)
    replay = dict(project_outcome(replay_raw))
    replay_wrapper = _wrapper_projection(worker, wrapper_fields)
    post_snapshot = snapshot_alfworld_worker(worker, wrapper_fields=wrapper_fields)
    post_artifact = write_alfworld_state_artifact(post_snapshot, output_dir=output_dir)
    comparisons = {
        "observation": first.get("observation") == replay.get("observation"),
        "reward": first.get("reward") == replay.get("reward"),
        "done": first.get("done") == replay.get("done"),
        "won": first.get("won") == replay.get("won"),
        "wrapper_state": first_wrapper == replay_wrapper,
    }
    for field in ("facts", "admissible_actions"):
        if field in first or field in replay:
            comparisons[field] = (
                field in first and field in replay and first[field] == replay[field]
            )
    receipt = AlfWorldRestoreReceipt(
        schema=ALFWORLD_RESTORE_RECEIPT_SCHEMA,
        state_snapshot_sha256=snapshot.state_snapshot_sha256,
        action=action,
        comparisons=comparisons,
        restore_verified=all(comparisons[name] for name in REQUIRED_RESTORE_COMPARISONS),
        first_outcome=first,
        replay_outcome=replay,
    )
    receipt_path = write_alfworld_restore_receipt(receipt, output_dir=output_dir)
    receipt_sha256 = _sha256_file(receipt_path)
    if not receipt.restore_verified:
        raise AlfWorldStateError(
            "ALFWorld restore probe failed; refusing to emit a verified state transition"
        )
    return AlfWorldVerifiedStep(
        outcome=replay,
        raw_outcome=replay_raw,
        snapshot_artifact=artifact,
        restore_receipt=receipt,
        restore_receipt_path=str(receipt_path),
        restore_receipt_sha256=receipt_sha256,
        post_snapshot_artifact=post_artifact,
    )


def state_capability_manifest(*, source: str, restore_verified: bool = False) -> dict[str, Any]:
    """Return the common sidecar contract without claiming a live probe ran."""

    return {
        "schema": ALFWORLD_STATE_CAPABILITY_SCHEMA,
        "source": source,
        "state_snapshot_field": "state_snapshot_sha256",
        "state_snapshot_artifact_required": True,
        "post_state_snapshot_artifact_required": True,
        "physical_call_counter_schema": ALFWORLD_PHYSICAL_CALL_COUNTER_SCHEMA,
        "restore_receipt_required": True,
        "observation_hash_allowed": False,
        "restore_probe_required": True,
        "restore_probe_policy": "only_duplicate_candidate_states_then_fail_closed",
        "restore_verified": restore_verified,
        "required_comparisons": list(REQUIRED_RESTORE_COMPARISONS),
        "identity_fields": [
            "state_snapshot_sha256",
            "goal_hash",
            "remaining_budget",
            "action_affordance_hash",
        ],
        "applies_to": [
            "grpo_u",
            "g2po_source_credit",
            "g2po_trajectory_normalized",
            "verified_state_dp",
            "graphgpo",
        ],
    }


def _wrapper_projection(worker: Any, fields: tuple[str, ...]) -> str:
    payload = {field: repr(getattr(worker, field)) for field in fields if hasattr(worker, field)}
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()


def _gamefiles_from_reset(result: Any) -> tuple[str, ...]:
    """Extract one immutable TextWorld gamefile from a worker reset result.

    Public ALFWorld wrappers expose both ``(obs, infos)`` and
    ``(text_obs, image_obs, infos)`` reset signatures.  We accept only an
    explicit ``extra.gamefile`` value; inferring it from an observation would
    silently turn the restore backend into observation grouping.
    """

    candidates: list[str] = []

    def visit(value: Any) -> None:
        if isinstance(value, Mapping):
            if "extra.gamefile" in value:
                raw_gamefile = value["extra.gamefile"]
                if isinstance(raw_gamefile, str):
                    candidates.append(raw_gamefile)
                elif isinstance(raw_gamefile, (list, tuple)):
                    candidates.extend(item for item in raw_gamefile if isinstance(item, str))
            for child in value.values():
                if isinstance(child, (Mapping, list, tuple)):
                    visit(child)
        elif isinstance(value, (list, tuple)):
            for child in value:
                if isinstance(child, (Mapping, list, tuple)):
                    visit(child)

    visit(result)
    normalized = tuple(dict.fromkeys(item.strip() for item in candidates if item.strip()))
    if len(normalized) != 1:
        raise AlfWorldStateError(
            "replay state requires exactly one explicit extra.gamefile from reset"
        )
    return normalized


def _scalar_sequence_value(value: Any, field: str) -> Any:
    if not isinstance(value, (list, tuple)) or len(value) != 1:
        raise AlfWorldStateError(f"TextWorld scalar worker requires exactly one {field} value")
    return value[0]


def _scalar_info_value(value: Any, field: str) -> Any:
    if isinstance(value, (list, tuple)):
        return _scalar_sequence_value(value, field)
    return value


def _canonical_sha256(value: Any) -> str:
    try:
        payload = json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    except (TypeError, ValueError) as error:
        raise AlfWorldStateError(
            f"verified transition metadata is not canonical JSON: {error!r}"
        ) from error
    return hashlib.sha256(payload).hexdigest()


def _is_sha256(value: str) -> bool:
    return len(value) == 64 and all(character in "0123456789abcdef" for character in value)


def _sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
