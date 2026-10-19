"""Live numeric parity against one pinned official G2PO credit implementation."""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType, SimpleNamespace

from lh_platform.training.agentic_rl_runtime import canonical_sha256
from lh_platform.training.dataset import sha256_file
from lh_platform.training.g2po.credit import (
    G2POCreditMode,
    G2POTransition,
    assign_g2po_credit,
)

G2PO_SOURCE_PARITY_SCHEMA = "lh_g2po_source_credit_parity/v1"
G2PO_SOURCE_PARITY_SCOPE = "source_credit_numbers_only_not_tn_or_actor_loss"

__all__ = [
    "G2PO_SOURCE_PARITY_SCHEMA",
    "G2PO_SOURCE_PARITY_SCOPE",
    "G2POSourceParityError",
    "G2POSourceParityReceipt",
    "audit_g2po_source_parity",
]


class G2POSourceParityError(RuntimeError):
    """Raised when source identity or live numeric parity does not hold."""


@dataclass(frozen=True)
class G2POSourceParityReceipt:
    schema: str
    passed: bool
    parity_scope: str
    source_revision: str
    source_file: str
    source_file_sha256: str
    fixture_path: str
    fixture_sha256: str
    row_count: int
    tolerance: float
    max_abs_error: float
    official_advantages: tuple[float, ...]
    native_advantages: tuple[float, ...]
    contract_sha256: str


def audit_g2po_source_parity(
    *,
    source_root: str | Path,
    expected_revision: str,
    expected_source_sha256: str,
    fixture_path: str | Path,
    tolerance: float = 1e-5,
) -> G2POSourceParityReceipt:
    """Execute pinned source credit and compare it with the native SOURCE mode.

    This proves only credit-number parity for the fixture. It deliberately does
    not claim parity for the actor loss, optimizer, rollout stack, or the
    trajectory-normalized treatment used by the P0 experiment.
    """

    if tolerance <= 0:
        raise G2POSourceParityError("source parity tolerance must be positive")
    root = Path(source_root).resolve()
    fixture = Path(fixture_path).resolve()
    source_relative = Path("g2po/core_g2po.py")
    source_file = root / source_relative
    actual_revision = _git_output(root, "rev-parse", "HEAD")
    if actual_revision != expected_revision:
        raise G2POSourceParityError(
            f"G2PO source revision mismatch: expected {expected_revision}, got {actual_revision}"
        )
    tracked_changes = _git_output(
        root,
        "status",
        "--porcelain",
        "--untracked-files=no",
    )
    if tracked_changes:
        raise G2POSourceParityError(
            "G2PO source checkout has tracked modifications: " + tracked_changes
        )
    if not source_file.is_file():
        raise G2POSourceParityError(f"G2PO source credit file is missing: {source_file}")
    actual_source_sha256 = sha256_file(source_file)
    if actual_source_sha256 != expected_source_sha256:
        raise G2POSourceParityError(
            "G2PO source credit file hash mismatch: "
            f"expected {expected_source_sha256}, got {actual_source_sha256}"
        )

    try:
        raw = json.loads(fixture.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise G2POSourceParityError(f"cannot load G2PO parity fixture: {error}") from error
    if raw.get("schema") != "lh_g2po_source_credit_fixture/v1":
        raise G2POSourceParityError("unsupported G2PO source parity fixture schema")
    if raw.get("source_commit") != actual_revision:
        raise G2POSourceParityError("G2PO parity fixture source revision mismatch")
    if raw.get("source_file") != source_relative.as_posix():
        raise G2POSourceParityError("G2PO parity fixture source file mismatch")
    raw_rows = raw.get("rows")
    if not isinstance(raw_rows, list) or not raw_rows:
        raise G2POSourceParityError("G2PO parity fixture requires transition rows")
    try:
        rows = tuple(G2POTransition(**row) for row in raw_rows)
    except (TypeError, ValueError) as error:
        raise G2POSourceParityError(f"invalid G2PO parity transition: {error}") from error

    source_core = _load_source_core(source_file)
    official = _compute_official_advantages(source_core, rows)
    native = tuple(
        float(item.advantage) for item in assign_g2po_credit(rows, mode=G2POCreditMode.SOURCE)
    )
    if len(official) != len(native):
        raise G2POSourceParityError("G2PO source/native parity row count differs")
    errors = tuple(abs(source - ours) for source, ours in zip(official, native, strict=True))
    max_abs_error = max(errors, default=0.0)
    contract = {
        "schema": G2PO_SOURCE_PARITY_SCHEMA,
        "parity_scope": G2PO_SOURCE_PARITY_SCOPE,
        "source_revision": actual_revision,
        "source_file": source_relative.as_posix(),
        "source_file_sha256": actual_source_sha256,
        "fixture_sha256": sha256_file(fixture),
        "tolerance": tolerance,
    }
    if max_abs_error > tolerance:
        raise G2POSourceParityError(
            "G2PO live source parity failed: "
            f"max_abs_error={max_abs_error} tolerance={tolerance}; "
            f"official={official}; native={native}"
        )
    return G2POSourceParityReceipt(
        schema=G2PO_SOURCE_PARITY_SCHEMA,
        passed=True,
        parity_scope=G2PO_SOURCE_PARITY_SCOPE,
        source_revision=actual_revision,
        source_file=source_relative.as_posix(),
        source_file_sha256=actual_source_sha256,
        fixture_path=str(fixture),
        fixture_sha256=contract["fixture_sha256"],
        row_count=len(rows),
        tolerance=tolerance,
        max_abs_error=max_abs_error,
        official_advantages=official,
        native_advantages=native,
        contract_sha256=canonical_sha256(contract),
    )


def _git_output(root: Path, *arguments: str) -> str:
    try:
        result = subprocess.run(
            ["git", *arguments],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError) as error:
        raise G2POSourceParityError(f"cannot inspect pinned G2PO source: {error}") from error
    return result.stdout.strip()


def _load_source_core(path: Path) -> ModuleType:
    """Load the exact source file while stubbing its unused DataProto type only."""

    module_name = "_lh_platform_pinned_g2po_core"
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise G2POSourceParityError(f"cannot construct source loader for {path}")
    module = importlib.util.module_from_spec(spec)
    previous_verl = sys.modules.get("verl")
    verl_stub = ModuleType("verl")
    verl_stub.DataProto = object
    sys.modules["verl"] = verl_stub
    try:
        spec.loader.exec_module(module)
    except Exception as error:
        raise G2POSourceParityError(f"cannot execute pinned G2PO credit source: {error}") from error
    finally:
        if previous_verl is None:
            sys.modules.pop("verl", None)
        else:
            sys.modules["verl"] = previous_verl
    return module


def _compute_official_advantages(
    source_core: ModuleType,
    rows: tuple[G2POTransition, ...],
) -> tuple[float, ...]:
    np = source_core.np
    torch = source_core.torch
    max_tokens = max(row.action_token_count for row in rows)
    response_mask = torch.zeros((len(rows), max_tokens), dtype=torch.float32)
    token_level_rewards = torch.zeros_like(response_mask)
    for index, row in enumerate(rows):
        response_mask[index, : row.action_token_count] = 1.0
        token_level_rewards[index, row.action_token_count - 1] = row.episode_reward
    index = np.array([row.group_uid for row in rows], dtype=object)
    trajectories = np.array([row.trajectory_id for row in rows], dtype=object)
    batch = SimpleNamespace(
        batch={"responses": torch.zeros_like(response_mask, dtype=torch.long)},
        non_tensor_batch={
            "episode_rewards": np.array([row.episode_reward for row in rows]),
            "anchor_obs": np.array([row.anchor_observation for row in rows], dtype=object),
            "traj_uid": trajectories,
            "uid": index,
        },
    )
    step_rewards, group_idx, next_group_idx = source_core.compute_group_aggregation_values(batch)
    advantages, _ = source_core.compute_g2po_outcome_advantage(
        token_level_rewards=token_level_rewards,
        step_rewards=step_rewards,
        response_mask=response_mask,
        index=index,
        traj_index=trajectories,
        group_idx=group_idx,
        next_group_idx=next_group_idx,
        is_action_valid=np.array([row.valid_action for row in rows], dtype=int),
        step_advantage_w=1.0,
        mode="mean_std_norm",
    )
    return tuple(float(advantages[index, 0].item()) for index in range(len(rows)))
