"""Explicit catalog of frozen datasets admitted to self-trace experiments."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from lh_platform.core.evidence_contract import canonical_sha256
from lh_platform.tracing.self_trace_controlled import (
    CONTROLLED_SELF_TRACE_DATASET_ID,
    ControlledSelfTraceCell,
    controlled_self_trace_cells,
    controlled_self_trace_dataset_manifest,
)
from lh_platform.tracing.self_trace_unseen import (
    CONTROLLED_SELF_TRACE_UNSEEN_DATASET_ID,
    CONTROLLED_SELF_TRACE_UNSEEN_REPLAY_CONTROLLER_ID,
    controlled_self_trace_unseen_cells,
    controlled_self_trace_unseen_dataset_manifest,
)


@dataclass(frozen=True)
class SelfTraceDatasetV1:
    """One immutable dataset contract plus its executable replay cells."""

    dataset_id: str
    cells: tuple[ControlledSelfTraceCell, ...]
    replay_controller_id: str
    _manifest_json: str

    @classmethod
    def build(
        cls,
        *,
        dataset_id: str,
        cells: tuple[ControlledSelfTraceCell, ...],
        replay_controller_id: str,
        manifest: dict[str, Any],
    ) -> SelfTraceDatasetV1:
        if manifest.get("dataset_id") != dataset_id:
            raise ValueError("self-trace dataset manifest identity mismatch")
        manifest_controller_id = manifest.get("replay_controller_id")
        if manifest_controller_id is not None and manifest_controller_id != replay_controller_id:
            raise ValueError("self-trace replay controller identity mismatch")
        if not cells:
            raise ValueError("self-trace dataset must contain at least one cell")
        case_ids = [cell.case_id for cell in cells]
        task_ids = [cell.template.task_id for cell in cells]
        if len(case_ids) != len(set(case_ids)) or len(task_ids) != len(set(task_ids)):
            raise ValueError("self-trace dataset case and task identities must be unique")
        manifest_case_ids = [
            row["template"]["case_id"] for row in manifest.get("cases", [])
        ]
        if manifest_case_ids != case_ids:
            raise ValueError("self-trace dataset manifest does not bind executable cells")
        return cls(
            dataset_id=dataset_id,
            cells=cells,
            replay_controller_id=replay_controller_id,
            _manifest_json=json.dumps(
                manifest,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ),
        )

    @property
    def manifest(self) -> dict[str, Any]:
        return json.loads(self._manifest_json)

    @property
    def sha256(self) -> str:
        return "sha256:" + canonical_sha256(self.manifest)


def _development_v2() -> SelfTraceDatasetV1:
    return SelfTraceDatasetV1.build(
        dataset_id=CONTROLLED_SELF_TRACE_DATASET_ID,
        cells=controlled_self_trace_cells(),
        replay_controller_id="controlled_self_trace_executable_replay/v2",
        manifest=controlled_self_trace_dataset_manifest(),
    )


def _unseen_v3() -> SelfTraceDatasetV1:
    return SelfTraceDatasetV1.build(
        dataset_id=CONTROLLED_SELF_TRACE_UNSEEN_DATASET_ID,
        cells=controlled_self_trace_unseen_cells(),
        replay_controller_id=CONTROLLED_SELF_TRACE_UNSEEN_REPLAY_CONTROLLER_ID,
        manifest=controlled_self_trace_unseen_dataset_manifest(),
    )


_DATASET_FACTORIES = {
    CONTROLLED_SELF_TRACE_DATASET_ID: _development_v2,
    CONTROLLED_SELF_TRACE_UNSEEN_DATASET_ID: _unseen_v3,
}


def available_self_trace_datasets() -> list[str]:
    return sorted(_DATASET_FACTORIES)


def self_trace_dataset(dataset_id: str) -> SelfTraceDatasetV1:
    try:
        factory = _DATASET_FACTORIES[dataset_id]
    except KeyError as error:
        raise ValueError(
            f"unknown self-trace dataset {dataset_id!r}; "
            f"available={available_self_trace_datasets()}"
        ) from error
    return factory()


__all__ = [
    "CONTROLLED_SELF_TRACE_UNSEEN_DATASET_ID",
    "SelfTraceDatasetV1",
    "available_self_trace_datasets",
    "self_trace_dataset",
]
