"""Canonical GraphGPO study-family API.

This package owns the first migrated GraphGPO-specific runtime slice:
subprocess preparation, grouped ALFWorld helpers, parity audits, progress
reporting, and the executable ``reset_probe`` submodule. Generic study
contracts remain in :mod:`lh_platform.studies`.
"""

from __future__ import annotations

from importlib import import_module
from typing import Any

_EXPORTS = {
    "GraphGPOStudyRunner": "lh_platform.studies.graphgpo.runner",
    "PreparedStudy": "lh_platform.studies.graphgpo.runner",
    "StudyLaunchBlocked": "lh_platform.studies.graphgpo.runner",
    "GRAPHGPO_ALLOWED_ABLATIONS": "lh_platform.studies.graphgpo.runner",
    "GRAPHGPO_PROGRESS_SCHEMA": "lh_platform.studies.graphgpo.progress",
    "GRAPHGPO_PARITY_AUDIT_SCHEMA": "lh_platform.studies.graphgpo.parity_audit",
    "build_graphgpo_progress_snapshot": "lh_platform.studies.graphgpo.progress",
    "build_graphgpo_parity_audit": "lh_platform.studies.graphgpo.parity_audit",
    "GroupedActorTopology": "lh_platform.studies.graphgpo.grouped_alfworld",
    "ScalarEnvironmentGroup": "lh_platform.studies.graphgpo.grouped_alfworld",
    "RecipeWorkerGroup": "lh_platform.studies.graphgpo.grouped_alfworld",
    "SequentialAlfWorldRolloutSource": "lh_platform.studies.graphgpo.grouped_alfworld",
    "grouped_actor_topology": "lh_platform.studies.graphgpo.grouped_alfworld",
    "collect_sequential_rollout_chunks": "lh_platform.studies.graphgpo.grouped_alfworld",
    "install_grouped_scalar_alfworld_runtime": "lh_platform.studies.graphgpo.grouped_alfworld",
    "install_sequential_scalar_alfworld_runtime": "lh_platform.studies.graphgpo.grouped_alfworld",
}


def __getattr__(name: str) -> Any:
    module_name = _EXPORTS.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module(module_name), name)
    globals()[name] = value
    return value


PUBLIC_EXPORTS = tuple(_EXPORTS)
ROOT_PUBLIC_EXPORTS = (
    "GraphGPOStudyRunner",
    "StudyLaunchBlocked",
    "GRAPHGPO_PROGRESS_SCHEMA",
    "GRAPHGPO_PARITY_AUDIT_SCHEMA",
    "build_graphgpo_progress_snapshot",
    "build_graphgpo_parity_audit",
)
__all__ = list(PUBLIC_EXPORTS)
