"""Frozen topology and metering declarations carried by every v0.1 run plan.

These identifiers describe the planned artifact and metering contracts. They do
not prove that a run directory or actual usage source satisfies those contracts.
"""

ARTIFACT_TOPOLOGY_ID_V0_1 = (
    "longhorizon_tracefix_bench/run_artifact_topology/0.1.0-dev.2"
)
METERING_CONTRACT_ID_V0_1 = "longhorizon_tracefix_bench/metering/0.1.0-dev.2"
WORK_ITEM_METERING_BOUNDARY_ID_V0_1 = "work_item_finalize_exclusion/v2"
RUN_METERING_BOUNDARY_ID_V0_1 = "finalizer_start_exclusion/v2"

__all__ = [
    "ARTIFACT_TOPOLOGY_ID_V0_1",
    "METERING_CONTRACT_ID_V0_1",
    "WORK_ITEM_METERING_BOUNDARY_ID_V0_1",
    "RUN_METERING_BOUNDARY_ID_V0_1",
]
