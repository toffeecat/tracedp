"""Define method identities shared by every managed-runtime lifecycle boundary."""

from lh_platform.test_time_adaptation.runtime_joint_policy import (
    FAILURE_DIAGNOSIS_JOINT_METHODS,
    JOINT_POLICY_METHODS,
    ROTATING_RESIDUAL_SOURCE_METHOD,
    SETTLEMENT_ONLY_JOINT_METHODS,
    TRACEDP_RESIDUAL_METHODS,
)

MANAGED_OBSERVATION_METHOD = "managed_observation_only_v1"
# Fresh-debugger arm: the same failure workspace, public history, model, tools and
# total recovery budget as the closed-loop methods, but the Runtime selects no probes
# or repairs and admits no candidates (system-level comparison against Full).
FRESH_DEBUGGER_METHOD = "fresh_debugger_v1"
SHARED_INPUT_PROTECTION_METHOD = "shared_input_protection_v1"
FIXED_TWO_STAGE_RUNTIME_METHOD = "fixed_two_stage_runtime_v1"
DYNAMIC_TRACEDP_V1 = "dynamic_tracedp_runtime_v1"
DYNAMIC_TRACEDP_V2 = "dynamic_tracedp_runtime_v2"
DYNAMIC_TRACEDP_V3 = "dynamic_tracedp_runtime_v3"
DYNAMIC_TRACEDP_V4 = "dynamic_tracedp_runtime_v4"

DYNAMIC_RUNTIME_METHODS = frozenset(
    {DYNAMIC_TRACEDP_V1, DYNAMIC_TRACEDP_V2, DYNAMIC_TRACEDP_V3, DYNAMIC_TRACEDP_V4}
)
POLICY_RUNTIME_METHODS = frozenset({*DYNAMIC_RUNTIME_METHODS, FIXED_TWO_STAGE_RUNTIME_METHOD})
JOINT_RUNTIME_METHODS = frozenset(JOINT_POLICY_METHODS)
INTERVENTION_BUDGET_METHODS = frozenset({*POLICY_RUNTIME_METHODS, *JOINT_RUNTIME_METHODS})
OUTPUT_PROGRESS_RUNTIME_METHODS = frozenset(
    {
        DYNAMIC_TRACEDP_V2,
        DYNAMIC_TRACEDP_V3,
        DYNAMIC_TRACEDP_V4,
        FIXED_TWO_STAGE_RUNTIME_METHOD,
        *TRACEDP_RESIDUAL_METHODS,
        ROTATING_RESIDUAL_SOURCE_METHOD,
        *(JOINT_RUNTIME_METHODS - FAILURE_DIAGNOSIS_JOINT_METHODS - SETTLEMENT_ONLY_JOINT_METHODS),
    }
)
OUTPUT_SEMANTIC_RUNTIME_METHODS = frozenset(
    {
        DYNAMIC_TRACEDP_V3,
        DYNAMIC_TRACEDP_V4,
        FIXED_TWO_STAGE_RUNTIME_METHOD,
        *JOINT_RUNTIME_METHODS,
    }
)
EFFECT_LEDGER_RUNTIME_METHODS = frozenset(
    {
        DYNAMIC_TRACEDP_V3,
        DYNAMIC_TRACEDP_V4,
        FIXED_TWO_STAGE_RUNTIME_METHOD,
        *JOINT_RUNTIME_METHODS,
    }
)
SUPPORTED_RUNTIME_METHODS = frozenset(
    {
        MANAGED_OBSERVATION_METHOD,
        FRESH_DEBUGGER_METHOD,
        SHARED_INPUT_PROTECTION_METHOD,
        *INTERVENTION_BUDGET_METHODS,
    }
)
