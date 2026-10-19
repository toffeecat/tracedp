"""Assembly boundary for the supplied tracer corpus."""

from __future__ import annotations

import subprocess
import sys
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path

from lh_platform.models.controlled_runtime import (
    CONTROLLED_RUNTIME,
    ControlledRuntimeIdentity,
    RuntimeModelPurpose,
    require_runtime_model_identity,
)
from lh_platform.tracing.integrations._runtime import NativeTraceRuntimeV1
from lh_platform.tracing.integrations.aegis import native_aegis_adapter
from lh_platform.tracing.integrations.agentdiagnose import native_agentdiagnose_adapter
from lh_platform.tracing.integrations.critictool import (
    CRITICTOOL_PROMPT_PROFILE_NO_COT,
    native_critictool_adapter,
)
from lh_platform.tracing.integrations.evotool import native_evotool_adapter
from lh_platform.tracing.integrations.expected_effect_debt_bidirectional_dominance import (
    expected_effect_debt_bidirectional_dominance_catalog_descriptor,
    native_expected_effect_debt_bidirectional_dominance_adapter,
)
from lh_platform.tracing.integrations.expected_effect_debt_independent_score import (
    expected_effect_debt_independent_score_catalog_descriptor,
    native_expected_effect_debt_independent_score_adapter,
)
from lh_platform.tracing.integrations.expected_effect_debt_rerank import (
    expected_effect_debt_rerank_catalog_descriptor,
    native_expected_effect_debt_rerank_adapter,
)
from lh_platform.tracing.integrations.longh_residual_causal import (
    longh_residual_causal_catalog_descriptor,
    native_longh_residual_causal_adapter,
)
from lh_platform.tracing.integrations.tracedp_context_static import (
    static_context_catalog_descriptors,
)
from lh_platform.tracing.integrations.traceelephant import native_traceelephant_adapters
from lh_platform.tracing.integrations.tracer_risk import native_tracer_risk_adapter
from lh_platform.tracing.integrations.whowhen import native_whowhen_adapters
from lh_platform.tracing.unified import TraceMethodAdapter, TraceMethodPlatform


def build_catalog_trace_method_platform() -> TraceMethodPlatform:
    """Expose all supplied methods, failing closed until a runtime is configured."""

    from lh_platform.tracing.integrations.blind_trace_baselines import (
        blind_trace_baseline_catalog_descriptors,
    )

    return TraceMethodPlatform.from_descriptors(
        (
            *TraceMethodPlatform.default_catalog(),
            *blind_trace_baseline_catalog_descriptors(),
            longh_residual_causal_catalog_descriptor(),
            *static_context_catalog_descriptors(),
        )
    )


def build_native_trace_method_platform(
    *,
    corpus_root: str | Path,
    model_id: str,
    base_url: str,
    api_key: str = "EMPTY",
    python_executable: str = sys.executable,
    max_tokens: int = 512,
    max_model_calls: int | None = None,
    timeout_seconds: int = 180,
    temperature: float = 0.0,
    generation_seed: int = 0,
    diagnostic_artifact_root: str | Path | None = None,
    critictool_prompt_profile: str = CRITICTOOL_PROMPT_PROFILE_NO_COT,
    provider: str = CONTROLLED_RUNTIME.provider,
    thinking: str = CONTROLLED_RUNTIME.thinking,
    runtime_purpose: RuntimeModelPurpose = "longh_controlled",
    expected_runtime_identity: ControlledRuntimeIdentity | None = None,
) -> TraceMethodPlatform:
    """Overlay admitted native implementations while retaining the full corpus."""

    # Import the LongH-only controls lazily to avoid coupling package discovery
    # to the test-time adaptation package during tracing package initialization.
    from lh_platform.tracing.integrations.blind_trace_baselines import (
        blind_trace_baseline_catalog_descriptors,
        deterministic_blind_trace_adapters,
        model_blind_trace_adapters,
    )

    # Fail before importing upstream code or constructing an API-backed runtime.
    require_runtime_model_identity(
        provider=provider,
        model=model_id,
        thinking=thinking,
        purpose=runtime_purpose,
        expected_identity=expected_runtime_identity,
    )
    runtime = NativeTraceRuntimeV1(
        model_id=model_id,
        base_url=base_url,
        api_key=api_key,
        python_executable=python_executable,
        max_tokens=max_tokens,
        max_model_calls=max_model_calls,
        timeout_seconds=timeout_seconds,
        temperature=temperature,
        generation_seed=generation_seed,
        diagnostic_artifact_root=(
            str(Path(diagnostic_artifact_root).resolve())
            if diagnostic_artifact_root is not None
            else None
        ),
    )
    factories: tuple[tuple[str, Callable[[], tuple[TraceMethodAdapter, ...]]], ...] = (
        ("controls", deterministic_blind_trace_adapters),
        (
            "longh_style_baselines",
            lambda: model_blind_trace_adapters(runtime=runtime),
        ),
        (
            "expected_effect_debt",
            lambda: (
                native_expected_effect_debt_rerank_adapter(runtime=runtime),
                native_expected_effect_debt_independent_score_adapter(runtime=runtime),
                native_expected_effect_debt_bidirectional_dominance_adapter(runtime=runtime),
            ),
        ),
        (
            "longh",
            lambda: (native_longh_residual_causal_adapter(runtime=runtime),),
        ),
        (
            "tracer",
            lambda: (
                native_tracer_risk_adapter(
                    corpus_root=corpus_root,
                    python_executable=python_executable,
                    timeout_seconds=timeout_seconds,
                ),
            ),
        ),
        (
            "who_when",
            lambda: native_whowhen_adapters(corpus_root=corpus_root, runtime=runtime),
        ),
        (
            "traceelephant",
            lambda: native_traceelephant_adapters(corpus_root=corpus_root, runtime=runtime),
        ),
        (
            "aegis",
            lambda: (native_aegis_adapter(corpus_root=corpus_root, runtime=runtime),),
        ),
        (
            "evotool",
            lambda: (native_evotool_adapter(corpus_root=corpus_root, runtime=runtime),),
        ),
        (
            "agentdiagnose",
            lambda: (native_agentdiagnose_adapter(corpus_root=corpus_root, runtime=runtime),),
        ),
        (
            "critictool",
            lambda: (
                native_critictool_adapter(
                    corpus_root=corpus_root,
                    runtime=runtime,
                    prompt_profile=critictool_prompt_profile,
                ),
            ),
        ),
    )
    adapters: list[TraceMethodAdapter] = []
    admission_failures: dict[str, str] = {}
    for corpus_method_id, factory in factories:
        try:
            adapters.extend(factory())
        except (OSError, RuntimeError, ValueError, subprocess.SubprocessError) as error:
            admission_failures[corpus_method_id] = (
                f"native admission failed: {type(error).__name__}: {error}"
            )
    base_catalog = (
        *TraceMethodPlatform.default_catalog(),
        *blind_trace_baseline_catalog_descriptors(),
        expected_effect_debt_rerank_catalog_descriptor(),
        expected_effect_debt_independent_score_catalog_descriptor(),
        expected_effect_debt_bidirectional_dominance_catalog_descriptor(),
        longh_residual_causal_catalog_descriptor(),
        *static_context_catalog_descriptors(),
    )
    catalog = tuple(
        replace(
            descriptor,
            unavailable_reason=admission_failures[descriptor.corpus_method_id],
        )
        if descriptor.corpus_method_id in admission_failures
        else descriptor
        for descriptor in base_catalog
    )
    replacement_ids = tuple(adapter.descriptor.method_id for adapter in adapters)
    platform = TraceMethodPlatform.overlay(
        catalog=catalog,
        adapters=tuple(adapters),
        replace_method_ids=replacement_ids,
    )
    represented = {descriptor.corpus_method_id for descriptor in platform.catalog()}
    expected = {descriptor.corpus_method_id for descriptor in base_catalog}
    if represented != expected:
        raise RuntimeError(
            "native tracer assembly changed corpus coverage: "
            f"missing={sorted(expected - represented)} "
            f"extra={sorted(represented - expected)}"
        )
    return platform


__all__ = [
    "build_catalog_trace_method_platform",
    "build_native_trace_method_platform",
]
