"""Method base class."""

from __future__ import annotations

import os
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

from lh_platform.core.method_capabilities import (
    INTEGRATION_MOCK,
    METHOD_INTEGRATION_LEVELS,
)
from lh_platform.core.types import Action, Observation, Task, Trajectory
from lh_platform.methods._skills import SkillsRegistry

# Backward-compatible export for adapter modules and coverage checks.  The
# canonical ordered vocabulary lives in ``core.method_capabilities``.
INTEGRATION_LEVELS = set(METHOD_INTEGRATION_LEVELS)


# Default location of the persistent skill bank. Methods can override.
DEFAULT_SKILL_STORE = Path(__file__).resolve().parents[2] / "data" / "skills.json"


class MethodBase(ABC):
    """A method is a policy that maps (task, history) -> Action.

    Subclasses are constructed by the registry; the registry passes
    model_id, model_registry (Registry), and model_kwargs (dict) so the
    method can lazily instantiate the model with the correct parameters
    (e.g. model name, temperature, base_url for local vLLM).
    """

    name: str = "base"
    integration_level: str = INTEGRATION_MOCK
    requires_benchmark: str = ""  # empty = generic; otherwise benchmark name required
    # ``None`` means this is a prompt-driven generic policy and it delegates
    # the concrete grammar to the task instruction.  A non-empty set means
    # the method emits only those action languages and Runner must reject an
    # incompatible benchmark before model dispatch.
    supported_action_contracts: frozenset[str] | None = None

    def __init__(
        self,
        model_id: str = "dummy",
        model_registry: Any | None = None,
        model_kwargs: dict[str, Any] | None = None,
        skills: SkillsRegistry | None = None,
        seed: int | None = None,
    ) -> None:
        self.model_id = model_id
        self.model_registry = model_registry
        self.model_kwargs = model_kwargs or {}
        self.seed = seed
        self._model_cache: Any | None = None
        if skills is not None:
            self.skills = skills
        else:
            store = Path(os.environ.get("LH_SKILL_STORE", str(DEFAULT_SKILL_STORE)))
            self.skills = SkillsRegistry(store_path=store)

    def _get_model(self) -> Any:
        if self._model_cache is None:
            if self.model_registry is None:
                raise RuntimeError("method has no model_registry")
            self._model_cache = self.model_registry.create(self.model_id, **self.model_kwargs)
        return self._model_cache

    def bind_model_proxy(self, model_proxy: Any) -> None:
        """Bind a Runner-owned model proxy before method execution.

        Strict local-model experiments execute this method in a dedicated
        JSON-RPC worker. The worker receives only its ``generate`` proxy,
        never the authority's event ledger or control channel. Keeping this
        hook on the common base class means existing adapters do not need a
        special code path while the Runner retains the provenance boundary.
        """

        if model_proxy is None:
            raise ValueError("model_proxy must not be None")
        self._model_cache = model_proxy

    def model_runtime_evidence(self) -> dict[str, Any]:
        """Expose actual model usage to the Runner evidence contract.

        A method that never called the model must not look equivalent to an
        observed local-model run.  Concrete model implementations may expose
        a richer ``runtime_evidence`` method; unknown models are recorded
        conservatively instead of guessed.
        """

        if self._model_cache is None:
            return {
                "model_instantiated": False,
                "dummy_model_used": False,
                "dummy_fallback_used": False,
            }
        producer = getattr(self._model_cache, "runtime_evidence", None)
        if callable(producer):
            try:
                evidence = producer()
            except Exception as error:
                return {
                    "model_instantiated": True,
                    "dummy_model_used": False,
                    "dummy_fallback_used": False,
                    "runtime_evidence_error": type(error).__name__,
                }
            if isinstance(evidence, dict):
                return dict(evidence)
        return {
            "model_instantiated": True,
            "dummy_model_used": False,
            "dummy_fallback_used": False,
            "backend": "unknown",
        }

    def model_generation_cursor(self) -> int:
        """Return the trusted model event cursor before one ``predict`` call.

        This deliberately lives on ``MethodBase`` instead of letting Runner
        inspect adapter internals.  Methods that use a non-platform model can
        still run, but their actions are marked unbound and cannot be exported
        as official SWE-bench model-generated patches.
        """

        if self._model_cache is None:
            return 0
        producer = getattr(self._model_cache, "generation_cursor", None)
        if not callable(producer):
            return 0
        try:
            value = producer()
        except Exception:
            return 0
        return (
            int(value)
            if isinstance(value, int) and not isinstance(value, bool) and value >= 0
            else 0
        )

    def model_generation_events_since(self, cursor: int) -> tuple[dict[str, Any], ...]:
        """Expose defensive copies of platform-model events to Runner only."""

        if self._model_cache is None:
            return ()
        producer = getattr(self._model_cache, "generation_events_since", None)
        if not callable(producer):
            return ()
        try:
            events = producer(cursor)
        except Exception:
            return ()
        if not isinstance(events, (list, tuple)):
            return ()
        return tuple(dict(event) for event in events if isinstance(event, dict))

    def format_skill_hints(self, task: Task, k: int = 3) -> str:
        """Return up to k skill hints for this task. Empty string if no match."""
        return self.skills.format_hints(task.instruction or "", k=k)

    def setup(self, task: Task) -> None:
        """Optional per-task setup hook (called after construction, before predict)."""
        return None

    def reset(self, task: Task) -> None:
        """Reset per-task state.

        By default clears nothing (stateless method). Stateful adapters
        (e.g. memory banks, episodic memory, role histories) MUST override
        and clear their internal state to avoid cross-task leakage.
        Called once per task, BEFORE setup(task).
        """
        return None

    @abstractmethod
    def predict(self, task: Task, trajectory: Trajectory) -> Action: ...

    def update(self, trajectory: Trajectory, action: Action, observation: Observation) -> None:
        """Optional post-step hook."""
        return None
