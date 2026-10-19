"""Lightweight skill library, inspired by Voyager's `SkillManager`.

A skill is a small named, reusable piece of knowledge a method can attach
to its prompt. The simplest case: a one-line tip
("always run `pytest -q` after editing a Python file"). The richer case:
a function the LLM can call by name with structured args.

In Voyager, the skill library is filled by the SkillManager agent at
runtime from successful trajectories. We don't run that agent here;
instead:
  1. The platform ships a default skill bank (`DEFAULT_SKILLS`) with
     a small set of general-purpose tips that match common long-horizon
     failure modes.
  2. Each method may register its own skills via
     `SkillsRegistry.register_skill(...)`; these are persisted in
     `lh_platform/data/skills.json` so the library grows across runs.
  3. `MethodBase` exposes a `skills: SkillsRegistry` attribute and the
     `format_skill_hints(n=3)` helper that returns the top-N most
     relevant skills for a given task description.

This is a platform-level idea, not method-specific. Methods that want
to ignore it can simply not call `format_skill_hints`.
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path

logger = logging.getLogger(__name__)


# Default skill bank. Keys are skill names, values are dicts with:
#   - description: short human-readable description (used for matching)
#   - body: the skill content (LLM-injected as a hint)
#   - tags: list of keywords (also used for matching)
DEFAULT_SKILLS: dict[str, dict[str, object]] = {
    "swe_repro_first": {
        "description": "On SWE-bench / repo fix tasks, write or run a failing test before editing source code.",
        "body": (
            "Before editing source, write a small failing test or run the existing "
            "FAIL_TO_PASS test. This anchors the fix in observable behavior."
        ),
        "tags": ["swe", "swebench", "fix", "repro", "test"],
    },
    "long_horizon_replan": {
        "description": "When stuck for several steps, write a brief re-plan rather than continuing the same line.",
        "body": (
            "If the same action type has failed twice in a row, pause and write a 1-2 sentence "
            "replan: what is the goal, what has been tried, what should be tried next."
        ),
        "tags": ["plan", "replan", "stuck", "stagnation"],
    },
    "tool_error_recover": {
        "description": "After a tool / shell / API error, inspect the error message before retrying.",
        "body": (
            "When a tool or shell call errors, read the error before retrying. "
            "If the error is the same, change the call (different args, different tool) "
            "rather than just resending."
        ),
        "tags": ["tool", "error", "retry", "shell"],
    },
    "gui_recover_observe": {
        "description": "On GUI / web tasks, take a fresh observation before retrying a click.",
        "body": (
            "If a click or type fails, re-read the page snapshot. The DOM may have updated; "
            "the original element selector is likely stale."
        ),
        "tags": ["gui", "web", "click", "dom"],
    },
    "context_compact": {
        "description": "When trajectory length grows, summarize the early steps rather than re-feeding them.",
        "body": (
            "Once a trajectory is long, replace the early steps with a short summary "
            "(goals, subgoals completed, failures) and keep only the last few steps verbatim."
        ),
        "tags": ["memory", "context", "compact", "summary"],
    },
}


def _tokenize(text: str) -> list[str]:
    return [t for t in re.findall(r"[a-z0-9_]+", text.lower()) if len(t) > 2]


def _score(task_text: str, skill: dict[str, object]) -> int:
    """Cheap keyword-overlap score. Higher = more relevant."""
    if not task_text:
        return 0
    task_tokens = set(_tokenize(task_text))
    if not task_tokens:
        return 0
    desc_tokens = set(_tokenize(str(skill.get("description", ""))))
    tag_tokens = set()
    for t in skill.get("tags", []) or []:
        tag_tokens.update(_tokenize(str(t)))
    overlap = task_tokens & (desc_tokens | tag_tokens)
    return len(overlap)


class SkillsRegistry:
    """Per-method (or global) skill library."""

    def __init__(self, store_path: Path | None = None) -> None:
        self._skills: dict[str, dict[str, object]] = dict(DEFAULT_SKILLS)
        self._store_path = store_path
        if store_path is not None and store_path.is_file():
            try:
                data = json.loads(store_path.read_text())
                if isinstance(data, dict):
                    for name, sk in data.items():
                        if isinstance(sk, dict):
                            self._skills[name] = sk
            except Exception:
                logger.exception("skills: failed to load skill store from %s", store_path)

    def list_skills(self) -> list[str]:
        return sorted(self._skills.keys())

    def get_skill(self, name: str) -> dict[str, object] | None:
        return self._skills.get(name)

    def register_skill(
        self, name: str, description: str, body: str, tags: list[str] | None = None
    ) -> None:
        self._skills[name] = {
            "description": description,
            "body": body,
            "tags": list(tags or []),
        }
        if self._store_path is not None:
            self._save()

    def remove_skill(self, name: str) -> bool:
        if name in self._skills and name not in DEFAULT_SKILLS:
            del self._skills[name]
            if self._store_path is not None:
                self._save()
            return True
        return False

    def _save(self) -> None:
        if self._store_path is None:
            return
        try:
            self._store_path.parent.mkdir(parents=True, exist_ok=True)
            self._store_path.write_text(json.dumps(self._skills, indent=2, ensure_ascii=False))
        except Exception:
            logger.exception("skills: failed to persist skill store to %s", self._store_path)

    def top_k(self, task_text: str, k: int = 3) -> list[dict[str, object]]:
        scored = [(s, _score(task_text, s)) for s in self._skills.values()]
        scored.sort(key=lambda kv: (-kv[1], kv[0].get("description", "")))
        return [s for s, _ in scored[: max(0, k)]]

    def format_hints(self, task_text: str, k: int = 3) -> str:
        """Return a prompt-friendly string of the top-k skill hints, or '' if none match."""
        picks = self.top_k(task_text, k=k)
        if not picks:
            return ""
        lines = ["# Skill hints"]
        for s in picks:
            lines.append(f"- {s.get('description', '')}")
        return "\n".join(lines)
