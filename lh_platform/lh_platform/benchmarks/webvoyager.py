"""WebVoyager benchmark adapter.

A local clone of the repository, which includes
`data/WebVoyager_data.jsonl` (643 tasks) and `run.py` (the native runner).

Validity is CONDITIONAL on the local clone being present:
  - If `data/WebVoyager_data.jsonl` exists: `eval_validity = "real_partial"`.
    `load()` reads real tasks; `step()` opens a real Playwright browser,
    navigates to the task URL, and returns page title + visible text as the
    observation. This is NOT the full WebVoyager agent loop (no
    screenshot+VLM, no BBOX clicks) — a text-only real browser interaction
    so long-horizon methods can be evaluated on real web environments.
  - If the local clone is ABSENT: the instance silently falls back to
    `smoke_only` synthetic tasks (`is_mock = True`, `has_real_data = False`,
    `eval_validity = "smoke_only"`). The class-level defaults describe the
    INTENDED adapter, not the active data source — read the instance
    attributes after construction to learn which mode is in effect.

Evidence boundary: a present jsonl only confirms the dataset is loaded; it
does NOT establish that the evaluator or method is paper-faithful.

To reach `real_native`, one would need: (1) an OpenAI API key with a
vision-capable model; (2) reuse of `WebVoyager/repo/run.py` (shells out to
GPT-4V); (3) BBOX-click action parsing in the action schema.
"""

from __future__ import annotations

import json
import random
import re
import warnings
from collections.abc import Callable
from pathlib import Path

from lh_platform.benchmarks._local import (
    eval_benchmark_path,
    read_jsonl,
)
from lh_platform.benchmarks.base import BenchmarkBase
from lh_platform.core.action_language import (
    ActionLanguage,
    ParsedAction,
    bind_task_action_language,
    parse_and_annotate,
)
from lh_platform.core.outcome import EVIDENCE_ABSENT, EVIDENCE_WEAK, OutcomeEvidence
from lh_platform.core.registry import BENCHMARKS
from lh_platform.core.types import Action, Observation, Task, TaskDomain


class WebVoyagerActionLanguage(ActionLanguage):
    name = "webvoyager"
    contract_id = "webvoyager_text/v1"

    def prompt_vocabulary(self, task: Task) -> str:
        del task
        return (
            "Output exactly one of:\n"
            "  click(<visible text>)\n"
            "  type(<search query or input>)\n"
            "  goto(<full URL>)\n"
            "  answer(<your final answer>)\n"
            "  done"
        )

    def conformance_examples(self, task: Task) -> tuple[str, ...]:
        del task
        return (
            "click(Book now)",
            "type(Tokyo hotel)",
            "goto(https://example.com)",
            "answer(complete)",
            "done",
        )

    def parse(self, action: Action) -> ParsedAction:
        a = (action.raw or "").strip()
        a_l = a.lower()

        if a_l.startswith("goto:"):
            return self._with_required_arg("goto", a.split(":", 1)[1].strip(), a)
        if a_l.startswith("click:"):
            return self._with_required_arg("click", a.split(":", 1)[1].strip(), a)
        if a_l.startswith("type:"):
            return self._with_required_arg("type", a.split(":", 1)[1].strip(), a)
        if a_l.startswith("answer:"):
            return self._with_required_arg("answer", a.split(":", 1)[1].strip(), a, terminal=True)
        if a_l.startswith("done") or a_l.startswith("stop"):
            return ParsedAction(name="answer", args={"text": a}, raw=a, terminal=True)

        m = re.search(r"(click|type|answer|goto)\s*\((.*?)\)", a, re.DOTALL)
        if m:
            verb, arg = m.group(1), m.group(2).strip().strip('"').strip("'")
            return self._with_required_arg(verb, arg, a, terminal=(verb == "answer"))

        return ParsedAction(
            name="invalid",
            raw=a,
            valid=False,
            error="action does not match WebVoyager Action Language",
        )

    @staticmethod
    def _with_required_arg(name: str, arg: str, raw: str, terminal: bool = False) -> ParsedAction:
        if not arg:
            return ParsedAction(
                name=name,
                raw=raw,
                valid=False,
                terminal=terminal,
                error=f"{name} requires a non-empty argument",
            )
        return ParsedAction(name=name, args={"text": arg}, raw=raw, terminal=terminal)


@BENCHMARKS.register("webvoyager")
class WebVoyagerBenchmark(BenchmarkBase):
    name = "webvoyager"
    # Class-level evidence is conservative because the dataset path is resolved
    # only when the concrete adapter is constructed. Runtime admission may
    # promote it to real_partial, never beyond that ceiling.
    eval_validity = "smoke_only"
    runtime_eval_validity_dynamic = True
    runtime_eval_validity_max = "real_partial"
    has_real_data = False
    has_native_harness = True
    supports_native_eval = True
    is_mock = True
    required_action_contract = WebVoyagerActionLanguage.contract_id

    def __init__(
        self, max_steps: int = 8, judge_fn: Callable[[str, Task], bool] | None = None
    ) -> None:
        self.max_steps = max_steps
        self._judge_fn = judge_fn
        self.load_path_used: str = ""
        self._data_path = (
            eval_benchmark_path("WebVoyager") / "repo" / "data" / "WebVoyager_data.jsonl"
        )
        self.dataset_path = self._data_path
        self.dataset_split = "WebVoyager_data.jsonl"
        # Class labels describe the intended adapter.  Instance labels record
        # the actual data source used in this run; otherwise a missing local
        # clone silently becomes a synthetic task while still looking real.
        if self._data_path.is_file():
            self.data_source = "webvoyager_jsonl"
            self.has_real_data = True
            self.eval_validity = "real_partial"
            self.is_mock = False
        else:
            self.data_source = "synthetic_fallback"
            self.has_real_data = False
            self.eval_validity = "smoke_only"
            self.is_mock = True
        self.task_selection_strategy = "file_order_first_n"
        self.task_selection_seed: int | None = None
        self.action_language = WebVoyagerActionLanguage()
        self._terminal_answers: dict[str, str] = {}
        self._playwright = None
        self._browser = None
        self._context = None
        self._page = None
        self._current_url: str | None = None
        self._active_task_id: str | None = None

    def load(self, n: int = 3) -> list[Task]:
        self.task_selection_strategy = "file_order_first_n"
        self.task_selection_seed = None
        self.load_path_used = "file_order_first_n"
        if not self._data_path.is_file():
            return self._synthetic(n)
        rows = read_jsonl(self._data_path, limit=n, strict=True)
        if not rows:
            raise ValueError("WebVoyager real dataset is empty")
        return self._tasks_from_rows(rows)

    def load_seeded(self, n: int, seed: int) -> list[Task]:
        """Select real tasks deterministically without relying on file order."""

        self.task_selection_strategy = "seeded_without_replacement"
        self.task_selection_seed = int(seed)
        self.load_path_used = f"seeded_without_replacement:seed={seed}"
        if not self._data_path.is_file():
            return self._synthetic(n)
        rows = read_jsonl(self._data_path, strict=True)
        if not rows:
            raise ValueError("WebVoyager real dataset is empty")
        order = list(range(len(rows)))
        random.Random(seed).shuffle(order)
        selected = [rows[index] for index in order[: min(n, len(order))]]
        return self._tasks_from_rows(selected)

    def _tasks_from_rows(self, rows: list[dict[str, object]]) -> list[Task]:
        tasks: list[Task] = []
        for i, r in enumerate(rows):
            inst = r.get("ques") or r.get("question") or r.get("instruction") or json.dumps(r)[:200]
            tasks.append(
                bind_task_action_language(
                    Task(
                        task_id=str(r.get("id") or r.get("web") or f"webvoyager_{i}"),
                        domain=TaskDomain.WEB,
                        instruction=inst,
                        max_steps=self.max_steps,
                        metadata={
                            "website": r.get("web", "?"),
                            "web_name": r.get("web_name", "?"),
                            "url": r.get("web", ""),
                            "row": r,
                        },
                    ),
                    self.action_language,
                )
            )
        return tasks

    def _synthetic(self, n: int) -> list[Task]:
        self.task_selection_strategy = "synthetic_fixture_sequence"
        self.load_path_used = "synthetic_fixture_sequence"
        self.data_source = "synthetic_fallback"
        self.has_real_data = False
        self.eval_validity = "smoke_only"
        self.is_mock = True
        return [
            bind_task_action_language(
                Task(
                    task_id=f"webvoyager_synth_{i}",
                    domain=TaskDomain.WEB,
                    instruction=f"Synthetic web task #{i}: search and book a hotel in Tokyo for 2 nights.",
                    max_steps=self.max_steps,
                ),
                self.action_language,
            )
            for i in range(n)
        ]

    # ---- Real environment (Playwright) ----
    def _ensure_browser(self) -> None:
        if self._browser is not None:
            return
        from playwright.sync_api import sync_playwright

        self._playwright = sync_playwright().start()
        self._browser = self._playwright.chromium.launch(headless=True)
        self._page = self._browser.new_page()

    def _navigate(self, url: str) -> None:
        if not url:
            return
        self._ensure_browser()
        if self._current_url == url:
            return
        try:
            self._page.goto(url, timeout=15000, wait_until="domcontentloaded")
            self._current_url = url
        except Exception as e:
            warnings.warn(f"[webvoyager] navigate {url} failed: {e}", stacklevel=2)

    def _start_task_session(self, task: Task, url: str) -> None:
        """Open a fresh browser context at the task's own start URL.

        Each task must begin without the previous task's cookies, storage or
        current page, and the environment must not re-navigate away from the
        URL the agent navigated to.  The stale terminal answer of an earlier
        episode of the same task is dropped here as well.
        """

        self._ensure_browser()
        if self._context is not None:
            try:
                self._context.close()
            except Exception as e:
                warnings.warn(f"[webvoyager] context close failed: {e}", stacklevel=2)
        self._context = self._browser.new_context()
        self._page = self._context.new_page()
        self._active_task_id = task.task_id
        self._terminal_answers.pop(task.task_id, None)
        self._current_url = None
        self._navigate(url)

    def step(self, task: Task, action: Action) -> Observation:
        parsed = parse_and_annotate(self.action_language, action)
        if not parsed.valid:
            return Observation(
                content=f"Invalid action: {parsed.error}",
                info={
                    "done": False,
                    "continue": True,
                    "parse_error": True,
                    "parsed_action": parsed.to_dict(),
                },
                is_error=True,
            )
        url = task.metadata.get("url") or task.metadata.get("website")
        if not url and not self._page:
            return Observation(
                content=f"[webvoyager mock] task={task.task_id} no URL",
                info={"done": False, "continue": True},
                is_error=True,
            )
        try:
            if self._active_task_id != task.task_id:
                self._start_task_session(task, url)
        except Exception as e:
            return Observation(
                content=f"[webvoyager] browser init failed: {e}",
                info={"done": False, "continue": True},
                is_error=True,
            )

        op = parsed.name
        arg = str(parsed.args.get("text", ""))

        if op == "answer":
            self._terminal_answers[task.task_id] = arg
            return Observation(
                content=f"final answer: {arg}",
                info={
                    "done": True,
                    "answer": arg,
                    "continue": False,
                    "parsed_action": parsed.to_dict(),
                },
            )

        if op == "goto":
            try:
                self._page.goto(arg, timeout=15000, wait_until="domcontentloaded")
                self._current_url = arg
            except Exception as e:
                return Observation(
                    content=f"goto failed: {e}",
                    info={"done": False, "continue": True},
                    is_error=True,
                )

        if op == "click":
            try:
                # Try to click first matching link/button
                loc = self._page.get_by_text(arg).first
                loc.click(timeout=4000)
            except Exception as e:
                return Observation(
                    content=f"click '{arg}' failed: {e}",
                    info={"done": False, "continue": True},
                    is_error=True,
                )

        if op == "type":
            try:
                # Type into first visible input/textarea
                self._page.keyboard.type(arg, delay=10)
            except Exception as e:
                return Observation(
                    content=f"type failed: {e}",
                    info={"done": False, "continue": True},
                    is_error=True,
                )

        # observe: return current page snapshot
        try:
            title = self._page.title() or ""
            text = self._page.evaluate(
                "() => document.body && document.body.innerText ? document.body.innerText.slice(0, 2000) : ''"
            )
            current_url = self._page.url
        except Exception as e:
            return Observation(
                content=f"observe failed: {e}",
                info={"done": False, "continue": True},
                is_error=True,
            )

        content = f"URL: {current_url}\nTITLE: {title}\nTEXT (first 2000 chars):\n{text}"
        return Observation(
            content=content,
            info={
                "done": False,
                "continue": True,
                "screenshot_path": None,
                "url": current_url,
                "title": title,
                "eval_validity": self.eval_validity,
            },
        )

    def outcome_evidence(self, task: Task) -> OutcomeEvidence:
        answer = self._terminal_answers.get(task.task_id)
        if answer and self._judge_fn is not None:
            try:
                success = self._judge_fn(answer, task)
            except Exception:
                success = False
            return OutcomeEvidence(
                success=success,
                reason="judge_fn verdict" if success else "judge_fn rejected answer",
                evidence_quality=EVIDENCE_WEAK,
                source="webvoyager_judge_fn",
                final_answer=answer,
            )
        if answer:
            return OutcomeEvidence(
                success=False,
                reason="agent submitted final answer without an independent correctness verdict",
                evidence_quality=EVIDENCE_WEAK,
                source="webvoyager_submission_diagnostic",
                final_answer=answer,
            )
        return OutcomeEvidence(
            success=False,
            reason="no terminal answer observed",
            evidence_quality=EVIDENCE_ABSENT,
            source="benchmark_close",
        )

    def close(self, artifacts_dir: Path | None = None) -> None:
        if self._context is not None:
            try:
                self._context.close()
            except Exception:
                pass
        if self._browser is not None:
            try:
                self._browser.close()
            except Exception:
                pass
        if self._playwright is not None:
            try:
                self._playwright.stop()
            except Exception:
                pass
        self._browser = None
        self._context = None
        self._page = None
        self._playwright = None
        self._current_url = None
        self._active_task_id = None
