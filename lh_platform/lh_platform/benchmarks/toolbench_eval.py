"""ToolBench adapter with strict input, environment, and verdict boundaries.

Expects a local ToolBench clone. The repo provides
  - data_example/instruction/{G1,G2,G3}_query.json   (queries)
  - data_example/answer/{G1,G2,G3}_answer/           (reference ChatGPT predictions)
  - toolbench/tooleval/*                              (offline evaluators; need API key)

Default validity level: `fixture_only`
  - `load()` reads real queries from `data_example/instruction/`.
  - `step()` is a real per-step loop:
      * Action grammar:
          Call(api_name=<ToolName.API>, parameters=<json-or-text>)
          Finish(<final answer>)
      * Observation: returns the page state (api_list + history + remaining_tools).
  - The model output is converted to the standard ToolBench answer format and
    written to `artifacts/toolbench_predictions.json` for offline eval.
  - A staged six-test-set corpus needs a repository-reviewed canonical source
    lock before it upgrades the *loader* to ``real_loader``.  Remote execution
    also needs a repository-reviewed identity lock for the exact HTTPS proxy;
    otherwise it is refused rather than labeled ``real_partial``.
    It is still not ToolEval and cannot emit an official ToolBench score.
  - The official `toolbench/tooleval/eval_pass_rate.py` needs GPT-3.5; this
    adapter ships a deterministic heuristic planning-match diagnostic that does
    not require any external API. Neither that heuristic nor a syntactic
    ``Finish`` is promoted to a ToolBench result.
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any

from lh_platform.benchmarks._local import (
    eval_benchmark_path,
)
from lh_platform.benchmarks.base import BenchmarkBase
from lh_platform.benchmarks.toolbench_official import (
    OFFICIAL_TEST_SETS,
    ToolBenchOfficialCorpus,
    ToolBenchOfficialDataError,
    load_official_corpus,
    resolve_official_data_root,
    verify_repository_rapidapi_proxy_identity,
)
from lh_platform.benchmarks.toolbench_remote import (
    UPSTREAM_MAX_OBSERVATION_LENGTH,
    UPSTREAM_MAX_RESPONSE_BYTES,
    UPSTREAM_RAPIDAPI_SERVICE_URL,
    UPSTREAM_RATE_LIMIT_DELAY_S,
    UPSTREAM_RATE_LIMIT_RECOVERY_DELAY_S,
    UPSTREAM_REQUEST_TIMEOUT_S,
    ToolBenchRapidAPIInvocation,
    ToolBenchRemoteClient,
    ToolBenchRemoteProtocolError,
)
from lh_platform.core.action_language import (
    ActionLanguage,
    ParsedAction,
    bind_task_action_language,
    parse_and_annotate,
)
from lh_platform.core.artifact_redaction import redact_artifact
from lh_platform.core.outcome import EVIDENCE_ABSENT, EVIDENCE_WEAK, OutcomeEvidence
from lh_platform.core.registry import BENCHMARKS
from lh_platform.core.types import Action, Observation, Task, TaskDomain

# ToolBench instruction files in order of recommended priority.
_INSTRUCTION_FILES = ["G1_query.json", "G2_query.json", "G3_query.json"]

# Action grammar regex. We accept:
#   Call(api_name=<name>, parameters=<json-or-text>)
#   Finish(<final-answer>)
#   Legacy form: api_name=<name>  (treated as a Call with no parameters)
_CALL_RE = re.compile(
    r"Call\s*\(\s*api_name\s*=\s*['\"]?([A-Za-z0-9_.]+)['\"]?",
    re.IGNORECASE,
)
_FINISH_RE = re.compile(
    r"Finish\s*\(\s*(.*?)\s*\)\s*$",
    re.IGNORECASE | re.DOTALL,
)
_PARAM_JSON_RE = re.compile(
    r"parameters\s*=\s*(\{.*?\})\s*\)",
    re.IGNORECASE | re.DOTALL,
)
_BARE_API_RE = re.compile(r"api_name\s*=\s*['\"]?([A-Za-z0-9_.]+)['\"]?", re.IGNORECASE)


def _parse_upstream_finish_arguments(action_input: str) -> dict[str, Any]:
    """Mirror ``rapidapi_wrapper._step``'s Finish argument handling.

    ToolBench first attempts ``json.loads(..., strict=False)``.  Its fallback
    then recovers the two function fields from malformed text so older
    function-calling models are not rejected before the environment sees the
    same status code.  The platform keeps the historical fixture-only
    ``Finish(answer)`` shortcut outside this helper; remote mode treats that
    shortcut as the upstream status-2 error it would actually receive.
    """

    try:
        decoded = json.loads(action_input, strict=False)
    except (TypeError, json.JSONDecodeError):
        decoded = {}
        if '"return_type": "' in action_input:
            if '"return_type": "give_answer"' in action_input:
                return_type = "give_answer"
            elif '"return_type": "give_up_and_restart"' in action_input:
                return_type = "give_up_and_restart"
            else:
                start = action_input.find('"return_type": "') + len('"return_type": "')
                return_type = action_input[start : action_input.find('",', start)]
            decoded["return_type"] = return_type
        if '"final_answer": "' in action_input:
            start = action_input.find('"final_answer": "') + len('"final_answer": "')
            decoded["final_answer"] = action_input[start:]
    # Upstream expects an object.  Treat a JSON scalar as the same status-2
    # contract error rather than letting a malformed model action crash the
    # unified runner.
    if not isinstance(decoded, dict):
        decoded = {}
    return decoded


class ToolBenchActionLanguage(ActionLanguage):
    name = "toolbench_eval"
    contract_id = "toolbench_call"

    def prompt_vocabulary(self, task: Task) -> str:
        remote = (task.metadata or {}).get("toolbench_execution_mode") == "upstream_remote"
        if remote:
            return (
                "Output exactly one Call(api_name=<exact function name>, "
                "parameters=<JSON object>) or "
                'Finish({"return_type":"give_answer",'
                '"final_answer":"..."}) / '
                'Finish({"return_type":"give_up_and_restart"}).'
            )
        return (
            "Output exactly one Call(api_name=<ToolName.API>, "
            "parameters=<json-or-text>) or Finish(<final answer>)."
        )

    def conformance_examples(self, task: Task) -> tuple[str, ...]:
        call = 'Call(api_name=Weather.Get, parameters={"city":"Tokyo"})'
        if (task.metadata or {}).get("toolbench_execution_mode") == "upstream_remote":
            return (
                call,
                'Finish({"return_type":"give_answer","final_answer":"done"})',
                'Finish({"return_type":"give_up_and_restart"})',
            )
        return (call, "Finish(done)")

    def parse(self, action: Action) -> ParsedAction:
        a = (action.raw or "").strip()
        m = _FINISH_RE.search(a)
        if m:
            finish_input = m.group(1).strip()
            if not finish_input:
                return ParsedAction(
                    name="finish",
                    raw=a,
                    valid=False,
                    terminal=True,
                    error="Finish requires a non-empty final answer",
                )
            if (
                finish_input.startswith("{")
                or '"return_type": "' in finish_input
                or '"final_answer": "' in finish_input
            ):
                finish = _parse_upstream_finish_arguments(finish_input)
                return_type = str(finish.get("return_type") or "").strip()
                args: dict[str, Any] = {
                    "return_type": return_type,
                    "upstream_finish_input": finish_input,
                }
                if not return_type:
                    args["finish_error"] = 'must have "return_type"'
                elif return_type not in {"give_answer", "give_up_and_restart"}:
                    args["finish_error"] = '"return_type" is not a valid choice'
                if return_type == "give_answer":
                    if "final_answer" not in finish:
                        args["finish_error"] = 'must have "final_answer"'
                    else:
                        # The original wrapper only tests field presence; it
                        # does not impose a non-empty string type.
                        args["answer"] = str(finish["final_answer"])
                return ParsedAction(name="finish", args=args, raw=a, terminal=True)
            # Retain the historical lightweight action format for fixture
            # tests, but make its translation into upstream function arguments
            # explicit in the recorded action metadata.
            answer = finish_input
            return ParsedAction(
                name="finish",
                args={
                    "return_type": "give_answer",
                    "answer": answer,
                    "finish_format": "legacy_text_translation",
                    "upstream_finish_input": json.dumps(
                        {"return_type": "give_answer", "final_answer": answer}
                    ),
                },
                raw=a,
                terminal=True,
            )

        m = _CALL_RE.search(a)
        if m:
            api_name = m.group(1).strip()
            params_match = _PARAM_JSON_RE.search(a)
            parameters: Any = {}
            if params_match:
                try:
                    parameters = json.loads(params_match.group(1))
                except Exception as e:
                    return ParsedAction(
                        name="call",
                        raw=a,
                        valid=False,
                        error=f"parameters must be valid JSON: {e}",
                    )
            return ParsedAction(
                name="call",
                args={"api_name": api_name, "parameters": parameters},
                raw=a,
            )

        m = _BARE_API_RE.search(a)
        if m:
            return ParsedAction(
                name="call",
                args={"api_name": m.group(1).strip(), "parameters": {}},
                raw=a,
            )

        return ParsedAction(
            name="invalid",
            raw=a,
            valid=False,
            error="action does not match ToolBench Action Language",
        )


def _load_queries(repo_path: Path, instruction_files: list[str] | None = None) -> list[dict]:
    """Load ToolBench queries from data_example/instruction/."""
    instr_dir = repo_path / "data_example" / "instruction"
    if not instr_dir.is_dir():
        return []
    files = instruction_files or _INSTRUCTION_FILES
    out: list[dict] = []
    for fname in files:
        fp = instr_dir / fname
        if not fp.is_file():
            continue
        try:
            rows = json.loads(fp.read_text())
        except Exception:
            continue
        if not isinstance(rows, list):
            continue
        group = fname.replace("_query.json", "")
        for r in rows:
            qid = r.get("query_id")
            if qid is None:
                continue
            out.append(
                {
                    "query_id": int(qid),
                    "query": r.get("query", ""),
                    "api_list": r.get("api_list", []),
                    "relevant_apis": r.get("relevant APIs", r.get("relevant_apis", [])),
                    "group": group,
                }
            )
    return out


def _parameter_names(parameters: list[Any]) -> str:
    """Render both fixture dicts and legacy string parameter lists safely."""

    names: list[str] = []
    for parameter in parameters:
        if isinstance(parameter, dict):
            names.append(str(parameter.get("name", "?")))
        else:
            names.append(str(parameter))
    return ",".join(names)


def _load_reference_answers(repo_path: Path) -> dict[int, dict]:
    """Load reference answers from data_example/answer/."""
    ans_dir = repo_path / "data_example" / "answer"
    if not ans_dir.is_dir():
        return {}
    out: dict[int, dict] = {}
    for group_dir in ans_dir.iterdir():
        if not group_dir.is_dir() or group_dir.name.startswith("."):
            continue
        for fp in group_dir.iterdir():
            if not fp.is_file() or not fp.name.endswith(".json"):
                continue
            m = re.match(r"(\d+)_", fp.name)
            if not m:
                continue
            try:
                qid = int(m.group(1))
                out[qid] = json.loads(fp.read_text())
            except Exception:
                continue
    return out


def _load_official_queries(corpus: ToolBenchOfficialCorpus) -> list[dict]:
    """Build adapter rows from the upstream-compatible function vocabulary.

    The official corpus has the original ToolBench ``api_list`` plus the
    function names constructed by its own ``rapidapi_wrapper`` convention.
    We retain both: the latter is what a model should emit, while the former
    remains useful for a clearly marked planning diagnostic.
    """

    out: list[dict] = []
    for test_set in OFFICIAL_TEST_SETS:
        for source_row in corpus.rows_for(test_set):
            query_id = str(source_row["query_id"])
            functions = corpus.available_tools(test_set, query_id)
            tool_invocations = corpus.rapidapi_invocations(test_set, query_id)
            api_list = source_row.get("api_list") or []
            action_functions = [fn for fn in functions if fn.get("name") != "Finish"]
            if len(action_functions) != len(api_list):
                raise ToolBenchOfficialDataError(
                    f"{test_set} query_id={query_id}: function vocabulary does not align with api_list"
                )
            aliases: dict[str, str] = {}
            reverse_aliases: dict[str, str] = {}
            for api, function in zip(api_list, action_functions, strict=False):
                legacy = f"{api.get('tool_name', '?')}.{api.get('api_name', '?')}"
                name = str(function["name"])
                aliases[name] = legacy
                # The upstream task data normally has no duplicate relevant
                # API names per query.  If it does, retain the first mapping
                # and keep the row usable only as a diagnostic.
                reverse_aliases.setdefault(legacy, name)
            relevant = source_row.get("relevant APIs", source_row.get("relevant_apis", [])) or []
            relevant_functions = []
            for item in relevant:
                if isinstance(item, (list, tuple)) and len(item) >= 2:
                    relevant_functions.append(
                        reverse_aliases.get(f"{item[0]}.{item[1]}", f"{item[0]}.{item[1]}")
                    )
            out.append(
                {
                    "query_id": query_id,
                    "query": source_row.get("query", ""),
                    "api_list": api_list,
                    "relevant_apis": relevant,
                    "relevant_function_names": relevant_functions,
                    "available_tools": functions,
                    "tool_invocations": tool_invocations,
                    "api_aliases": aliases,
                    "group": test_set,
                    "test_set": test_set,
                }
            )
    return out


@BENCHMARKS.register("toolbench_eval")
class ToolBenchEvalBenchmark(BenchmarkBase):
    name = "toolbench_eval"
    # Both fixture and remote modes parse ToolBench Call/Finish actions.  The
    # Finish argument detail is carried in task metadata, but a Python-code
    # action policy is incompatible with either mode.
    required_action_contract = "toolbench_call"
    eval_validity = "fixture_only"
    has_real_data = True
    has_native_harness = False  # upstream eval harness needs GPT-3.5
    supports_native_eval = False
    is_mock = False
    # Config matrix admission must defer the validity decision to the instance:
    # `repo/data/` may be absent locally but staged on the server.  Runner
    # re-applies the same gates after loading that concrete instance.
    runtime_eval_validity_dynamic = True
    # A complete manifest can upgrade to ``real_loader``.  The instance can
    # reach ``real_partial`` only with an explicitly reviewed source anchor,
    # proxy identity anchor, live ToolBench key, and remote protocol.
    # ``real_native`` remains unavailable because ToolEval is not integrated.
    runtime_eval_validity_max = "real_partial"

    def __init__(
        self,
        max_steps: int = 8,
        group: str = "G1",
        data_root: str | Path | None = None,
        execution_mode: str = "synthetic",
        toolbench_key_env: str = "TOOLBENCH_KEY",
        service_url: str = UPSTREAM_RAPIDAPI_SERVICE_URL,
        request_timeout_s: float = UPSTREAM_REQUEST_TIMEOUT_S,
        request_rate_limit_delay_s: float = UPSTREAM_RATE_LIMIT_DELAY_S,
        rate_limit_recovery_delay_s: float = UPSTREAM_RATE_LIMIT_RECOVERY_DELAY_S,
        max_observation_length: int = UPSTREAM_MAX_OBSERVATION_LENGTH,
        max_response_bytes: int = UPSTREAM_MAX_RESPONSE_BYTES,
        observ_compress_method: str = "truncate",
    ) -> None:
        self.max_steps = max_steps
        self._bench_dir = eval_benchmark_path("ToolBench")
        self._repo = self._bench_dir / "repo"
        self._queries: list[dict] = []
        self._reference: dict[int, dict] = {}
        self._official_corpus: ToolBenchOfficialCorpus | None = None
        self._official_data_error: ToolBenchOfficialDataError | None = None
        self._official_root: Path | None = None
        self._official_root_source = "none"
        self.dataset_source_verified: bool | None = None
        self.dataset_source_verification_reason: str | None = None
        self.dataset_source_lock_sha256: str | None = None
        self.remote_proxy_identity_verified: bool | None = None
        self.remote_proxy_identity_reason: str | None = None
        self.remote_proxy_identity_lock_sha256: str | None = None
        self._execution_mode = str(execution_mode or "").strip()
        if self._execution_mode not in {"synthetic", "upstream_remote"}:
            raise ValueError("ToolBench execution_mode must be 'synthetic' or 'upstream_remote'")
        self._toolbench_key_env = str(toolbench_key_env or "").strip()
        self._remote: ToolBenchRemoteClient | None = None
        self._remote_error: str | None = None
        # A full manifest-locked data root is sufficient for the loader; the
        # platform must not require an unrelated local ToolBench code clone on
        # an evaluation server.  The clone is only needed for data_example and
        # its legacy reference files.
        root, root_source = resolve_official_data_root(self._repo, data_root)
        self._official_root = root
        self._official_root_source = root_source
        if root is not None:
            try:
                self._official_corpus = load_official_corpus(
                    root,
                )
            except ToolBenchOfficialDataError as error:
                self._official_data_error = error
        if self._official_corpus is not None:
            self._queries = _load_official_queries(self._official_corpus)
        elif self._official_data_error is None and self._repo.is_dir():
            self._queries = _load_queries(self._repo)
        if self._repo.is_dir():
            self._reference = _load_reference_answers(self._repo)
        if self._official_corpus is not None:
            # A manifest protects local bytes; only a separately reviewed
            # source lock establishes that those bytes belong to the intended
            # upstream release.
            self.data_source = (
                "toolbench_canonical_source_locked_six_testsets"
                if self._official_corpus.source_verified
                else "toolbench_manifest_locked_unverified_source"
            )
            self.has_real_data = True
            self.eval_validity = (
                "real_loader" if self._official_corpus.source_verified else "unverified_loader"
            )
            self.is_mock = False
            # The manifest is the portable, immutable input checksum.  A
            # directory hash would neither be stable nor identify the exact
            # six-set roster the run consumed.
            self.dataset_path = self._official_corpus.manifest_path
            self.dataset_manifest_path = self._official_corpus.manifest_path
            self.dataset_manifest_sha256 = self._official_corpus.manifest_sha256
            self.dataset_revision = self._official_corpus.source_reference
            self.dataset_source_verified = self._official_corpus.source_verified
            self.dataset_source_verification_reason = (
                self._official_corpus.source_verification_reason
            )
            self.dataset_source_lock_sha256 = self._official_corpus.source_lock_sha256
            self.dataset_split = "ToolBench six upstream-named test sets (manifest-locked)"
            self.environment_fingerprint = "synthetic_tool_acknowledgement/v1"
            self.evaluator_version = None
        elif self._official_data_error is not None:
            # An explicit data root or upstream-standard repo/data exists but
            # lacks a valid manifest.  Falling back to `data_example` here
            # would let a server run look like it used its downloaded corpus.
            self.data_source = "toolbench_official_invalid"
            self.has_real_data = False
            self.eval_validity = "smoke_only"
            self.is_mock = True
            self.dataset_path = None
        else:
            # The data_example fixture is useful for action-format and
            # planning diagnostics, but it is not the official evaluator.
            self.data_source = "toolbench_data_example" if self._queries else "none"
            self.has_real_data = bool(self._queries)
            self.eval_validity = "fixture_only" if self._queries else "smoke_only"
            self.is_mock = not bool(self._queries)
            self.dataset_path = None
        self._group = group
        self._configure_execution_mode(
            service_url=service_url,
            request_timeout_s=request_timeout_s,
            request_rate_limit_delay_s=request_rate_limit_delay_s,
            rate_limit_recovery_delay_s=rate_limit_recovery_delay_s,
            max_observation_length=max_observation_length,
            max_response_bytes=max_response_bytes,
            observ_compress_method=observ_compress_method,
        )
        self._per_query_predictions: dict[Any, list[dict]] = {}
        self._per_query_steps: dict[Any, int] = {}
        self._iter_seen_tasks: list[Task] = []
        self._last_summary: dict[str, Any] = {}
        self._last_per_task_metrics: dict[Any, dict] = {}
        self.action_language = ToolBenchActionLanguage()

    def _configure_execution_mode(
        self,
        *,
        service_url: str,
        request_timeout_s: float,
        request_rate_limit_delay_s: float,
        rate_limit_recovery_delay_s: float,
        max_observation_length: int,
        max_response_bytes: int,
        observ_compress_method: str,
    ) -> None:
        """Enable only an explicitly admissible non-synthetic environment.

        A full data manifest and a key are separate requirements.  Never
        degrade an explicitly requested remote run into synthetic execution:
        the instance keeps a lower validity tier and ``load`` raises before a
        method/model can be dispatched.
        """

        if self._execution_mode == "synthetic":
            return
        if self._official_corpus is None:
            self._remote_error = (
                "upstream_remote requires a manifest-locked full ToolBench corpus; "
                "stage the official six-test-set data first"
            )
            self.data_source = "toolbench_remote_unavailable"
            self.eval_validity = "fixture_only" if self._queries else "smoke_only"
            self.supports_native_eval = False
            self.is_mock = True
            return
        if not self._official_corpus.source_verified:
            self._remote_error = (
                "upstream_remote requires a reviewed canonical ToolBench source lock; "
                "a manifest/source_reference alone is not release provenance"
            )
            self.data_source = "toolbench_manifest_locked_unverified_source"
            self.eval_validity = "unverified_loader"
            self.supports_native_eval = False
            return
        proxy_identity = verify_repository_rapidapi_proxy_identity(
            service_url=service_url,
        )
        self.remote_proxy_identity_verified = proxy_identity.status == "verified"
        self.remote_proxy_identity_reason = proxy_identity.reason
        self.remote_proxy_identity_lock_sha256 = proxy_identity.sha256
        if proxy_identity.status != "verified":
            self._remote_error = (
                "upstream_remote requires a repository-reviewed RapidAPI proxy identity lock "
                f"for this exact service_url: {proxy_identity.reason}"
            )
            self.data_source = "toolbench_manifest_locked_remote_identity_unverified"
            self.eval_validity = "real_loader"
            self.supports_native_eval = False
            return
        if not self._toolbench_key_env:
            self._remote_error = "upstream_remote requires a non-empty toolbench_key_env name"
            self.data_source = "toolbench_remote_key_env_invalid"
            self.eval_validity = "real_loader"
            self.supports_native_eval = False
            return
        key = os.environ.get(self._toolbench_key_env, "").strip()
        if not key:
            self._remote_error = (
                f"upstream_remote requires ${self._toolbench_key_env} in the process environment"
            )
            self.data_source = "toolbench_manifest_locked_remote_key_missing"
            self.eval_validity = "real_loader"
            self.supports_native_eval = False
            return
        proxy_lock = proxy_identity.payload
        if proxy_lock is None:
            self._remote_error = "reviewed RapidAPI proxy identity lock lost its payload"
            self.data_source = "toolbench_manifest_locked_remote_identity_invalid"
            self.eval_validity = "real_loader"
            self.supports_native_eval = False
            return
        try:
            remote = ToolBenchRemoteClient(
                toolbench_key=key,
                service_url=service_url,
                request_timeout_s=request_timeout_s,
                rate_limit_delay_s=request_rate_limit_delay_s,
                rate_limit_recovery_delay_s=rate_limit_recovery_delay_s,
                max_observation_length=max_observation_length,
                max_response_bytes=max_response_bytes,
                observ_compress_method=observ_compress_method,
                expected_tls_certificate_sha256=str(proxy_lock["tls_certificate_sha256"]),
            )
        except ToolBenchRemoteProtocolError as error:
            self._remote_error = str(error)
            self.data_source = "toolbench_manifest_locked_remote_config_invalid"
            self.eval_validity = "real_loader"
            self.supports_native_eval = False
            return
        self._remote = remote
        self.data_source = "toolbench_manifest_locked_upstream_remote_protocol"
        self.eval_validity = "real_partial"
        self.supports_native_eval = True
        self.is_mock = False
        self.environment_fingerprint = remote.environment_fingerprint
        self.evaluator_version = "toolbench_rapidapi_wrapper_protocol/v1;ToolEval=absent"

    def load(self, n: int = 3) -> list[Task]:
        if self._official_data_error is not None:
            raise self._official_data_error
        if self._remote_error is not None:
            raise ToolBenchRemoteProtocolError(self._remote_error)
        if self._official_corpus is not None:
            rows = self._official_rows_for_group()
        elif self._group in {"all", "*"}:
            rows = list(self._queries)
        else:
            rows = [q for q in self._queries if q.get("group") == self._group]
        if not rows:
            # A configured group that matches no staged row must fail closed:
            # silently scoring a mixed corpus under the requested group label
            # makes the recorded roster disagree with the executed one.
            raise ValueError(
                f"ToolBench group {self._group!r} matched no rows in the staged corpus"
            )
        rows = rows[: max(0, n)]
        tasks: list[Task] = []
        for r in rows:
            qid = r["query_id"]
            inst = self._format_instruction(r)
            task_group = str(r.get("test_set") or r.get("group") or self._group)
            tasks.append(
                bind_task_action_language(
                    Task(
                        task_id=f"toolbench_{task_group}_{qid}",
                        domain=TaskDomain.TOOL,
                        instruction=inst,
                        max_steps=self.max_steps,
                        metadata={
                            "query_id": qid,
                            "group": task_group,
                            "query": r["query"],
                            "api_list": r["api_list"],
                            "relevant_apis": r["relevant_apis"],
                            "row": r,
                            "test_set": r.get("test_set"),
                            "available_tools": r.get("available_tools"),
                            "tool_invocations": r.get("tool_invocations", {}),
                            "api_aliases": r.get("api_aliases", {}),
                            "relevant_function_names": r.get("relevant_function_names", []),
                            # ``TOOL`` is not enough to recover the executable
                            # grammar.  Paper-faithful methods consult this
                            # explicit task contract so an upstream-remote task
                            # never receives the fixture-only Finish syntax.
                            "toolbench_execution_mode": self._execution_mode,
                            "toolbench_finish_contract": (
                                "upstream_function_json"
                                if self._execution_mode == "upstream_remote"
                                else "legacy_text"
                            ),
                        },
                    ),
                    self.action_language,
                )
            )
        self._iter_seen_tasks = list(tasks)
        return tasks

    def _official_rows_for_group(self) -> list[dict]:
        if self._group in {"all", "*"}:
            return list(self._queries)
        if self._group in OFFICIAL_TEST_SETS:
            return [row for row in self._queries if row.get("test_set") == self._group]
        if self._group in {"G1", "G2", "G3"}:
            return [
                row
                for row in self._queries
                if str(row.get("test_set", "")).startswith(f"{self._group}_")
            ]
        raise ToolBenchOfficialDataError(
            f"unknown ToolBench group {self._group!r}; use all, G1/G2/G3, or one of {OFFICIAL_TEST_SETS}"
        )

    def _format_instruction(self, row: dict) -> str:
        available_tools = row.get("available_tools")
        if isinstance(available_tools, list):
            tool_lines = []
            for function in available_tools:
                if not isinstance(function, dict):
                    continue
                name = str(function.get("name", "?"))
                parameters = function.get("parameters") or {}
                required = ",".join(parameters.get("required") or [])
                optional = ",".join(parameters.get("optional") or [])
                tool_lines.append(f"- {name} (required=[{required}], optional=[{optional}])")
            tool_block = "\n".join(tool_lines) if tool_lines else "(no functions)"
            execution_contract = (
                "Each Call is sent through ToolBench's upstream rapidapi_wrapper HTTP protocol. "
                "A Finish action only ends this trajectory syntactically; ToolEval has not judged it.\n"
                "Respond with one of:\n"
                "  Call(api_name=<exact function name>, parameters=<JSON object>)\n"
                '  Finish({"return_type":"give_answer","final_answer":"..."})\n'
                '  Finish({"return_type":"give_up_and_restart"})'
                if self._remote is not None
                else "This adapter currently records a planning trace only: it does not execute RapidAPI.\n"
                "Respond with one of:\n"
                "  Call(api_name=<exact function name>, parameters=<json-or-text>)\n"
                "  Finish(<final answer>)"
            )
            return (
                f"Query: {row['query']}\n"
                f"Upstream ToolBench function vocabulary:\n{tool_block}\n" + execution_contract
            )
        api_lines = []
        for a in row["api_list"]:
            req_params = _parameter_names(a.get("required_parameters") or [])
            opt_params = _parameter_names(a.get("optional_parameters") or [])
            api_lines.append(
                f"- {a.get('tool_name', '?')}.{a.get('api_name', '?')} "
                f"(method={a.get('method', '?')}, required=[{req_params}], "
                f"optional=[{opt_params}])"
            )
        api_block = "\n".join(api_lines) if api_lines else "(no APIs)"
        return (
            f"Query: {row['query']}\n"
            f"Available APIs (you may call multiple):\n{api_block}\n"
            f"Respond with one of:\n"
            f"  Call(api_name=<ToolName.API>, parameters=<json-or-text>)\n"
            f"  Finish(<final answer>)"
        )

    def _api_list_remaining(self, task: Task, called: set[str]) -> list[dict]:
        available = task.metadata.get("available_tools")
        if isinstance(available, list):
            return [
                function
                for function in available
                if isinstance(function, dict)
                and function.get("name") not in called
                and function.get("name") != "Finish"
            ]
        all_apis = task.metadata.get("api_list", [])
        out = []
        for a in all_apis:
            key = f"{a.get('tool_name', '?')}.{a.get('api_name', '?')}"
            if key not in called:
                out.append(a)
        return out

    @staticmethod
    def _task_state_key(task: Task) -> Any:
        """Return an unambiguous state key without breaking fixture artifacts."""

        query_id = task.metadata.get("query_id")
        test_set = task.metadata.get("test_set")
        return f"{test_set}:{query_id}" if test_set else query_id

    def step(self, task: Task, action: Action) -> Observation:
        parsed = parse_and_annotate(self.action_language, action)
        state_key = self._task_state_key(task)
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
        if parsed.name == "finish":
            return self._finish_observation(state_key, parsed)
        if parsed.name == "call":
            steps = self._per_query_steps.get(state_key, 0) + 1
            self._per_query_steps[state_key] = steps
            api_name = str(parsed.args.get("api_name", ""))
            calls = self._per_query_predictions.setdefault(state_key, [])
            call_record = {
                "role": "Action",
                "api_name": api_name,
                "parameters": parsed.args.get("parameters", {}),
            }
            calls.append(call_record)
            if self._remote is not None:
                return self._step_upstream_remote(
                    task,
                    parsed,
                    calls,
                    call_record,
                )
            called = {c.get("api_name", "") for c in calls if c.get("role") == "Action"}
            remaining = self._api_list_remaining(task, called)
            remaining_lines = []
            for candidate in remaining:
                if "name" in candidate:
                    remaining_lines.append(f"- {candidate.get('name', '?')}")
                else:
                    remaining_lines.append(
                        f"- {candidate.get('tool_name', '?')}.{candidate.get('api_name', '?')}"
                    )
            parameters = parsed.args.get("parameters", {})
            return Observation(
                content=(
                    f"Tool call accepted: {api_name}({parameters})\n"
                    f"Remaining APIs:\n"
                    + ("\n".join(remaining_lines) if remaining_lines else "(none)")
                ),
                info={
                    "done": False,
                    "continue": True,
                    "calls": calls,
                    "remaining_apis": remaining,
                    "called": list(called),
                    "parsed_action": parsed.to_dict(),
                },
            )
        raise AssertionError(f"unhandled parsed action: {parsed.name}")

    def _finish_observation(self, state_key: Any, parsed: ParsedAction) -> Observation:
        """Apply ToolBench's ``Finish`` function semantics without judging truth."""

        calls = self._per_query_predictions.setdefault(state_key, [])
        finish_error = str(parsed.args.get("finish_error") or "")
        if (
            self._remote is not None
            and parsed.args.get("finish_format") == "legacy_text_translation"
        ):
            # Fixture mode retains the convenience syntax, but the upstream
            # wrapper receives that literal text and returns status 2 because
            # it lacks the required function argument object.
            finish_error = 'must have "return_type"'
        if finish_error:
            calls.append(
                {
                    "role": "Finish",
                    "return_type": parsed.args.get("return_type"),
                    "upstream_finish_input": parsed.args.get("upstream_finish_input"),
                    "finish_error": finish_error,
                }
            )
            return Observation(
                content=json.dumps({"error": finish_error}),
                is_error=True,
                info={
                    "done": False,
                    "success": False,
                    "continue": True,
                    "toolbench_status_code": 2,
                    "calls": calls,
                    "parsed_action": parsed.to_dict(),
                },
            )
        return_type = str(parsed.args.get("return_type") or "give_answer")
        finish_record: dict[str, Any] = {
            "role": "Finish",
            "return_type": return_type,
            "upstream_finish_input": parsed.args.get("upstream_finish_input"),
        }
        if parsed.args.get("finish_format"):
            finish_record["finish_format"] = parsed.args["finish_format"]
        if return_type == "give_up_and_restart":
            calls.append(finish_record)
            content = (
                json.dumps({"response": "chose to give up and restart"})
                if self._remote is not None
                else "Finish: give_up_and_restart"
            )
            return Observation(
                content=content,
                info={
                    "done": True,
                    "success": False,
                    "continue": False,
                    "toolbench_status_code": 4,
                    "calls": calls,
                    "parsed_action": parsed.to_dict(),
                },
            )
        answer = str(parsed.args.get("answer", ""))
        finish_record["role"] = "Final Answer"
        finish_record["content"] = answer
        calls.append(finish_record)
        content = (
            json.dumps({"response": "successfully giving the final answer."})
            if self._remote is not None
            else f"Finish: {answer}"
        )
        return Observation(
            content=content,
            info={
                "done": True,
                # ToolBench's wrapper marks a syntactic Finish as internally
                # successful. The platform never turns that into benchmark
                # success because no ToolEval verdict exists yet.
                "success": False,
                "answer": answer,
                "continue": False,
                "toolbench_status_code": 3,
                "calls": calls,
                "parsed_action": parsed.to_dict(),
            },
        )

    def _step_upstream_remote(
        self,
        task: Task,
        parsed: ParsedAction,
        calls: list[dict],
        call_record: dict,
    ) -> Observation:
        """Execute one manifest-bound ToolBench remote action.

        The original wrapper resolves the first staged function whose name
        ends with the model-emitted action name.  The staged invocation map
        preserves that function order, so this bridge reproduces it without
        ever routing outside the manifest-bound task vocabulary.
        """

        assert self._remote is not None
        api_name = str(parsed.args.get("api_name", ""))
        raw_invocations = task.metadata.get("tool_invocations") or {}
        invocation_name = api_name
        invocation_raw = (
            raw_invocations.get(api_name) if isinstance(raw_invocations, dict) else None
        )
        if invocation_raw is None and isinstance(raw_invocations, dict):
            for candidate_name, candidate_invocation in raw_invocations.items():
                if str(candidate_name).endswith(api_name):
                    invocation_name = str(candidate_name)
                    invocation_raw = candidate_invocation
                    break
        if not isinstance(invocation_raw, dict):
            execution = self._remote.invalid_function(api_name)
            call_record["invocation_error"] = "not in manifest-bound function vocabulary"
        else:
            try:
                invocation = ToolBenchRapidAPIInvocation.from_mapping(invocation_raw)
                parameters = parsed.args.get("parameters", {})
                execution = self._remote.execute(invocation, json.dumps(parameters))
                call_record["upstream_invocation"] = invocation.public_dict()
                # Preserve the raw model action in ``api_name`` for audit,
                # but use the exact staged function selected by upstream
                # suffix routing for validity/recall metrics.
                call_record["resolved_api_name"] = invocation_name
                if invocation_name != api_name:
                    call_record["upstream_suffix_resolved_function"] = invocation_name
            except ToolBenchRemoteProtocolError as error:
                execution = self._remote.invalid_function(api_name)
                call_record["invocation_error"] = str(error)
        call_record["tool_execution"] = "upstream_remote"
        call_record["toolbench_status_code"] = execution.status_code
        call_record["upstream_receipt"] = execution.receipt
        called = {c.get("api_name", "") for c in calls if c.get("role") == "Action"}
        remaining = self._api_list_remaining(task, called)
        return Observation(
            content=execution.observation,
            is_error=execution.is_error,
            info={
                "done": False,
                "continue": True,
                "calls": calls,
                "remaining_apis": remaining,
                "called": list(called),
                "tool_execution": "upstream_remote",
                "toolbench_status_code": execution.status_code,
                "upstream_receipt": execution.receipt,
                "parsed_action": parsed.to_dict(),
            },
        )

    def _per_task_metrics(self, task: Task) -> dict[str, Any]:
        state_key = self._task_state_key(task)
        calls = self._per_query_predictions.get(state_key, [])
        relevant = task.metadata.get("relevant_apis") or []
        api_names_called = [
            c.get("resolved_api_name", c.get("api_name", ""))
            for c in calls
            if c.get("role") == "Action"
        ]
        aliases = task.metadata.get("api_aliases") or {}
        valid_apis = (
            set(aliases)
            if aliases
            else {
                f"{a.get('tool_name')}.{a.get('api_name')}"
                for a in task.metadata.get("api_list", [])
            }
        )
        invalid_calls = [n for n in api_names_called if n and n not in valid_apis]
        called_set = set(api_names_called)
        n_calls = len(api_names_called)
        # A method that emitted no tool call has no defined call-accuracy; report
        # it as absent instead of letting the empty denominator score as 1.0.
        tool_call_accuracy = (1.0 - len(invalid_calls) / n_calls) if n_calls else None
        invalid_tool_rate = (len(invalid_calls) / n_calls) if n_calls else None
        relevant_set = set(task.metadata.get("relevant_function_names") or [])
        if not relevant_set:
            relevant_set = {f"{t}.{a}" for t, a in relevant}
        recall = len(called_set & relevant_set) / len(relevant_set) if relevant_set else 0.0
        finished = any(c.get("role") == "Final Answer" for c in calls)
        gave_up = any(
            c.get("role") == "Finish" and c.get("return_type") == "give_up_and_restart"
            for c in calls
        )
        remote_calls = [
            c
            for c in calls
            if c.get("role") == "Action" and c.get("tool_execution") == "upstream_remote"
        ]
        remote_error_calls = [
            c for c in remote_calls if int(c.get("toolbench_status_code", 0) or 0) != 0
        ]
        heuristic_plan_matched = bool(finished and recall >= 1.0 and not invalid_calls)
        return {
            # This is deliberately not a pass/fail verdict. ToolBench's
            # official ToolEval was not invoked and no tool was executed.
            "heuristic_plan_matched": heuristic_plan_matched,
            "tool_call_accuracy": tool_call_accuracy,
            "invalid_tool_rate": invalid_tool_rate,
            "relevant_api_recall": recall,
            "finished": finished,
            "gave_up": gave_up,
            "n_calls": n_calls,
            "n_upstream_remote_calls": len(remote_calls),
            "n_upstream_remote_error_calls": len(remote_error_calls),
            "calls": calls,
        }

    def close(self, artifacts_dir: Path | None = None) -> None:
        # Write artifacts to run directory when provided, otherwise to repo dir
        out_dir = artifacts_dir or self._bench_dir / "artifacts"
        out_dir.mkdir(parents=True, exist_ok=True)
        results = []
        remote_receipts = []
        per_task_metrics: dict[Any, dict] = {}
        for task in self._iter_seen_tasks:
            qid = task.metadata.get("query_id")
            state_key = self._task_state_key(task)
            calls = self._per_query_predictions.get(state_key, [])
            # If the method never produced a single Call/Finish (e.g. it
            # speaks free-form text), fabricate an empty calls list so the
            # per-task metrics are still well-defined.
            self._per_query_predictions.setdefault(state_key, calls)
            m = self._per_task_metrics(task)
            per_task_metrics[state_key] = m
            aliases = task.metadata.get("api_aliases") or {}
            valid_apis = (
                set(aliases)
                if aliases
                else {
                    f"{a.get('tool_name')}.{a.get('api_name')}"
                    for a in task.metadata.get("api_list", [])
                }
            )
            api_names_called = [
                c.get("resolved_api_name", c.get("api_name", ""))
                for c in calls
                if c.get("role") == "Action"
            ]
            results.append(
                {
                    "task_key": state_key,
                    "query_id": qid,
                    "test_set": task.metadata.get("test_set"),
                    "calls": calls,
                    "heuristic_plan_matched": m["heuristic_plan_matched"],
                    "n_calls": m["n_calls"],
                    "n_upstream_remote_calls": m["n_upstream_remote_calls"],
                    "n_upstream_remote_error_calls": m["n_upstream_remote_error_calls"],
                    "invalid_calls": [n for n in api_names_called if n and n not in valid_apis],
                }
            )
            for call_index, call in enumerate(calls):
                receipt = call.get("upstream_receipt")
                if isinstance(receipt, dict):
                    remote_receipts.append(
                        {
                            "task_key": state_key,
                            "query_id": qid,
                            "test_set": task.metadata.get("test_set"),
                            "call_index": call_index,
                            "api_name": call.get("api_name"),
                            "receipt": receipt,
                        }
                    )
        n_matched = sum(1 for m in per_task_metrics.values() if m["heuristic_plan_matched"])
        n_total = len(per_task_metrics)
        self._last_per_task_metrics = per_task_metrics
        # Call-accuracy means are conditioned on tasks that actually emitted a
        # tool call; the pairing of value and denominator is published together.
        accuracy_values = [
            m["tool_call_accuracy"]
            for m in per_task_metrics.values()
            if m["tool_call_accuracy"] is not None
        ]
        invalid_rate_values = [
            m["invalid_tool_rate"]
            for m in per_task_metrics.values()
            if m["invalid_tool_rate"] is not None
        ]
        summary = {
            # This is deliberately not ``pass_rate``: no tool was executed
            # and the upstream GPT-based ToolEval verifier was not invoked.
            "heuristic_plan_match_rate": (n_matched / n_total) if n_total else 0.0,
            "n_heuristic_matches": n_matched,
            "n_total": n_total,
            "mean_tool_call_accuracy": (
                sum(accuracy_values) / len(accuracy_values) if accuracy_values else None
            ),
            "n_tool_call_accuracy_defined": len(accuracy_values),
            "mean_invalid_tool_rate": (
                sum(invalid_rate_values) / len(invalid_rate_values)
                if invalid_rate_values
                else None
            ),
            "n_invalid_tool_rate_defined": len(invalid_rate_values),
            "mean_relevant_api_recall": (
                sum(m["relevant_api_recall"] for m in per_task_metrics.values()) / n_total
                if n_total
                else 0.0
            ),
            "mean_n_calls": (
                sum(m["n_calls"] for m in per_task_metrics.values()) / n_total if n_total else 0.0
            ),
            "n_finished": sum(1 for m in per_task_metrics.values() if m["finished"]),
            "n_gave_up": sum(1 for m in per_task_metrics.values() if m["gave_up"]),
            "n_upstream_remote_calls": sum(
                m["n_upstream_remote_calls"] for m in per_task_metrics.values()
            ),
            "n_upstream_remote_error_calls": sum(
                m["n_upstream_remote_error_calls"] for m in per_task_metrics.values()
            ),
        }
        (out_dir / "toolbench_predictions.json").write_text(
            json.dumps(
                redact_artifact({"summary": summary, "results": results}),
                indent=2,
                ensure_ascii=False,
            )
        )
        if self._remote is not None or remote_receipts:
            (out_dir / "toolbench_remote_receipts.json").write_text(
                json.dumps(
                    redact_artifact({"summary": summary, "receipts": remote_receipts}),
                    indent=2,
                    ensure_ascii=False,
                )
            )
        (out_dir / "toolbench_predictions.md").write_text(_render_summary_md(summary))
        self._last_summary = summary

    @property
    def last_summary(self) -> dict[str, Any]:
        return dict(self._last_summary)

    def outcome_evidence(self, task: Task) -> OutcomeEvidence:
        state_key = self._task_state_key(task)
        m = self._last_per_task_metrics.get(state_key) or self._per_task_metrics(task)
        calls = list(m.get("calls") or [])
        final_answers = [c.get("content", "") for c in calls if c.get("role") == "Final Answer"]
        relevant = task.metadata.get("relevant_apis") or []
        relevant_apis = [f"{tool}.{api}" for tool, api in relevant]
        if not m.get("finished"):
            return OutcomeEvidence(
                success=False,
                reason="no Finish observed",
                evidence_quality=EVIDENCE_ABSENT,
                source="benchmark_close",
                tool_calls=calls,
                relevant_apis=relevant_apis,
                extra={
                    "relevant_api_recall": m.get("relevant_api_recall", 0.0),
                    "invalid_tool_rate": m.get("invalid_tool_rate"),
                    "n_upstream_remote_calls": m.get("n_upstream_remote_calls", 0),
                },
            )
        has_remote_execution = bool(m.get("n_upstream_remote_calls"))
        reason = (
            "ToolBench upstream remote protocol executed; ToolEval verdict is absent"
            if has_remote_execution
            else (
                "ToolBench format/planning heuristic matched relevant APIs"
                if m.get("heuristic_plan_matched")
                else "ToolBench planning diagnostic did not match"
            )
        )
        return OutcomeEvidence(
            # A weak heuristic signal is never an observed benchmark success.
            success=False,
            reason=reason,
            evidence_quality=EVIDENCE_WEAK,
            source=(
                "toolbench_upstream_remote_protocol"
                if has_remote_execution
                else "toolbench_heuristic_diagnostic"
            ),
            final_answer=final_answers[-1] if final_answers else None,
            tool_calls=calls,
            relevant_apis=relevant_apis,
            extra={
                "tool_call_accuracy": m.get("tool_call_accuracy", 0.0),
                "invalid_tool_rate": m.get("invalid_tool_rate", 0.0),
                "relevant_api_recall": m.get("relevant_api_recall", 0.0),
                "n_calls": m.get("n_calls", 0),
                "heuristic_plan_matched": bool(m.get("heuristic_plan_matched")),
                "n_upstream_remote_calls": m.get("n_upstream_remote_calls", 0),
                "n_upstream_remote_error_calls": m.get("n_upstream_remote_error_calls", 0),
            },
        )


def _render_summary_md(summary: dict) -> str:
    def rendered(value: object, format_spec: str) -> str:
        return format(value, format_spec) if isinstance(value, (int, float)) else "n/a"

    return (
        "# ToolBench planning diagnostic summary\n\n"
        "This artifact is not ToolBench official evaluation and does not report a pass rate.\n\n"
        f"- heuristic_plan_match_rate: {summary['heuristic_plan_match_rate']:.3f}\n"
        f"- n_heuristic_matches: {summary['n_heuristic_matches']} / {summary['n_total']}\n"
        "- mean_tool_call_accuracy: "
        f"{rendered(summary['mean_tool_call_accuracy'], '.3f')} "
        "(defined for "
        f"{summary['n_tool_call_accuracy_defined']} / {summary['n_total']} tasks)\n"
        "- mean_invalid_tool_rate: "
        f"{rendered(summary['mean_invalid_tool_rate'], '.3f')} "
        "(defined for "
        f"{summary['n_invalid_tool_rate_defined']} / {summary['n_total']} tasks)\n"
        f"- mean_relevant_api_recall: {summary['mean_relevant_api_recall']:.3f}\n"
        f"- mean_n_calls: {summary['mean_n_calls']:.2f}\n"
        f"- n_finished: {summary['n_finished']} / {summary['n_total']}\n"
        f"- n_gave_up: {summary['n_gave_up']} / {summary['n_total']}\n"
        f"- n_upstream_remote_calls: {summary['n_upstream_remote_calls']}\n"
        f"- n_upstream_remote_error_calls: {summary['n_upstream_remote_error_calls']}\n"
    )
