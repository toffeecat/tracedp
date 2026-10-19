"""SWE-bench Lite benchmark adapter.

This adapter implements the three-tier validity ladder documented in
`configs/datasets/swebench_lite.json`:

  fixture_only  : eval_validity = smoke_only, data_source = tiny_fixture
                  (4 hand-picked instances in
                   lh_platform/data/fixtures/swebench_lite_tiny.jsonl).
                  Local development + unittests ONLY.
  real_loader   : eval_validity = real_loader, data_source = swebench_lite_full
                  (downloaded by lh_platform/scripts/download_swebench_lite.py
                   into data/swe-bench-lite.jsonl under the repository root).
                  Exactly 300 instances plus a self-hashed pinned manifest.
                  No evaluation harness yet.
  real_native   : not set by this adapter.  A separate official-evaluation
                  receipt may use it only after a full canonical roster,
                  runner-bound predictions, a verified harness identity, and
                  a successful upstream Docker report all agree.

This adapter never synthesizes a pass_rate. step() does not run any test
suite, docker image, or git-apply heuristic; it only stages the instance
metadata so downstream methods can produce PatchPrediction records.

``Task`` objects are agent-visible.  Gold patches and hidden test metadata
therefore stay in this benchmark instance's private row store and are never
copied into ``Task.metadata`` or its instruction.  Mixing those two data
planes would let an agent read the answer it is meant to be evaluated on.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from lh_platform.benchmarks._local import lh_root
from lh_platform.benchmarks.base import BenchmarkBase
from lh_platform.core.evidence_contract import canonical_sha256
from lh_platform.core.registry import BENCHMARKS
from lh_platform.core.types import Action, Observation, Task, TaskDomain

REQUIRED_FIELDS = [
    "instance_id",
    "repo",
    "base_commit",
    "problem_statement",
    "test_patch",
    "patch",
    "FAIL_TO_PASS",
    "PASS_TO_PASS",
]

# These are the only raw-row fields an agent needs in order to identify and
# prepare a patch for an instance.  The remaining required fields are evaluator
# private: in particular ``patch`` is the gold patch and ``test_patch`` carries
# hidden-test information.
AGENT_VISIBLE_METADATA_FIELDS = (
    "instance_id",
    "repo",
    "base_commit",
)

FIXTURE_PATH_DEFAULT = (
    Path(__file__).resolve().parents[2] / "data" / "fixtures" / "swebench_lite_tiny.jsonl"
)

DOWNLOADED_PATH_DEFAULT = (
    lh_root() / "data" / "swe-bench-lite.jsonl"
)
DOWNLOADED_MANIFEST_SCHEMA = "lh_swebench_download_manifest/v1"
SWE_BENCH_LITE_EXPECTED_INSTANCES = 300


def downloaded_manifest_path(path: Path) -> Path:
    """Return the immutable sidecar required before calling data full."""

    return path.with_suffix(path.suffix + ".manifest.json")


def _read_jsonl(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    out: list[dict] = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except Exception:
                continue
            out.append(row)
    return out


def _validate(rows: list[dict]) -> tuple[list[dict], list[dict]]:
    valid: list[dict] = []
    bad: list[dict] = []
    for r in rows:
        missing = [f for f in REQUIRED_FIELDS if f not in r]
        if missing:
            bad.append({"instance_id": r.get("instance_id", "?"), "missing": missing})
        else:
            valid.append(r)
    return valid, bad


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_full_download(
    data_path: Path,
    manifest_path: Path | None = None,
) -> tuple[bool, str, dict[str, Any] | None]:
    """Verify the exact full-data contract before admitting ``real_loader``.

    Existence is not evidence of the 300-instance Lite split.  A downloaded
    file must have a paired, self-hashed manifest that pins its source commit,
    byte checksum, and exact roster.  Any mismatch deliberately downgrades to
    the local fixture rather than creating a deceptively real experiment.
    """

    manifest_path = manifest_path or downloaded_manifest_path(data_path)
    if data_path.is_symlink() or not data_path.is_file():
        return False, "downloaded JSONL missing or unsafe", None
    if manifest_path.is_symlink() or not manifest_path.is_file():
        return False, "downloaded JSONL manifest missing or unsafe", None
    rows = _read_jsonl(data_path)
    valid, bad = _validate(rows)
    if bad or len(valid) != SWE_BENCH_LITE_EXPECTED_INSTANCES or len(rows) != len(valid):
        return False, "downloaded JSONL is not exactly 300 valid rows", None
    ids = [str(row.get("instance_id") or "") for row in valid]
    if not all(ids) or len(set(ids)) != SWE_BENCH_LITE_EXPECTED_INSTANCES:
        return False, "downloaded JSONL lacks 300 unique instance IDs", None
    try:
        raw = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False, "downloaded JSONL manifest is not valid JSON", None
    if not isinstance(raw, Mapping):
        return False, "downloaded JSONL manifest is not an object", None
    if raw.get("schema") != DOWNLOADED_MANIFEST_SCHEMA:
        return False, "downloaded JSONL manifest schema mismatch", None
    declared = str(raw.get("artifact_sha256") or "")
    if declared != canonical_sha256(
        {key: value for key, value in raw.items() if key != "artifact_sha256"}
    ):
        return False, "downloaded JSONL manifest artifact hash mismatch", None
    if raw.get("dataset_name") != "SWE-bench/SWE-bench_Lite" or raw.get("split") != "test":
        return False, "downloaded JSONL manifest dataset/split mismatch", None
    if raw.get("expected_n_instances") != SWE_BENCH_LITE_EXPECTED_INSTANCES:
        return False, "downloaded JSONL manifest denominator mismatch", None
    revision = str(raw.get("source_revision") or "")
    if not re.fullmatch(r"[0-9a-f]{7,64}", revision):
        return False, "downloaded JSONL manifest lacks immutable source revision", None
    checksum = f"sha256:{_sha256_file(data_path)}"
    if raw.get("dataset_checksum") != checksum:
        return False, "downloaded JSONL bytes do not match manifest checksum", None
    roster = raw.get("instance_ids")
    if not isinstance(roster, list) or roster != sorted(ids):
        return False, "downloaded JSONL manifest roster mismatch", None
    if raw.get("roster_sha256") != canonical_sha256(sorted(ids)):
        return False, "downloaded JSONL manifest roster hash mismatch", None
    return (
        True,
        "ok",
        {
            "dataset_revision": revision,
            "dataset_manifest_path": manifest_path,
            "dataset_manifest_sha256": f"sha256:{_sha256_file(manifest_path)}",
            "dataset_checksum": checksum,
        },
    )


@BENCHMARKS.register("swebench_lite")
class SWEBenchLiteBenchmark(BenchmarkBase):
    """SWE-bench Lite adapter with three-tier validity ladder."""

    name = "swebench_lite"
    # This staging adapter accepts a unified-diff candidate, not arbitrary
    # executable Python.  Official harness evaluation consumes the same patch
    # contract when it is added later.
    required_action_contract = "swe_unified_diff"
    has_real_data = False
    has_native_harness = False
    supports_native_eval = False
    is_mock = False

    def __init__(
        self,
        max_steps: int = 30,
        fixture_path: Path | None = None,
        downloaded_path: Path | None = None,
        downloaded_manifest: Path | None = None,
    ) -> None:
        self.max_steps = max_steps
        self._fixture_path = Path(fixture_path) if fixture_path else FIXTURE_PATH_DEFAULT
        self._downloaded_path = (
            Path(downloaded_path) if downloaded_path else DOWNLOADED_PATH_DEFAULT
        )
        self._downloaded_manifest = (
            Path(downloaded_manifest)
            if downloaded_manifest
            else downloaded_manifest_path(self._downloaded_path)
        )
        self._resolve_data_source()
        self._seen_tasks: list[Task] = []
        # Deliberately private.  Do not add an accessor: Methods receive Task
        # objects, and the benchmark must never provide them gold patches or
        # held-out test data through the agent-facing API.
        self._private_rows_by_instance: dict[str, dict] = {}
        self._last_summary: dict = {}

    def _resolve_data_source(self) -> None:
        """Pick the most authoritative data source available.

        Order:
          1. downloaded_path  -> data_source = swebench_lite_full, eval_validity = real_loader
          2. fixture_path     -> data_source = tiny_fixture, eval_validity = smoke_only
          3. neither          -> data_source = none, eval_validity = smoke_only
        """
        # These fields feed the experiment evidence contract.  Without the
        # resolved file path, a full-data run could not bind a dataset
        # checksum; without the split, downstream prediction receipts could
        # not state which official task partition their IDs refer to.
        self.dataset_path = None
        self.dataset_split = "test"
        self.dataset_revision = None
        self.dataset_manifest_path = None
        self.dataset_manifest_sha256 = None
        self.dataset_integrity_error = None
        full_ok, full_reason, full_metadata = validate_full_download(
            self._downloaded_path,
            self._downloaded_manifest,
        )
        if full_ok and full_metadata is not None:
            self.data_source = "swebench_lite_full"
            self.eval_validity = "real_loader"
            self.has_real_data = True
            self.dataset_path = self._downloaded_path
            self.dataset_revision = full_metadata["dataset_revision"]
            self.dataset_manifest_path = full_metadata["dataset_manifest_path"]
            self.dataset_manifest_sha256 = full_metadata["dataset_manifest_sha256"]
            return
        self.dataset_integrity_error = full_reason
        if self._fixture_path.is_file():
            rows = _read_jsonl(self._fixture_path)
            valid, _ = _validate(rows)
            if valid:
                self.data_source = "tiny_fixture"
                self.eval_validity = "smoke_only"
                self.has_real_data = False
                self.dataset_path = self._fixture_path
                return
        self.data_source = "none"
        self.eval_validity = "smoke_only"
        self.has_real_data = False

    def _resolve_rows(self) -> list[dict]:
        if self.data_source == "swebench_lite_full":
            return _read_jsonl(self._downloaded_path)
        if self.data_source == "tiny_fixture":
            return _read_jsonl(self._fixture_path)
        return []

    def load(self, n: int = 3) -> list[Task]:
        rows = self._resolve_rows()
        valid, bad = _validate(rows)
        rows = valid[: max(0, n)]
        self._private_rows_by_instance = {str(row["instance_id"]): dict(row) for row in rows}
        tasks: list[Task] = []
        for r in rows:
            tasks.append(
                Task(
                    task_id=str(r["instance_id"]),
                    domain=TaskDomain.SWE,
                    instruction=self._format_instruction(r),
                    max_steps=self.max_steps,
                    metadata={key: r[key] for key in AGENT_VISIBLE_METADATA_FIELDS}
                    | {
                        "data_source": self.data_source,
                    },
                )
            )
        self._seen_tasks = list(tasks)
        return tasks

    def _format_instruction(self, row: dict) -> str:
        return (
            f"SWE-bench Lite instance {row['instance_id']}\n"
            f"Repo: {row['repo']} (base_commit={row['base_commit'][:12]})\n\n"
            f"Problem statement:\n{row['problem_statement']}\n\n"
            f"Produce a unified-diff patch (with `diff --git` headers) that fixes "
            f"the issue. The harness will apply your patch on top of base_commit, "
            f"run FAIL_TO_PASS tests, and assert they pass."
        )

    def step(self, task: Task, action: Action) -> Observation:
        # The adapter never executes code or invokes a test suite; it only
        # stages the patch candidate for downstream evaluation. Real eval
        # is the job of the official SWE-bench harness, not this adapter.
        return Observation(
            content=(
                f"[swebench_lite stage] instance_id={task.metadata.get('instance_id')} "
                f"received action (len={len(action.raw or '')})"
            ),
            info={
                "done": False,
                "continue": True,
                "data_source": self.data_source,
                "eval_validity": self.eval_validity,
                "patch_text": (action.info or {}).get("patch_text"),
            },
        )

    def close(self) -> None:
        # Real_native is reserved for the official Docker harness; this
        # adapter never sets it.
        summary = {
            "data_source": self.data_source,
            "eval_validity": self.eval_validity,
            "n_loaded": len(self._seen_tasks),
            "patch_count": sum(
                1
                for t in self._seen_tasks
                # placeholder; metrics collected at method level
                for _ in [None]
            ),
            "pass_rate": None,
            "note": (
                "fixture_only: tiny_fixture MUST NOT be used as SOTA evidence."
                if self.data_source == "tiny_fixture"
                else (
                    "real_loader: full SWE-bench Lite loaded; no harness invoked."
                    if self.data_source == "swebench_lite_full"
                    else "no data source available; swebench_lite not loaded."
                )
            ),
        }
        self._last_summary = summary

    @property
    def last_summary(self) -> dict:
        return dict(self._last_summary)
