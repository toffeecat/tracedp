"""Fail-closed ToolBench official-corpus boundary.

ToolBench's upstream evaluator consumes six named test sets and their exact
query-ID rosters.  ``data_example`` is deliberately insufficient: it is a
small format fixture, not a substitute for the released ``data/`` directory.

This module owns only the *input-data* contract.  A complete local corpus is
not automatically the released corpus: it becomes ``real_loader`` only after
it is checked against a reviewed canonical source lock.  It does not execute
RapidAPI tools or invoke the GPT-based ToolEval evaluator.  Those are separate
boundaries; no function here produces a ToolBench score.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
import tempfile
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

from lh_platform.core.evidence_contract import canonical_sha256

# Exact order used by ToolBench's ``toolbench/tooleval/utils.py``.  Keeping it
# here avoids a dangerous "whatever files happen to exist" evaluation roster.
OFFICIAL_TEST_SETS: tuple[str, ...] = (
    "G1_instruction",
    "G1_category",
    "G1_tool",
    "G2_instruction",
    "G2_category",
    "G3_instruction",
)
OFFICIAL_CORPUS_MANIFEST_SCHEMA = "lh_toolbench_official_corpus_manifest/v1"
OFFICIAL_CORPUS_MANIFEST_FILENAME = "toolbench_official_corpus_manifest.v1.json"
CANONICAL_SOURCE_LOCK_SCHEMA = "lh_toolbench_canonical_source_lock/v1"
RAPIDAPI_PROXY_IDENTITY_LOCK_SCHEMA = "lh_toolbench_rapidapi_proxy_identity_lock/v1"
UPSTREAM_REPOSITORY = "https://github.com/OpenBMB/ToolBench"
_DESCRIPTION_MAX_LENGTH = 256
_TYPE_MAP = {"NUMBER": "integer", "STRING": "string", "BOOLEAN": "boolean"}
_RESERVED_FUNCTION_NAMES = frozenset({"from", "class", "return", "false", "true", "id", "and"})
_NON_STANDARD = re.compile(r"[^\u4e00-\u9fa5^a-z^A-Z^0-9^_]")
_REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
_CANONICAL_SOURCE_LOCK_RELATIVE_PATH = Path("configs/locks/toolbench_canonical_source_lock.v1.json")
_RAPIDAPI_PROXY_IDENTITY_LOCK_RELATIVE_PATH = Path(
    "configs/locks/toolbench_rapidapi_proxy_identity_lock.v1.json"
)


class ToolBenchOfficialDataError(RuntimeError):
    """Raised when released ToolBench data cannot prove the six-set contract."""


@dataclass(frozen=True)
class RepositoryLockVerification:
    """Result of checking a repository-owned ToolBench trust anchor.

    A data-directory lock is only evidence about one operator's local files.
    ``verified`` here means that a fixed lock path is present, HEAD-tracked,
    structurally reviewed, and consistent with the candidate data/configuration.
    """

    status: str
    path: Path
    reason: str
    sha256: str | None = None
    payload: Mapping[str, Any] | None = None


@dataclass(frozen=True)
class ToolBenchOfficialCorpus:
    """A manifest-verified, read-only view of ToolBench's six test sets."""

    data_root: Path
    manifest_path: Path
    manifest_sha256: str
    source_reference: str
    source_verified: bool
    source_verification_reason: str
    source_lock_path: Path | None
    source_lock_sha256: str | None
    rows_by_test_set: Mapping[str, tuple[dict[str, Any], ...]]
    rosters_by_test_set: Mapping[str, tuple[str, ...]]
    available_tools_by_test_set: Mapping[str, Mapping[str, tuple[dict[str, Any], ...]]]

    @property
    def test_sets(self) -> tuple[str, ...]:
        return OFFICIAL_TEST_SETS

    @property
    def total_roster_count(self) -> int:
        return sum(len(ids) for ids in self.rosters_by_test_set.values())

    def rows_for(self, test_set: str) -> tuple[dict[str, Any], ...]:
        _require_test_set(test_set)
        return tuple(_clone_json(row) for row in self.rows_by_test_set[test_set])

    def available_tools(self, test_set: str, query_id: str | int) -> list[dict[str, Any]]:
        _require_test_set(test_set)
        query_key = _normalize_query_id(query_id, context=f"{test_set} query id")
        try:
            tools = self.available_tools_by_test_set[test_set][query_key]
        except KeyError as error:
            raise ToolBenchOfficialDataError(
                f"{test_set}: no staged tool contract for query_id={query_key!r}"
            ) from error
        return _clone_json(list(tools))

    def function_aliases(self, test_set: str, query_id: str | int) -> dict[str, str]:
        """Map upstream function names back to their source ``Tool.API`` pair.

        This is diagnostic metadata for the unified adapter; ToolBench's
        external evaluator still sees the canonical function names returned by
        :meth:`available_tools`.
        """

        _require_test_set(test_set)
        query_key = _normalize_query_id(query_id, context=f"{test_set} query id")
        row = next(
            (
                item
                for item in self.rows_by_test_set[test_set]
                if str(item["query_id"]) == query_key
            ),
            None,
        )
        if row is None:
            raise ToolBenchOfficialDataError(
                f"{test_set}: no staged row for query_id={query_key!r}"
            )
        functions = self.available_tools(test_set, query_key)
        actions = [function for function in functions if function.get("name") != "Finish"]
        api_list = row.get("api_list") or []
        if len(actions) != len(api_list):
            raise ToolBenchOfficialDataError(
                f"{test_set} query_id={query_key}: function/API alignment changed after staging"
            )
        return {
            str(function["name"]): f"{api.get('tool_name', '?')}.{api.get('api_name', '?')}"
            for api, function in zip(api_list, actions, strict=False)
        }

    def rapidapi_invocations(
        self,
        test_set: str,
        query_id: str | int,
    ) -> dict[str, dict[str, str]]:
        """Return the exact source fields consumed by ``rapidapi_wrapper``.

        ``available_tools`` gives the function vocabulary shown to the model.
        The upstream remote environment additionally needs the category, the
        normalized tool JSON filename, and the normalized API name.  Derive
        those fields from the same staged files used to build the function
        vocabulary so a model action cannot be routed through a guessed or
        unrelated API endpoint.
        """

        _require_test_set(test_set)
        query_key = _normalize_query_id(query_id, context=f"{test_set} query id")
        row = next(
            (
                item
                for item in self.rows_by_test_set[test_set]
                if str(item["query_id"]) == query_key
            ),
            None,
        )
        if row is None:
            raise ToolBenchOfficialDataError(
                f"{test_set}: no staged row for query_id={query_key!r}"
            )
        functions = [
            function
            for function in self.available_tools(test_set, query_key)
            if function.get("name") != "Finish"
        ]
        api_list = row.get("api_list") or []
        if len(functions) != len(api_list):
            raise ToolBenchOfficialDataError(
                f"{test_set} query_id={query_key}: function/API alignment changed after staging"
            )
        tool_root = self.data_root / "toolenv" / "tools"
        out: dict[str, dict[str, str]] = {}
        for api, function in zip(api_list, functions, strict=False):
            if not isinstance(api, Mapping):
                raise ToolBenchOfficialDataError(
                    f"{test_set} query_id={query_key}: api_list contains a non-object"
                )
            function_name = str(function.get("name") or "").strip()
            category = str(api.get("category_name") or "").strip()
            api_name = str(api.get("api_name") or "").strip()
            tool_name = str(api.get("tool_name") or "").strip()
            if not function_name or not category or not api_name or not tool_name:
                raise ToolBenchOfficialDataError(
                    f"{test_set} query_id={query_key}: incomplete RapidAPI invocation source fields"
                )
            tool_file = _find_tool_file(tool_root, category, tool_name)
            expected_function, _ = _api_to_upstream_function(
                api,
                tool_root,
                test_set,
                query_key,
            )
            if expected_function.get("name") != function_name:
                raise ToolBenchOfficialDataError(
                    f"{test_set} query_id={query_key}: staged function vocabulary changed before RapidAPI routing"
                )
            if function_name in out:
                raise ToolBenchOfficialDataError(
                    f"{test_set} query_id={query_key}: duplicate RapidAPI function {function_name!r}"
                )
            out[function_name] = {
                "function_name": function_name,
                "category": category,
                # ``rapidapi_wrapper`` receives the standardized filename
                # stem through ``tool_descriptions``, not the display name.
                "tool_name": tool_file.stem,
                "api_name": _change_name(_standardize(api_name)),
            }
        return _clone_json(out)


@dataclass(frozen=True)
class _CorpusScan:
    rows_by_test_set: Mapping[str, tuple[dict[str, Any], ...]]
    rosters_by_test_set: Mapping[str, tuple[str, ...]]
    available_tools_by_test_set: Mapping[str, Mapping[str, tuple[dict[str, Any], ...]]]
    manifest_test_sets: tuple[dict[str, Any], ...]


def default_manifest_path(data_root: str | Path) -> Path:
    return Path(data_root).expanduser().resolve() / OFFICIAL_CORPUS_MANIFEST_FILENAME


def repository_canonical_source_lock_path() -> Path:
    """Return the only source-provenance lock path allowed for real claims.

    The path deliberately lives in the LongHorizon repository rather than
    beside a downloaded corpus.  An operator can stage local data and a local
    manifest, but cannot make that data ``real_loader`` by generating a lock
    next to it.
    """

    return (_REPOSITORY_ROOT / _CANONICAL_SOURCE_LOCK_RELATIVE_PATH).resolve()


def repository_rapidapi_proxy_identity_lock_path() -> Path:
    """Return the fixed reviewed identity lock required for remote execution."""

    return (_REPOSITORY_ROOT / _RAPIDAPI_PROXY_IDENTITY_LOCK_RELATIVE_PATH).resolve()


def resolve_official_data_root(
    repo_path: str | Path,
    explicit_data_root: str | Path | None = None,
) -> tuple[Path | None, str]:
    """Resolve a full-data location without silently treating it as staged.

    ``explicit_data_root`` has priority, followed by ``LH_TOOLBENCH_DATA_ROOT``
    and the upstream-standard ``<ToolBench repo>/data`` location.  Callers
    must still call :func:`load_official_corpus`; a directory's mere existence
    is not evidence that it is the released corpus.
    """

    if explicit_data_root is not None:
        return Path(explicit_data_root).expanduser().resolve(), "explicit"
    from_environment = os.getenv("LH_TOOLBENCH_DATA_ROOT", "").strip()
    if from_environment:
        return Path(from_environment).expanduser().resolve(), "environment"
    default = Path(repo_path).expanduser().resolve() / "data"
    if default.exists():
        return default, "upstream_default"
    return None, "none"


def stage_official_corpus(
    data_root: str | Path,
    *,
    source_reference: str,
    manifest_path: str | Path | None = None,
    overwrite: bool = False,
) -> Path:
    """Freeze a complete corpus and bind it to the repository trust root if present.

    ``source_reference`` is intentionally required.  It should identify the
    user-downloaded upstream archive/revision.  A bare string is only a
    declaration; it never establishes ``real_loader``.  To prove the corpus
    against a known release, a maintainer must review and commit the fixed
    repository-owned canonical source lock.  Arbitrary lock paths are not
    accepted because they allow the producer of a corpus to self-authorize it.
    """

    root = _require_data_root(data_root)
    source = str(source_reference or "").strip()
    if not source:
        raise ToolBenchOfficialDataError(
            "source_reference is required; record the upstream archive/revision used"
        )
    target = (
        Path(manifest_path).expanduser().resolve() if manifest_path else default_manifest_path(root)
    )
    if target.exists() and not overwrite:
        raise ToolBenchOfficialDataError(
            f"refusing to overwrite existing corpus manifest: {target}; pass overwrite=True deliberately"
        )
    scan = _scan_corpus(root)
    source_verification = _stage_source_verification(
        source_reference=source,
        scan=scan,
    )
    payload = {
        "schema": OFFICIAL_CORPUS_MANIFEST_SCHEMA,
        "upstream_repository": UPSTREAM_REPOSITORY,
        "source_reference": source,
        "test_sets": list(scan.manifest_test_sets),
        "total_roster_count": sum(len(ids) for ids in scan.rosters_by_test_set.values()),
        "roster_contract_sha256": canonical_sha256(
            {name: list(scan.rosters_by_test_set[name]) for name in OFFICIAL_TEST_SETS}
        ),
        "source_verification": source_verification,
        "note": (
            "This manifest proves a complete local ToolBench input corpus. "
            "Without a validated canonical source lock it is unverified_loader only. "
            "It does not prove native tool execution or ToolEval completion."
        ),
    }
    payload["artifact_sha256"] = canonical_sha256(payload)
    _write_json_atomic(target, payload, overwrite=overwrite)
    return target


def load_official_corpus(
    data_root: str | Path,
    *,
    manifest_path: str | Path | None = None,
) -> ToolBenchOfficialCorpus:
    """Load a revalidated staged corpus and report source-authentication state.

    Callers may inspect an unverified staged corpus for diagnostics, but must
    require ``corpus.source_verified`` before advertising ``real_loader`` or
    dispatching upstream remote execution.
    """

    root = _require_data_root(data_root)
    target = (
        Path(manifest_path).expanduser().resolve() if manifest_path else default_manifest_path(root)
    )
    if not target.is_file():
        raise ToolBenchOfficialDataError(
            f"ToolBench full data is not staged: missing immutable manifest {target}; "
            "run `lh-platform stage-toolbench-full` after downloading the official data/ release"
        )
    raw = _read_json_mapping(target, label="ToolBench corpus manifest")
    if raw.get("schema") != OFFICIAL_CORPUS_MANIFEST_SCHEMA:
        raise ToolBenchOfficialDataError(
            f"unexpected ToolBench corpus manifest schema: {raw.get('schema')!r}"
        )
    declared = str(raw.get("artifact_sha256") or "")
    calculated = canonical_sha256({k: v for k, v in raw.items() if k != "artifact_sha256"})
    if declared != calculated:
        raise ToolBenchOfficialDataError("ToolBench corpus manifest artifact_sha256 mismatch")
    source = str(raw.get("source_reference") or "").strip()
    if not source:
        raise ToolBenchOfficialDataError("ToolBench corpus manifest lacks source_reference")
    if raw.get("upstream_repository") != UPSTREAM_REPOSITORY:
        raise ToolBenchOfficialDataError("ToolBench corpus manifest upstream_repository mismatch")

    scan = _scan_corpus(root)
    _validate_manifest_against_scan(raw, scan)
    source_verified, source_reason, lock_path, lock_sha256 = _load_source_verification(
        source_reference=source,
        staged_manifest=raw,
        scan=scan,
    )
    return ToolBenchOfficialCorpus(
        data_root=root,
        manifest_path=target,
        manifest_sha256=_sha256_file(target),
        source_reference=source,
        source_verified=source_verified,
        source_verification_reason=source_reason,
        source_lock_path=lock_path,
        source_lock_sha256=lock_sha256,
        rows_by_test_set=scan.rows_by_test_set,
        rosters_by_test_set=scan.rosters_by_test_set,
        available_tools_by_test_set=scan.available_tools_by_test_set,
    )


def _expected_corpus_lock(scan: _CorpusScan) -> dict[str, Any]:
    """Return the immutable roster and file-hash contract a source lock signs."""

    return {
        "test_sets": list(scan.manifest_test_sets),
        "total_roster_count": sum(len(ids) for ids in scan.rosters_by_test_set.values()),
        "roster_contract_sha256": canonical_sha256(
            {name: list(scan.rosters_by_test_set[name]) for name in OFFICIAL_TEST_SETS}
        ),
    }


def _stage_source_verification(
    *,
    source_reference: str,
    scan: _CorpusScan,
) -> dict[str, Any]:
    """Record whether this corpus matches the fixed reviewed source anchor."""

    verification = _verify_repository_canonical_source_lock(
        source_reference=source_reference,
        scan=scan,
    )
    payload = {
        "status": verification.status,
        "reason": verification.reason,
        "source_lock_schema": CANONICAL_SOURCE_LOCK_SCHEMA,
        "source_lock_path": _repository_relative_path(verification.path),
    }
    if verification.sha256 is not None:
        payload["source_lock_sha256"] = verification.sha256
    if verification.status == "verified" and verification.payload is not None:
        payload["source_archive_sha256"] = str(verification.payload["source_archive_sha256"])
        payload["source_revision"] = str(verification.payload["source_revision"])
    return payload


def _load_source_verification(
    *,
    source_reference: str,
    staged_manifest: Mapping[str, Any],
    scan: _CorpusScan,
) -> tuple[bool, str, Path | None, str | None]:
    """Re-validate the repository trust root at load time.

    A data manifest records the evidence used at staging time, but it may have
    been staged before a reviewed anchor existed.  It cannot be upgraded by a
    later arbitrary file; the current HEAD-tracked anchor must match both the
    manifest and the local corpus again.
    """

    staged = staged_manifest.get("source_verification")
    if not isinstance(staged, Mapping):
        raise ToolBenchOfficialDataError("ToolBench corpus manifest lacks source_verification")
    verification = _verify_repository_canonical_source_lock(
        source_reference=source_reference,
        scan=scan,
    )
    if verification.status != "verified":
        return False, verification.reason, verification.path, verification.sha256
    if staged.get("status") != "verified":
        return (
            False,
            "corpus manifest was staged before the repository canonical source lock was verified; restage it",
            verification.path,
            verification.sha256,
        )
    if str(staged.get("source_lock_sha256") or "") != verification.sha256:
        return (
            False,
            "corpus manifest does not bind the current repository canonical source lock",
            verification.path,
            verification.sha256,
        )
    return True, "repository canonical source lock verified", verification.path, verification.sha256


def _repository_head_blob(path: Path) -> bytes | None:
    """Return the checked-out HEAD bytes for a fixed repository-owned lock."""

    try:
        repository_root = _REPOSITORY_ROOT.resolve()
        relative = path.resolve().relative_to(repository_root).as_posix()
    except ValueError:
        return None
    try:
        result = subprocess.run(
            ["git", "-C", str(repository_root), "show", f"HEAD:{relative}"],
            capture_output=True,
            check=False,
        )
    except OSError:
        return None
    return result.stdout if result.returncode == 0 else None


def _repository_relative_path(path: Path) -> str:
    try:
        return path.resolve().relative_to(_REPOSITORY_ROOT.resolve()).as_posix()
    except ValueError:
        return str(path)


def _read_repository_lock(
    *,
    path: Path,
    schema: str,
    label: str,
) -> RepositoryLockVerification:
    """Read a fixed HEAD-tracked reviewed lock, never an operator-provided one."""

    if not path.is_file() or path.is_symlink():
        return RepositoryLockVerification(
            status="missing",
            path=path,
            reason=f"repository-reviewed {label} is not installed",
        )
    head_bytes = _repository_head_blob(path)
    try:
        local_bytes = path.read_bytes()
    except OSError as error:
        return RepositoryLockVerification(
            status="untrusted",
            path=path,
            reason=f"cannot read repository-reviewed {label}: {type(error).__name__}",
        )
    if head_bytes is None or head_bytes != local_bytes:
        return RepositoryLockVerification(
            status="untrusted",
            path=path,
            reason=f"{label} is not identical to the checked-out HEAD blob",
        )
    try:
        raw = _read_json_mapping(path, label=label)
    except ToolBenchOfficialDataError as error:
        return RepositoryLockVerification(
            status="malformed",
            path=path,
            reason=str(error),
        )
    if raw.get("schema") != schema:
        return RepositoryLockVerification(
            status="malformed",
            path=path,
            reason=f"{label} schema mismatch",
        )
    if raw.get("candidate_only") is True or raw.get("review_status") != "approved":
        return RepositoryLockVerification(
            status="untrusted",
            path=path,
            reason=f"{label} has not been explicitly approved after review",
        )
    if not isinstance(raw.get("review"), Mapping):
        return RepositoryLockVerification(
            status="malformed",
            path=path,
            reason=f"{label} lacks a structured review record",
        )
    declared = str(raw.get("artifact_sha256") or "")
    calculated = canonical_sha256(
        {key: value for key, value in raw.items() if key != "artifact_sha256"}
    )
    if declared != calculated:
        return RepositoryLockVerification(
            status="malformed",
            path=path,
            reason=f"{label} artifact_sha256 mismatch",
        )
    return RepositoryLockVerification(
        status="verified",
        path=path,
        reason=f"repository-reviewed {label} is HEAD-tracked and approved",
        sha256=declared,
        payload=raw,
    )


def _verify_repository_canonical_source_lock(
    *,
    source_reference: str,
    scan: _CorpusScan,
) -> RepositoryLockVerification:
    """Verify data provenance against the one repository-reviewed source lock."""

    verification = _read_repository_lock(
        path=repository_canonical_source_lock_path(),
        schema=CANONICAL_SOURCE_LOCK_SCHEMA,
        label="ToolBench canonical source lock",
    )
    if verification.status != "verified" or verification.payload is None:
        return verification
    raw_lock = verification.payload
    if raw_lock.get("upstream_repository") != UPSTREAM_REPOSITORY:
        return _lock_mismatch(verification, "canonical source lock upstream_repository mismatch")
    if str(raw_lock.get("source_reference") or "").strip() != source_reference:
        return _lock_mismatch(verification, "canonical source lock source_reference mismatch")
    revision = str(raw_lock.get("source_revision") or "").strip()
    if not re.fullmatch(r"[0-9a-f]{40}", revision):
        return _lock_mismatch(
            verification,
            "canonical source lock must record a full 40-hex immutable source revision",
        )
    archive_sha256 = str(raw_lock.get("source_archive_sha256") or "").strip().lower()
    if not re.fullmatch(r"[0-9a-f]{64}", archive_sha256):
        return _lock_mismatch(
            verification,
            "canonical source lock source_archive_sha256 must be a 64-character SHA-256",
        )
    review = raw_lock["review"]
    if review.get("source_revision") != revision:
        return _lock_mismatch(verification, "canonical source review source_revision mismatch")
    source_url = str(review.get("source_url") or "")
    if not source_url.startswith(f"{UPSTREAM_REPOSITORY}/"):
        return _lock_mismatch(
            verification,
            "canonical source review source_url is not an official immutable ToolBench source",
        )
    expected = raw_lock.get("expected_corpus")
    if not isinstance(expected, Mapping) or _clone_json(expected) != _expected_corpus_lock(scan):
        return _lock_mismatch(
            verification,
            "canonical source lock does not match this ToolBench corpus roster/file contract",
        )
    return verification


def verify_repository_rapidapi_proxy_identity(
    *,
    service_url: str,
) -> RepositoryLockVerification:
    """Verify that remote execution targets the reviewed ToolBench proxy only.

    Merely using HTTPS authenticates transport, not the protocol service or
    implementation.  A real-partial claim therefore requires a separately
    reviewed, HEAD-tracked proxy identity lock bound to the exact endpoint.
    """

    verification = _read_repository_lock(
        path=repository_rapidapi_proxy_identity_lock_path(),
        schema=RAPIDAPI_PROXY_IDENTITY_LOCK_SCHEMA,
        label="ToolBench RapidAPI proxy identity lock",
    )
    if verification.status != "verified" or verification.payload is None:
        return verification
    raw_lock = verification.payload
    endpoint = str(service_url or "").strip()
    if raw_lock.get("upstream_repository") != UPSTREAM_REPOSITORY:
        return _lock_mismatch(verification, "RapidAPI proxy lock upstream_repository mismatch")
    if raw_lock.get("service_url") != endpoint:
        return _lock_mismatch(verification, "RapidAPI proxy lock service_url mismatch")
    endpoint_sha256 = hashlib.sha256(endpoint.encode("utf-8")).hexdigest()
    if raw_lock.get("service_url_sha256") != endpoint_sha256:
        return _lock_mismatch(verification, "RapidAPI proxy lock service_url_sha256 mismatch")
    protocol_revision = str(raw_lock.get("protocol_source_revision") or "").strip()
    if not re.fullmatch(r"[0-9a-f]{40}", protocol_revision):
        return _lock_mismatch(
            verification,
            "RapidAPI proxy lock must record a full 40-hex immutable ToolBench protocol revision",
        )
    expected_protocol_source_url = f"{UPSTREAM_REPOSITORY}/tree/{protocol_revision}"
    if raw_lock.get("protocol_source_url") != expected_protocol_source_url:
        return _lock_mismatch(
            verification,
            "RapidAPI proxy lock protocol_source_url must bind the declared ToolBench protocol revision",
        )
    proxy_repository = str(raw_lock.get("proxy_repository") or "").strip()
    if not re.fullmatch(r"https://github\.com/[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", proxy_repository):
        return _lock_mismatch(
            verification,
            "RapidAPI proxy lock proxy_repository must be an immutable-reviewable GitHub repository URL",
        )
    revision = str(raw_lock.get("proxy_revision") or "").strip()
    if not re.fullmatch(r"[0-9a-f]{40}", revision):
        return _lock_mismatch(
            verification,
            "RapidAPI proxy lock must record a full 40-hex immutable proxy revision",
        )
    expected_proxy_source_url = f"{proxy_repository}/tree/{revision}"
    if raw_lock.get("proxy_source_url") != expected_proxy_source_url:
        return _lock_mismatch(
            verification,
            "RapidAPI proxy lock proxy_source_url must bind the declared proxy revision",
        )
    implementation_path = str(raw_lock.get("proxy_implementation_path") or "").strip()
    try:
        relative_implementation_path = PurePosixPath(implementation_path)
    except (TypeError, ValueError):
        relative_implementation_path = PurePosixPath()
    if (
        not implementation_path
        or relative_implementation_path.is_absolute()
        or not relative_implementation_path.parts
        or any(part in {"", ".", ".."} for part in relative_implementation_path.parts)
    ):
        return _lock_mismatch(
            verification,
            "RapidAPI proxy lock proxy_implementation_path must be a safe repository-relative file path",
        )
    expected_implementation_source_url = (
        f"{proxy_repository}/blob/{revision}/{relative_implementation_path.as_posix()}"
    )
    if raw_lock.get("proxy_implementation_source_url") != expected_implementation_source_url:
        return _lock_mismatch(
            verification,
            "RapidAPI proxy lock proxy_implementation_source_url must bind the reviewed implementation file",
        )
    implementation_sha256 = str(raw_lock.get("proxy_implementation_sha256") or "").strip().lower()
    if not re.fullmatch(r"[0-9a-f]{64}", implementation_sha256):
        return _lock_mismatch(
            verification,
            "RapidAPI proxy lock proxy_implementation_sha256 must be a 64-character SHA-256",
        )
    tls_certificate_sha256 = str(raw_lock.get("tls_certificate_sha256") or "").strip().lower()
    if not re.fullmatch(r"[0-9a-f]{64}", tls_certificate_sha256):
        return _lock_mismatch(
            verification,
            "RapidAPI proxy lock tls_certificate_sha256 must be a 64-character SHA-256",
        )
    review = raw_lock["review"]
    if review.get("protocol_source_revision") != protocol_revision:
        return _lock_mismatch(
            verification,
            "RapidAPI proxy review protocol_source_revision mismatch",
        )
    if review.get("protocol_source_url") != expected_protocol_source_url:
        return _lock_mismatch(
            verification,
            "RapidAPI proxy review protocol_source_url mismatch",
        )
    if review.get("proxy_revision") != revision:
        return _lock_mismatch(verification, "RapidAPI proxy review proxy_revision mismatch")
    if review.get("proxy_source_url") != expected_proxy_source_url:
        return _lock_mismatch(
            verification,
            "RapidAPI proxy review proxy_source_url mismatch",
        )
    if review.get("proxy_implementation_path") != relative_implementation_path.as_posix():
        return _lock_mismatch(
            verification,
            "RapidAPI proxy review proxy_implementation_path mismatch",
        )
    if review.get("proxy_implementation_source_url") != expected_implementation_source_url:
        return _lock_mismatch(
            verification,
            "RapidAPI proxy review proxy_implementation_source_url mismatch",
        )
    if review.get("proxy_implementation_sha256") != implementation_sha256:
        return _lock_mismatch(
            verification,
            "RapidAPI proxy review proxy_implementation_sha256 mismatch",
        )
    return verification


def _lock_mismatch(
    verification: RepositoryLockVerification,
    reason: str,
) -> RepositoryLockVerification:
    return RepositoryLockVerification(
        status="mismatch",
        path=verification.path,
        reason=reason,
        sha256=verification.sha256,
        payload=verification.payload,
    )


def _scan_corpus(root: Path) -> _CorpusScan:
    tool_root = root / "toolenv" / "tools"
    if not tool_root.is_dir():
        raise ToolBenchOfficialDataError(
            f"ToolBench full data lacks toolenv/tools required by upstream function vocabulary: {tool_root}"
        )

    rows_by_test_set: dict[str, tuple[dict[str, Any], ...]] = {}
    rosters_by_test_set: dict[str, tuple[str, ...]] = {}
    functions_by_test_set: dict[str, dict[str, tuple[dict[str, Any], ...]]] = {}
    records: list[dict[str, Any]] = []
    for test_set in OFFICIAL_TEST_SETS:
        instruction_path = root / "test_instruction" / f"{test_set}.json"
        if not instruction_path.is_file():
            raise ToolBenchOfficialDataError(
                f"ToolBench corpus is incomplete: missing {test_set} instruction file {instruction_path}"
            )
        rows = _read_instruction_rows(instruction_path, test_set)
        roster_path = _resolve_roster_path(root, test_set)
        roster = _read_roster(roster_path, test_set)
        by_id = {str(row["query_id"]): row for row in rows}
        missing = [query_id for query_id in roster if query_id not in by_id]
        if missing:
            preview = ", ".join(missing[:5])
            raise ToolBenchOfficialDataError(
                f"{test_set}: roster query IDs missing from instruction data: {preview}"
            )
        selected = tuple(_clone_json(by_id[query_id]) for query_id in roster)
        functions_for_set: dict[str, tuple[dict[str, Any], ...]] = {}
        tool_files: dict[str, str] = {}
        for row in selected:
            functions, used_tool_files = _available_tools_for_row(row, tool_root, test_set)
            query_id = str(row["query_id"])
            functions_for_set[query_id] = tuple(functions)
            for path in used_tool_files:
                relative = path.relative_to(root).as_posix()
                tool_files[relative] = _sha256_file(path)
        rows_by_test_set[test_set] = selected
        rosters_by_test_set[test_set] = tuple(roster)
        functions_by_test_set[test_set] = functions_for_set
        records.append(
            {
                "name": test_set,
                "instruction_path": instruction_path.relative_to(root).as_posix(),
                "instruction_sha256": _sha256_file(instruction_path),
                "instruction_count": len(rows),
                "roster_path": roster_path.relative_to(root).as_posix(),
                "roster_sha256": _sha256_file(roster_path),
                "roster_count": len(roster),
                "roster_ids_sha256": canonical_sha256(list(roster)),
                "selected_rows_sha256": canonical_sha256(list(selected)),
                "function_contract_sha256": canonical_sha256(functions_for_set),
                "tool_files": [
                    {"path": path, "sha256": digest} for path, digest in sorted(tool_files.items())
                ],
            }
        )
    return _CorpusScan(
        rows_by_test_set=rows_by_test_set,
        rosters_by_test_set=rosters_by_test_set,
        available_tools_by_test_set=functions_by_test_set,
        manifest_test_sets=tuple(records),
    )


def _validate_manifest_against_scan(raw: Mapping[str, Any], scan: _CorpusScan) -> None:
    records = raw.get("test_sets")
    if not isinstance(records, list) or [
        entry.get("name") if isinstance(entry, Mapping) else None for entry in records
    ] != list(OFFICIAL_TEST_SETS):
        raise ToolBenchOfficialDataError(
            "ToolBench corpus manifest test-set roster is incomplete or reordered"
        )
    actual = list(scan.manifest_test_sets)
    for expected, observed in zip(records, actual, strict=False):
        if not isinstance(expected, Mapping):
            raise ToolBenchOfficialDataError(
                "ToolBench corpus manifest contains a non-object test-set record"
            )
        for field in (
            "name",
            "instruction_path",
            "instruction_sha256",
            "instruction_count",
            "roster_path",
            "roster_sha256",
            "roster_count",
            "roster_ids_sha256",
            "selected_rows_sha256",
            "function_contract_sha256",
            "tool_files",
        ):
            if expected.get(field) != observed.get(field):
                raise ToolBenchOfficialDataError(
                    f"ToolBench corpus manifest mismatch for {expected.get('name')!r} field {field}: "
                    "local data changed after staging"
                )
    expected_total = sum(len(ids) for ids in scan.rosters_by_test_set.values())
    if raw.get("total_roster_count") != expected_total:
        raise ToolBenchOfficialDataError("ToolBench corpus manifest total_roster_count mismatch")
    expected_roster_hash = canonical_sha256(
        {name: list(scan.rosters_by_test_set[name]) for name in OFFICIAL_TEST_SETS}
    )
    if raw.get("roster_contract_sha256") != expected_roster_hash:
        raise ToolBenchOfficialDataError(
            "ToolBench corpus manifest roster_contract_sha256 mismatch"
        )


def _available_tools_for_row(
    row: Mapping[str, Any],
    tool_root: Path,
    test_set: str,
) -> tuple[list[dict[str, Any]], set[Path]]:
    api_list = row.get("api_list")
    if not isinstance(api_list, list) or not api_list:
        raise ToolBenchOfficialDataError(
            f"{test_set} query_id={row.get('query_id')!r}: api_list must be a non-empty list"
        )
    tools: list[dict[str, Any]] = []
    tool_files: set[Path] = set()
    names: set[str] = set()
    for api in api_list:
        if not isinstance(api, Mapping):
            raise ToolBenchOfficialDataError(
                f"{test_set} query_id={row.get('query_id')!r}: api_list contains a non-object"
            )
        function, tool_file = _api_to_upstream_function(
            api, tool_root, test_set, row.get("query_id")
        )
        name = str(function["name"])
        if name in names:
            raise ToolBenchOfficialDataError(
                f"{test_set} query_id={row.get('query_id')!r}: duplicate upstream function name {name!r}"
            )
        names.add(name)
        tools.append(function)
        tool_files.add(tool_file)
    tools.append(_finish_function())
    return tools, tool_files


def _api_to_upstream_function(
    api: Mapping[str, Any],
    tool_root: Path,
    test_set: str,
    query_id: Any,
) -> tuple[dict[str, Any], Path]:
    category = str(api.get("category_name") or "").strip()
    tool_name = str(api.get("tool_name") or "").strip()
    api_name = str(api.get("api_name") or "").strip()
    if not category or not tool_name or not api_name:
        raise ToolBenchOfficialDataError(
            f"{test_set} query_id={query_id!r}: api entry lacks category_name/tool_name/api_name"
        )
    tool_file = _find_tool_file(tool_root, category, tool_name)
    raw_tool = _read_json_mapping(tool_file, label=f"ToolBench tool definition {tool_file}")
    # ``rapidapi_wrapper.fetch_api_json`` resolves the filename from the
    # normalized instruction tool name and then trusts that file's metadata;
    # it does not require byte-for-byte equality of the human-readable names.
    source_api = _find_source_api(raw_tool, api_name, tool_file)
    standard_tool_name = tool_file.stem
    pure_api_name = _change_name(_standardize(api_name))
    function_name = f"{pure_api_name}_for_{standard_tool_name}"[-64:]
    description = f'This is the subfunction for tool "{standard_tool_name}", you can use this tool.'
    api_description = str(source_api.get("description") or api.get("api_description") or "").strip()
    if api_description:
        truncated = api_description.replace(api_name, function_name)[:_DESCRIPTION_MAX_LENGTH]
        description += f'The description of this function is: "{truncated}"'
    parameters = {"type": "object", "properties": {}, "required": [], "optional": []}
    for required in _parameter_list(source_api, "required_parameters", tool_file, api_name):
        name, descriptor = _parameter_descriptor(required)
        parameters["properties"][name] = descriptor
        parameters["required"].append(name)
    for optional in _parameter_list(source_api, "optional_parameters", tool_file, api_name):
        name, descriptor = _parameter_descriptor(optional)
        parameters["properties"][name] = descriptor
        parameters["optional"].append(name)
    return {
        "name": function_name,
        "description": description,
        "parameters": parameters,
    }, tool_file


def _find_tool_file(tool_root: Path, category: str, tool_name: str) -> Path:
    category_root = tool_root / category
    direct = category_root / f"{_standardize(tool_name)}.json"
    if direct.is_file():
        return direct
    # This is intentionally the same path contract as upstream
    # ``rapidapi_wrapper.fetch_api_json``.  Searching neighboring JSON files
    # would make a malformed source directory look usable while producing a
    # function vocabulary that the original environment would never expose.
    raise ToolBenchOfficialDataError(
        f"missing ToolBench tool definition for {category}/{tool_name}; expected {direct}"
    )


def _find_source_api(
    raw_tool: Mapping[str, Any], api_name: str, tool_file: Path
) -> Mapping[str, Any]:
    api_list = raw_tool.get("api_list")
    if not isinstance(api_list, list):
        raise ToolBenchOfficialDataError(f"ToolBench tool definition lacks api_list: {tool_file}")
    # Upstream compares ``change_name(standardize(name))`` rather than the
    # source strings.  Released records can differ in punctuation/case while
    # still mapping to the same RapidAPI function.
    expected = _change_name(_standardize(api_name))
    normalized = [
        entry
        for entry in api_list
        if isinstance(entry, Mapping)
        and _change_name(_standardize(str(entry.get("name") or ""))) == expected
    ]
    if len(normalized) != 1:
        raise ToolBenchOfficialDataError(
            f"ToolBench tool definition {tool_file} has no unique normalized api "
            f"matching {api_name!r}"
        )
    return normalized[0]


def _parameter_list(
    api: Mapping[str, Any],
    key: str,
    tool_file: Path,
    api_name: str,
) -> list[Mapping[str, Any]]:
    raw = api.get(key, [])
    if raw is None:
        return []
    if not isinstance(raw, list) or not all(isinstance(item, Mapping) for item in raw):
        raise ToolBenchOfficialDataError(
            f"ToolBench tool definition {tool_file} has malformed {key} for {api_name!r}"
        )
    return list(raw)


def _parameter_descriptor(raw: Mapping[str, Any]) -> tuple[str, dict[str, Any]]:
    name = _change_name(_standardize(str(raw.get("name") or "")))
    if not name:
        raise ToolBenchOfficialDataError("ToolBench tool parameter has an empty name")
    parameter_type = _TYPE_MAP.get(str(raw.get("type") or ""), "string")
    descriptor: dict[str, Any] = {
        "type": parameter_type,
        "description": str(raw.get("description") or "")[:_DESCRIPTION_MAX_LENGTH],
    }
    default = raw.get("default", "")
    if len(str(default)) != 0:
        descriptor["example_value"] = default
    return name, descriptor


def _finish_function() -> dict[str, Any]:
    # Byte-for-byte field semantics from rapidapi_wrapper in the upstream repo.
    return {
        "name": "Finish",
        "description": (
            "If you believe that you have obtained a result that can answer the task, "
            "please call this function to provide the final answer. Alternatively, if "
            "you recognize that you are unable to proceed with the task in the current "
            "state, call this function to restart. Remember: you must ALWAYS call this "
            "function at the end of your attempt, and the only part that will be shown "
            "to the user is the final answer, so it should contain sufficient information."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "return_type": {
                    "type": "string",
                    "enum": ["give_answer", "give_up_and_restart"],
                },
                "final_answer": {
                    "type": "string",
                    "description": 'The final answer you want to give the user. You should have this field if "return_type"=="give_answer"',
                },
            },
            "required": ["return_type"],
        },
    }


def _read_instruction_rows(path: Path, test_set: str) -> list[dict[str, Any]]:
    raw = _read_json(path, label=f"{test_set} instruction data")
    if not isinstance(raw, list) or not raw:
        raise ToolBenchOfficialDataError(
            f"{test_set}: instruction data must be a non-empty JSON list"
        )
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    for index, item in enumerate(raw):
        if not isinstance(item, Mapping):
            raise ToolBenchOfficialDataError(
                f"{test_set}: instruction row {index} is not an object"
            )
        query_id = _normalize_query_id(
            item.get("query_id"), context=f"{test_set} instruction row {index}"
        )
        if query_id in seen:
            raise ToolBenchOfficialDataError(f"{test_set}: duplicate query_id {query_id!r}")
        seen.add(query_id)
        row = _clone_json(dict(item))
        row["query_id"] = query_id
        if not str(row.get("query") or "").strip():
            raise ToolBenchOfficialDataError(f"{test_set} query_id={query_id!r}: query is empty")
        rows.append(row)
    return rows


def _resolve_roster_path(root: Path, test_set: str) -> Path:
    candidates: list[Path] = []
    for directory in ("test_query_ids", "test_ids"):
        for filename in (f"{test_set}_test_query_ids.json", f"{test_set}.json"):
            candidate = root / directory / filename
            if candidate.is_file():
                candidates.append(candidate)
    if not candidates:
        raise ToolBenchOfficialDataError(
            f"ToolBench corpus is incomplete: missing query-ID roster for {test_set} "
            "under test_query_ids/ or test_ids/"
        )
    normalized = [_read_roster(path, test_set) for path in candidates]
    if any(ids != normalized[0] for ids in normalized[1:]):
        raise ToolBenchOfficialDataError(
            f"conflicting ToolBench query-ID rosters found for {test_set}: {candidates}"
        )
    return candidates[0]


def _read_roster(path: Path, test_set: str) -> list[str]:
    raw = _read_json(path, label=f"{test_set} query-ID roster")
    if isinstance(raw, Mapping):
        values = None
        for key in ("query_ids", "test_query_ids", "ids"):
            if key in raw:
                values = raw[key]
                break
        if values is None:
            raise ToolBenchOfficialDataError(
                f"{test_set}: roster object must contain query_ids/test_query_ids/ids"
            )
        raw = values
    if not isinstance(raw, list) or not raw:
        raise ToolBenchOfficialDataError(
            f"{test_set}: query-ID roster must be a non-empty JSON list"
        )
    ids = [_normalize_query_id(value, context=f"{test_set} roster") for value in raw]
    if len(set(ids)) != len(ids):
        raise ToolBenchOfficialDataError(f"{test_set}: query-ID roster contains duplicates")
    return ids


def _require_data_root(data_root: str | Path) -> Path:
    root = Path(data_root).expanduser().resolve()
    if not root.is_dir():
        raise ToolBenchOfficialDataError(f"ToolBench data root does not exist: {root}")
    return root


def _require_test_set(test_set: str) -> None:
    if test_set not in OFFICIAL_TEST_SETS:
        raise ToolBenchOfficialDataError(
            f"unknown ToolBench test set {test_set!r}; expected one of {OFFICIAL_TEST_SETS}"
        )


def _normalize_query_id(value: Any, *, context: str) -> str:
    if isinstance(value, bool) or value is None:
        raise ToolBenchOfficialDataError(f"{context}: query_id is missing or invalid")
    if isinstance(value, int):
        return str(value)
    if isinstance(value, str) and value.strip():
        return value.strip()
    raise ToolBenchOfficialDataError(f"{context}: query_id must be an int or non-empty string")


def _standardize(value: str) -> str:
    normalized = _NON_STANDARD.sub("_", value)
    normalized = re.sub(r"(_){2,}", "_", normalized).lower().strip("_")
    if normalized and normalized[0].isdigit():
        normalized = f"get_{normalized}"
    return normalized


def _change_name(value: str) -> str:
    return f"is_{value}" if value in _RESERVED_FUNCTION_NAMES else value


def _read_json(path: Path, *, label: str) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ToolBenchOfficialDataError(f"cannot read {label} at {path}: {error}") from error


def _read_json_mapping(path: Path, *, label: str) -> Mapping[str, Any]:
    raw = _read_json(path, label=label)
    if not isinstance(raw, Mapping):
        raise ToolBenchOfficialDataError(f"{label} must be a JSON object: {path}")
    return raw


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _clone_json(value: Any) -> Any:
    return json.loads(json.dumps(value, ensure_ascii=False))


def _write_json_atomic(path: Path, payload: Mapping[str, Any], *, overwrite: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and not overwrite:
        raise ToolBenchOfficialDataError(f"refusing to overwrite existing file: {path}")
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=path.parent, prefix=f".{path.name}.", delete=False
    ) as handle:
        staged = Path(handle.name)
        json.dump(payload, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    try:
        if path.exists() and not overwrite:
            raise ToolBenchOfficialDataError(f"refusing to overwrite existing file: {path}")
        os.replace(staged, path)
    except Exception:
        staged.unlink(missing_ok=True)
        raise
