"""Method Adapter Lifecycle helpers.

This module owns the common repo-import lifecycle for method adapters:
resolve the local upstream repo path, probe imports, and enforce mode honesty.
"""

from __future__ import annotations

import sys
from collections.abc import Callable, Iterable
from dataclasses import asdict, dataclass, field
from pathlib import Path

from lh_platform.methods._local import has_repo, method_repo_path


@dataclass
class ProbeResult:
    success: bool
    error: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class RepoContext:
    slug: str
    repo_path: Path | None
    import_path: Path | None
    probe_result: ProbeResult
    supported_paths: list[str] = field(default_factory=list)
    import_failures: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        data = asdict(self)
        data["repo_path"] = str(self.repo_path) if self.repo_path is not None else None
        data["import_path"] = str(self.import_path) if self.import_path is not None else None
        return data


_PROBE_CACHE: dict[tuple[str, str], ProbeResult] = {}


def clear_probe_cache() -> None:
    _PROBE_CACHE.clear()


def resolve_repo_context(
    *,
    slug: str,
    lookup_slug: str | None = None,
    explicit_repo: Path | None = None,
    import_path: Path | None = None,
    import_path_for_repo: Callable[[Path], Path] | None = None,
    probe_import: Callable[[], bool],
    supported_paths: Iterable[str],
) -> RepoContext:
    """Resolve a method repo and probe whether its upstream package imports."""
    repo_path = (
        Path(explicit_repo) if explicit_repo is not None else _resolve_repo(lookup_slug or slug)
    )
    if import_path is not None:
        probe_path = import_path
    elif repo_path is not None and import_path_for_repo is not None:
        probe_path = import_path_for_repo(repo_path)
    else:
        probe_path = repo_path
    probe_result = _probe(slug=slug, import_path=probe_path, probe_import=probe_import)
    import_failures = [] if probe_result.success else [probe_result.error]
    return RepoContext(
        slug=slug,
        repo_path=repo_path,
        import_path=probe_path,
        probe_result=probe_result,
        supported_paths=list(supported_paths),
        import_failures=import_failures,
    )


def require_native_available(
    repo_context: RepoContext,
    *,
    mode: str,
    exc_type: type[RuntimeError],
    import_label: str,
) -> None:
    """Fail fast when native mode was requested but the upstream probe failed."""
    if mode != "native" or repo_context.probe_result.success:
        return
    path = repo_context.import_path or repo_context.repo_path
    raise exc_type(
        f"{import_label} not importable from {path!s}; "
        f"refusing to silently fall back to prompt mode."
    )


def bind_repo_import(
    *,
    slug: str,
    lookup_slug: str,
    probe_import: Callable[[], bool],
    supported_paths: Iterable[str] = ("prompt_wrapper",),
    explicit_repo: Path | None = None,
    import_path: Path | None = None,
    import_path_for_repo: Callable[[Path], Path] | None = None,
) -> RepoContext:
    """ADR 0004 entry for repo_import adapters.

    Prefer this over ``_local.add_repo_to_syspath``.  ``probe_import`` runs
    after the resolved import path is on ``sys.path`` (when present).
    """

    return resolve_repo_context(
        slug=slug,
        lookup_slug=lookup_slug,
        explicit_repo=explicit_repo,
        import_path=import_path,
        import_path_for_repo=import_path_for_repo,
        probe_import=probe_import,
        supported_paths=supported_paths,
    )


def _resolve_repo(lookup_slug: str) -> Path | None:
    if has_repo(lookup_slug):
        return method_repo_path(lookup_slug)
    return None


def _probe(
    *,
    slug: str,
    import_path: Path | None,
    probe_import: Callable[[], bool],
) -> ProbeResult:
    cache_key = (slug, str(import_path) if import_path is not None else "")
    if cache_key in _PROBE_CACHE:
        return _PROBE_CACHE[cache_key]

    added_path = ""
    if import_path is not None and import_path.is_dir():
        added_path = str(import_path.resolve())
        if added_path not in sys.path:
            sys.path.insert(0, added_path)

    try:
        success = bool(probe_import())
        result = ProbeResult(success=success, error="" if success else "probe returned false")
    except Exception as e:
        result = ProbeResult(success=False, error=repr(e))

    if added_path and not result.success:
        try:
            sys.path.remove(added_path)
        except ValueError:
            pass

    _PROBE_CACHE[cache_key] = result
    return result
