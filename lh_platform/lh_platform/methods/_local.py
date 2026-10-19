"""Local-method adapter helper."""

from __future__ import annotations

import sys
from pathlib import Path


def lh_root() -> Path:
    """Root of the LongHorizon project (where method/ and eval-benchmark/ live)."""
    here = Path(__file__).resolve()
    p = here
    for _ in range(8):
        if (p / "method").is_dir() and (p / "eval-benchmark").is_dir():
            return p
        if (p / "old" / "method").is_dir() and (p / "old" / "eval-benchmark").is_dir():
            return p / "old"
        p = p.parent
    return here


def method_repo_path(slug: str) -> Path:
    """Resolve a method repo by exact name match (case-insensitive) under method/.

    Returns the directory itself if it contains a `repo/` subdir (we then
    return the `repo/` path so callers can `import` directly), or the
    directory itself if it only has `paper.pdf`.
    """
    found = find_method_repo(slug)
    if found is not None:
        return found
    # Fallback: walk and find the directory that matches by name (case-insensitive)
    root = lh_root() / "method"
    if not root.is_dir():
        return root / slug
    target = slug.lower()
    for d in root.rglob("*"):
        if d.is_dir() and d.name.lower() == target:
            if (d / "repo").is_dir():
                return d / "repo"
            return d
    return root / slug


def add_repo_to_syspath(repo: Path) -> bool:
    """Deprecated: use ``lifecycle.bind_repo_import`` / ``resolve_repo_context``.

    Kept only for emergency/out-of-tree callers.  Adapter modules must not
    call this — ADR 0004 makes Method Adapter Lifecycle the sole sys.path seam.
    """

    import warnings

    warnings.warn(
        "add_repo_to_syspath is deprecated; use lh_platform.methods.lifecycle.bind_repo_import",
        DeprecationWarning,
        stacklevel=2,
    )
    if not repo.is_dir():
        return False
    sp = str(repo.resolve())
    if sp not in sys.path:
        sys.path.insert(0, sp)
    return True


def has_repo(slug: str) -> bool:
    """True if the method's local `repo/` directory is present.

    Accepts both layouts: <method>/repo/ exists OR method_repo_path
    already points at the `repo/` directory itself.
    """
    found = find_method_repo(slug)
    if found is not None:
        return True
    p = method_repo_path(slug)
    if p.is_dir():
        # method_repo_path may already return the repo/ subdir
        if p.name == "repo":
            return True
        if (p / "repo").is_dir():
            return True
    return False


def find_method_repo(slug: str):
    """Robust lookup that walks method/ and finds a directory whose name
    contains `slug` (case-insensitive) and has a `repo/` subdirectory.
    """
    root = lh_root() / "method"
    if not root.is_dir():
        return None
    target = slug.lower()
    for d in root.rglob("*"):
        if d.is_dir() and d.name.lower() == target and (d / "repo").is_dir():
            return d / "repo"
    return None


def placeholder_note(slug: str, expected_url: str) -> str:
    return f"[{slug}] local repo not found; expected clone URL: {expected_url}"
