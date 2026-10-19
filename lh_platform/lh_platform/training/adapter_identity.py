"""Content-addressed identity for a PEFT adapter checkpoint.

This module owns adapter provenance.  Callers receive one typed, validated
record instead of re-parsing checkpoint manifests at each experiment seam.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

from lh_platform.training.transformers_peft_policy import (
    adapter_artifact_tree_sha256,
)


class AdapterIdentityError(ValueError):
    """Raised when an adapter manifest does not bind its checkpoint files."""


@dataclass(frozen=True)
class AdapterIdentity:
    path: str
    checkpoint_sha256: str
    adapter_artifact_sha256: str
    base_checkpoint_sha256: str

    def to_dict(self) -> dict[str, str]:
        return asdict(self)


def _require_sha256(value: object, *, field: str) -> str:
    digest = str(value)
    if len(digest) != 64 or any(character not in "0123456789abcdef" for character in digest):
        raise AdapterIdentityError(f"adapter manifest lacks a valid {field}")
    return digest


def load_adapter_identity(path: str | Path) -> AdapterIdentity:
    resolved = Path(path).resolve()
    try:
        manifest = json.loads((resolved / "checkpoint_manifest.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise AdapterIdentityError(f"cannot read adapter checkpoint manifest: {error}") from error
    if not isinstance(manifest, dict):
        raise AdapterIdentityError("adapter checkpoint manifest must be an object")
    checkpoint_sha256 = _require_sha256(
        manifest.get("checkpoint_sha256"), field="checkpoint_sha256"
    )
    expected_artifact = _require_sha256(
        manifest.get("adapter_artifact_sha256"), field="adapter_artifact_sha256"
    )
    actual_artifact = adapter_artifact_tree_sha256(resolved)
    if expected_artifact != actual_artifact:
        raise AdapterIdentityError("adapter manifest does not bind its artifact files")
    return AdapterIdentity(
        path=str(resolved),
        checkpoint_sha256=checkpoint_sha256,
        adapter_artifact_sha256=actual_artifact,
        base_checkpoint_sha256=_require_sha256(
            manifest.get("base_checkpoint_sha256"),
            field="base_checkpoint_sha256",
        ),
    )


__all__ = [
    "AdapterIdentity",
    "AdapterIdentityError",
    "load_adapter_identity",
]
