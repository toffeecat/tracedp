"""Canonical model-I/O hashes independent of GPU or backend state."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

_TOKENIZER_CONTRACT_FILES = (
    "tokenizer_config.json",
    "special_tokens_map.json",
    "tokenizer.json",
    "tokenizer.model",
    "vocab.json",
    "merges.txt",
    "added_tokens.json",
    "chat_template.jinja",
    "processor_config.json",
    "preprocessor_config.json",
    "generation_config.json",
)


def canonical_sha256(value: Any) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":"),
        default=str,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def chat_template_sha256(tokenizer: Any) -> str:
    template = getattr(tokenizer, "chat_template", None)
    if template is None:
        raise ValueError("tokenizer has no pinned chat template")
    return canonical_sha256(template)


def tokenizer_artifact_fingerprint(model_path: str | Path) -> dict[str, Any]:
    root = Path(model_path).resolve()
    if not root.is_dir():
        raise ValueError("tokenizer artifact root is not a directory")
    files = {}
    for name in _TOKENIZER_CONTRACT_FILES:
        path = root / name
        if path.is_file():
            files[name] = hashlib.sha256(path.read_bytes()).hexdigest()
    if not files:
        raise ValueError("tokenizer artifact root has no contract files")
    return {
        "files": files,
        "artifact_sha256": canonical_sha256(files),
    }


__all__ = [
    "canonical_sha256",
    "chat_template_sha256",
    "tokenizer_artifact_fingerprint",
]
