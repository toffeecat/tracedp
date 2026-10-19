"""Redact credentials from persisted experiment artifacts and diagnostics.

Every storage path should pass structured values through ``redact_artifact``.
The narrower ``redact_secrets`` and ``redact_text`` functions remain useful at
runtime boundaries where callers need to preserve a mapping or message shape.
This module owns sanitization only; it does not decide experiment credibility.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from pathlib import Path
from typing import Any
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

_SECRET_KEYS = frozenset(
    {
        "api_key",
        "api-key",
        "toolbench_key",
        "toolbench-key",
        "x_api_key",
        "authorization",
        "bearer_token",
        "api_token",
        "access_token",
        "x_auth_token",
        "cookie",
        "set_cookie",
        "client_secret",
        "signature",
        "token",
        "secret",
        "password",
    }
)
_SECRET_KEY_NORMALIZED = frozenset(re.sub(r"[^a-z0-9]", "", key.lower()) for key in _SECRET_KEYS)
_URL_VALUE_KEYS = frozenset({"baseurl", "url", "endpoint", "endpointurl"})
_HEADER_KEYS = frozenset({"headers", "header"})
_SAFE_HEADER_KEYS = frozenset({"accept", "contenttype", "useragent"})
_URL_IN_TEXT_RE = re.compile(
    r"[a-z][a-z0-9+.-]*://(?:<redacted>|[^\s<>\"'])+",
    re.IGNORECASE,
)
_HEADER_LINE_RE = re.compile(
    r"(?im)^(\s*(?:authorization|proxy-authorization|x-api-key|x-auth-token|"
    r"bearer-token|cookie|set-cookie)\s*:\s*)([^\r\n]*)$"
)
_INLINE_SECRET_RE = re.compile(
    r"""(?ix)
    (?P<prefix>
        ["']?(?:api[_-]?key|toolbench[_-]?key|x[_-]?api[_-]?key|authorization|bearer[_-]?token|
        api[_-]?token|access[_-]?token|x[_-]?auth[_-]?token|cookie|client[_-]?secret|
        signature|token|secret|password)["']?\s*[:=]\s*
    )
    (?P<value>
        (?:Bearer\s+)?(?:
            "(?:[^"\\]|\\.)*" |
            '(?:[^'\\]|\\.)*' |
            [^\s,;&\]\[}]+
        )
    )
    """,
    re.VERBOSE,
)


def redact_secrets(value: Any, *, public_url_query_keys: frozenset[str] = frozenset()) -> Any:
    """Return a JSON-safe copy that cannot leak credentials into artifacts."""

    if isinstance(value, Mapping):
        out: dict[str, Any] = {}
        for key, item in value.items():
            key_text = str(key)
            normalized_key = re.sub(r"[^a-z0-9]", "", key_text.lower())
            if normalized_key in _HEADER_KEYS and isinstance(item, Mapping):
                out[key_text] = _redact_headers(item)
            elif normalized_key in _SECRET_KEY_NORMALIZED:
                out[key_text] = "<redacted>"
            elif normalized_key in _URL_VALUE_KEYS and isinstance(item, str):
                out[key_text] = _redact_url(item, public_url_query_keys=public_url_query_keys)
            else:
                out[key_text] = redact_secrets(item, public_url_query_keys=public_url_query_keys)
        return out
    if isinstance(value, (list, tuple)):
        return [redact_secrets(item, public_url_query_keys=public_url_query_keys) for item in value]
    if isinstance(value, Path):
        return str(value)
    return value


def _redact_headers(headers: Mapping[Any, Any]) -> dict[str, Any]:
    """Preserve only harmless HTTP headers in persisted artifacts."""

    out: dict[str, Any] = {}
    for key, value in headers.items():
        key_text = str(key)
        normalized_key = re.sub(r"[^a-z0-9]", "", key_text.lower())
        if normalized_key in _SAFE_HEADER_KEYS:
            out[key_text] = redact_text(str(value))
        else:
            out[key_text] = "<redacted>"
    return out


def _redact_url(value: str, *, public_url_query_keys: frozenset[str] = frozenset()) -> str:
    """Remove credentials and preserve only explicitly public query values."""

    try:
        parsed = urlsplit(value)
    except ValueError:
        without_fragment, fragment_separator, _fragment = value.partition("#")
        prefix, separator, _query = without_fragment.partition("?")
        prefix = re.sub(r"(?<=://)[^/@]+@", "<redacted>@", prefix)
        return (
            prefix
            + ("?<redacted>" if separator else "")
            + ("#<redacted>" if fragment_separator else "")
        )
    if not parsed.scheme and not parsed.netloc:
        return value
    netloc = parsed.netloc
    if "@" in netloc:
        netloc = f"<redacted>@{netloc.rsplit('@', 1)[1]}"
    query = []
    for key, item in parse_qsl(parsed.query, keep_blank_values=True):
        normalized_key = re.sub(r"[^a-z0-9]", "", key.lower())
        public = key in public_url_query_keys and normalized_key not in _SECRET_KEY_NORMALIZED
        query.append((key, redact_text(item) if public else "<redacted>"))
    return urlunsplit(
        (
            parsed.scheme,
            netloc,
            parsed.path,
            urlencode(query),
            "<redacted>" if parsed.fragment else "",
        )
    )


def redact_text(value: str, *, public_url_query_keys: frozenset[str] = frozenset()) -> str:
    """Redact credentials embedded in exception messages, traces, and logs."""

    def redact_url(match: re.Match[str]) -> str:
        candidate = match.group(0)
        suffix = ""
        while candidate and candidate[-1] in ".,;:)]}":
            suffix = candidate[-1] + suffix
            candidate = candidate[:-1]
        return _redact_url(candidate, public_url_query_keys=public_url_query_keys) + suffix

    text = _URL_IN_TEXT_RE.sub(redact_url, str(value))
    text = _HEADER_LINE_RE.sub(r"\1<redacted>", text)

    def redact_inline(match: re.Match[str]) -> str:
        value_text = match.group("value")
        quote = value_text[0] if value_text[:1] in {"'", '"'} else ""
        return f"{match.group('prefix')}{quote}<redacted>{quote}"

    return _INLINE_SECRET_RE.sub(redact_inline, text)


def redact_artifact(value: Any, *, public_url_query_keys: frozenset[str] = frozenset()) -> Any:
    """Sanitize artifacts; callers may explicitly retain public search parameters."""

    sanitized = redact_secrets(value, public_url_query_keys=public_url_query_keys)
    if isinstance(sanitized, Mapping):
        return {
            str(key): redact_artifact(item, public_url_query_keys=public_url_query_keys)
            for key, item in sanitized.items()
        }
    if isinstance(sanitized, list):
        return [
            redact_artifact(item, public_url_query_keys=public_url_query_keys) for item in sanitized
        ]
    if isinstance(sanitized, tuple):
        return [
            redact_artifact(item, public_url_query_keys=public_url_query_keys) for item in sanitized
        ]
    if isinstance(sanitized, str):
        return redact_text(sanitized, public_url_query_keys=public_url_query_keys)
    return sanitized


__all__ = ["redact_artifact", "redact_secrets", "redact_text"]
