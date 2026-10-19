"""Stable public facade for process-isolated generation authority."""

from lh_platform.core._generation_process import (
    GenerationAuthority,
    IsolatedMethodProxy,
)
from lh_platform.core._generation_protocol import (
    GENERATION_AUTHORITY_SCHEMA,
    AuthorityModelProxy,
    GenerationAuthorityError,
)

__all__ = [
    "AuthorityModelProxy",
    "GENERATION_AUTHORITY_SCHEMA",
    "GenerationAuthority",
    "GenerationAuthorityError",
    "IsolatedMethodProxy",
]
