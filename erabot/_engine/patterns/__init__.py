"""Tree-sitter pattern registry for multi-language LLM call detection.

Maps file extensions to S-expression query strings for the corresponding
tree-sitter language grammar.
"""

import logging
from typing import Optional

from .python import PYTHON_PATTERNS
from .typescript import TYPESCRIPT_PATTERNS
from .javascript import JAVASCRIPT_PATTERNS

logger = logging.getLogger(__name__)

# Extension -> (language_name, pattern_string)
_EXTENSION_REGISTRY: dict[str, tuple[str, str]] = {
    ".py": ("python", PYTHON_PATTERNS),
    ".ts": ("typescript", TYPESCRIPT_PATTERNS),
    ".tsx": ("typescript", TYPESCRIPT_PATTERNS),
    ".js": ("javascript", JAVASCRIPT_PATTERNS),
    ".jsx": ("javascript", JAVASCRIPT_PATTERNS),
}


def get_patterns_for_extension(ext: str) -> Optional[str]:
    """Get S-expression query patterns for a file extension.

    Args:
        ext: File extension including dot (e.g. '.py', '.ts', '.js')

    Returns:
        S-expression query string if the extension is supported, None otherwise.
    """
    entry = _EXTENSION_REGISTRY.get(ext)
    if entry is None:
        return None
    return entry[1]


def get_language_name_for_extension(ext: str) -> Optional[str]:
    """Get tree-sitter language name for a file extension.

    Args:
        ext: File extension including dot (e.g. '.py', '.ts', '.js')

    Returns:
        Language name string (e.g. 'python', 'typescript') or None.
    """
    entry = _EXTENSION_REGISTRY.get(ext)
    if entry is None:
        return None
    return entry[0]


def get_supported_extensions() -> set[str]:
    """Return set of all supported file extensions."""
    return set(_EXTENSION_REGISTRY.keys())


__all__ = [
    "get_patterns_for_extension",
    "get_language_name_for_extension",
    "get_supported_extensions",
    "PYTHON_PATTERNS",
    "TYPESCRIPT_PATTERNS",
    "JAVASCRIPT_PATTERNS",
]
