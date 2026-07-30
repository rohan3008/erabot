"""JavaScript tree-sitter S-expression queries for LLM SDK call detection.

JavaScript and TypeScript share the same AST node types for function calls
(call_expression, member_expression, property_identifier), so the patterns
are identical. Re-exported from the TypeScript module.
"""

from .typescript import TYPESCRIPT_PATTERNS as JAVASCRIPT_PATTERNS

__all__ = ["JAVASCRIPT_PATTERNS"]
