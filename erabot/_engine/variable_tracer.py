"""Variable tracer for LLM call argument analysis.

Given a detected LLM call site and the full file content, traces the
messages= / content= keyword arguments to determine what text is actually
being sent to the LLM.

Three trace outcomes map to estimation confidence levels:
  - literal_found   -> "high"   + "argument_content_analysis"
  - variable_traced -> "medium" + "variable_trace_analysis"
  - dynamic         -> "low"    + "call_site_static_analysis"

This module uses tree-sitter AST analysis to:
1. Find the keyword argument (messages, content, prompt, input) in the call
2. If the value is a literal (string, list, dict) -> extract the text directly
3. If the value is an identifier -> search backward for an assignment and extract
4. If the value is dynamic (function call, attribute access, etc.) -> return dynamic
"""

import logging
import os
from dataclasses import dataclass
from typing import Optional

from tree_sitter import Language, Parser
from tree_sitter_language_pack import get_language

logger = logging.getLogger(__name__)

# Keyword argument names that carry prompt content
_CONTENT_KEYWORDS_PYTHON = frozenset({
    "messages", "content", "prompt", "input", "text",
    "query", "question", "system", "user",
})

_CONTENT_KEYWORDS_TYPESCRIPT = frozenset({
    "messages", "content", "prompt", "input", "text",
    "query", "question", "system", "user",
})

# Language name -> content keyword set
_CONTENT_KEYWORDS = {
    "python": _CONTENT_KEYWORDS_PYTHON,
    "typescript": _CONTENT_KEYWORDS_TYPESCRIPT,
    "javascript": _CONTENT_KEYWORDS_TYPESCRIPT,
}

# Node types that indicate a literal value
_LITERAL_TYPES_PYTHON = frozenset({
    "string", "integer", "float", "true", "false", "none",
    "list", "dictionary", "tuple", "set",
    "concatenated_string", "f_string",
})

_LITERAL_TYPES_TYPESCRIPT = frozenset({
    "string", "number", "true", "false", "null", "undefined",
    "array", "object", "template_string",
})

_LITERAL_TYPES = {
    "python": _LITERAL_TYPES_PYTHON,
    "typescript": _LITERAL_TYPES_TYPESCRIPT,
    "javascript": _LITERAL_TYPES_TYPESCRIPT,
}

# Node types that indicate a dynamic value (cannot be statically resolved)
_DYNAMIC_TYPES = frozenset({
    "call", "call_expression",
    "await_expression",
    "attribute", "member_expression",
    "subscript", "subscript_expression",
    "binary_operator", "binary_expression",
    "conditional_expression", "ternary_expression",
    "yield_expression", "yield",
    "lambda", "arrow_function",
    "generator_expression", "list_comprehension",
    "dictionary_comprehension", "set_comprehension",
})

# Assignment node types by language
_ASSIGNMENT_TYPES = {
    "python": frozenset({"assignment", "augmented_assignment"}),
    "typescript": frozenset({"lexical_declaration", "variable_declaration"}),
    "javascript": frozenset({"lexical_declaration", "variable_declaration"}),
}


@dataclass
class TraceResult:
    """Result of tracing an LLM call's content argument."""

    # One of: "literal_found", "variable_traced", "dynamic", "no_content_arg"
    trace_outcome: str

    # The traced content text (if resolved), or None
    traced_content: Optional[str] = None

    # The keyword name that was traced (e.g. "messages", "content")
    keyword_name: Optional[str] = None

    # The estimation method to use based on trace outcome
    estimation_method: str = "call_site_static_analysis"

    # The estimation confidence to use based on trace outcome
    estimation_confidence: str = "low"


def _detect_language(file_path: str) -> Optional[str]:
    """Detect tree-sitter language name from file extension."""
    ext = os.path.splitext(file_path)[1].lower()
    ext_map = {
        ".py": "python",
        ".ts": "typescript",
        ".tsx": "typescript",
        ".js": "javascript",
        ".jsx": "javascript",
    }
    return ext_map.get(ext)


def _get_parser(lang_name: str) -> Optional[Parser]:
    """Get a tree-sitter parser for the given language."""
    try:
        language = get_language(lang_name)
        return Parser(language)
    except Exception as e:
        logger.warning("Failed to get parser for %s: %s", lang_name, e)
        return None


def _find_call_node_at_line(root_node, start_line: int, end_line: int, content: str):
    """Find the call AST node that matches the scanner finding's line range.

    Walks the AST to find a call/call_expression node whose line range
    overlaps with the given start_line..end_line (0-indexed rows).
    """
    call_types = {"call", "call_expression"}
    best = None

    def visit(node):
        nonlocal best
        node_start = node.start_point[0]
        node_end = node.end_point[0]

        if node.type in call_types:
            # Check overlap: node range overlaps with finding range
            if node_start <= end_line and node_end >= start_line:
                # Prefer the node closest to the finding's start
                if best is None or abs(node_start - start_line) < abs(best.start_point[0] - start_line):
                    best = node

        for child in node.children:
            visit(child)

    visit(root_node)
    return best


def _find_keyword_arg_python(call_node, content_keywords: frozenset, content: str):
    """Find a content-bearing keyword argument in a Python call node.

    Checks keyword arguments first (messages=..., content=..., prompt=...).
    Falls back to the first positional argument for methods like
    generate_content("text") where the content is a positional arg.

    Returns (keyword_name, value_node) or (None, None).
    """
    # Look for argument_list child
    for child in call_node.children:
        if child.type == "argument_list":
            # First pass: check keyword arguments
            for arg in child.children:
                if arg.type == "keyword_argument":
                    children = [c for c in arg.children if c.type not in ("=", ",")]
                    if len(children) >= 2:
                        key_node = children[0]
                        val_node = children[1]
                        key_text = content[key_node.start_byte:key_node.end_byte]
                        if key_text in content_keywords:
                            return key_text, val_node

            # Second pass: if no keyword match, use first positional argument
            # This handles generate_content("text"), invoke("text"), etc.
            for arg in child.children:
                if arg.type not in ("(", ")", ",", "keyword_argument", "**", "dictionary_splat"):
                    # This is a positional argument
                    return "content", arg
    return None, None


def _find_keyword_arg_typescript(call_node, content_keywords: frozenset, content: str):
    """Find a content-bearing keyword in a TypeScript/JS call's object argument.

    In TS/JS, LLM calls typically pass an object literal:
    openai.chat.completions.create({model: 'gpt-4o', messages: [...]})

    Returns (keyword_name, value_node) or (None, None).
    """
    # Look for arguments child
    for child in call_node.children:
        if child.type == "arguments":
            for arg in child.children:
                if arg.type == "object":
                    # Object has pair children: property_identifier: value
                    for pair in arg.children:
                        if pair.type == "pair":
                            pair_children = [c for c in pair.children if c.type not in (":", ",")]
                            if len(pair_children) >= 2:
                                key_node = pair_children[0]
                                val_node = pair_children[1]
                                key_text = content[key_node.start_byte:key_node.end_byte]
                                if key_text in content_keywords:
                                    return key_text, val_node
    return None, None


def _classify_value_node(value_node, lang_name: str) -> str:
    """Classify a value node as 'literal', 'identifier', or 'dynamic'.

    Returns one of: 'literal', 'identifier', 'dynamic'
    """
    node_type = value_node.type
    literal_types = _LITERAL_TYPES.get(lang_name, set())

    if node_type in literal_types:
        return "literal"
    if node_type == "identifier":
        return "identifier"
    if node_type in _DYNAMIC_TYPES:
        return "dynamic"

    # For f-strings and template strings, treat as literal (we can extract text)
    if node_type in ("f_string", "template_string", "concatenated_string"):
        return "literal"

    # Default: dynamic (cannot resolve statically)
    return "dynamic"


def _search_backward_for_assignment_python(root_node, var_name: str, before_line: int, content: str) -> Optional[str]:
    """Search backward from a line for an assignment to var_name in Python.

    Scans all assignment nodes in the file that appear before before_line,
    picks the closest one, and returns the assigned value as text.
    """
    candidates = []

    def visit(node):
        if node.type == "assignment":
            # assignment: identifier = value
            children = [c for c in node.children if c.type not in ("=", ":", "type")]
            if len(children) >= 2:
                target = children[0]
                value = children[-1]
                if target.type == "identifier":
                    target_name = content[target.start_byte:target.end_byte]
                    if target_name == var_name and node.start_point[0] < before_line:
                        candidates.append((node.start_point[0], value))

        for child in node.children:
            visit(child)

    visit(root_node)

    if not candidates:
        return None

    # Pick the assignment closest to (but before) the target line
    candidates.sort(key=lambda x: x[0], reverse=True)
    _, value_node = candidates[0]

    # Check if the assigned value is itself dynamic
    # We only return the text for literals and simple structures
    value_type = value_node.type
    literal_types = _LITERAL_TYPES.get("python", set())

    if value_type in literal_types or value_type == "identifier":
        return content[value_node.start_byte:value_node.end_byte]

    # For dynamic assignments, still return the text but mark as traced
    # The caller should check if this looks like a literal
    return content[value_node.start_byte:value_node.end_byte]


def _search_backward_for_assignment_typescript(root_node, var_name: str, before_line: int, content: str) -> Optional[str]:
    """Search backward for a variable declaration/assignment in TypeScript/JS.

    Handles: const x = ..., let x = ..., var x = ...
    """
    candidates = []

    def visit(node):
        if node.type in ("lexical_declaration", "variable_declaration"):
            for child in node.children:
                if child.type == "variable_declarator":
                    declarator_children = [c for c in child.children if c.type not in ("=", ":", "type_annotation")]
                    if len(declarator_children) >= 2:
                        name_node = declarator_children[0]
                        value_node = declarator_children[-1]
                        if name_node.type == "identifier":
                            name_text = content[name_node.start_byte:name_node.end_byte]
                            if name_text == var_name and node.start_point[0] < before_line:
                                candidates.append((node.start_point[0], value_node))

        for child in node.children:
            visit(child)

    visit(root_node)

    if not candidates:
        return None

    candidates.sort(key=lambda x: x[0], reverse=True)
    _, value_node = candidates[0]
    return content[value_node.start_byte:value_node.end_byte]


def _is_value_literal(value_text: str, lang_name: str, root_node, content: str) -> bool:
    """Check if a traced value text represents a literal (not a function call or variable)."""
    # Quick heuristic: if it starts with a literal indicator
    stripped = value_text.strip()
    if not stripped:
        return False

    # String literals
    if stripped[0] in ('"', "'", '`', 'f"', "f'"):
        return True
    if stripped.startswith('"""') or stripped.startswith("'''"):
        return True

    # List/array literals
    if stripped[0] == '[':
        return True

    # Dict/object literals
    if stripped[0] == '{':
        return True

    # Tuple
    if stripped[0] == '(' and lang_name == "python":
        return True

    # Number literals
    try:
        float(stripped)
        return True
    except (ValueError, OverflowError):
        pass

    # Boolean/None/null/undefined
    if stripped in ("True", "False", "None", "true", "false", "null", "undefined"):
        return True

    return False


def trace_arguments(
    file_path: str,
    file_content: str,
    finding_start_line: int,
    finding_end_line: int,
) -> TraceResult:
    """Trace LLM call arguments to determine content for token estimation.

    Given a file and a finding's line range (1-indexed), finds the LLM call
    in the AST and traces its content-bearing keyword argument (messages,
    content, prompt, etc.) to determine what text is being sent to the LLM.

    Args:
        file_path: Path to the source file (used for language detection).
        file_content: Full source code content of the file.
        finding_start_line: 1-indexed start line of the LLM call finding.
        finding_end_line: 1-indexed end line of the LLM call finding.

    Returns:
        TraceResult with trace outcome, optional traced content, and
        recommended estimation method/confidence.
    """
    lang_name = _detect_language(file_path)
    if lang_name is None:
        return TraceResult(
            trace_outcome="dynamic",
            estimation_method="call_site_static_analysis",
            estimation_confidence="low",
        )

    parser = _get_parser(lang_name)
    if parser is None:
        return TraceResult(
            trace_outcome="dynamic",
            estimation_method="call_site_static_analysis",
            estimation_confidence="low",
        )

    try:
        tree = parser.parse(file_content.encode("utf-8"))
    except Exception as e:
        logger.warning("Failed to parse %s: %s", file_path, e)
        return TraceResult(
            trace_outcome="dynamic",
            estimation_method="call_site_static_analysis",
            estimation_confidence="low",
        )

    root = tree.root_node

    # Convert from 1-indexed to 0-indexed for tree-sitter
    start_line_0 = finding_start_line - 1
    end_line_0 = finding_end_line - 1

    # Step 1: Find the call node
    call_node = _find_call_node_at_line(root, start_line_0, end_line_0, file_content)
    if call_node is None:
        return TraceResult(
            trace_outcome="dynamic",
            estimation_method="call_site_static_analysis",
            estimation_confidence="low",
        )

    # Step 2: Find the content keyword argument
    content_keywords = _CONTENT_KEYWORDS.get(lang_name, set())

    if lang_name == "python":
        keyword_name, value_node = _find_keyword_arg_python(call_node, content_keywords, file_content)
    else:
        keyword_name, value_node = _find_keyword_arg_typescript(call_node, content_keywords, file_content)

    if keyword_name is None or value_node is None:
        return TraceResult(
            trace_outcome="no_content_arg",
            estimation_method="call_site_static_analysis",
            estimation_confidence="low",
        )

    # Step 3: Classify the value node
    classification = _classify_value_node(value_node, lang_name)

    if classification == "literal":
        # Direct literal - extract the text
        traced_text = file_content[value_node.start_byte:value_node.end_byte]
        return TraceResult(
            trace_outcome="literal_found",
            traced_content=traced_text,
            keyword_name=keyword_name,
            estimation_method="argument_content_analysis",
            estimation_confidence="high",
        )

    if classification == "identifier":
        # Variable reference - trace backward
        var_name = file_content[value_node.start_byte:value_node.end_byte]

        if lang_name == "python":
            traced_text = _search_backward_for_assignment_python(
                root, var_name, start_line_0, file_content
            )
        else:
            traced_text = _search_backward_for_assignment_typescript(
                root, var_name, start_line_0, file_content
            )

        if traced_text is not None:
            # Check if the traced value is itself a literal
            if _is_value_literal(traced_text, lang_name, root, file_content):
                return TraceResult(
                    trace_outcome="variable_traced",
                    traced_content=traced_text,
                    keyword_name=keyword_name,
                    estimation_method="variable_trace_analysis",
                    estimation_confidence="medium",
                )
            else:
                # Variable found but assigned a dynamic value
                return TraceResult(
                    trace_outcome="dynamic",
                    traced_content=traced_text,
                    keyword_name=keyword_name,
                    estimation_method="call_site_static_analysis",
                    estimation_confidence="low",
                )

        # Variable not found in scope
        return TraceResult(
            trace_outcome="dynamic",
            keyword_name=keyword_name,
            estimation_method="call_site_static_analysis",
            estimation_confidence="low",
        )

    # Dynamic value (function call, attribute access, etc.)
    return TraceResult(
        trace_outcome="dynamic",
        keyword_name=keyword_name,
        estimation_method="call_site_static_analysis",
        estimation_confidence="low",
    )
