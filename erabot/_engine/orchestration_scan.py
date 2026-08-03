"""Static LangGraph orchestration risk-flag analyzer (orchestration pillar v1).

Lifts LangGraph graph construction from source into a thin sketch and emits
CANDIDATE risk flags — a flag claims a risk detectable in code, never a proven
saving (a cap may exist at runtime via config we can't see). Regex-based and
dependency-free so it vendors cleanly into the free OSS wedge.
"""
from __future__ import annotations

import re

_NODE = re.compile(r"""add_node\(\s*['"]([^'"]+)['"]""")
# source and target may be quoted node names OR bare START/END constants
_EDGE = re.compile(r"""add_edge\(\s*(['"]?)(\w+)\1\s*,\s*(['"]?)(\w+)\3""")
_COND = re.compile(r"""add_conditional_edges\(\s*['"]([^'"]+)['"]""")
_ENTRY = re.compile(r"""set_entry_point\(\s*['"]([^'"]+)['"]""")
_INVOKE = re.compile(r"\.(invoke|stream|ainvoke|astream)\(")
_RECURSION = re.compile(r"recursion_limit")
_TOOLS_COND = re.compile(r"tools_condition")
_USES_LANGGRAPH = re.compile(r"langgraph|StateGraph|add_node|add_edge")


def _flag(path, kind, detail):
    return {"path": path, "kind": kind, "detail": detail,
            "evidence_grade": "static", "confidence": "candidate"}


def scan_orchestration(files: list[dict]) -> list[dict]:
    """files: [{'path', 'content'}]. Returns candidate risk-flag dicts."""
    flags: list[dict] = []
    for f in files:
        src = f.get("content", "") or ""
        if not _USES_LANGGRAPH.search(src):
            continue
        path = f.get("path", "")
        nodes = set(_NODE.findall(src))
        _edges = _EDGE.findall(src)                 # (q, source, q, target)
        edge_sources = {m[1] for m in _edges}
        edge_targets = {m[3] for m in _edges}
        cond_sources = set(_COND.findall(src))
        entry = set(_ENTRY.findall(src))
        has_cap = bool(_RECURSION.search(src))
        # A conditional edge or a tools_condition edge can re-enter a node — a
        # candidate loop. (Regex can't see the routing targets, only that a
        # branch exists, which is enough to flag "loop present, no cap".)
        has_loop = bool(_TOOLS_COND.search(src)) or bool(cond_sources)
        invoked = bool(_INVOKE.search(src))

        if has_loop and not has_cap:
            flags.append(_flag(path, "unbounded_loop",
                "a conditional/tool loop can re-enter a node with no recursion_limit set "
                "— a candidate for an unbounded agentic loop"))
        if invoked and not has_cap:
            flags.append(_flag(path, "missing_iteration_cap",
                "the graph is invoked with no recursion_limit in any config — no bound on "
                "iterations (candidate; a cap may be set elsewhere)"))
        # Unreachable-branch detection is only reliable when there are NO
        # conditional edges: their routing targets are a dict/function we can't
        # read statically, so a node they route to would look unreachable. When
        # any conditional edge exists, stay silent rather than false-positive.
        if not cond_sources:
            reachable = edge_targets | edge_sources | entry
            for n in sorted(nodes - reachable):
                flags.append(_flag(path, "unreachable_branch",
                    f"node '{n}' is declared but never targeted by an edge or entry point "
                    f"— dead branch (candidate)"))
    return flags
