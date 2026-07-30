"""Deterministic recall extension — REC-01 (framework idioms).

Grounded in the July-2026 false-negative categorization across 520 labeled
repos (3,049 call sites, scanner recall 0.42): the dominant tractable miss
category was agent/chain framework idioms — agent.run(...), chain.invoke(...),
agent.stream(...), crew.kickoff(...) — where the billable call happens inside
framework code and no SDK name appears at the call site.

Design notes (measured, not guessed):
- A companion "local wrapper resolution" pass was built and REMOVED: measured
  precision 0.141 on the labeled corpus — the labeling methodology does not
  count wrapper call sites as separate billable sites. Kept out deliberately.
- String/docstring guard and tool-API receiver deny were added after FP
  sampling (print("agent.run()"), serper_api.invoke) showed those leaks.

Only ADDS findings on lines the primary scanner didn't already cover.
Disable with DETECTION_EXTENSIONS_ENABLED=false.
"""
from __future__ import annotations

import os
import re

_RECEIVER = r"(?P<recv>[A-Za-z_][A-Za-z0-9_]*)"
_METHODS = r"(?P<meth>run|arun|stream|astream|invoke|ainvoke|kickoff|query|aquery|chat|achat)"
_IDIOM = re.compile(rf"\b{_RECEIVER}\.{_METHODS}\s*\(")

# Classic false-positive receivers (Flask app.run, asyncio.run, runners, HTTP).
_RECV_DENY = {
    "app", "application", "server", "flask", "uvicorn", "asyncio", "loop",
    "thread", "process", "pool", "executor", "task", "job", "worker", "queue",
    "subprocess", "os", "sys", "router", "self", "cls", "test", "unittest",
    "pytest", "db", "session", "cursor", "browser", "page", "driver",
}
# Tool/retrieval API receivers — .invoke() on these is not a model call.
_RECV_DENY_SUFFIX = re.compile(r"(api|tool|tools|retriever|search|scraper|parser)$", re.I)
# Receivers that strongly indicate an LLM framework object. Used for whole-file
# agent-context scanning. Anchored to whole identifier tokens (\b...\b) so
# `agent`/`crew`/`chain` match as tokens, not as substrings of
# `agentic_pipeline`/`engine`/etc. (which are not agent objects).
_RECV_STRONG = re.compile(r"\b(agent|chain|crew|llm|copilot|assistant|bot)\b", re.I)

# Name-component test for a call RECEIVER. A bare \b scan of the raw receiver
# misses compound names — `agent_executor`, `llm_chain`, `research_agent`,
# `myAgent` — because `_` is a word char and camelCase has no \b. Split the
# receiver identifier on `_` and camelCase boundaries into lowercased components
# and treat it as strong iff any component is a known agent-object token. This
# RESTORES compound receivers while still rejecting `agentic` (component
# `agentic` != `agent`), `pipeline`, `engine`.
_STRONG_COMPONENTS = {
    "agent", "chain", "crew", "llm", "copilot", "assistant", "bot", "executor",
}


def _recv_components(recv: str) -> set[str]:
    """Lowercased name components of a receiver identifier (split on `_` and camelCase)."""
    spaced = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", "_", recv)
    return {p.lower() for p in spaced.split("_") if p}


def _tiles_into_strong(s: str) -> bool:
    """True iff `s` decomposes end-to-end into a run of _STRONG_COMPONENTS tokens.

    Recovers an agent object written as one lowercase word with no `_`/camelCase
    boundary — `llmchain` -> llm+chain, `agentexecutor` -> agent+executor — which
    _recv_components cannot split. Requires the WHOLE string to tile from the
    start, so partial hits (`blockchain`, `keychain`, `chainsaw`, `toolchain`,
    `robot`) never qualify: none begins with a strong token.
    """
    n = len(s)
    reachable = [True] + [False] * n
    for i in range(n):
        if reachable[i]:
            for tok in _STRONG_COMPONENTS:
                if s.startswith(tok, i):
                    reachable[i + len(tok)] = True
    return reachable[n]


def _recv_is_strong(recv: str) -> bool:
    """Is the receiver a strong agent/LLM object by name-component match?"""
    comps = _recv_components(recv)
    if comps & _STRONG_COMPONENTS:
        return True
    # Pure lowercase concatenation (one component, no boundary the split can use)
    # that tiles entirely into agent tokens — `llmchain`, `agentexecutor`.
    return len(comps) == 1 and _tiles_into_strong(next(iter(comps)))

_COMMENT = re.compile(r"^\s*(#|//|/\*|\*|\"\"\"|'''|>>>|`)")
_DEF_LINE = re.compile(r"^\s*(async\s+)?(def|function|class)\b")

# Agent-framework call forms the generic receiver.method idiom misses:
# OpenAI Agents SDK Runner.run(...), LangGraph compiled-graph .invoke on an `app`/`graph`
# receiver, AutoGen initiate_chat. These are billable agent-loop entry points.
# NOTE: only actual call/run forms belong here. `StateGraph(...)` is a graph
# CONSTRUCTOR — it makes zero LLM calls — so it is a context signal only
# (_FRAMEWORK_CTX below), never a billable finding.
_FRAMEWORK_CALL = re.compile(
    r"\b(Runner\s*\.\s*run|initiate_chat|a?run_stream)\s*\(|"
    r"\b(app|graph|workflow)\s*\.\s*(invoke|ainvoke|stream|astream)\s*\(")

# Self-identifying agent-framework signals (the structural group-1 forms above).
# Their presence anywhere in a file establishes agent context, which the generic
# app/graph/workflow.<method> idiom (group 2) requires before it can be trusted.
_FRAMEWORK_CTX = re.compile(r"\b(Runner|StateGraph|initiate_chat|a?run_stream)\b")

_LOOP_BOUND_RE = re.compile(
    r"\b(max_iterations|max_turns|max_steps|max_iter|recursion_limit|maxSteps|maxTurns)\s*=\s*(\d+)")


def extract_loop_bound(line: str):
    """Return the agent-loop iteration cap declared on this line, or None."""
    m = _LOOP_BOUND_RE.search(line or "")
    return int(m.group(2)) if m else None


def _in_string(line: str, pos: int) -> bool:
    """Is position `pos` inside a string literal on this line? (quote-parity heuristic)"""
    prefix = line[:pos]
    return (prefix.count('"') % 2 == 1) or (prefix.count("'") % 2 == 1) or ("`" in prefix)


def framework_idiom_pass(path: str, content: str, covered: set[int]) -> list[dict]:
    findings = []
    added_lines: set[int] = set()
    # File-level agent context: a strong agent receiver token or a self-identifying
    # framework signal anywhere in the file. Ambiguous methods (.invoke/.ainvoke,
    # which boto3/command-pattern/generic clients also use) are trusted only when
    # this holds — mirroring how the group-2 app/graph/workflow branch is gated.
    agent_ctx = bool(_RECV_STRONG.search(content) or _FRAMEWORK_CTX.search(content))
    for i, line in enumerate(content.splitlines(), 1):
        if i in covered or _COMMENT.match(line) or _DEF_LINE.match(line):
            continue
        m = _IDIOM.search(line)
        if not m:
            continue
        if _in_string(line, m.start()):
            continue
        recv, meth = m.group("recv"), m.group("meth")
        if recv.lower() in _RECV_DENY or _RECV_DENY_SUFFIX.search(recv):
            continue
        # Strong receivers (by name-component) qualify for any method. Weak
        # receivers qualify only for framework-unambiguous methods: kickoff
        # (crewAI-specific) and astream are safe unconditionally; invoke/ainvoke
        # are ambiguous and additionally require file-level agent context.
        if not _recv_is_strong(recv):
            if meth in {"invoke", "ainvoke"}:
                if not agent_ctx:
                    continue
            elif meth not in {"kickoff", "astream"}:
                continue
        findings.append({
            "file_path": path, "line": i, "end_line": i,
            "text": line.strip()[:400], "provider": "framework",
            "method_name": f"{recv}.{meth}", "model": None, "file_content": content,
            "source": "framework_idiom", "loop_bound": extract_loop_bound(line),
        })
        added_lines.add(i)

    # Structural agent-framework call forms the generic idiom above misses
    # (e.g. Runner.run, app.invoke on a LangGraph-compiled graph, initiate_chat).
    # The generic app/graph/workflow.<method> branch (group 2) uses receivers that
    # also appear in plain non-LLM code (FastAPI handlers, plotting graphs), so it
    # is trusted only when the file shows agent context — a strong agent receiver
    # or a self-identifying framework signal (agent_ctx, computed above).
    # Group-1 structural forms need no gate.
    for i, line in enumerate(content.splitlines(), 1):
        if i in covered or i in added_lines or _COMMENT.match(line) or _DEF_LINE.match(line):
            continue
        m = _FRAMEWORK_CALL.search(line)
        if not m:
            continue
        if _in_string(line, m.start()):
            continue
        if m.group(2) and not agent_ctx:
            continue
        method_name = m.group(1) or f"{m.group(2)}.{m.group(3)}"
        findings.append({
            "file_path": path, "line": i, "end_line": i,
            "text": line.strip()[:400], "provider": "framework",
            "method_name": method_name.strip(), "model": None, "file_content": content,
            "source": "framework_idiom", "loop_bound": extract_loop_bound(line),
        })
    return findings


def apply_extensions(files: list[dict], primary: list[dict]) -> list[dict]:
    """Run the idiom pass; return ONLY the additional findings."""
    if os.environ.get("DETECTION_EXTENSIONS_ENABLED", "true").lower() == "false":
        return []
    covered_by_file: dict[str, set[int]] = {}
    for c in primary:
        s = covered_by_file.setdefault(c["file_path"], set())
        s.update(range(int(c["line"]), int(c.get("end_line", c["line"])) + 1))
    # Learned LLM-wrapper registry (compounding recall): match wrapper method names the
    # backstop has taught us on prior audits, deterministically and for free.
    try:
        from erabot._engine.wrapper_registry import learned_wrapper_pass
    except Exception:  # noqa: BLE001 — registry is optional; never break the scan
        learned_wrapper_pass = None
    extra: list[dict] = []
    for f in files:
        path, content = f.get("path", ""), f.get("content", "")
        if not (path and content):
            continue
        covered = covered_by_file.get(path, set())
        found = framework_idiom_pass(path, content, covered)
        extra.extend(found)
        if learned_wrapper_pass is not None:
            # don't double-report lines the idiom pass already covered
            covered = covered | {c["line"] for c in found}
            extra.extend(learned_wrapper_pass(path, content, covered))
    return extra
