# engine/turn_model.py
"""Estimate how many times a call site fires per user task (agent loop turns).

Static heuristic, zero-integration: an explicit agent-loop bound
(max_iterations / max_turns / max_steps / max_iter / recursion_limit) wins; else a
surrounding for/while loop gets a conservative band; else a single call (1). Returns
a band {calls_per_task, basis, low, high} — the point estimate plus an uncertainty
range, mirroring how volume is already an assumption in the cost model.
"""
from __future__ import annotations
import re

# Explicit agent-loop iteration bounds, in priority order. Value is the cap.
_BOUND_RE = re.compile(
    r"\b(max_iterations|max_turns|max_steps|max_iter|recursion_limit|maxSteps|maxTurns)\s*=\s*(\d+)")
_LOOP_RE = re.compile(r"^\s*(for|while)\b")
_LOOP_DEFAULT = 4      # conservative point estimate for an unbounded agent loop
_LOOP_BAND = (2, 8)    # low/high band for an unbounded loop
# Sanity ceiling on inferred turns-per-task. Agent loops realistically top out in
# the low tens; a larger bound (range(10000), max_turns=9999) is almost always a
# batch/data loop — that's call VOLUME, not agent turns, and multiplying per-site
# cost by it produces an absurd, dominating estimate. Cap it; real turn counts
# need traces anyway.
_MAX_INFERRED_TURNS = 50


def _cap(n: int) -> int:
    return min(int(n), _MAX_INFERRED_TURNS)


def _indent(line: str) -> int:
    return len(line) - len(line.lstrip())


def _enclosing_loop_header(lines: list[str], idx: int):
    """Scope-aware: the header line of the nearest for/while loop enclosing the
    call at `idx` (0-based), or None. Walk upward for the nearest header at a lower
    indent than the current bar: a for/while means enclosed (return its text); a
    def/class means we hit the function boundary with no enclosing loop; any other
    block header (if/with/try) moves the bar up and we keep looking. This ignores
    sibling/earlier loops that don't actually enclose the call."""
    if idx < 0 or idx >= len(lines):
        return None
    bar = _indent(lines[idx])
    for j in range(idx - 1, -1, -1):
        ln = lines[j]
        if not ln.strip():
            continue
        ind = _indent(ln)
        if ind < bar:
            head = ln.lstrip()
            if re.match(r"(for|while|async\s+for)\b", head):
                return head.rstrip()
            if re.match(r"(def|async\s+def|class)\b", head):
                return None
            bar = ind  # if/with/try/elif/else — a loop may still enclose this block
    return None


def _enclosing_loop(lines: list[str], idx: int) -> bool:
    return _enclosing_loop_header(lines, idx) is not None


_RANGE_RE = re.compile(r"\bfor\b.*\bin\s+range\s*\(([^)]*)\)")


def _resolve_int_name(name: str, code: str):
    """Resolve an integer bound for `name` from a function default arg
    (name=6 / name: int = 6) or a plain assignment (name = 6) in `code`. Comments
    and strings are blanked first so a value in a docstring never counts. Ignores
    comparisons (==, >=). Returns None if not resolvable to a literal int."""
    stripped = _strip_noncode(code or "")
    m = re.search(rf"\b{re.escape(name)}\s*(?::[^,=()]+)?(?<![=!<>])=(?!=)\s*(\d+)\b",
                  stripped)
    return int(m.group(1)) if m else None


def _loop_range_bound(header: str, code: str):
    """Iteration count for a `for ... in range(...)` header, or None when it isn't
    a range loop or the bound can't be resolved to a literal int (while loops,
    for-over-iterable, range(1, n+1) expressions)."""
    m = _RANGE_RE.search(header or "")
    if not m:
        return None
    args = [a.strip() for a in m.group(1).split(",") if a.strip()]
    if not args or len(args) > 3:
        return None

    def resolve(tok: str):
        if re.fullmatch(r"\d+", tok):
            return int(tok)
        if re.fullmatch(r"[A-Za-z_]\w*", tok):
            return _resolve_int_name(tok, code)
        return None  # an expression like n+1 — don't evaluate

    if len(args) == 1:
        return resolve(args[0])
    start, stop = resolve(args[0]), resolve(args[1])
    if start is None or stop is None:
        return None
    n = stop - start
    return n if n >= 1 else None


# A call whose receiver is an agent/executor/crew object: `agent.run(...)`,
# `crew.kickoff(...)`, `executor.invoke(...)`. Used to link the call to the bound
# declared on that object's construction.
_RECEIVER_CALL = re.compile(
    r"\b(\w+)\s*\.\s*(?:kickoff|run|arun|invoke|ainvoke|execute|aexecute|stream|astream)\s*\(")
# The bound kwargs recognised on a `var = SomeCtor(...)` construction.
_CTOR_ASSIGN = re.compile(r"\b(\w+)\s*=\s*\w[\w.]*\s*\(")
_KWARG_BOUND = re.compile(
    r"\b(max_iter|max_iterations|max_turns|max_steps|recursion_limit|maxSteps|maxTurns)"
    r"\s*=\s*(\d+)")


def _strip_noncode(text: str) -> str:
    """Blank out string-literal contents and ``#``-comments so bound-regex matching
    only sees real code. Preserves length and newlines (positions map 1:1 to the
    original), so a match's offset still yields the correct line number.

    A ``max_iterations=1000`` inside a comment or a string must never be read as a
    real loop cap."""
    out: list[str] = []
    quote = None
    i, n = 0, len(text)
    while i < n:
        c = text[i]
        if c == "\n":            # strings/comments don't cross physical lines here
            quote = None
            out.append("\n")
            i += 1
            continue
        if quote:
            if c == "\\" and i + 1 < n and text[i + 1] != "\n":
                out.append("  ")
                i += 2
                continue
            out.append(" ")
            if c == quote:
                quote = None
            i += 1
            continue
        if c in ("'", '"'):
            quote = c
            out.append(" ")
            i += 1
            continue
        if c == "#":
            j = text.find("\n", i)
            if j == -1:
                j = n
            out.append(" " * (j - i))
            i = j
            continue
        out.append(c)
        i += 1
    return "".join(out)


def _call_receiver(line: str):
    m = _RECEIVER_CALL.search(line or "")
    return m.group(1) if m else None


def extract_construction_bounds(code: str) -> dict:
    """Map ``{receiver_var -> [(line, bound_name, N), ...]}`` for agent/crew/executor
    objects built with an explicit loop cap anywhere in the file.

    Each construction is recorded with its line number so a call can be linked to the
    NEAREST PRECEDING construction of its receiver (a reassigned var no longer has the
    last construction in the file attributed to every one of its calls). The
    constructor's argument list is scanned with balanced-paren awareness so a bound
    only counts when it belongs to the OUTER constructor — e.g.
    ``Crew(agent=Agent(tools=[]), max_iter=25)`` resolves to 25, and a nested
    ``Agent(max_iter=99)`` is not stolen. Comments/strings are blanked first."""
    code = code or ""
    stripped = _strip_noncode(code)
    out: dict[str, list] = {}
    for m in _CTOR_ASSIGN.finditer(stripped):
        var = m.group(1)
        open_paren = m.end() - 1  # index of the '('
        depth = 0
        j = open_paren
        L = len(stripped)
        while j < L:
            ch = stripped[j]
            if ch == "(":
                depth += 1
            elif ch == ")":
                depth -= 1
                if depth == 0:
                    break
            j += 1
        args = stripped[open_paren + 1:j]
        # Take a bound kwarg only at the top level of this constructor's arg list.
        for bm in _KWARG_BOUND.finditer(args):
            if args.count("(", 0, bm.start()) - args.count(")", 0, bm.start()) == 0:
                line = code.count("\n", 0, m.start()) + 1
                out.setdefault(var, []).append(
                    (line, bm.group(1), max(1, int(bm.group(2)))))
                break
    return out


def estimate_calls_per_task(code: str, call_line: int, end_line: int | None = None) -> dict:
    """`code` is a slice (often the whole file); `call_line` is 1-based within it.
    `end_line` (1-based, optional) is the last line of the call statement — for a
    multiline call the bound may sit on a continuation line, so the on-call-line
    bound is searched over the whole call span, not just the start line."""
    lines = (code or "").splitlines()
    idx = max(0, min(call_line - 1, len(lines) - 1)) if lines else 0
    if end_line is None:
        end_idx = idx
    else:
        end_idx = max(idx, min(int(end_line) - 1, len(lines) - 1)) if lines else idx
    call_ln = _strip_noncode(lines[idx]) if lines else _strip_noncode(code or "")
    call_span = _strip_noncode("\n".join(lines[idx:end_idx + 1])) if lines else call_ln
    # 1) bound declared ON the call statement, e.g. Runner.run(agent, input, max_turns=5)
    #    — searched across the full (comment/string-stripped) call span.
    m = _BOUND_RE.search(call_span)
    if m:
        n = _cap(max(1, int(m.group(2))))
        return {"calls_per_task": n, "basis": f"config:{m.group(1)}", "low": 1, "high": n}
    # 2) bound on the call receiver's CONSTRUCTION (receiver-matched, anywhere in file) —
    #    only attributed when the constructed var IS this call's receiver, using the
    #    nearest construction that PRECEDES this call line.
    recv = _call_receiver(call_ln)
    if recv:
        cands = [t for t in extract_construction_bounds(code or "").get(recv, [])
                 if t[0] <= call_line]
        if cands:
            _line, name, n = max(cands, key=lambda t: t[0])
            n = _cap(n)
            return {"calls_per_task": n, "basis": f"config:{name}@construction",
                    "low": 1, "high": n}
    # 3) is the call actually INSIDE an enclosing for/while (by scope)?
    header = _enclosing_loop_header(lines, idx) if lines else None
    if header is not None:
        # Read a real `range(N)` bound (literal / default-arg / assignment) if we
        # can; otherwise fall back to the conservative band.
        bound = _loop_range_bound(header, code or "")
        if bound is not None:
            bound = _cap(bound)
            return {"calls_per_task": bound, "basis": "loop:range", "low": 1, "high": bound}
        return {"calls_per_task": _LOOP_DEFAULT, "basis": "loop",
                "low": _LOOP_BAND[0], "high": _LOOP_BAND[1]}
    # 4) single call
    return {"calls_per_task": 1, "basis": "single", "low": 1, "high": 1}
