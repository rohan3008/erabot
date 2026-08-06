"""`erabot estimate` — free, local LLM-cost estimate. No upload, no signup."""
from __future__ import annotations

import json as _json
import re
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.table import Table

from erabot._engine.scanner import scan_files_for_llm_calls
from erabot._engine.token_estimator import calculate_finding_cost, EstimationConfig

app = typer.Typer(add_completion=False, help="Free, local LLM-cost estimate for your codebase.")
console = Console()


@app.callback()
def _root() -> None:
    """erabot — free, local LLM-cost estimator. Nothing leaves your machine."""

_CODE_EXT = {".py", ".ts", ".tsx", ".js", ".jsx", ".mjs"}
_SKIP_DIRS = {"node_modules", ".git", ".venv", "venv", "env", "dist", "build",
              "__pycache__", "site-packages", ".next", ".tox", "vendor", ".mypy_cache"}
# Test / eval / example trees are excluded by default — otherwise a benchmark
# harness shows up as a company's "top cost site", which is misleading. Opt back
# in with --include-tests.
_TEST_DIRS = {"test", "tests", "testing", "example", "examples", "eval", "evals",
              "evaluation", "evaluations", "benchmark", "benchmarks", "docs", "doc",
              "sample", "samples", "demo", "demos", "e2e", "fixtures", "notebooks"}
_TEST_FILE_SUFFIXES = ("_test.py", "_eval.py", "_evaluate.py", "_evaluation.py",
                       "_benchmark.py", "_bench.py")


def _is_test_path(p: Path) -> bool:
    if any(part.lower() in _TEST_DIRS for part in p.parts):
        return True
    name = p.name.lower()
    return (name.startswith(("test_", "eval_", "benchmark_")) or name == "conftest.py"
            or name.endswith(_TEST_FILE_SUFFIXES))
# Flagship models: expensive tiers that are prime downgrade candidates. This
# share is volume-independent, so it's the credible headline even when absolute
# spend is only modeled.
_FLAGSHIP = ("gpt-4o", "gpt-4-", "gpt-4.5", "gpt-4-turbo", "o1", "o3",
             "claude-3-opus", "claude-opus", "gemini-1.5-pro", "gemini-2.5-pro")
# Cheap tiers that share a flagship prefix (gpt-4o-mini ⊃ "gpt-4o") but are NOT
# downgrade candidates — they ARE the downgrade. Matched as delimited TOKENS, not
# substrings, so "gemini" (⊃ "mini") isn't mistaken for a utility model. Checked
# first so the headline flagship-share never miscounts an already-cheap model.
_UTILITY_MARKERS = frozenset(
    ("mini", "nano", "flash", "lite", "haiku", "8b", "small", "instant"))


def _is_flagship(model: str) -> bool:
    m = (model or "").lower()
    if _UTILITY_MARKERS.intersection(re.split(r"[-_.\s]+", m)):
        return False
    return any(f in m for f in _FLAGSHIP)


def _collect_files(target: Path, include_tests: bool = False) -> tuple[list[dict], int]:
    """Returns (files, n_test_files_skipped)."""
    files: list[dict] = []
    skipped = 0
    if target.is_file():
        return [{"path": str(target), "content": target.read_text(errors="replace")}], 0
    for p in target.rglob("*"):
        if p.suffix.lower() not in _CODE_EXT:
            continue
        if any(part in _SKIP_DIRS for part in p.parts):
            continue
        if not include_tests and _is_test_path(p):
            skipped += 1
            continue
        try:
            files.append({"path": str(p), "content": p.read_text(errors="replace")})
        except (OSError, UnicodeError):
            continue
    return files, skipped


@app.command()
def estimate(
    path: str = typer.Argument(".", help="File or directory to scan."),
    calls_per_month: Optional[int] = typer.Option(
        None, "--calls-per-month", "-c",
        help="Invocations per month per call site. Set this to your real volume for a real number "
             "(default 10,000 is only an assumption)."),
    turns_per_task: Optional[int] = typer.Option(
        None, "--turns-per-task", "-t",
        help="Agent-loop turns per task — how many times a loop call site fires per user "
             "request. Agentic systems fire many calls per task; without this, an agent "
             "site's cost is a PER-TURN floor. Not visible from code — set it or connect traces."),
    completion_ratio: Optional[float] = typer.Option(
        None, "--completion-ratio", help="Completion tokens as a fraction of input tokens (default 0.5)."),
    as_json: bool = typer.Option(False, "--json", help="Emit machine-readable JSON instead of a table."),
    include_tests: bool = typer.Option(
        False, "--include-tests",
        help="Include test/eval/example/benchmark files (excluded by default — they'd "
             "otherwise show up as your top cost sites)."),
) -> None:
    """Scan a codebase for LLM API calls and estimate monthly spend — locally."""
    target = Path(path)
    if not target.exists():
        console.print(f"[red]Path not found:[/red] {path}")
        raise typer.Exit(2)

    files, skipped_tests = _collect_files(target, include_tests=include_tests)
    calls = scan_files_for_llm_calls(files)
    if not calls:
        console.print("[yellow]No LLM API calls detected.[/yellow] "
                      "(erabot scans Python/TS/JS for OpenAI, Anthropic, Gemini, LangChain, etc.)")
        raise typer.Exit(0)
    if skipped_tests and not as_json:
        console.print(f"[dim]Excluded {skipped_tests} test/eval/example file(s) so they don't "
                      f"skew the numbers — pass --include-tests to include them.[/dim]")

    # Is this an AGENTIC codebase? A LangGraph loop means each loop call site fires
    # MANY times per task, re-sending the harness each turn — so a flat per-call
    # estimate is the wrong model (cost = tasks × turns-per-run × per-call, and
    # turns-per-run isn't visible in code). Detect it and reframe honestly.
    from erabot._engine.orchestration_scan import scan_orchestration
    oflags = scan_orchestration(files)
    is_agentic = any(f["kind"] in ("unbounded_loop", "missing_iteration_cap") for f in oflags)

    # Collapse duplicate detections at the same call site (the scanner can emit
    # more than one pattern match for a single call) — one call site, one row.
    _seen, _deduped = set(), []
    for c in calls:
        k = (str(c.get("file_path", "")), c.get("line"))
        if k in _seen:
            continue
        _seen.add(k)
        _deduped.append(c)
    calls = _deduped

    # CREDIBILITY: attach file_content so the estimator can trace real prompt
    # tokens where the prompt is a literal (app code) instead of flooring to the
    # bare call-site snippet.
    content_by_path = {f["path"]: f["content"] for f in files}
    for c in calls:
        c.setdefault("file_content", content_by_path.get(str(c.get("file_path", "")), ""))

    explicit_volume = calls_per_month is not None
    config = EstimationConfig(
        calls_per_month=calls_per_month,
        completion_ratio=completion_ratio,
        calls_per_task=turns_per_task,
        _volume_explicit=explicit_volume,
        _completion_ratio_explicit=completion_ratio is not None,
        _turns_explicit=turns_per_task is not None,
    )
    resolved_calls = config.resolved_calls_per_month()
    turns = config.resolved_calls_per_task()

    # Per-site agent-loop turns, inferred statically (framework-agnostic): an
    # explicit bound (max_turns/max_iter/recursion_limit) wins; else a surrounding
    # for/while loop → ~4× (band 2–8); else 1. This makes the estimate account for
    # agentic loop cost instead of assuming one call per site. A global
    # --turns-per-task overrides it. See turn_model.
    from erabot._engine.turn_model import estimate_calls_per_task
    rows, total, flagship_total, traced_sites = [], 0.0, 0.0, 0
    loop_sites, inferred_turns = 0, []
    for c in calls:
        est = calculate_finding_cost(c, config)          # config turns=explicit or 1
        mc = float(est.get("monthly_cost_usd", 0) or 0)
        if config._turns_explicit:
            site_turns, basis = turns, "user"            # already applied by config
        else:
            tm = estimate_calls_per_task(c.get("file_content", ""), c.get("line") or 1, c.get("end_line"))
            site_turns = max(1, int(tm.get("calls_per_task", 1)))
            basis = tm.get("basis", "single")
            mc *= site_turns                             # apply inferred loop turns
        if site_turns > 1:
            loop_sites += 1
            inferred_turns.append(site_turns)
        model = est.get("model") or c.get("model") or ""
        if not model or model.lower() in ("unknown", "?"):
            model = "model unresolved"
        in_tok = int(est.get("input_tokens", 0) or 0)
        if in_tok > 60:  # heuristic: real prompt content was traced, not just the code snippet
            traced_sites += 1
        total += mc
        if _is_flagship(model):
            flagship_total += mc
        rows.append({"file": str(c.get("file_path", "")), "line": c.get("line"),
                     "provider": c.get("provider") or "?", "model": model,
                     "mo": mc, "traced": in_tok > 60, "turns": site_turns, "basis": basis})
    rows.sort(key=lambda r: r["mo"], reverse=True)
    fshare = (flagship_total / total * 100) if total else 0.0
    # Agentic if any call site loops (statically-inferred turns > 1) or a LangGraph
    # loop flag fired — framework-agnostic, not LangGraph-only.
    is_agentic = is_agentic or loop_sites > 0
    avg_turns = round(sum(inferred_turns) / len(inferred_turns), 1) if inferred_turns else 1
    if avg_turns == int(avg_turns):  # 4.0 → 4 for cleaner display
        avg_turns = int(avg_turns)

    if as_json:
        console.print_json(_json.dumps({
            "call_sites": len(calls), "files": len(files),
            "estimated_monthly_usd": round(total, 2), "calls_per_month_per_site": resolved_calls,
            "volume_assumed": not explicit_volume, "flagship_spend_pct": round(fshare, 1),
            "sites_with_traced_prompts": traced_sites,
            "agentic": is_agentic,
            "turns_per_task": turns if config._turns_explicit else avg_turns,
            "turns_assumed": not config._turns_explicit,
            "turns_inferred_statically": (not config._turns_explicit) and loop_sites > 0,
            "loop_call_sites": loop_sites, "orchestration_risks": len(oflags),
            "top": rows[:15],
        }))
        return

    # ---- honest, credible output ----
    vol_note = (f"at [bold]{resolved_calls:,}[/bold] calls/mo per site"
                + ("" if explicit_volume else " [dim](assumed — pass --calls-per-month for your real volume)[/dim]"))
    # Lead with the money and the flagship-share (both credible + repo-independent);
    # the raw site count is a rough static pass, so it comes last and labeled.
    if config._turns_explicit:
        turn_note = f" × [bold]{turns}[/bold] turns/task"
    elif loop_sites:
        turn_note = f" [dim](incl. ~{avg_turns}× inferred loop turns)[/dim]"
    else:
        turn_note = ""
    console.print(f"\n  Estimated [bold]${total:,.0f}/mo[/bold] on LLM calls {vol_note}{turn_note}")
    if is_agentic and not config._turns_explicit:
        console.print(
            "  [bold yellow]⚑ This looks agentic[/bold yellow] [dim]— it has agent loops, so those "
            f"call sites fire many times per task. We read the loops and estimated their turns "
            f"(~{avg_turns}× per task, usually 2–8), so the number above [bold]already includes[/bold] "
            "them. Static code can't show the exact turn count, though — pass [bold]--turns-per-task[/bold] "
            "for your real number, or run the full audit to measure it from traces.[/dim]"
        )
    if fshare >= 1:
        console.print(
            f"  [bold yellow]⚑ {fshare:.0f}% of that runs on flagship models[/bold yellow]"
            " [dim]— prime downgrade candidates (the full audit says which are safe to switch)[/dim]"
        )
    console.print(
        f"  [dim]{len(calls)} candidate call sites across {len(files)} files — a fast static pass; "
        f"the full audit's semantic layer refines this.[/dim]\n"
    )
    console.print("  [bold]Top cost sites[/bold]")

    table = Table(show_header=True, header_style="dim")
    table.add_column("Call site"); table.add_column("Model"); table.add_column("Est. $/mo", justify="right")
    for r in rows[:10]:
        flag = " ⚑" if _is_flagship(r["model"]) else ""
        table.add_row(f"{Path(r['file']).name}:{r['line']}", r["model"] + flag, f"${r['mo']:,.2f}")
    console.print(table)

    # Honest caveat where prompts aren't visible statically.
    untraced = len(rows) - traced_sites
    if untraced > 0:
        console.print(
            f"\n[dim]{untraced} of {len(rows)} call sites build their prompt at runtime, so this is a "
            f"lower bound — connect Helicone/Langfuse/OTel for measured spend.[/dim]"
        )
    # Orchestration risks in agent graphs (LangGraph), already scanned up top:
    # loops with no cap, dead branches. Candidate flags — the paid audit proves & fixes.
    if oflags:
        console.print(
            f"\n  ⚑ [bold]{len(oflags)}[/bold] orchestration risk(s) in your agent graph "
            f"(uncapped loops / missing caps / dead branches). The full audit proves & fixes them."
        )

    console.print(
        "\n[dim]Free local scan — nothing left your machine. For findings + apply-ready fixes,"
        " run the full audit at [/dim][bold]https://erabot.ai[/bold]\n"
    )


def main() -> None:
    app()


if __name__ == "__main__":
    main()
