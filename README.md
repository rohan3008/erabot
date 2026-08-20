# erabot

[![PyPI](https://img.shields.io/pypi/v/erabot)](https://pypi.org/project/erabot/)
[![Python](https://img.shields.io/pypi/pyversions/erabot)](https://pypi.org/project/erabot/)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue)](LICENSE)
[![Downloads](https://img.shields.io/pypi/dm/erabot)](https://pypi.org/project/erabot/)

**Find the cost risks in your AI agent's code — the runaway loops and missing caps that token counters miss. Locally, in seconds.**

Most "LLM cost" tools count tokens. `erabot` also reads your agent's **orchestration** — the LangGraph loops and branches where cost actually runs away — and flags the ones with no cap set. Plus the usual: every LLM call site (OpenAI, Anthropic, Gemini, LangChain, LlamaIndex…) and its estimated monthly cost. Runs **entirely on your machine** — no upload, no signup, no account.

```bash
pipx install erabot        # or: pip install erabot
erabot estimate .
```

_Requires Python ≥ 3.10 and a recent pip (`pip install --upgrade pip`)._

```
  148 LLM call sites across 92 files
  Estimated $2,140/mo at 10,000 calls/mo per site (assumed — pass --calls-per-month for your real volume)
  ⚑ 81% of estimated spend runs on flagship models (prime downgrade candidates)

  Call site                    Model            Est. $/mo
  planner.py:88                gpt-4o ⚑            $412.00
  summarize.py:12              gpt-4o ⚑            $301.00
  classify.py:44               gpt-4o ⚑            $188.00
  ...

  ⚑ 2 orchestration risk(s) in your agent graph (unbounded loops / missing caps).
    researcher.py:40  a conditional loop with no recursion_limit — relies on the
                      default cap (10007), so a runaway loop can cost up to that.

  Free local scan — nothing left your machine. For findings + apply-ready fixes,
  run the full audit at https://erabot.ai
```

## Why the loop risks matter

A LangGraph loop with no explicit `recursion_limit` falls back to the framework default (**10007** iterations). One badly-conditioned agent can quietly burn thousands of LLM calls before it stops. Generic token counters can't see this — it's a property of the *graph*, not the prompt. `erabot` flags:

- **uncapped loops** — a conditional / tool loop that relies on the default cap
- **missing iteration caps** — a looping graph invoked with no `recursion_limit`
- **dead branches** — nodes declared but never reachable

### Agent-loop cost (any framework)

Separately — and *not* limited to LangGraph — `erabot` detects when a call site sits inside an agent loop (a `while`/`for` loop, or a bounded `range(max_turns)`) and multiplies its cost by the inferred turns per task. A call that looks cheap once may run 4–8× per task. Where the code pins the cap (`max_turns=6`) erabot reads it; otherwise it uses a conservative band and says so. This works across raw SDK loops, CrewAI, and others — the estimate flags `⚑ This looks agentic` when it fires.

### These flags are measured, not vibes

Run on 5 real LangGraph repos (`langgraph`, `open_deep_research`, and 3 others — 97 graph-defining files): the loop flags fire only where a loop actually exists with no explicit cap — **100% precision on that corpus, and 0 false flags on graphs that already set a `recursion_limit`.**

And the call-site detection itself was measured on **9 real AI codebases** (including held-out repos it wasn't tuned on): **~97% precision**, with 100% recall on unambiguous SDK calls. Full method, numbers, and known false positives: [`docs/precision-eval.md`](docs/precision-eval.md).

## Get a real cost number, not a guess

Static analysis can't see how often each call fires, so the default assumes a volume. Set your real traffic:

```bash
erabot estimate . --calls-per-month 285000     # your actual monthly call volume
erabot estimate . --json                        # machine-readable output for CI
```

Where your prompts are literals in the code, erabot reads the real token counts. Where the prompt is built at runtime, the estimate is a **lower bound** — connect Helicone / Langfuse / OpenTelemetry for measured spend.

## Honest limits

- The `$/mo` figure is **modeled** on an assumed call volume until you pass `--calls-per-month`; treat it as a shape, not a bill.
- Orchestration flags are **candidates** — a cap may be set elsewhere than erabot can see statically. They tell you where to look, not that you're definitely wrong.
- The graph **risk flags** (uncapped loops / dead branches) currently cover **LangGraph** only; other graph frameworks are on the roadmap. (Agent-loop *cost* detection, above, is framework-agnostic.)

## What's free vs. what's not

| Free (this tool) | Full audit — [erabot.ai](https://erabot.ai) |
|---|---|
| Detect LLM call sites + estimate `$/mo`, model mix, flagship share | Diagnosed findings + root cause |
| Flag agent-loop cost risks (uncapped loops, missing caps) | **Prove** which calls are safe to downgrade / which loops are safe to cap |
| 100% local, no signup | Shadow-verified % savings, enterprise dashboard, CI cost-gate |

Detection is open source (MIT). The engine that *proves* a fix is safe — before you ship it — is the paid product.

## What it detects

Python, TypeScript, and JavaScript for call sites (direct SDK calls, LangChain chains, LlamaIndex query engines, common wrapper patterns), and LangGraph graph construction for the loop-risk flags — all via tree-sitter AST analysis.

---

MIT licensed. `erabot` never sends your code anywhere.
