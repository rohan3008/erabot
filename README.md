# erabot

**Find out what your codebase spends on LLM API calls — locally, in seconds.**

`erabot` scans your source for LLM calls (OpenAI, Anthropic, Gemini, LangChain, LlamaIndex…), estimates the monthly cost, and shows you where the money goes. It runs **entirely on your machine** — no upload, no signup, no account.

```bash
pipx install erabot        # or: pip install erabot
erabot estimate .
```

```
  148 LLM call sites across 92 files
  Estimated $2,140/mo at 10,000 calls/mo per site (assumed — pass --calls-per-month for your real volume)
  ⚑ 81% of estimated spend runs on flagship models (prime downgrade candidates)

  Call site                    Model            Est. $/mo
  planner.py:88                gpt-4o ⚑            $412.00
  summarize.py:12              gpt-4o ⚑            $301.00
  classify.py:44               gpt-4o ⚑            $188.00
  ...

  Free local scan — nothing left your machine. For findings + apply-ready fixes,
  run the full audit at https://erabot.ai
```

## Get a real number, not a guess

Static analysis can't see how often each call fires, so the default assumes a volume. Set your real traffic:

```bash
erabot estimate . --calls-per-month 285000     # your actual monthly call volume
erabot estimate . --json                        # machine-readable output for CI
```

Where your prompts are literals in the code, erabot reads the real token counts. Where the prompt is built at runtime, the estimate is a **lower bound** — connect Helicone / Langfuse / OpenTelemetry for measured spend.

## What's free vs. what's not

| Free (this tool) | Full audit — [erabot.ai](https://erabot.ai) |
|---|---|
| Detect LLM call sites | Diagnosed findings + root cause |
| Estimate `$/mo`, model mix, flagship share | Apply-ready fixes + patches |
| 100% local, no signup | Shadow-verified % savings, enterprise dashboard, CI cost-gate |

Detection is open source (MIT). The agentic audit — which tells you *which* calls are safe to downgrade and writes the fix — is the paid product.

## What it detects

Python, TypeScript, and JavaScript. Direct SDK calls, framework idioms (LangChain chains, LlamaIndex query engines), and common wrapper patterns, via tree-sitter AST analysis.

---

MIT licensed. `erabot` never sends your code anywhere.
