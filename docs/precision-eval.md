# erabot — detection accuracy (pre-launch eval)

_2026-08-06. Honest accuracy numbers for the free `erabot estimate` scanner,
measured on real open-source AI codebases with the exact functions the tool
ships. Purpose: publish what works and what doesn't before open-sourcing._

## What was measured

**Call-site precision** — of the LLM call sites erabot reports, how many are
genuinely billable model calls (vs. a database query, a tool call, a file
upload, etc.)? Precision is what kills adoption if it's low: a tool that counts
your `db.query()` calls as LLM spend loses trust immediately.

## Method

Two corpora, nine real repos:

- **Tuning set (5):** onyx, gpt-researcher, crewAI, aider, open_deep_research
- **Held-out set (4):** autogen, haystack, llama_index, langchainjs — used to
  check the fixes generalize rather than overfit the tuning set.

For each, a random seeded sample of detected sites was hand-classified against
the real source line as a real model call or a false positive. Test/example
files are excluded (as the tool does by default).

## Headline result

| Corpus | Call-site precision |
|---|---|
| Held-out (unseen repos) | **~97%** (autogen 10/10, haystack 10/10, llama_index 10/10 on the final sample) |
| Recall on unambiguous SDK calls | **100%** (20/20 `chat.completions.create` / `messages.create` / `litellm.completion` / `generate_content`) |

Getting here required fixing several real false-positive classes the raw
patterns produced — measured, then removed:

| False positive | Why it isn't an LLM call | Fix |
|---|---|---|
| `db_session.query(...)`, `shard.query(...)` | SQLAlchemy / vector-store / pandas queries | keep `.query()` only for a LlamaIndex query **engine** |
| `tool.ainvoke(...)`, `tavily_search.ainvoke(...)` | LangChain gives tools/search the same Runnable interface as models | deny tool/search/retriever receivers (by name component; an `llm` component wins) |
| `shtab.complete(...)` | shell tab-completion | keep `.complete()` only for LLM-ish receivers |
| `client.files.create/list/retrieve` | file management, no inference | drop non-inference methods + namespaces |
| `model.count_tokens(...)` | local token counting | removed from the pattern |
| Text after an em-dash/emoji | byte-offset vs. char-offset slicing corrupted call text | slice `node.text`, not `content[byte:byte]` |
| `chrome.runtime.sendMessage(...)` | browser-extension messaging, not Gemini `chat.sendMessage()` | keep `sendMessage` only for LLM-ish receivers |

Across the corpus these false positives were **~68% of all raw detections** —
so the cleanup roughly tripled precision.

## Multi-language

TypeScript/JavaScript detection works end-to-end (langchainjs: 84 sites across
~1,020 files, no crashes) and shares the same receiver gates as Python.

## Honest limits

- **Precision is measured, and it varies by codebase — it is not a single
  universal number.** ~97% held on our corpus, but a scan of Onyx (which ships a
  browser extension) initially hit ~77% because `chrome.runtime.sendMessage`
  collided with the Gemini `chat.sendMessage()` pattern — a false-positive class
  the corpus didn't contain. We gated it and Onyx recovered to ~95%. New real
  codebases can surface new patterns; the fix-and-remeasure loop is how they're
  closed. The known residuals are also framework *internals* (e.g. LangChain's
  own `Runnable.invoke` plumbing) that only appear when scanning a framework's
  source, not a normal app.
- **Recall is measured only on unambiguous direct-SDK calls** (100% there).
  Calls hidden behind custom wrappers or dynamic dispatch aren't counted here —
  those are harder and are where the paid engine's LLM backstop helps.
- **`$/mo` is modeled** on an assumed call volume until you pass
  `--calls-per-month`; treat it as a shape, not a bill.
- **Agent-loop turns are inferred statically** (a `range(N)` bound where N
  resolves, else a conservative band) — real turn counts need traces.
- These are open-source *framework/library* repos; a real app's top cost sites
  will be its own code, not a framework's plumbing.

## Reproduce

The detectors are the shipped `erabot._engine` functions. Tests pinning every
fix above live in `tests/` (`test_scanner_query_precision.py`,
`test_scanner_invoke_precision.py`, `test_scanner_noninference.py`,
`test_scanner_encoding.py`, `test_scanner_ts_precision.py`).
