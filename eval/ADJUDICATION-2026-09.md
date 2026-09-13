# Adjudicated precision — agentic engine, planted-truth suite (Sept 2026)

**Headline: adjudicated precision = 21/24 = 0.875.** The raw proxy (0.375) was a
floor, not a measurement — 12 of the 15 "extras" are real, distinct, correctly-
hedged findings the ground truth simply didn't enumerate. The genuine failure
modes are small and specific: **1 speculative finding (4%)** and **2
double-counts (8%)** — both with concrete engine fixes below.

## Method (and the bias caveat)

The planted-truth suite scores recall exactly (we hid the issues, so we know the
answer key) but its precision proxy counts every un-enumerated finding as false.
This adjudication labels each of the 15 extras from the committed post-fix run
(`results-agentic.json`, raw audits included) against the actual repo source.

**Adjudicated by the suite's author — bias is structural.** Mitigations: the
criteria below are stated before the verdicts; every verdict carries its
reasoning; the raw audits are committed so anyone can re-adjudicate and disagree
line-by-line. Strictness rules: a finding only counts LEGIT if the pattern is
verifiably present in the file, the fix is safe-or-hedged, and it is not the same
waste as an already-matched planted finding re-billed under another name.

Labels: **LEGIT** (real, distinct, correctly hedged) · **DOUBLE-COUNT** (real
waste, but an alternative remedy for dollars already claimed by a matched
finding) · **SPECULATIVE** (assumes facts not in evidence).

## Verdicts — all 15 extras

| Repo | Finding | $/mo | Verdict | Reasoning |
|---|---|---|---|---|
| 02-trivial-classify | over_generation: no max_tokens on a one-word reply | 0.38 | **LEGIT** | Pattern present (no cap, system prompt demands 'spam'/'ham'); `max_tokens=5` is behavior-safe; distinct axis from the matched model finding. |
| 03-uncached-faq | caching: TTL-cache POLICY_DOC "if bloat isn't trimmed" | 89.39 | **DOUBLE-COUNT** | An *alternative remedy* for the same waste the matched caching/context finding already prices. Real pattern, redundant billing. |
| 03-uncached-faq | other: replace the LLM call with a template (KB is one fact) | 132.57 | **LEGIT** | Correct and the *best* recommendation on this repo: the knowledge base is literally one sentence ×400 — the call is eliminable. Distinct class (elimination), properly framed as a candidate. |
| 03-uncached-faq | model_mismatch: sonnet-4 for a bounded single-fact FAQ | 88.38 | **LEGIT** | Real pattern; independent axis (model choice vs caching); fix explicitly gated on shadow_verify — no blind swap. |
| 04-agent-loop | oversized_context: history grows unboundedly, full resend per turn | 78.00 | **LEGIT** | Verifiably true (`history.append` inside `while True`); distinct from the loop-cap issue (prompt growth per turn ≠ unbounded turns). |
| 04-agent-loop | other: no max_iterations bound on the loop | 75.00 | **DOUBLE-COUNT** | This *is* the planted unbounded-loop issue, re-emitted under 'other' after the matched retry_loop finding claimed it. |
| 04-agent-loop | model_mismatch: gpt-4o hardcoded for every loop turn | 252.00 | **LEGIT** | Real; recommendation explicitly marked UNVERIFIED / route turns 2-N; distinct axis. |
| 05-reembed | batching: sync per-doc embedding at startup vs Batch API | 7.61 | **LEGIT** | Loop issues one sync call per doc; Batch API (~50% cheaper, off critical path) is a correct, distinct fix from the re-embed dedup. |
| 05-reembed | model_mismatch: 3-large where 3-small is ~6.5× cheaper | 12.87 | **LEGIT** | Real hardcoded model; hedged rollout; distinct axis. |
| 06-multi-issue | over_generation: no max_tokens/stop on a one-line summary | 2.50 | **LEGIT** | Present; `max_tokens=30, stop=["\n"]` enforces the already-requested format. |
| 07-batch-miss | dedup: no dedup guard for repeated descriptions | 0.01 | **SPECULATIVE** | Assumes duplicate inputs not evidenced anywhere in the repo; immaterial dollars. The one fabrication-class miss. |
| 07-batch-miss | over_generation: uncapped tokens for a one-word output | 0.04 | **LEGIT** | Pattern present; fix safe; honest tiny dollars. |
| 08-eval-frontier | batching: nightly 10k-pair judge loop, sync instead of Batch API | 265.50 | **LEGIT** | The docstring states the nightly 10k volume; Batch API halves an offline job's cost; distinct from the matched model finding. |
| 08-eval-frontier | over_generation: no cap on a PASS/FAIL judge output | 153.00 | **LEGIT** | Present; `max_tokens=3` is exactly right for the task. |
| 09-double-call | model_mismatch: sonnet-4 for bounded summarization | 24.91 | **LEGIT** | Real; explicitly "do NOT ship a blind swap — shadow_verify first"; distinct from the matched duplicate-call finding. |

**Tally:** 12 LEGIT · 2 DOUBLE-COUNT · 1 SPECULATIVE.

## The numbers, honestly framed

- **Adjudicated precision: (9 matched + 12 legit) / 24 = 0.875**
- **Fabrication-class rate: 1/24 ≈ 4%** (a speculative $0.01 finding)
- **Double-count rate: 2/24 ≈ 8%** (real waste, redundantly billed)
- Recall (exact, planted): 9/9 · honest-nulls: 2/2 clean · invariants: 0 violations · stability: 1.00

## What the misses teach (engine work, both cheap)

1. **One finding per waste.** Both double-counts are *alternative remedies for
   already-claimed dollars* emitted as separate findings. Fix: an emit_finding
   instruction — "if a cheaper remedy exists for the SAME waste, put it in the
   fix field of the one finding; never emit two findings for one waste." Would
   have made this suite 23/24 ≈ 0.96.
2. **Evidence gate for input-distribution assumptions.** The speculative dedup
   finding assumed duplicate inputs with no evidence and priced it at $0.01.
   Fix: require observed evidence (or an explicit assumption flag + zero
   dollars) for findings that depend on input distributions.
3. **Vocabulary gap: `elimination`.** The best finding on 03 (replace the call
   with a template) had to land in `other`. Candidate ninth category.
