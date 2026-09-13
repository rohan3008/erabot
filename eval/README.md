# The planted-truth eval — check our homework

This is the evaluation suite behind the numbers we quote for the erabot audit
engine. Ground truth is exact by construction: we wrote ten tiny codebases and
deliberately planted known cost mistakes in them (wrong model for a trivial
task, an uncached 4k-token prompt, an unbounded agent loop, re-embedding on
every run, a duplicated call, a missed batch, a frontier model in an eval
script). Two of the ten are traps with nothing wrong — one has no LLM code, one
has LLM code that is already efficient — so any finding on those is the tool
inventing problems.

## What's here

- `repos/` — the ten planted codebases. The answer key (`GROUND_TRUTH` in
  `run_agentic_eval.py`) says exactly what was hidden where.
- `run_agentic_eval.py` — the grader: recall on planted issues (file overlap +
  category family), false-positive count, a savings<=spend invariant check, and
  a run-twice stability probe.
- `results-2026-09-04-prefix.json` / `-postfix.json` — two full runs including
  raw audits: before and after we fixed a category-naming bug the suite itself
  caught (stability 0.25 → 1.00).
- `ADJUDICATION-2026-09.md` — every "extra" finding judged against the source,
  verdict by verdict, including the ones that were our engine's fault.

## Headline results (post-fix run)

- Recall on planted issues: **9/9**
- Findings on the two clean repos: **0** (no invented problems)
- Savings-accounting violations: **0**
- Run-to-run category stability: **1.00** (was 0.25 before the fix — the ugly
  number is in the prefix file, kept on purpose)
- Adjudicated precision: **0.875** — the adjudication doc shows the receipts,
  including the 3 findings that did NOT survive scrutiny (2 double-counts, 1
  speculative), and the engine changes they triggered.

## Honesty notes

- The runner drives the commercial audit engine, so running it yourself
  requires an erabot audit environment; the *methodology, ground truth, planted
  repos, raw outputs, and adjudication* are all here to inspect and argue with.
- The adjudication was done by the suite's author. Mitigation: stated criteria,
  per-finding reasoning, and committed raw audits so anyone can re-adjudicate
  and disagree line by line. If you do, open an issue — that is the point.
