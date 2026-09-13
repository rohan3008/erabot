"""Agentic-engine grading suite — evals the auditor CLIENTS actually run.

Planted-truth methodology: each repo under repos/ plants known cost issues, so
ground truth is exact by construction (unlike inferring truth for real repos).

Per repo:
  recall    — fraction of planted issues the audit found (file overlap + category family)
  fp_count  — findings that map to no planted issue (on these tiny repos, extras
              are near-certainly false positives; the two null repos make any
              finding an FP by definition)
  invariant — identified savings <= reported spend (the $505-on-$213 class)
Aggregate + a variance probe: repo 06 runs twice; we report the Jaccard overlap
of its category sets (the known run-to-run stability concern).

Usage:  python tests/eval-agentic/run_agentic_eval.py [--max-turns 24] [--out results.json]
Runs the REAL agentic auditor (claude CLI auth) — costs real minutes + dollars;
agent cost is summed and reported.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from pathlib import Path

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parents[1] / "backend"))

from engine.agentic_auditor import run_agentic_audit  # noqa: E402

# Category families: a planted issue matches a finding if the finding's
# category OR title contains any family keyword (case-insensitive) AND the
# finding touches the planted file. Vocabulary from observed agentic output.
FAMILIES = {
    "model": ["model_mismatch", "model mismatch", "model_downgrade", "model downgrade",
              "wrong model", "frontier", "cheaper model", "overkill"],
    "caching": ["caching", "cache", "uncached", "prompt caching"],
    "context": ["oversized", "context", "prompt bloat", "duplicat", "repetit",
                "token waste", "compression"],
    "loop": ["retry-loop", "loop", "unbounded", "runaway", "uncapped", "turn cap"],
    "dedup": ["dedup", "re-embed", "reembed", "redundant", "repeated call",
              "identical call", "double"],
    "batching": ["batch", "sequential", "concurren", "parallel"],
    "elimination": ["eliminat", "template", "remove the call", "llm not needed"],
}

# repo -> list of planted issues (file, acceptable families). Empty list = a
# null repo where ANY finding is a false positive.
GROUND_TRUTH: dict[str, list[dict]] = {
    "01-clean-api": [],
    "02-trivial-classify": [{"file": "moderate.py", "families": ["model"]}],
    "03-uncached-faq": [{"file": "faq.py", "families": ["caching", "context", "elimination"]}],
    "04-agent-loop": [{"file": "agent.py", "families": ["loop"]}],
    "05-reembed": [{"file": "index.py", "families": ["dedup", "caching"]}],
    "06-multi-issue": [
        {"file": "summarize.py", "families": ["model"]},
        {"file": "summarize.py", "families": ["context", "caching"]},
    ],
    "07-batch-miss": [{"file": "tagger.py", "families": ["batching"]}],
    "08-eval-frontier": [{"file": "eval_answers.py", "families": ["model"]}],
    "09-double-call": [{"file": "handler.py", "families": ["dedup", "caching"]}],
    "10-efficient-llm": [],
}

VARIANCE_REPO = "06-multi-issue"  # audited twice; report category-set Jaccard


def _matches(finding: dict, planted: dict) -> bool:
    files = " ".join(str(f) for f in (finding.get("files") or []))
    if planted["file"] not in files:
        return False
    hay = ((finding.get("category") or "") + " " + (finding.get("title") or "")).lower()
    return any(kw in hay for fam in planted["families"] for kw in FAMILIES[fam])


def grade(repo_name: str, audit: dict) -> dict:
    planted = GROUND_TRUTH[repo_name]
    findings = audit.get("findings") or []
    hits, used = [], set()
    for p in planted:
        hit = next((i for i, f in enumerate(findings)
                    if i not in used and _matches(f, p)), None)
        if hit is not None:
            used.add(hit)
        hits.append(hit is not None)
    fp = len(findings) - len(used)
    identified = round(sum(float(f.get("monthly_savings_usd") or 0) for f in findings), 2)
    spend = float(audit.get("total_monthly_spend_usd") or 0)
    return {
        "repo": repo_name,
        "planted": len(planted), "found": sum(hits), "missed": len(planted) - sum(hits),
        "fp_count": fp, "findings_total": len(findings),
        "identified_usd": identified, "spend_usd": spend,
        "invariant_ok": identified <= spend or identified == 0,
        "stopped_reason": audit.get("stopped_reason"),
        "agent_cost_usd": audit.get("agent_cost_usd"),
        "categories": sorted({(f.get("category") or "?") for f in findings}),
    }


async def _audit_one(repo_dir: Path, max_turns: int, tag: str = "") -> tuple[str, dict]:
    name = repo_dir.name + tag
    print(f"[{time.strftime('%H:%M:%S')}] auditing {name} …", flush=True)
    audit = await run_agentic_audit(repo_path=str(repo_dir), repo_name=repo_dir.name,
                                    max_turns=max_turns)
    print(f"[{time.strftime('%H:%M:%S')}] done {name}: "
          f"{len(audit.get('findings') or [])} findings "
          f"({audit.get('stopped_reason')})", flush=True)
    return name, audit


async def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-turns", type=int, default=24)
    ap.add_argument("--out", default=str(_HERE / "results-agentic.json"))
    ap.add_argument("--concurrency", type=int, default=3)
    args = ap.parse_args()

    repos = sorted((_HERE / "repos").iterdir())
    jobs = [(d, "") for d in repos if d.is_dir()]
    jobs.append((_HERE / "repos" / VARIANCE_REPO, "#rerun"))

    sem = asyncio.Semaphore(args.concurrency)

    async def bounded(d, tag):
        async with sem:
            return await _audit_one(d, args.max_turns, tag)

    results = await asyncio.gather(*(bounded(d, t) for d, t in jobs),
                                   return_exceptions=True)

    audits: dict[str, dict] = {}
    for r in results:
        if isinstance(r, Exception):
            print(f"AUDIT ERROR (excluded from grading, counted below): {r}")
            continue
        audits[r[0]] = r[1]

    graded = [grade(n.split("#")[0], a) for n, a in audits.items() if "#" not in n]
    errored = len(jobs) - len(audits)

    tot_planted = sum(g["planted"] for g in graded)
    tot_found = sum(g["found"] for g in graded)
    tot_fp = sum(g["fp_count"] for g in graded)
    tot_findings = sum(g["findings_total"] for g in graded)
    nulls = [g for g in graded if g["planted"] == 0]
    agent_cost = round(sum(float(g.get("agent_cost_usd") or 0) for g in graded), 2)

    # variance probe
    variance = None
    a1, a2 = audits.get(VARIANCE_REPO), audits.get(VARIANCE_REPO + "#rerun")
    if a1 and a2:
        c1 = {(f.get("category") or "?") for f in (a1.get("findings") or [])}
        c2 = {(f.get("category") or "?") for f in (a2.get("findings") or [])}
        union = c1 | c2
        variance = {"run1": sorted(c1), "run2": sorted(c2),
                    "jaccard": round(len(c1 & c2) / len(union), 2) if union else 1.0}

    summary = {
        "repos_graded": len(graded), "audits_errored": errored,
        "recall": round(tot_found / tot_planted, 3) if tot_planted else None,
        "planted": tot_planted, "found": tot_found,
        "false_positives": tot_fp, "findings_total": tot_findings,
        "precision_proxy": round((tot_findings - tot_fp) / tot_findings, 3) if tot_findings else None,
        "null_repos_clean": sum(1 for g in nulls if g["findings_total"] == 0),
        "null_repos_total": len(nulls),
        "invariant_violations": sum(1 for g in graded if not g["invariant_ok"]),
        "category_stability_jaccard": (variance or {}).get("jaccard"),
        "total_agent_cost_usd": agent_cost,
    }
    # Raw audits included so extras can be adjudicated (titles/root-causes) —
    # the first run saved only grades and adjudication needed a re-run.
    out = {"summary": summary, "per_repo": graded, "variance": variance,
           "raw_audits": audits}
    Path(args.out).write_text(json.dumps(out, indent=2))

    print("\n===== AGENTIC ENGINE — GRADED =====")
    for k, v in summary.items():
        print(f"  {k}: {v}")
    print(f"results → {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
