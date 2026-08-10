"""The flagship-model-share is only shown as reliable when a meaningful fraction
of call sites have an EXPLICIT model literal. Real code sets the model at runtime,
so the metric must carry its own confidence (model_resolution_rate) rather than
report a split computed from a tiny, biased subset."""
import json
import pathlib
import subprocess
import sys

_REPO = pathlib.Path(__file__).resolve().parents[1]  # so `-m erabot.cli` resolves


def _scan(tmp_path):
    # stdout carries clean JSON; the "excluded N files" note goes to stderr.
    # --include-tests: pytest's tmp dir is named after the test fn (starts with
    # "test"), which _is_test_path would otherwise exclude, scanning zero files.
    r = subprocess.run(
        [sys.executable, "-m", "erabot.cli", "estimate", str(tmp_path),
         "--json", "--include-tests"],
        capture_output=True, text=True, cwd=str(_REPO),
    )
    assert r.returncode == 0 and r.stdout.strip(), f"rc={r.returncode} err={r.stderr}"
    return json.loads(r.stdout)


def test_reliable_when_models_are_explicit(tmp_path):
    (tmp_path / "a.py").write_text(
        "c = OpenAI()\n"
        'c.chat.completions.create(model="gpt-4o", messages=m)\n'
        'c.chat.completions.create(model="gpt-4o", messages=m)\n'
        'c.chat.completions.create(model="gpt-4o-mini", messages=m)\n'
    )
    d = _scan(tmp_path)
    assert d["model_resolution_rate"] == 1.0
    assert d["flagship_share_reliable"] is True
    assert d["flagship_spend_pct"] > 0


def test_unreliable_when_models_are_runtime(tmp_path):
    (tmp_path / "b.py").write_text(
        "def run(llm):\n"
        "    llm.invoke(messages)\n"
        "    llm.invoke(messages)\n"
        "    llm.invoke(messages)\n"
    )
    d = _scan(tmp_path)
    assert d["resolved_model_sites"] == 0
    assert d["model_resolution_rate"] == 0.0
    assert d["flagship_share_reliable"] is False
