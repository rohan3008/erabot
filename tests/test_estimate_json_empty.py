"""`erabot estimate --json` must emit JSON even when no LLM calls are found.

The README recommends --json for CI. A clean repo used to print the plain-text
"No LLM API calls detected." line instead, so any script parsing the output
failed on precisely the result it was hoping for.
"""
import json

from typer.testing import CliRunner

from erabot.cli import app

runner = CliRunner()


def _run(args):
    result = runner.invoke(app, args)
    assert result.exit_code == 0, result.output
    return result.output


def test_json_on_a_repo_with_no_llm_calls_is_valid_json(tmp_path):
    (tmp_path / "util.py").write_text("def add(a, b):\n    return a + b\n")
    payload = json.loads(_run(["estimate", str(tmp_path), "--json"]))
    assert payload["call_sites"] == 0
    assert payload["estimated_monthly_usd"] == 0
    assert payload["top"] == []
    # "files" counts what the collector kept as LLM candidates, not every file.
    assert isinstance(payload["files"], int)


def test_empty_json_has_the_same_keys_as_a_populated_result(tmp_path):
    empty = tmp_path / "empty"
    empty.mkdir()
    (empty / "util.py").write_text("x = 1\n")
    full = tmp_path / "full"
    full.mkdir()
    (full / "app.py").write_text(
        "from openai import OpenAI\n"
        "client = OpenAI()\n"
        "r = client.chat.completions.create(model='gpt-4o', messages=[])\n"
    )
    empty_keys = set(json.loads(_run(["estimate", str(empty), "--json"])))
    full_keys = set(json.loads(_run(["estimate", str(full), "--json"])))
    assert empty_keys == full_keys


def test_plain_output_on_no_calls_is_unchanged(tmp_path):
    (tmp_path / "util.py").write_text("x = 1\n")
    assert "No LLM API calls detected" in _run(["estimate", str(tmp_path)])
