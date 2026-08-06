"""Test/eval/example files are excluded from the cost scan by default. This pins
the hyphen-separator case (test-openapi-key.py) that an underscore-only prefix
check missed."""
from pathlib import Path

from erabot.cli import _is_test_path


def test_hyphenated_test_file_excluded():
    assert _is_test_path(Path("backend/test-openapi-key.py"))


def test_hyphenated_eval_and_benchmark_files_excluded():
    assert _is_test_path(Path("scripts/eval-runner.py"))
    assert _is_test_path(Path("perf/benchmark-suite.py"))


def test_underscore_prefixes_still_excluded():
    assert _is_test_path(Path("tests/test_scanner.py"))
    assert _is_test_path(Path("conftest.py"))
    assert _is_test_path(Path("foo_eval.py"))


def test_real_product_files_not_excluded():
    for p in ("planner.py", "latest_model.py", "contest_service.py",
              "greatest.py"):
        assert not _is_test_path(Path(p)), p
