"""turn_model should read a `for ... in range(N)` bound instead of always
falling back to the 4x loop band. N may be a literal, a function default arg,
or a module/local assignment. Unresolvable bounds (runtime iterables, while
loops, name+1 expressions) correctly keep the conservative band."""
from erabot._engine.turn_model import estimate_calls_per_task as est


def test_range_literal():
    code = "for _ in range(5):\n    r = agent.run(m)\n"
    d = est(code, 2)
    assert d["calls_per_task"] == 5 and d["basis"] == "loop:range"


def test_range_two_literals_is_span():
    code = "for i in range(2, 7):\n    r = agent.run(x)\n"
    assert est(code, 2)["calls_per_task"] == 5  # 7 - 2


def test_range_default_arg():
    code = ("def f(messages, max_turns=6):\n"
            "    for _ in range(max_turns):\n"
            "        r = llm.invoke(x)\n")
    assert est(code, 3)["calls_per_task"] == 6


def test_range_typed_default_arg():
    code = ("def f(n: int = 8):\n"
            "    for _ in range(n):\n"
            "        r = llm.invoke(x)\n")
    assert est(code, 3)["calls_per_task"] == 8


def test_range_module_assignment():
    code = ("N = 10\n"
            "for i in range(N):\n"
            "    r = model.generate_content(x)\n")
    assert est(code, 3)["calls_per_task"] == 10


def test_runtime_iterable_keeps_band():
    code = "for item in items:\n    r = model.invoke(item)\n"
    d = est(code, 2)
    assert d["calls_per_task"] == 4 and d["basis"] == "loop"


def test_while_loop_keeps_band():
    code = "while True:\n    r = model.invoke(x)\n"
    d = est(code, 2)
    assert d["calls_per_task"] == 4 and d["basis"] == "loop"


def test_unresolvable_range_name_keeps_band():
    code = "for i in range(k):\n    r = agent.run(x)\n"
    d = est(code, 2)
    assert d["calls_per_task"] == 4 and d["basis"] == "loop"


def test_range_name_plus_one_keeps_band():
    # range(1, n+1) — the stop is an expression we don't evaluate → band
    code = "for i in range(1, n + 1):\n    r = agent.run(x)\n"
    d = est(code, 2)
    assert d["calls_per_task"] == 4 and d["basis"] == "loop"


def test_explicit_bound_on_call_still_wins():
    code = "r = Runner.run(agent, data, max_turns=5)\n"
    assert est(code, 1)["calls_per_task"] == 5


def test_single_call_unchanged():
    assert est("r = llm.invoke(x)\n", 1)["calls_per_task"] == 1
