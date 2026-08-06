"""LangChain gives tools, search, retrievers, and models the same .invoke/.ainvoke
Runnable interface, and .complete collides with shell-completion (shtab.complete).
These pin the receiver-gate that keeps model calls and drops the rest. Also covers
files.create (OpenAI file upload, no inference)."""
from erabot._engine.scanner import detect_llm_calls


def _m(code):
    return [f["method_name"] for f in detect_llm_calls("x.py", code)]


# --- invoke / ainvoke on non-LLM Runnables ---
def test_tool_ainvoke_not_flagged():
    assert detect_llm_calls("x.py", "r = tool.ainvoke(args, config)\n") == []


def test_search_ainvoke_not_flagged():
    for recv in ("duckduckgo_search", "tavily_search", "web_search", "webSearch"):
        assert detect_llm_calls("x.py", f"r = {recv}.ainvoke(q)\n") == [], recv


def test_retriever_and_parser_invoke_not_flagged():
    assert detect_llm_calls("x.py", "r = retriever.invoke(q)\n") == []
    assert detect_llm_calls("x.py", "r = output_parser.invoke(text)\n") == []


def test_research_receiver_not_wrongly_denied():
    # 'research' CONTAINS 'search' as a substring but not as a name component —
    # component matching must keep this real call (no substring cardinal sin).
    assert len(detect_llm_calls("x.py", "r = research_chain.invoke(x)\n")) == 1


def test_llm_and_chain_invoke_still_flagged():
    assert _m("r = llm.invoke(m)\n") == ["invoke"]
    assert _m("r = chain.ainvoke(m)\n") == ["ainvoke"]


def test_llm_with_tools_receiver_kept():
    # 'llm_with_tools' has a 'tools' component but also 'llm' — LLM signal wins
    assert _m("r = await llm_with_tools.ainvoke(messages)\n") == ["ainvoke"]


def test_chained_ainvoke_receiver_unparseable_is_kept():
    # model.with_config(...).ainvoke(...) — receiver isn't a bare identifier, keep
    assert len(detect_llm_calls("x.py", "r = model.with_config(c).ainvoke(m)\n")) == 1


# --- complete / acomplete ---
def test_shtab_complete_not_flagged():
    assert detect_llm_calls("x.py", "print(shtab.complete(parser, shell=sh))\n") == []


def test_llm_complete_still_flagged():
    assert _m("r = llm.complete('hi')\n") == ["complete"]
    assert _m("r = self.model.acomplete('hi')\n") == ["acomplete"]


# --- files.create (upload, no inference) ---
def test_files_create_not_flagged():
    code = "f = client.files.create(file=x, purpose='fine-tune')\n"
    assert detect_llm_calls("x.py", code) == []


def test_chat_completions_create_still_flagged():
    code = "r = client.chat.completions.create(model='gpt-4o', messages=m)\n"
    assert _m(code) == ["create"]
