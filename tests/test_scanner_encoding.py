"""tree-sitter reports byte offsets; the source is a str. Slicing content[byte:byte]
corrupts call text once any non-ASCII char (em-dash, smart quote, accent, emoji)
precedes a call site — which broke receiver gates and model extraction. These pin
the byte-correct slice."""
from erabot._engine.scanner import detect_llm_calls

# an em-dash (3 bytes) in a docstring shifts every later byte offset by +2
_PREFIX = '"""Ada Lovelace — first programmer. Café résumé façade."""\n'


def test_tool_invoke_dropped_despite_leading_nonascii():
    code = _PREFIX + "result = tool.invoke(**args)\n"
    assert detect_llm_calls("x.py", code) == []


def test_model_extracted_despite_leading_nonascii():
    code = _PREFIX + "r = client.chat.completions.create(model='gpt-4o', messages=m)\n"
    findings = detect_llm_calls("x.py", code)
    assert len(findings) == 1
    assert findings[0].get("model") == "gpt-4o"


def test_query_engine_kept_despite_leading_nonascii():
    code = _PREFIX + "resp = query_engine.query('hi')\n"
    assert len(detect_llm_calls("x.py", code)) == 1


def test_datastore_query_dropped_despite_leading_nonascii():
    code = _PREFIX + "rows = db_session.query(User).all()\n"
    assert detect_llm_calls("x.py", code) == []


def test_multiple_nonascii_before_call():
    # several multibyte chars — 🤖 is 4 bytes; slice must still land right
    code = '"""🤖 agent — café."""\nout = search_tool.ainvoke(q)\n'
    assert detect_llm_calls("x.py", code) == []
