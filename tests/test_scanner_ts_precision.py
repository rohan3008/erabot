"""TypeScript/JavaScript share the pattern set. The same non-inference cleanup as
Python must hold here: files.list / files.retrieve (management) dropped, countTokens
dropped, real inference kept. The invoke/complete receiver gate is language-agnostic
(runs in detect_llm_calls), so it applies to TS too."""
from erabot._engine.scanner import detect_llm_calls


def _m(code):
    return [f["method_name"] for f in detect_llm_calls("x.ts", code)]


def test_files_list_retrieve_not_flagged_ts():
    assert detect_llm_calls("x.ts", "const f = await client.files.list(query);\n") == []
    assert detect_llm_calls("x.ts", "const f = await client.files.retrieve(id);\n") == []


def test_count_tokens_not_flagged_ts():
    assert detect_llm_calls("x.ts", "const n = model.countTokens(msgs);\n") == []


def test_tool_invoke_not_flagged_ts():
    # language-agnostic receiver gate
    assert detect_llm_calls("x.ts", "const r = await tool.invoke(args);\n") == []


def test_chat_completions_create_still_flagged_ts():
    code = "const r = await this.client.chat.completions.create({ model: 'gpt-4o' });\n"
    assert _m(code) == ["create"]


def test_images_generate_still_flagged_ts():
    assert _m("const r = this.client.images.generate(fields);\n") == ["generate"]


def test_chat_model_invoke_still_flagged_ts():
    assert _m("const r = await chatModel.invoke(messages);\n") == ["invoke"]
