"""Not every SDK method is a billable inference call. count_tokens is local token
counting; messages.list / runs.retrieve / files.delete / batches.cancel are
management/retrieval. None incur completion cost, so a cost scan must not flag them.
Surfaced by the held-out eval (AutoGen count_tokens, Azure messages.list)."""
from erabot._engine.scanner import detect_llm_calls


def _m(code):
    return [f["method_name"] for f in detect_llm_calls("x.py", code)]


def test_count_tokens_not_flagged():
    assert detect_llm_calls("x.py", "n = self._model_client.count_tokens(messages)\n") == []
    assert detect_llm_calls("x.py", "n = llm.count_tokens(msgs)\n") == []


def test_management_chain_methods_not_flagged():
    for code in (
        "async for m in client.beta.threads.messages.list(tid):\n    pass\n",
        "r = client.beta.threads.runs.retrieve(tid, rid)\n",
        "client.files.delete(fid)\n",
        "client.batches.cancel(bid)\n",
    ):
        assert detect_llm_calls("x.py", code) == [], code


def test_inference_chain_methods_still_flagged():
    assert _m("r = client.chat.completions.create(model='gpt-4o', messages=m)\n") == ["create"]
    assert _m("r = client.messages.create(model='claude-3-5-sonnet', messages=m)\n") == ["create"]
    assert _m("r = client.images.generate(prompt='a cat')\n") == ["generate"]


def test_genai_generate_content_still_flagged():
    assert _m("r = model.generate_content('hi')\n") == ["generate_content"]


def test_assistants_api_setup_not_flagged():
    # thread/assistant creation is setup, not inference
    assert detect_llm_calls("x.py", "t = client.beta.threads.create()\n") == []
    assert detect_llm_calls("x.py", "a = client.beta.assistants.create(model='gpt-4o')\n") == []


def test_assistants_api_run_still_flagged():
    # runs.create starts the model — that IS inference
    assert _m("r = client.beta.threads.runs.create(thread_id=t, assistant_id=a)\n") == ["create"]
