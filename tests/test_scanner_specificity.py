"""Specificity: non-AI code that shares method names must not be flagged.
Surfaced by scanning pandas (0 FPs), sqlalchemy (post_load.invoke), and pipecat
(genai.types.*Config constructors, handler.invoke)."""
from erabot._engine.scanner import detect_llm_calls


def _m(code):
    return [f["method_name"] for f in detect_llm_calls("x.py", code)]


# --- FP class 1: SDK-namespace config/type constructors (PascalCase methods) ---
def test_genai_config_constructors_not_flagged():
    for code in (
        "config = genai.types.GenerateContentConfig(temperature=0.7)\n",
        "vc = genai.types.PrebuiltVoiceConfig(voice_name='Kore')\n",
        "sc = genai.types.SpeechConfig(voice_config=vc)\n",
        "m = genai.GenerativeModel('gemini-1.5-pro')\n",   # client/model ctor, not a call
    ):
        assert detect_llm_calls("x.py", code) == [], code


def test_real_genai_call_still_flagged():
    assert _m("r = model.generate_content('hi')\n") == ["generate_content"]


# --- FP class 2: generic .invoke on event/callback handlers ---
def test_handler_invoke_not_flagged():
    assert detect_llm_calls("x.py", "r = await handler.invoke(args, self)\n") == []
    assert detect_llm_calls("x.py", "r = dispatcher.invoke(evt)\n") == []
    # NOTE: post_load.invoke (SQLAlchemy events) tokenizes to {post, load} and is
    # NOT caught — an accepted rare long-tail miss; not worth broadening the deny set.


# --- real SDK-namespace calls (lowercase methods) still flagged ---
def test_chain_and_client_calls_still_flagged():
    assert _m("r = client.chat.completions.create(model='gpt-4o')\n") == ["create"]
    assert _m("r = client.audio.transcriptions.create(file=f)\n") == ["create"]
