"""`.query()` is not unique to LLM SDKs (Qdrant, Pinecone, pandas, SQLAlchemy all
have it), so a bare `X.query(...)` over-fires. These tests pin the precision fix:
data-store receivers are dropped; a LlamaIndex query engine is still detected.
"""
from erabot._engine.scanner import detect_llm_calls


def _methods(code):
    return [f["method_name"] for f in detect_llm_calls("x.py", code)]


def test_vector_db_query_not_flagged():
    # the exact false positive seen in crewAI's qdrant storage
    code = "results = shard.query(QueryRequest(vector=v, top_k=5))\n"
    assert detect_llm_calls("x.py", code) == []


def test_common_datastore_receivers_query_not_flagged():
    # includes COMPOUND names (db_session, sf_client, sync_db) an exact-match
    # blocklist would miss — these are the real receivers seen in Onyx.
    for recv in ("db", "database", "client", "collection", "cursor",
                 "session", "db_session", "sf_client", "sync_db",
                 "probe_session", "table", "store", "qdrant", "pinecone",
                 "weaviate", "chroma", "milvus", "redis", "mongo", "df"):
        code = f"r = {recv}.query(something)\n"
        assert detect_llm_calls("x.py", code) == [], f"{recv}.query should not flag"


def test_async_datastore_query_not_flagged():
    assert detect_llm_calls("x.py", "r = client.aquery(req)\n") == []


def test_llamaindex_query_engine_still_flagged():
    code = 'resp = query_engine.query("What is the revenue?")\n'
    findings = detect_llm_calls("x.py", code)
    assert len(findings) == 1
    assert findings[0]["method_name"] == "query"


def test_llamaindex_query_engine_attribute_receiver_still_flagged():
    # self.query_engine.query(...) — receiver is an attribute chain
    code = 'resp = self.query_engine.query("hi")\n'
    findings = detect_llm_calls("x.py", code)
    assert len(findings) == 1


def test_llamaindex_chained_as_query_engine_still_flagged():
    # index.as_query_engine().query("...") — receiver is a chained call
    code = 'resp = index.as_query_engine().query("hi")\n'
    findings = detect_llm_calls("x.py", code)
    assert len(findings) == 1


def test_chat_engine_query_still_flagged():
    code = 'resp = chat_engine.query("hi")\n'
    assert len(detect_llm_calls("x.py", code)) == 1


def test_other_llm_methods_unaffected():
    # the fix must only touch query/aquery, not the rest of the method pattern
    code = "resp = model.generate_content('hi')\nout = chain.invoke(x)\n"
    assert set(_methods(code)) == {"generate_content", "invoke"}
