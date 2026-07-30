"""Python tree-sitter S-expression queries for LLM SDK call detection.

Detects calls to: OpenAI, Anthropic, Google (genai), Cohere, Mistral, Groq,
Together AI, AWS Bedrock, LangChain, LlamaIndex.
Uses structural AST queries instead of fragile regex to avoid false positives
on comments, strings, and non-LLM function calls.
"""

# Three complementary pattern groups:
#
# 1. SDK Module Pattern: Matches calls where a known SDK module name (openai, anthropic, etc.)
#    appears as the root object in a chained attribute access. Catches:
#    - openai.ChatCompletion.create(...)
#    - anthropic.Anthropic().messages.create(...)
#    - genai.GenerativeModel(...)
#
# 2. Chain Pattern: Matches calls where a known API namespace (completions, messages, etc.)
#    appears as an intermediate in the chain, with a known method at the end. Catches:
#    - client.chat.completions.create(...)
#    - client.messages.create(...)
#    - client.embeddings.create(...)
#    This handles cases where the variable name is user-chosen (e.g. `client`, `ai`, `llm`).
#
# 3. Method Pattern: Matches calls to uniquely-named LLM methods that are unlikely to appear
#    in non-LLM code. Catches:
#    - model.generate_content(...)
#    - chain.invoke(...)
#    - query_engine.query(...)

PYTHON_PATTERNS = """
; --- SDK Module Pattern ---
; Matches: openai.ChatCompletion.create(), anthropic.Anthropic(), genai.GenerativeModel()
(call
  function: (attribute
    object: (attribute
      object: (identifier) @sdk_module)
    attribute: (identifier) @method_name)
  (#match? @sdk_module "^(anthropic|openai|genai|google|cohere|mistralai|mistral|groq|together|bedrock|langchain|llama_index|llamaindex)$")
) @llm_call

; --- Chain Pattern ---
; Matches: *.completions.create(), *.messages.create(), *.embeddings.create()
(call
  function: (attribute
    object: (attribute
      attribute: (identifier) @chain_part)
    attribute: (identifier) @method_name)
  (#match? @chain_part "^(completions|messages|chat|embeddings|models|images|audio|speech|files|fine_tuning|batches|moderations|responses|assistants|threads|runs|transcriptions|translations)$")
  (#match? @method_name "^(create|acreate|list|retrieve|delete|generate|cancel|stream|parse|edit|edits|submit_tool_outputs)$")
) @llm_call

; --- Method Pattern ---
; Matches unique LLM method names: generate_content, embed_content, invoke,
; complete, query, invoke_model, chat_stream, completion (LiteLLM).
; Generic verbs like stream/parse are deliberately NOT here — they are only
; matched via the chain pattern above (gated by completions/messages/responses/...)
; so json.parse(), response.stream(), etc. are not flagged as LLM calls.
(call
  function: (attribute
    attribute: (identifier) @method_name)
; Additions are LLM-SDK-specific enough to preserve precision (measured 0.96):
; stream_chat/astream_chat (LlamaIndex chat engines), chat_completion (HF/router).
; Generic bare functions (completion/embed/moderation) were tested and REVERTED —
; they matched non-LLM same-named functions and dropped precision to 0.88.
  (#match? @method_name "^(generate_content|generate_content_async|embed_content|embed_content_async|count_tokens|invoke|ainvoke|complete|acomplete|completion|acompletion|query|aquery|invoke_model|invoke_model_with_response_stream|chat_stream|stream_chat|astream_chat|chat_completion)$")
) @llm_call
"""
