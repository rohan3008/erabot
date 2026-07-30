"""TypeScript tree-sitter S-expression queries for LLM SDK call detection.

Detects calls to: OpenAI, Anthropic, Google, LangChain, and other LLM SDKs.
TypeScript uses different AST node types than Python:
  - call_expression (not call)
  - member_expression (not attribute)
  - property_identifier (not identifier for attribute names)
  - arguments (not argument_list)
"""

TYPESCRIPT_PATTERNS = """
; --- SDK Module Pattern ---
; Matches: openai.completions.create(), anthropic.messages.create()
(call_expression
  function: (member_expression
    object: (member_expression
      object: (identifier) @sdk_module
      property: (property_identifier) @sdk_namespace)
    property: (property_identifier) @method_name)
  (#match? @sdk_module "^(anthropic|openai|genai|cohere|google|mistral|groq|together|bedrock)$")
) @llm_call

; --- Chain Pattern ---
; Matches: *.completions.create(), *.messages.create()
(call_expression
  function: (member_expression
    object: (member_expression
      property: (property_identifier) @chain_part)
    property: (property_identifier) @method_name)
  (#match? @chain_part "^(completions|messages|chat|embeddings|models|images|audio|speech|files|responses|assistants|threads|runs|moderations|transcriptions|translations)$")
  (#match? @method_name "^(create|list|retrieve|delete|generate|cancel|stream|parse|edit)$")
) @llm_call

; --- Method Pattern ---
; Matches unique LLM method names: generateContent, invoke, complete, sendMessage
(call_expression
  function: (member_expression
    property: (property_identifier) @method_name)
  (#match? @method_name "^(generateContent|generateContentStream|invoke|complete|sendMessage|countTokens|embedContent|invokeModel|chatStream|streamText|generateText)$")
) @llm_call

; --- Bare Function Pattern (Vercel AI SDK) ---
; Matches: generateText({model, prompt}), streamText(...), generateObject(...)
; These names are AI-SDK-specific; generic embed/embedMany reverted (FP risk).
(call_expression
  function: (identifier) @method_name
  (#match? @method_name "^(generateText|streamText|generateObject|streamObject)$")
) @llm_call
"""
