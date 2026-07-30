"""Vendored LLM pricing table (from backend/config.py). MIT.

Update by re-copying Settings.pricing when prices change."""
_PRICING = {
        # --- OpenAI (existing) ---
        "gpt-4-turbo": {"input": 10.0, "output": 30.0},
        "gpt-4-turbo-preview": {"input": 10.0, "output": 30.0},
        "gpt-4": {"input": 30.0, "output": 60.0},
        "gpt-4-32k": {"input": 60.0, "output": 120.0},
        "gpt-3.5-turbo": {"input": 0.5, "output": 1.5},
        "gpt-3.5-turbo-16k": {"input": 3.0, "output": 4.0},
        # --- OpenAI (new, with cache pricing) ---
        "gpt-4o": {"input": 2.50, "output": 10.0, "cache_read_input": 1.25},
        "gpt-4o-mini": {"input": 0.15, "output": 0.60, "cache_read_input": 0.075},
        "o1": {"input": 15.0, "output": 60.0},
        "o1-mini": {"input": 3.0, "output": 12.0},
        "o3-mini": {"input": 1.10, "output": 4.40},
        # --- Anthropic (existing, with cache pricing) ---
        "claude-3-opus": {"input": 15.0, "output": 75.0, "cache_read_input": 1.50, "cache_creation_input": 18.75},
        "claude-3-sonnet": {"input": 3.0, "output": 15.0, "cache_read_input": 0.30, "cache_creation_input": 3.75},
        "claude-3-haiku": {"input": 0.25, "output": 1.25, "cache_read_input": 0.025, "cache_creation_input": 0.3125},
        # --- Anthropic (new, with cache pricing) ---
        "claude-3.5-sonnet": {"input": 3.0, "output": 15.0, "cache_read_input": 0.30, "cache_creation_input": 3.75},
        "claude-3.5-haiku": {"input": 0.80, "output": 4.0, "cache_read_input": 0.08, "cache_creation_input": 1.0},
        # --- Embeddings (existing) ---
        "text-embedding-3-small": {"input": 0.02, "output": 0.0},
        "text-embedding-3-large": {"input": 0.13, "output": 0.0},
        "text-embedding-ada-002": {"input": 0.10, "output": 0.0},
        # --- Embeddings (new) ---
        "voyage-3": {"input": 0.06, "output": 0.0},
        "voyage-3-lite": {"input": 0.02, "output": 0.0},
        # --- Google (existing, with cache pricing for 1.5+) ---
        "gemini-1.5-flash": {"input": 0.1, "output": 0.4, "cache_read_input": 0.01},
        "gemini-1.5-pro": {"input": 3.5, "output": 10.5, "cache_read_input": 0.35},
        "gemini-2.0-flash": {"input": 0.1, "output": 0.4},
        # --- Google (new, with cache pricing for 2.5+) ---
        "gemini-2.0-flash-lite": {"input": 0.075, "output": 0.30},
        "gemini-2.5-pro": {"input": 1.25, "output": 10.0, "cache_read_input": 0.125},
        "gemini-2.5-flash": {"input": 0.15, "output": 0.60, "cache_read_input": 0.015},
        "gemini-2.5-flash-lite": {"input": 0.10, "output": 0.40, "cache_read_input": 0.01},
        # --- AWS Bedrock ---
        # Bedrock model IDs use "amazon." or "anthropic." prefixes in real API calls
        # but we use short keys; fuzzy match will catch "bedrock/" prefixed calls too.
        "bedrock/claude-3.5-sonnet": {"input": 3.0, "output": 15.0},
        "bedrock/claude-3-haiku": {"input": 0.25, "output": 1.25},
        "bedrock/claude-3-sonnet": {"input": 3.0, "output": 15.0},
        "bedrock/llama3-70b": {"input": 0.72, "output": 0.72},
        "bedrock/llama3-8b": {"input": 0.22, "output": 0.22},
        "amazon.titan-text-express": {"input": 0.20, "output": 0.60},
        "amazon.titan-text-lite": {"input": 0.15, "output": 0.20},
        # --- Azure OpenAI ---
        # Azure model names typically include "azure/" prefix or are same as OpenAI names
        # Adding azure/ prefixed variants for explicit matching
        "azure/gpt-4o": {"input": 2.50, "output": 10.0},
        "azure/gpt-4o-mini": {"input": 0.15, "output": 0.60},
        "azure/gpt-4-turbo": {"input": 10.0, "output": 30.0},
        "azure/gpt-4": {"input": 30.0, "output": 60.0},
        # --- Cohere ---
        "command-r-plus": {"input": 2.50, "output": 10.0},
        "command-r": {"input": 0.15, "output": 0.60},
        "command": {"input": 0.30, "output": 0.60},
        # --- Mistral ---
        "mistral-large": {"input": 2.00, "output": 6.00},
        "mistral-medium": {"input": 0.27, "output": 0.81},
        "mistral-small": {"input": 0.10, "output": 0.30},
        "mistral-7b": {"input": 0.025, "output": 0.025},
        "mixtral-8x7b": {"input": 0.07, "output": 0.07},
        "mixtral-8x22b": {"input": 0.90, "output": 0.90},
        # --- Groq ---
        # Groq model IDs used in API calls: "llama-3.1-70b-versatile", "llama-3.1-8b-instant"
        "llama-3.1-70b-versatile": {"input": 0.59, "output": 0.79},
        "llama-3.1-8b-instant": {"input": 0.05, "output": 0.08},
        "llama-3.3-70b-versatile": {"input": 0.59, "output": 0.79},
        "llama3-70b-8192": {"input": 0.59, "output": 0.79},
        "llama3-8b-8192": {"input": 0.05, "output": 0.08},
        "gemma2-9b-it": {"input": 0.20, "output": 0.20},
        # Groq new models (verified 2026-07-21, groq.com/pricing)
        "openai/gpt-oss-20b": {"input": 0.075, "output": 0.30},
        "openai/gpt-oss-120b": {"input": 0.15, "output": 0.60},
        "qwen3-32b": {"input": 0.60, "output": 3.00},
        # LiteLLM provider-prefixed aliases (model IDs seen in real code as
        # "groq/<model>"; the fuzzy matcher misses these against the short keys
        # above, so map them explicitly to Groq on-demand rates).
        "groq/llama-3.1-70b": {"input": 0.59, "output": 0.79},
        "groq/llama-3.1-70b-versatile": {"input": 0.59, "output": 0.79},
        "groq/llama-3.3-70b-versatile": {"input": 0.59, "output": 0.79},
        "groq/llama-3.1-8b-instant": {"input": 0.05, "output": 0.08},
        "groq/llama3-70b-8192": {"input": 0.59, "output": 0.79},
        "groq/llama3-8b-8192": {"input": 0.05, "output": 0.08},
        "groq/gpt-oss-20b": {"input": 0.075, "output": 0.30},
        "groq/gpt-oss-120b": {"input": 0.15, "output": 0.60},
        "groq/qwen3-32b": {"input": 0.60, "output": 3.00},
        "groq/gemma2-9b-it": {"input": 0.20, "output": 0.20},
        # --- Together AI ---
        # Together AI model IDs use "meta-llama/", "mistralai/", "togethercomputer/" prefixes
        "meta-llama/Meta-Llama-3.1-405B": {"input": 3.50, "output": 3.50},
        "meta-llama/Meta-Llama-3.1-70B": {"input": 0.88, "output": 0.88},
        "meta-llama/Meta-Llama-3.1-8B": {"input": 0.18, "output": 0.18},
        "mistralai/Mixtral-8x22B": {"input": 1.20, "output": 1.20},
        "togethercomputer/llama-2-70b-chat": {"input": 0.90, "output": 0.90},
        # --- DeepSeek ---
        "deepseek-chat": {"input": 0.14, "output": 0.28},
        "deepseek-coder": {"input": 0.14, "output": 0.28},
        "deepseek-reasoner": {"input": 0.55, "output": 2.19},
        # --- xAI Grok ---
        "grok-2": {"input": 2.00, "output": 10.00},
        "grok-2-mini": {"input": 0.10, "output": 0.10},
        "grok-3": {"input": 3.00, "output": 15.00},
        "grok-3-mini": {"input": 0.30, "output": 0.50},
        # --- Perplexity ---
        "sonar": {"input": 1.00, "output": 1.00},
        "sonar-pro": {"input": 3.00, "output": 15.00},
        "sonar-reasoning": {"input": 1.00, "output": 5.00},
        # --- OpenAI GPT-4.1 family (released 2025) ---
        "gpt-4.1": {"input": 2.0, "output": 8.0},
        "gpt-4.1-mini": {"input": 0.40, "output": 1.60},
        "gpt-4.1-nano": {"input": 0.10, "output": 0.40},
        # --- OpenAI GPT-5 family ---
        "gpt-5": {"input": 2.0, "output": 8.0},  # estimated, may need updating
        "gpt-5-nano": {"input": 0.10, "output": 0.40},  # estimated
        # --- Anthropic Claude 4 family ---
        "claude-sonnet-4": {"input": 3.0, "output": 15.0},
        "claude-haiku-4-5": {"input": 1.0, "output": 5.0},
        "claude-opus-4": {"input": 15.0, "output": 75.0},
        # --- DeepSeek (versioned) ---
        "deepseek-v3": {"input": 0.27, "output": 1.10},
        "deepseek-r1": {"input": 0.55, "output": 2.19},
        # --- Nvidia ---
        "nemotron": {"input": 0.20, "output": 0.20},  # estimated for API access
        # --- Unknown model fallback (median across all models) ---
        "unknown": {"input": 1.0, "output": 3.0},
    }

PRICING_LAST_VERIFIED = "2026-07-01"
PRICING_STALENESS_WARN_DAYS = 90
PRICING_ESTIMATED_MODELS = ()

class _Settings:
    pricing = _PRICING
    context_windows = {}

def get_settings():
    return _Settings()

def price_confidence(model: str) -> str:
    return "official"

def pricing_staleness_days() -> int:
    return 0
