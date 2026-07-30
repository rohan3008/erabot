"""Token estimation module for static code cost analysis.

Given scanner findings (detected LLM API calls in source code), estimates:
  - Token counts per call site using provider-appropriate tokenizers
  - Monthly cost projections based on configurable usage assumptions
  - Per-file, per-provider, and per-model cost breakdowns

This handles the "estimate cost from code" path, distinct from the
proxy-based CostCalculator which counts actual runtime tokens.

Usage assumptions (overridable via EstimationConfig):
  - DEFAULT_CALLS_PER_MONTH: assumed calls/month per call site
  - COMPLETION_RATIO: completion tokens = input_tokens * ratio
"""

import re
import logging
from dataclasses import dataclass, field
from typing import Optional

import tiktoken

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class EstimationConfig:
    """Configuration for usage assumption overrides and custom pricing."""

    # How many times per month each detected call site is invoked.
    # None means the caller did not supply a value; 1000 will be used but
    # volume_assumed=True will be stamped on each finding so consumers can
    # display appropriate uncertainty caveats.
    calls_per_month: Optional[int] = None

    # Estimated completion tokens as a fraction of input tokens.
    # None means the caller did not supply a value; 0.5 will be used but
    # completion_ratio_assumed=True will be stamped on each finding so consumers
    # can display appropriate uncertainty caveats.
    completion_ratio: Optional[float] = None

    # Custom pricing overrides: model -> {"input": float, "output": float} per 1M tokens
    custom_pricing: dict = field(default_factory=dict)

    # Internal flag — set by EstimationConfig(calls_per_month=N, _volume_explicit=True)
    # when the caller explicitly provided a volume. Do not set via JSON/dict construction.
    _volume_explicit: bool = False

    # Internal flag — set by EstimationConfig(completion_ratio=R, _completion_ratio_explicit=True)
    # when the caller explicitly provided a completion ratio. Do not set via JSON/dict construction.
    _completion_ratio_explicit: bool = False

    # How many times this call site fires per task (agent loop turns). None -> 1 used,
    # turns_assumed=True stamped. Multiplies calls_per_month (tasks/mo) x calls_per_task.
    calls_per_task: Optional[int] = None
    _turns_explicit: bool = False

    def resolved_calls_per_month(self) -> int:
        """Return the effective calls-per-month value (10000 if none supplied)."""
        return self.calls_per_month if self.calls_per_month is not None else 10000

    def volume_assumed(self) -> bool:
        """True when calls_per_month was not explicitly provided by the caller."""
        return self.calls_per_month is None and not self._volume_explicit

    def resolved_completion_ratio(self) -> float:
        """Return the effective completion ratio (0.5 if none supplied)."""
        return self.completion_ratio if self.completion_ratio is not None else 0.5

    def completion_ratio_assumed(self) -> bool:
        """True when completion_ratio was not explicitly provided by the caller."""
        return self.completion_ratio is None and not self._completion_ratio_explicit

    def resolved_calls_per_task(self) -> int:
        # Clamp to >= 1: a nonsensical explicit 0/negative would otherwise zero or
        # negate every dollar. Unset -> 1; explicit 4 -> 4.
        return max(1, self.calls_per_task) if self.calls_per_task is not None else 1

    def turns_assumed(self) -> bool:
        # Turns are "assumed" unless the caller EXPLICITLY provided them (measured or
        # user-set). A statically-inferred value (turn_model) still sets calls_per_task
        # but is an assumption, so gate only on the explicit flag.
        return not self._turns_explicit


# ---------------------------------------------------------------------------
# Provider / model detection helpers
# ---------------------------------------------------------------------------

# Regex to extract model string from code.
# Matches Python kwargs (model="gpt-4o"), TypeScript object properties (model: "gpt-4"),
# and LangChain constructor args (model_name="gpt-4").
# Supports both = and : assignment operators to cover Python and JS/TS syntax.
_MODEL_RE = re.compile(
    r"""(?:model|engine|model_name)\s*[:=]\s*["']([a-zA-Z0-9._\-:/ ]+)["']""",
    re.IGNORECASE,
)

# Common model aliases used in code (shorthand → canonical)
_MODEL_ALIASES: dict[str, str] = {
    "gpt4": "gpt-4",
    "gpt4o": "gpt-4o",
    "gpt-4-turbo-preview": "gpt-4-turbo",
    "claude-3": "claude-3-sonnet",
    "claude3": "claude-3-sonnet",
    "claude": "claude-3-sonnet",
    "gemini": "gemini-2.0-flash",
    "gemini-pro": "gemini-1.5-pro",
    "gemini-flash": "gemini-2.0-flash",
}

# Provider keyword → canonical provider name
_PROVIDER_MAP: dict[str, str] = {
    "openai": "openai",
    "anthropic": "anthropic",
    "google": "google",
    "genai": "google",
    "generativemodel": "google",
    "langchain": "langchain",
    "llamaindex": "llamaindex",
    "llama_index": "llamaindex",
    "cohere": "cohere",
    "mistral": "mistral",
}

# Model prefix → provider
_MODEL_PROVIDER_PREFIX: list[tuple[str, str]] = [
    ("gpt-", "openai"),
    ("o1", "openai"),
    ("o3", "openai"),
    ("text-embedding", "openai"),
    ("claude-", "anthropic"),
    ("gemini-", "google"),
    ("command-", "cohere"),
    ("mistral-", "mistral"),
    ("voyage-", "anthropic"),  # Voyage is Anthropic-affiliated
]

# tiktoken encoding name by provider / model prefix
_ENCODING_MAP: dict[str, str] = {
    # OpenAI uses cl100k_base for GPT-4 and o-series; o200k_base for GPT-4o+
    "gpt-4o": "o200k_base",
    "gpt-4o-mini": "o200k_base",
    "o1": "o200k_base",
    "o1-mini": "o200k_base",
    "o3-mini": "o200k_base",
    # Fallback for everything else.
    # NOTE: cl100k_base is exact for OpenAI GPT-4-era models but is only an
    # APPROXIMATION for Anthropic and Google models — eval 2026-06-10 measured
    # 10-25% divergence on code-heavy text, not the ~5% previously claimed.
    "__default__": "cl100k_base",
}

# Per-provider tokenizer uncertainty bands (percentage).
# cl100k_base is the exact tokenizer for OpenAI GPT-4-era models (5% covers
# minor version drift).  For Anthropic (Claude) and Google (Gemini/Gemma) it
# is an approximation: eval 2026-06-10 measured 10-25% divergence on
# code-heavy text, so we report 25% to be conservative.  Completely unknown
# models get 40% because we don't even know which tokenizer family they use.
_TOKENIZER_UNCERTAINTY_PCT: dict[str, int] = {
    "openai": 5,
    "anthropic": 25,
    "google": 25,
    "__unknown__": 40,
}


def _provider_for_model(model: str) -> str:
    """Map a model string to one of the _TOKENIZER_UNCERTAINTY_PCT keys.

    Uses startswith/substring matching against known model-name prefixes:
      - gpt- / o1 / o3 / text-embedding / azure  → openai
      - claude-                                   → anthropic
      - gemini- / gemma-                          → google
      - anything else                             → __unknown__
    """
    m = model.lower()
    if (
        m.startswith("gpt-")
        or m.startswith("o1")
        or m.startswith("o3")
        or m.startswith("text-embedding")
        or m.startswith("azure")
    ):
        return "openai"
    if m.startswith("claude-"):
        return "anthropic"
    if m.startswith("gemini-") or m.startswith("gemma-"):
        return "google"
    return "__unknown__"


def tokenizer_uncertainty_pct(model: str) -> int:
    """Return the tokenizer uncertainty percentage for *model*.

    Reflects how much the tiktoken cl100k_base (or o200k_base) token count
    can diverge from the model's actual token count:
      - openai (gpt-*, o1*, o3*, text-embedding-*): 5%
        cl100k_base / o200k_base are the exact tokenizers for these models.
      - anthropic (claude-*): 25%
        cl100k_base is an approximation; eval 2026-06-10 shows 10-25%
        divergence on code-heavy text.
      - google (gemini-*, gemma-*): 25%
        Same as Anthropic — different SentencePiece-based tokenizer family.
      - unknown: 40%
        No information about the tokenizer family; widest honest band.
    """
    provider = _provider_for_model(model)
    return _TOKENIZER_UNCERTAINTY_PCT.get(provider, _TOKENIZER_UNCERTAINTY_PCT["__unknown__"])


def _resolve_encoding(model: str) -> tiktoken.Encoding:
    """Return a tiktoken Encoding for the given model name."""
    # Exact match first
    enc_name = _ENCODING_MAP.get(model)
    if enc_name:
        return tiktoken.get_encoding(enc_name)

    # Prefix match
    for prefix, enc in _ENCODING_MAP.items():
        if model.startswith(prefix):
            return tiktoken.get_encoding(enc)

    # Try tiktoken native mapping (works for GPT-3.5/4 variants)
    try:
        return tiktoken.encoding_for_model(model)
    except KeyError:
        pass

    return tiktoken.get_encoding(_ENCODING_MAP["__default__"])


def _has_exact_encoding(model: str) -> bool:
    """Check if the model has an exact tiktoken encoding (not the __default__ fallback).

    Returns True for OpenAI models (gpt-4o, gpt-4o-mini, o1, o1-mini, o3-mini,
    gpt-4, gpt-3.5-turbo) that have known exact encodings.
    Returns False for Anthropic, Google, and other models that use the
    cl100k_base approximation.
    """
    if not model:
        return False

    # Check exact match in _ENCODING_MAP (excluding the __default__ key)
    if model in _ENCODING_MAP and model != "__default__":
        return True

    # Check prefix match in _ENCODING_MAP (excluding __default__)
    for prefix in _ENCODING_MAP:
        if prefix != "__default__" and model.startswith(prefix):
            return True

    # Try tiktoken native mapping (works for GPT-3.5/4 variants)
    try:
        tiktoken.encoding_for_model(model)
        return True
    except KeyError:
        return False


def detect_model_from_text(text: str) -> Optional[str]:
    """Extract a model identifier from a code snippet.

    Scans for patterns like ``model="gpt-4o"`` or ``engine='claude-3-sonnet'``.
    Also infers model type from API usage patterns (e.g., embeddings.create → embedding model).
    Returns the first match (lowercased), normalised through alias table, or None.
    """
    matches = _MODEL_RE.findall(text)
    if matches:
        raw = matches[0].strip().lower()
        return _MODEL_ALIASES.get(raw, raw)

    # Infer from API usage when model name is in a variable
    text_lower = text.lower()
    if "embedding" in text_lower:
        if "ada" in text_lower:
            return "text-embedding-ada-002"
        return "text-embedding-3-small"
    if "image" in text_lower or "dall" in text_lower:
        return "gpt-4o"
    if "whisper" in text_lower or "audio" in text_lower:
        return "gpt-4o"

    return None


def provider_for_model(model: str) -> str:
    """Infer the provider name from a model string."""
    model_lower = model.lower()
    for prefix, provider in _MODEL_PROVIDER_PREFIX:
        if model_lower.startswith(prefix):
            return provider
    # Fallback: check substrings
    for keyword, provider in _PROVIDER_MAP.items():
        if keyword in model_lower:
            return provider
    return "unknown"


def provider_for_finding(finding: dict) -> str:
    """Return the provider for a scanner finding dict.

    Tries model-based detection first, falls back to the scanner's own
    ``provider`` field.
    """
    model = detect_model_from_text(finding.get("text", ""))
    if model:
        p = provider_for_model(model)
        if p != "unknown":
            return p
    return finding.get("provider", "unknown")


# ---------------------------------------------------------------------------
# Core token counting
# ---------------------------------------------------------------------------

def count_tokens(text: str, model: str = "gpt-4o") -> int:
    """Count tokens in *text* using the appropriate tokenizer for *model*.

    For OpenAI models (gpt-*, o1*, o3*) tiktoken is exact (within ~5%).
    For Anthropic (claude-*) and Google (gemini-*, gemma-*) models tiktoken
    cl100k_base is used as an approximation — eval 2026-06-10 measured
    10-25% divergence on code-heavy text.  Use tokenizer_uncertainty_pct()
    to get the honest per-provider uncertainty band.
    """
    try:
        enc = _resolve_encoding(model)
        return len(enc.encode(text))
    except Exception as e:
        logger.warning("Token count failed for model %s: %s; using char/4 estimate", model, e)
        return max(1, len(text) // 4)


def count_tokens_for_provider(text: str, provider: str, model: str = "") -> int:
    """Count tokens with awareness of which provider/model is being used.

    Picks the right tokenizer variant based on provider and optional model.
    """
    if not model:
        # Pick a representative model for the provider
        model = {
            "openai": "gpt-4o",
            "anthropic": "claude-3-sonnet",
            "google": "gemini-2.0-flash",
            "langchain": "gpt-4o",
            "llamaindex": "gpt-4o",
        }.get(provider, "unknown")
    return count_tokens(text, model)


# ---------------------------------------------------------------------------
# Context window lookup
# ---------------------------------------------------------------------------

# Default context window for unknown models (conservative estimate)
_DEFAULT_CONTEXT_WINDOW = 128_000


def _get_context_window(model: str) -> int:
    """Return the context window size (max tokens) for a model.

    Looks up the model in ``config.context_windows`` with exact match first,
    then fuzzy prefix match.  Returns ``_DEFAULT_CONTEXT_WINDOW`` for
    unknown models.
    """
    from erabot._engine._pricing import get_settings
    context_windows = get_settings().context_windows

    # Exact match
    if model in context_windows:
        return context_windows[model]

    # Fuzzy prefix/substring match (same strategy as _get_pricing)
    for known, window in context_windows.items():
        if known in model or model.startswith(known.rsplit("-", 1)[0]):
            return window

    return _DEFAULT_CONTEXT_WINDOW


# ---------------------------------------------------------------------------
# Token waste estimation
# ---------------------------------------------------------------------------

# Waste detection patterns with descriptive categories
_VERBOSE_SYSTEM_PATTERNS = [
    "you are a helpful assistant",
    "you are an ai assistant",
    "as a large language model",
    "please provide a detailed",
    "in your response, make sure to",
    "step by step",
    "let me think about this",
    "i want you to act as",
    "you must always",
    "remember to always",
]

_REDUNDANT_INSTRUCTION_PATTERNS = [
    "respond in json format",
    "format your response as json",
    "output valid json",
    "return a json object",
    "please ensure",
    "do not include any",
    "make sure you",
    "it is important that",
    "please note that",
]

_JSON_MODE_KEYWORDS = [
    "response_format",
    "json_object",
    "json_mode",
    "structured_output",
]


def _estimate_token_waste(text: str, model: str) -> tuple[int, list[str]]:
    """Estimate wasted tokens in a code snippet / prompt text.

    Analyses common waste patterns:
    1. **verbose_system_prompt** -- boilerplate preambles that add tokens
       without improving quality.
    2. **redundant_instructions** -- repeated formatting / behavioral
       instructions that could be consolidated.
    3. **json_mode_overhead** -- manual JSON formatting instructions when
       the model supports native JSON mode.

    Returns:
        (tokens_wasted, waste_categories) where ``tokens_wasted`` is an
        *estimated* count and ``waste_categories`` is a list of labels.
    """
    text_lower = text.lower()
    waste_categories: list[str] = []
    total_wasted = 0

    # 1. Verbose system prompt detection
    verbose_hits = sum(1 for p in _VERBOSE_SYSTEM_PATTERNS if p in text_lower)
    if verbose_hits >= 2:
        # Estimate ~15 tokens per verbose phrase
        wasted = verbose_hits * 15
        total_wasted += wasted
        waste_categories.append("verbose_system_prompt")

    # 2. Redundant instructions
    redundant_hits = sum(1 for p in _REDUNDANT_INSTRUCTION_PATTERNS if p in text_lower)
    if redundant_hits >= 2:
        wasted = redundant_hits * 12
        total_wasted += wasted
        waste_categories.append("redundant_instructions")

    # 3. JSON mode overhead -- manual JSON instructions when model supports native JSON mode
    json_instruction_hits = sum(1 for p in _JSON_MODE_KEYWORDS if p in text_lower)
    manual_json_hits = sum(
        1 for p in ["respond in json", "format your response as json", "output valid json", "return a json"]
        if p in text_lower
    )
    if manual_json_hits >= 1 and json_instruction_hits == 0:
        # User is instructing JSON manually instead of using response_format
        wasted = manual_json_hits * 20
        total_wasted += wasted
        waste_categories.append("json_mode_overhead")

    return total_wasted, waste_categories


# ---------------------------------------------------------------------------
# Cost lookup
# ---------------------------------------------------------------------------

def _get_pricing(model: str, config: EstimationConfig) -> dict:
    """Return per-1M-token pricing for *model*, with config overrides taking priority."""
    # Custom overrides first
    if model in config.custom_pricing:
        return config.custom_pricing[model]

    from erabot._engine._pricing import get_settings
    pricing = get_settings().pricing

    # Exact match
    if model in pricing:
        return pricing[model]

    # Prefix / substring fuzzy match
    for known, p in pricing.items():
        if known in model or model.startswith(known.rsplit("-", 1)[0]):
            return p

    # Model-class-aware fallback: embedding models are far cheaper than chat models
    model_lower = model.lower()
    if any(kw in model_lower for kw in ("embed", "voyage", "e5-", "bge-")):
        logger.warning("No pricing for embedding model '%s'; using embedding default ($0.10/1M)", model)
        return {"input": 0.10, "output": 0.0}
    if any(kw in model_lower for kw in ("whisper", "tts", "audio", "speech")):
        logger.warning("No pricing for audio model '%s'; using audio default ($0.06/min)", model)
        return {"input": 0.06, "output": 0.0}

    logger.warning("No pricing found for model '%s'; using conservative chat default ($5/$15)", model)
    return {"input": 5.0, "output": 15.0}


def calculate_finding_cost(
    finding: dict,
    config: EstimationConfig,
) -> dict:
    """Compute estimated monthly cost for a single scanner finding.

    Uses variable tracing (Phase 33) when file_content is available:
    traces content-bearing arguments (messages, content, prompt) to extract
    actual prompt text for more accurate token counting.

    Returns a dict with:
      - model: detected or inferred model name
      - provider: inferred provider
      - input_tokens: tokens in the code call site or traced content
      - completion_tokens: estimated completion tokens
      - cost_per_call_usd: cost for one invocation
      - monthly_cost_usd: projected monthly cost
      - calls_per_month: assumption used
      - estimation_method: how the estimate was produced
      - estimation_confidence: low/medium/high
    """
    text = finding.get("text", "")
    detected_model = detect_model_from_text(text)
    model_detected_explicitly = bool(detected_model)
    model = detected_model or ""
    provider = provider_for_finding(finding)

    # Fall back to a representative model for the provider if none found in code
    if not model:
        # Check if the call is an embedding call based on method/text
        call_text = finding.get("text", "").lower()
        is_embedding = "embedding" in call_text
        model = {
            "openai": "text-embedding-3-small" if is_embedding else "gpt-4o",
            "anthropic": "claude-3-sonnet",
            "google": "gemini-2.0-flash",
            "langchain": "gpt-4o",
            "llamaindex": "gpt-4o",
        }.get(provider, "unknown")

    # Phase 33: Variable tracing for better token estimation
    # When file_content is available, trace the content argument to get
    # actual prompt text instead of counting call-site code tokens.
    estimation_method = "call_site_static_analysis"
    estimation_confidence = "low"
    token_text = text  # Default: count tokens in the call-site code

    file_content = finding.get("file_content")
    if file_content:
        try:
            from erabot._engine.variable_tracer import trace_arguments

            trace_result = trace_arguments(
                file_path=finding.get("file_path", ""),
                file_content=file_content,
                finding_start_line=finding.get("line", 0),
                finding_end_line=finding.get("end_line", finding.get("line", 0)),
            )

            if trace_result.traced_content is not None:
                token_text = trace_result.traced_content
                estimation_method = trace_result.estimation_method
                estimation_confidence = trace_result.estimation_confidence
            elif trace_result.trace_outcome != "no_content_arg":
                # Dynamic or unresolved — keep call-site text but use tracer's method info
                estimation_method = trace_result.estimation_method
                estimation_confidence = trace_result.estimation_confidence
        except Exception as e:
            logger.debug("Variable tracing failed for %s:%s: %s",
                         finding.get("file_path", ""), finding.get("line", 0), e)

    # If tracing didn't improve confidence, fall back to Phase 29 logic
    if estimation_method == "call_site_static_analysis":
        if model_detected_explicitly and _has_exact_encoding(detected_model):
            estimation_confidence = "medium"
        else:
            estimation_confidence = "low"

    # Unknown model: force low confidence regardless of tracing outcome
    if model == "unknown":
        estimation_confidence = "low"

    input_tokens = count_tokens(token_text, model)
    completion_tokens = max(1, int(input_tokens * config.resolved_completion_ratio()))

    pricing = _get_pricing(model, config)
    cost_per_call = (
        (input_tokens / 1_000_000) * pricing["input"]
        + (completion_tokens / 1_000_000) * pricing["output"]
    )
    _resolved_calls = config.resolved_calls_per_month()
    _resolved_turns = config.resolved_calls_per_task()
    monthly_cost = round(cost_per_call * _resolved_calls * _resolved_turns, 6)

    # Phase 37: Context window utilization
    context_window_size = _get_context_window(model)
    total_tokens_per_call = input_tokens + completion_tokens
    context_utilization_pct = round(
        (total_tokens_per_call / context_window_size * 100) if context_window_size > 0 else 0.0,
        2,
    )
    tokens_available = max(0, context_window_size - total_tokens_per_call)

    # Phase 37: Token waste estimation
    tokens_wasted, waste_categories = _estimate_token_waste(token_text, model)

    from erabot._engine._pricing import price_confidence as _price_confidence
    _conf = _price_confidence(model)

    return {
        "file_path": finding.get("file_path", ""),
        "line": finding.get("line", 0),
        "model": model,
        "provider": provider,
        "input_tokens": input_tokens,
        "completion_tokens": completion_tokens,
        "cost_per_call_usd": round(cost_per_call, 8),
        "monthly_cost_usd": monthly_cost,
        "calls_per_month": _resolved_calls,
        "price_confidence": _conf,
        # Explicit call-volume assumption tracking (eval 2026-06-10):
        # assumed_calls_per_month mirrors calls_per_month for clarity in API responses.
        # volume_assumed=True when the 10,000 default was used; False when the caller
        # supplied an explicit value. This lets consumers flag cost estimates that are
        # dominated by an unverified assumption.
        "assumed_calls_per_month": _resolved_calls,
        "volume_assumed": config.volume_assumed(),
        "calls_per_task": _resolved_turns,
        "turns_assumed": config.turns_assumed(),
        "effective_calls_per_month": _resolved_calls * _resolved_turns,
        # Explicit completion-ratio assumption tracking (DS-1, eval 2026-06-12):
        # assumed_completion_ratio mirrors the resolved ratio for API response clarity.
        # completion_ratio_assumed=True when the 0.5 default was used; False when the
        # caller supplied an explicit value.
        "assumed_completion_ratio": config.resolved_completion_ratio(),
        "completion_ratio_assumed": config.completion_ratio_assumed(),
        "estimation_method": estimation_method,
        "estimation_confidence": estimation_confidence,
        # Per-provider tokenizer accuracy: cl100k_base is exact for OpenAI but
        # diverges 10-25% on code for Anthropic/Google (eval 2026-06-10).
        "tokenizer_uncertainty_pct": tokenizer_uncertainty_pct(model),
        # Phase 37: Efficiency metrics
        "context_window_size": context_window_size,
        "context_utilization_pct": context_utilization_pct,
        "tokens_available": tokens_available,
        "tokens_wasted": tokens_wasted,
        "waste_categories": waste_categories,
    }


# ---------------------------------------------------------------------------
# Aggregate / summary helpers
# ---------------------------------------------------------------------------

@dataclass
class TokenEstimationResult:
    """Full estimation result for a set of scanner findings."""

    # Per-finding details
    findings: list[dict] = field(default_factory=list)

    # Aggregated totals
    total_monthly_cost_usd: float = 0.0
    total_input_tokens_per_call: int = 0
    total_completion_tokens_per_call: int = 0

    # Confidence bands — range based on per-finding confidence
    cost_lower_bound_usd: float = 0.0
    cost_upper_bound_usd: float = 0.0

    # Confidence distribution
    confidence_counts: dict[str, int] = field(default_factory=lambda: {"high": 0, "medium": 0, "low": 0})

    # Breakdowns
    by_provider: dict[str, dict] = field(default_factory=dict)
    by_model: dict[str, dict] = field(default_factory=dict)
    by_file: dict[str, dict] = field(default_factory=dict)

    # Config snapshot
    calls_per_month_assumption: int = 10000
    completion_ratio_assumption: float = 0.5

    # Phase 37: Aggregate efficiency metrics
    efficiency_metrics: Optional[dict] = None

    def __getitem__(self, index):
        """Allow list-style indexing (e.g. result[0]) as a convenience alias for
        result.findings[index] — some callers treat the result as the per-finding list."""
        return self.findings[index]


def estimate_findings(
    findings: list[dict],
    config: Optional[EstimationConfig] = None,
) -> TokenEstimationResult:
    """Estimate token costs for a list of scanner findings.

    Args:
        findings: Raw list of dicts returned by ``scan_files_for_llm_calls``.
        config: Optional usage assumption overrides and custom pricing.

    Returns:
        ``TokenEstimationResult`` with per-finding details and roll-up breakdowns.
    """
    if config is None:
        config = EstimationConfig()

    # DS-4: Emit a staleness warning when the pricing table is overdue for review.
    from erabot._engine._pricing import pricing_staleness_days, PRICING_STALENESS_WARN_DAYS, PRICING_LAST_VERIFIED
    _stale_days = pricing_staleness_days()
    if _stale_days > PRICING_STALENESS_WARN_DAYS:
        logger.warning(
            "Pricing table is %d days old (last verified %s, warn threshold %d days). "
            "Cost estimates may be inaccurate — run scripts/refresh_pricing.py and update "
            "config.py + PRICING_LAST_VERIFIED.",
            _stale_days, PRICING_LAST_VERIFIED, PRICING_STALENESS_WARN_DAYS,
        )

    result = TokenEstimationResult(
        calls_per_month_assumption=config.resolved_calls_per_month(),
        completion_ratio_assumption=config.resolved_completion_ratio(),
    )

    from erabot._engine.turn_model import estimate_calls_per_task
    from erabot._engine.history_model import detect_history_growth

    for finding in findings:
        _code = finding.get("file_content") or finding.get("text") or ""
        # Cost is computed from the caller's config UNCHANGED: the calls_per_task
        # multiplier applies ONLY when a caller explicitly supplied a task-based
        # volume (config.calls_per_task is not None). The default path keeps
        # resolved_calls_per_task()==1, so monthly_cost == cost_per_call * calls_per_month.
        est = calculate_finding_cost(finding, config)
        # Agent turn-awareness is INFORMATIONAL only: surface the statically-inferred
        # turns (agent-loop bound / for-while loop / single call) without letting them
        # silently multiply the reported dollar figure.
        _tm = estimate_calls_per_task(
            _code,
            call_line=int(finding.get("line", 1) or 1),
            end_line=finding.get("end_line"),
        )
        est["inferred_calls_per_task"] = _tm["calls_per_task"]
        est["turn_basis"] = _tm["basis"]
        est["history_growth"] = detect_history_growth(_code)["history_growth"]
        result.findings.append(est)

        # Totals
        result.total_monthly_cost_usd += est["monthly_cost_usd"]
        result.total_input_tokens_per_call += est["input_tokens"]
        result.total_completion_tokens_per_call += est["completion_tokens"]

        # By provider
        prov = est["provider"]
        if prov not in result.by_provider:
            result.by_provider[prov] = {"monthly_cost_usd": 0.0, "call_sites": 0, "models": set()}
        result.by_provider[prov]["monthly_cost_usd"] += est["monthly_cost_usd"]
        result.by_provider[prov]["call_sites"] += 1
        result.by_provider[prov]["models"].add(est["model"])

        # By model
        mdl = est["model"]
        if mdl not in result.by_model:
            result.by_model[mdl] = {"monthly_cost_usd": 0.0, "call_sites": 0, "provider": prov}
        result.by_model[mdl]["monthly_cost_usd"] += est["monthly_cost_usd"]
        result.by_model[mdl]["call_sites"] += 1

        # By file
        fp = est["file_path"]
        if fp not in result.by_file:
            result.by_file[fp] = {"monthly_cost_usd": 0.0, "call_sites": 0}
        result.by_file[fp]["monthly_cost_usd"] += est["monthly_cost_usd"]
        result.by_file[fp]["call_sites"] += 1

    result.total_monthly_cost_usd = round(result.total_monthly_cost_usd, 6)

    # Compute confidence bands — multiply low-confidence findings by error factors
    # HIGH: ±10%, MEDIUM: ±50%, LOW: 0.1x–5x range
    _BAND_FACTORS = {"high": (0.9, 1.1), "medium": (0.5, 1.5), "low": (0.1, 5.0)}
    lower = 0.0
    upper = 0.0
    for est in result.findings:
        conf = est.get("estimation_confidence", "low")
        result.confidence_counts[conf] = result.confidence_counts.get(conf, 0) + 1
        lo_factor, hi_factor = _BAND_FACTORS.get(conf, (0.1, 5.0))
        lower += est["monthly_cost_usd"] * lo_factor
        upper += est["monthly_cost_usd"] * hi_factor
    result.cost_lower_bound_usd = round(lower, 6)
    result.cost_upper_bound_usd = round(upper, 6)

    # Convert model sets to sorted lists for JSON serializability
    for prov_data in result.by_provider.values():
        prov_data["models"] = sorted(prov_data["models"])

    # Phase 37: Compute aggregate efficiency metrics
    result.efficiency_metrics = _compute_efficiency_metrics(result.findings, config)

    return result


# ---------------------------------------------------------------------------
# Right-sizing model map (smaller-context alternatives)
# ---------------------------------------------------------------------------

_RIGHT_SIZE_MAP: dict[str, str] = {
    # Models with very large context windows -> smaller alternatives
    "gemini-1.5-pro": "gemini-2.0-flash",       # 2M -> 1M
    "gemini-2.5-pro": "gemini-2.5-flash",        # 1M -> 1M (cheaper)
    "claude-3-opus": "claude-3-haiku",            # 200K -> 200K (much cheaper)
    "claude-3-sonnet": "claude-3-haiku",          # 200K -> 200K (cheaper)
    "claude-3.5-sonnet": "claude-3.5-haiku",      # 200K -> 200K (cheaper)
    "o1": "gpt-4o-mini",                          # 200K -> 128K (much cheaper)
    "gpt-4-32k": "gpt-4o-mini",                   # 32K -> 128K (much cheaper)
    "gpt-4-turbo": "gpt-4o-mini",                 # 128K -> 128K (cheaper)
    "gpt-4o": "gpt-4o-mini",                      # 128K -> 128K (cheaper)
}


def _compute_efficiency_metrics(findings: list[dict], config: EstimationConfig) -> dict:
    """Compute aggregate token efficiency metrics from per-finding data.

    Returns a dict matching the ``TokenEfficiencyMetrics`` schema.
    """
    if not findings:
        return {
            "avg_context_utilization_pct": 0.0,
            "max_context_utilization_pct": 0.0,
            "min_context_utilization_pct": 0.0,
            "total_tokens_wasted": 0,
            "waste_savings_usd": 0.0,
            "right_sizing_savings_usd": 0.0,
            "total_efficiency_savings_usd": 0.0,
            "efficiency_grade": "B",
            "recommendations": [],
            "waste_by_category": {},
        }

    utilizations = [est.get("context_utilization_pct", 0.0) for est in findings]
    avg_util = round(sum(utilizations) / len(utilizations), 2) if utilizations else 0.0
    max_util = round(max(utilizations), 2) if utilizations else 0.0
    min_util = round(min(utilizations), 2) if utilizations else 0.0

    # Total tokens wasted + category breakdown
    total_wasted = sum(est.get("tokens_wasted", 0) for est in findings)
    waste_by_cat: dict[str, int] = {}
    for est in findings:
        wasted = est.get("tokens_wasted", 0)
        for cat in est.get("waste_categories", []):
            waste_by_cat[cat] = waste_by_cat.get(cat, 0) + wasted

    # Waste savings: cost of wasted tokens per month
    # Use average pricing across findings as approximation
    waste_savings = 0.0
    for est in findings:
        wasted = est.get("tokens_wasted", 0)
        if wasted > 0 and est.get("input_tokens", 0) > 0:
            # Proportional cost: (wasted / input_tokens) * monthly_cost
            ratio = wasted / est["input_tokens"]
            waste_savings += est.get("monthly_cost_usd", 0.0) * ratio
    waste_savings = round(waste_savings, 6)

    # Right-sizing savings: for findings with < 10% context utilization,
    # compute savings from switching to a cheaper model
    right_sizing_savings = 0.0
    for est in findings:
        util = est.get("context_utilization_pct", 0.0)
        if util < 10.0 and est.get("model", "") in _RIGHT_SIZE_MAP:
            alt_model = _RIGHT_SIZE_MAP[est["model"]]
            alt_pricing = _get_pricing(alt_model, config)
            current_pricing = _get_pricing(est["model"], config)
            input_tokens = est.get("input_tokens", 0)
            completion_tokens = est.get("completion_tokens", 0)
            # Scale by calls_per_month AND calls_per_task, matching the multiplier
            # calculate_finding_cost() applies to monthly_cost. Default path keeps
            # resolved_calls_per_task()==1, so right-sizing numbers are unchanged.
            _eff_calls = config.resolved_calls_per_month() * config.resolved_calls_per_task()
            current_cost = (
                (input_tokens / 1_000_000) * current_pricing["input"]
                + (completion_tokens / 1_000_000) * current_pricing["output"]
            ) * _eff_calls
            alt_cost = (
                (input_tokens / 1_000_000) * alt_pricing["input"]
                + (completion_tokens / 1_000_000) * alt_pricing["output"]
            ) * _eff_calls
            if current_cost > alt_cost:
                right_sizing_savings += current_cost - alt_cost
    right_sizing_savings = round(right_sizing_savings, 6)

    total_efficiency_savings = round(waste_savings + right_sizing_savings, 6)

    # Efficiency grade based on average context utilization
    if avg_util >= 80.0:
        grade = "A"
    elif avg_util >= 50.0:
        grade = "B"
    elif avg_util >= 20.0:
        grade = "C"
    elif avg_util >= 10.0:
        grade = "D"
    else:
        grade = "F"

    # Build recommendations
    recommendations: list[str] = []
    if avg_util < 10.0:
        recommendations.append(
            f"Average context utilization is only {avg_util:.1f}%. "
            "Consider using smaller, cheaper models for most calls."
        )
    if total_wasted > 0:
        recommendations.append(
            f"Estimated {total_wasted:,} tokens wasted per call on boilerplate. "
            "Consolidate system prompts and remove redundant instructions."
        )
    if "verbose_system_prompt" in waste_by_cat:
        recommendations.append(
            "Detected verbose system prompts. Use concise role descriptions "
            "to save tokens without reducing quality."
        )
    if "json_mode_overhead" in waste_by_cat:
        recommendations.append(
            "Manual JSON formatting instructions detected. Use the model's native "
            "response_format parameter instead for cleaner, cheaper JSON output."
        )
    if right_sizing_savings > 0:
        recommendations.append(
            f"Right-sizing models to match actual usage could save ~${right_sizing_savings:.2f}/mo."
        )
    if not recommendations:
        recommendations.append("Context window utilization looks healthy. No immediate efficiency gains detected.")

    return {
        "avg_context_utilization_pct": avg_util,
        "max_context_utilization_pct": max_util,
        "min_context_utilization_pct": min_util,
        "total_tokens_wasted": total_wasted,
        "waste_savings_usd": waste_savings,
        "right_sizing_savings_usd": right_sizing_savings,
        "total_efficiency_savings_usd": total_efficiency_savings,
        "efficiency_grade": grade,
        "recommendations": recommendations,
        "waste_by_category": waste_by_cat,
    }


def estimate_files(
    files: list[dict],
    config: Optional[EstimationConfig] = None,
) -> TokenEstimationResult:
    """Run scanner + estimator on a normalized file list.

    Convenience wrapper: scans for LLM calls, then estimates costs.

    Args:
        files: List of ``{"path": str, "content": str}`` dicts (normalizer contract).
        config: Optional usage assumption overrides.

    Returns:
        ``TokenEstimationResult`` populated from all detected call sites.
    """
    from erabot._engine.scanner import scan_files_for_llm_calls

    findings = scan_files_for_llm_calls(files)
    return estimate_findings(findings, config)


# ---------------------------------------------------------------------------
# Savings potential helpers
# ---------------------------------------------------------------------------

def savings_if_downgraded(
    result: TokenEstimationResult,
    config: Optional[EstimationConfig] = None,
) -> list[dict]:
    """For each detected model, compute potential savings by switching to a cheaper alternative.

    Returns a list of dicts with:
      - current_model, alternative_model, provider
      - current_monthly_cost_usd, alternative_monthly_cost_usd
      - savings_usd, savings_pct
    """
    if config is None:
        config = EstimationConfig()

    # Downgrade map: expensive model → cheaper alternative
    _DOWNGRADE_SUGGESTIONS: dict[str, str] = {
        "gpt-4": "gpt-4o-mini",
        "gpt-4-turbo": "gpt-4o-mini",
        "gpt-4o": "gpt-4o-mini",
        "o1": "o3-mini",
        "claude-3-opus": "claude-3-haiku",
        "claude-3-sonnet": "claude-3-haiku",
        "claude-3.5-sonnet": "claude-3.5-haiku",
        "gemini-1.5-pro": "gemini-2.0-flash",
        "gemini-2.5-pro": "gemini-2.5-flash",
    }

    suggestions = []
    for model, data in result.by_model.items():
        # Skip unknown models — can't recommend a downgrade if we don't know
        # what model is being used. Users should connect the erabot proxy
        # for accurate model detection.
        if model == "unknown":
            continue

        alt = _DOWNGRADE_SUGGESTIONS.get(model)
        # Fuzzy match: model names may include version suffixes
        # (e.g., "claude-3-opus-20240229" should match "claude-3-opus")
        if not alt:
            for key, value in _DOWNGRADE_SUGGESTIONS.items():
                if model.startswith(key) and model != value:
                    alt = value
                    break
        if not alt:
            continue

        current_pricing = _get_pricing(model, config)
        alt_pricing = _get_pricing(alt, config)

        # Use average token sizes from the estimation result
        call_sites = data["call_sites"]
        if call_sites == 0:
            continue

        avg_input = result.total_input_tokens_per_call / max(len(result.findings), 1)
        avg_completion = result.total_completion_tokens_per_call / max(len(result.findings), 1)

        current_cost_per_call = (
            (avg_input / 1_000_000) * current_pricing["input"]
            + (avg_completion / 1_000_000) * current_pricing["output"]
        )
        alt_cost_per_call = (
            (avg_input / 1_000_000) * alt_pricing["input"]
            + (avg_completion / 1_000_000) * alt_pricing["output"]
        )

        # Scale by calls_per_month AND calls_per_task, matching the multiplier
        # calculate_finding_cost() applies to monthly_cost, so savings reconcile
        # against reported spend. Default path keeps resolved_calls_per_task()==1.
        _eff_calls = config.resolved_calls_per_month() * config.resolved_calls_per_task()
        current_monthly = round(current_cost_per_call * _eff_calls * call_sites, 6)
        alt_monthly = round(alt_cost_per_call * _eff_calls * call_sites, 6)
        savings = round(current_monthly - alt_monthly, 6)
        savings_pct = round((savings / current_monthly * 100) if current_monthly > 0 else 0, 1)

        suggestions.append({
            "current_model": model,
            "alternative_model": alt,
            "provider": data["provider"],
            "current_monthly_cost_usd": current_monthly,
            "alternative_monthly_cost_usd": alt_monthly,
            "savings_usd": savings,
            "savings_pct": savings_pct,
            "call_sites": call_sites,
        })

    return sorted(suggestions, key=lambda x: x["savings_usd"], reverse=True)
