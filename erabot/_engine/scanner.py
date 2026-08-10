"""Tree-sitter multi-language scanner for LLM SDK call detection.

Replaces the old filesystem-based CodeScanner with AST-based detection that
operates on the normalized file list contract: List[{"path": str, "content": str}].

Detects LLM SDK calls (OpenAI, Anthropic, Google, LangChain, LlamaIndex, LiteLLM)
across Python, TypeScript, and JavaScript using structural tree-sitter queries.
"""

import logging
import os
import re
from typing import Optional

from tree_sitter import Language, Parser, Query, QueryCursor, QueryError
from tree_sitter_language_pack import get_language

from erabot._engine.patterns import (
    get_language_name_for_extension,
    get_patterns_for_extension,
    get_supported_extensions,
)

logger = logging.getLogger(__name__)

# --- Language Registry ---
# Cache Language objects to avoid repeated construction.
_LANGUAGE_CACHE: dict[str, Language] = {}


def _get_language(ext: str) -> Optional[Language]:
    """Get cached tree-sitter Language object for a file extension."""
    lang_name = get_language_name_for_extension(ext)
    if lang_name is None:
        return None

    if lang_name not in _LANGUAGE_CACHE:
        try:
            _LANGUAGE_CACHE[lang_name] = get_language(lang_name)
        except Exception as e:
            logger.error("Failed to load tree-sitter language '%s': %s", lang_name, e)
            return None

    return _LANGUAGE_CACHE[lang_name]


# --- Provider Inference ---
_PROVIDER_KEYWORDS = {
    # OpenAI
    "openai": "openai",
    "completions": "openai",
    "embeddings": "openai",
    "responses": "openai",
    # Anthropic
    "anthropic": "anthropic",
    "messages": "anthropic",
    # Google
    "genai": "google",
    "google": "google",
    "generativemodel": "google",
    "generate_content": "google",
    "generatecontent": "google",
    "embed_content": "google",
    "embedcontent": "google",
    # Cohere
    "cohere": "cohere",
    "co.chat": "cohere",
    "co.embed": "cohere",
    # Mistral
    "mistral": "mistral",
    "mistralclient": "mistral",
    "mistralai": "mistral",
    # Groq
    "groq": "groq",
    # Together AI
    "together": "together",
    # AWS Bedrock
    "bedrock": "bedrock",
    "bedrock-runtime": "bedrock",
    "invoke_model": "bedrock",
    "invokemodel": "bedrock",
    # Azure OpenAI
    "azureopenai": "azure",
    "azure_endpoint": "azure",
    # Frameworks
    "langchain": "langchain",
    "chatopenai": "langchain",
    "llama_index": "llamaindex",
    "llamaindex": "llamaindex",
    # LiteLLM (must be checked before generic keywords like "completion")
    "litellm": "litellm",
    "litellm.completion": "litellm",
}


def _captures_in_node(captures: dict[str, list], call_node) -> dict[str, list]:
    """Filter a query's capture lists to only nodes inside *call_node*'s span.

    tree-sitter returns captures for the WHOLE file, so a file containing calls
    to two providers would otherwise leak the first call's sdk_module/chain_part
    into every call's provider inference (e.g. an Anthropic messages.create call
    tagged 'openai' because an OpenAI completions.create appeared earlier).
    Scoping to the call node's byte range fixes that.
    """
    start, end = call_node.start_byte, call_node.end_byte
    scoped: dict[str, list] = {}
    for name, nodes in captures.items():
        scoped[name] = [n for n in nodes if start <= n.start_byte and n.end_byte <= end]
    return scoped


def _infer_provider(call_text: str, captures: dict[str, list]) -> str:
    """Infer the LLM provider from the captured node text and sibling captures.

    `captures` should be scoped to the call node (see _captures_in_node) so the
    sdk_module/chain_part hints belong to THIS call, not another call earlier in
    the file.
    """
    text_lower = call_text.lower()

    # Check sdk_module captures first (most reliable)
    for node in captures.get("sdk_module", []):
        module_name = node.text.decode("utf-8") if isinstance(node.text, bytes) else str(node.text)
        provider = _PROVIDER_KEYWORDS.get(module_name.lower())
        if provider:
            return provider

    # Check chain_part captures
    for node in captures.get("chain_part", []):
        chain_name = node.text.decode("utf-8") if isinstance(node.text, bytes) else str(node.text)
        provider = _PROVIDER_KEYWORDS.get(chain_name.lower())
        if provider:
            return provider

    # Fall back to keyword scanning in the full call text.
    # Sort by keyword length descending so longer/more specific keywords
    # match before shorter ambiguous ones (e.g. "litellm" before "messages").
    for keyword, provider in sorted(_PROVIDER_KEYWORDS.items(), key=lambda x: len(x[0]), reverse=True):
        if keyword in text_lower:
            return provider

    return "unknown"


def _extract_method_name_for_node(call_node, captures: dict[str, list]) -> str:
    """Extract the method name from the specific call node, not globally.

    Searches method_name captures that are descendants of the given call_node,
    falling back to the last attribute child of the call's function expression.
    """
    # First try: find a method_name capture that is a descendant of this call node
    for node in captures.get("method_name", []):
        if call_node.start_byte <= node.start_byte <= call_node.end_byte:
            name = node.text.decode("utf-8") if isinstance(node.text, bytes) else str(node.text)
            return name

    # Fallback: walk the call node's function child to find the leaf attribute name
    func = None
    for child in call_node.children:
        if child.type in ("attribute", "member_expression"):
            func = child
            break
    if func is not None:
        # The method name is the last identifier child of the attribute node
        for child in reversed(func.children):
            if child.type == "identifier" or child.type == "property_identifier":
                name = child.text.decode("utf-8") if isinstance(child.text, bytes) else str(child.text)
                return name

    return "unknown"


# --- Metadata Extractors ---
_RE_STREAMING = re.compile(r"stream\s*=\s*True|\.stream\s*\(", re.IGNORECASE)
_RE_VISION = re.compile(r"""["']type["']\s*:\s*["']image_url["']|["']type["']\s*[=:]\s*["']image["']""", re.IGNORECASE)
_RE_TOOL_USE = re.compile(r"(tools|functions)\s*=\s*\[", re.IGNORECASE)
_RE_BATCH = re.compile(r"batches\.create", re.IGNORECASE)

# --- Model Extraction ---
# Regex to extract model string from code text.
# Matches Python kwargs (model="gpt-4o"), TypeScript object properties (model: "gpt-4"),
# and LangChain constructor args (model_name="gpt-4").
# Supports both = and : assignment operators to cover Python and JS/TS syntax.
# Defined locally (mirrors token_estimator._MODEL_RE) to avoid circular imports.
_MODEL_RE = re.compile(
    r"""(?:model|engine|model_name)\s*[:=]\s*["']([a-zA-Z0-9._\-:/ ]+)["']""",
    re.IGNORECASE,
)

# Common model aliases (shorthand -> canonical). Mirrors token_estimator._MODEL_ALIASES.
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


def _extract_model(call_text: str) -> Optional[str]:
    """Extract a model name from detected LLM call text.

    Scans for patterns like ``model="gpt-4o"`` or ``model: "claude-3-opus"``
    using the same regex as token_estimator.detect_model_from_text.
    Returns the first match (lowercased), normalised through the alias table,
    or None if no model string literal is found.
    """
    matches = _MODEL_RE.findall(call_text)
    if matches:
        raw = matches[0].strip().lower()
        return _MODEL_ALIASES.get(raw, raw)
    return None


# `.query()`/`.aquery()` is matched as an LLM call for LlamaIndex query engines
# (query_engine.query("...")), but the name is NOT unique to LLM SDKs — vector
# stores (Qdrant, Pinecone, Weaviate, Chroma), SQL (SQLAlchemy db_session.query),
# and dataframes (pandas) all expose .query(). A blocklist of receiver names can't
# keep up with compound names (db_session, sf_client, sync_db), so invert it: the
# LlamaIndex call is the query ENGINE, so keep query/aquery only when the receiver
# expression names an engine and drop everything else.
_QUERY_METHOD_RE = re.compile(r"\.\s*a?query\s*\(")


def _is_nonllm_query(method_name: str, call_text: str) -> bool:
    """True when a query/aquery match is a data-store call, not an LLM query.

    Keep it only for a LlamaIndex query engine — query_engine.query(...),
    self.query_engine.query(...), index.as_query_engine().query(...) — where the
    receiver expression contains "engine". Everything else (db_session.query,
    shard.query, collection.query, df.query) is a data lookup, so it's dropped."""
    if method_name not in ("query", "aquery"):
        return False
    m = _QUERY_METHOD_RE.search(call_text or "")
    receiver = call_text[:m.start()] if m else (call_text or "")
    return "engine" not in receiver.lower()


# LangChain gives tools, search, retrievers, and parsers the same .invoke/.ainvoke
# Runnable interface as models, so `tool.ainvoke(...)` / `search.ainvoke(...)` match
# the method pattern but are not LLM calls. Deny by receiver NAME COMPONENT (split
# on _ and camelCase) so "research"/"research_chain" (contains "search" only as a
# substring) is NOT wrongly denied.
_INVOKE_DENY_COMPONENTS = frozenset({
    "tool", "tools", "search", "retriever", "parser", "scraper", "api",
    "loader", "splitter",
    # event/callback dispatch — handler.invoke, dispatcher.invoke, etc.
    "handler", "handlers", "callback", "callbacks", "listener", "dispatcher",
    "emitter", "hook", "hooks", "middleware", "event", "events",
})
# .complete/.acomplete is LlamaIndex `llm.complete(...)` but collides with common
# non-LLM methods (shtab.complete shell-completion, futures). Keep it only for an
# LLM-ish receiver.
_LLM_RECV_COMPONENTS = frozenset({
    "llm", "model", "models", "client", "llama", "llamaindex", "chat", "oai",
    "openai", "anthropic", "claude", "gpt", "gemini", "cohere", "mistral",
    "completion", "engine", "assistant", "agent", "copilot", "bot", "ai",
})


def _receiver_components(recv: str) -> set:
    """Lowercased name components of a receiver identifier (split on _ and camelCase)."""
    spaced = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", "_", recv or "")
    return {p.lower() for p in spaced.split("_") if p}


def _receiver_before(call_text: str, method_name: str) -> str:
    """The bare identifier immediately before `.method(`, or "" when the receiver is
    a chained call / attribute the regex can't resolve (in which case we keep it)."""
    m = re.search(rf"(\w+)\s*\.\s*{re.escape(method_name)}\s*\(", call_text or "")
    return m.group(1) if m else ""


def _ambiguous_method_is_nonllm(method_name: str, call_text: str) -> bool:
    """Gate methods shared with non-LLM objects. invoke/ainvoke: drop tool/search/
    retriever/parser receivers. complete/acomplete: keep only LLM-ish receivers."""
    if method_name in ("invoke", "ainvoke"):
        recv = _receiver_before(call_text, method_name)
        if not recv:
            return False  # chained/attribute receiver — keep (usually a real model call)
        comps = _receiver_components(recv)
        if comps & _LLM_RECV_COMPONENTS:
            return False  # llm_with_tools, chat_model — LLM signal wins over "tools"
        return bool(comps & _INVOKE_DENY_COMPONENTS)
    # complete/acomplete collide with shell-completion; sendMessage (Gemini
    # chat.sendMessage) collides with chrome.runtime.sendMessage / tabs.sendMessage
    # / EventEmitter. Keep only for an LLM-ish receiver.
    if method_name in ("complete", "acomplete", "sendMessage"):
        recv = _receiver_before(call_text, method_name)
        if not recv:
            return False
        return not (_receiver_components(recv) & _LLM_RECV_COMPONENTS)
    return False


def _extract_call_metadata(call_text: str) -> dict:
    """Extract metadata flags from a detected LLM call's text.

    Returns a dict with optional keys: streaming, has_vision,
    image_tokens_estimated, tools_detected, batch_mode.
    """
    meta: dict = {}

    # Model extraction
    model = _extract_model(call_text)
    if model is not None:
        meta["model"] = model

    # Streaming detection
    if _RE_STREAMING.search(call_text):
        meta["streaming"] = True

    # Vision / image detection
    vision_matches = _RE_VISION.findall(call_text)
    if vision_matches:
        meta["has_vision"] = True
        # Estimate 170 tokens per image reference (OpenAI low-detail default)
        meta["image_tokens_estimated"] = 170 * len(vision_matches)

    # Tool / function calling detection
    tool_matches = _RE_TOOL_USE.findall(call_text)
    if tool_matches:
        # Count tool/function defs by looking for dict-like structures after tools=[
        # Simple heuristic: count opening braces at top level of the list
        tool_count = call_text.count("{") if tool_matches else 0
        meta["tools_detected"] = max(tool_count, len(tool_matches))

    # Batch mode detection
    if _RE_BATCH.search(call_text):
        meta["batch_mode"] = True

    return meta


# --- Core Detection ---
def detect_llm_calls(file_path: str, content: str) -> list[dict]:
    """Detect LLM SDK calls in a single file using tree-sitter AST queries.

    Args:
        file_path: Path of the file (used for extension detection and finding metadata).
        content: Source code content of the file.

    Returns:
        List of finding dicts with keys:
            file_path, line, end_line, text, provider, method_name
    """
    ext = os.path.splitext(file_path)[1].lower()
    language = _get_language(ext)
    if language is None:
        return []

    patterns = get_patterns_for_extension(ext)
    if not patterns:
        return []

    # Parse the source code into an AST
    parser = Parser(language)
    tree = parser.parse(content.encode("utf-8"))

    # Create and execute query
    try:
        query = Query(language, patterns)
    except QueryError as e:
        logger.error(
            "tree-sitter query compilation failed for %s (ext=%s): %s",
            file_path, ext, e,
        )
        return []

    cursor = QueryCursor(query)
    captures = cursor.captures(tree.root_node)

    # Extract llm_call captures and build findings
    llm_call_nodes = captures.get("llm_call", [])
    if not llm_call_nodes:
        return []

    # Deduplicate: a single call may be captured by multiple pattern groups
    seen_ranges: set[tuple[int, int]] = set()
    findings: list[dict] = []

    for node in llm_call_nodes:
        node_range = (node.start_byte, node.end_byte)
        if node_range in seen_ranges:
            continue
        seen_ranges.add(node_range)

        # node.text is the exact source bytes for this node. Do NOT slice
        # content[start_byte:end_byte] — content is a str but tree-sitter offsets
        # are BYTE offsets, so any non-ASCII char earlier in the file (em-dash,
        # smart quote, emoji) shifts the slice and corrupts call_text — which
        # broke receiver gates and model extraction.
        call_text = node.text.decode("utf-8", errors="ignore")
        scoped_captures = _captures_in_node(captures, node)
        provider = _infer_provider(call_text, scoped_captures)
        method_name = _extract_method_name_for_node(node, captures)

        # A PascalCase method is a class/config constructor, not an LLM API call:
        # genai.types.GenerateContentConfig(...), genai.GenerativeModel(...). Real
        # LLM API methods are never PascalCase. (The SDK-module pattern otherwise
        # matches any call under a known SDK namespace.)
        if method_name[:1].isupper():
            continue
        # Drop data-store .query()/.aquery() (Qdrant/pandas/SQL), keep LLM engines.
        if _is_nonllm_query(method_name, call_text):
            continue
        # Drop invoke/ainvoke on tools/search and complete on non-LLM receivers.
        if _ambiguous_method_is_nonllm(method_name, call_text):
            continue

        model = _extract_model(call_text)

        finding = {
            "file_path": file_path,
            "line": node.start_point[0] + 1,  # 1-indexed
            "end_line": node.end_point[0] + 1,
            "text": call_text,
            "provider": provider,
            "method_name": method_name,
            "model": model,
            "file_content": content,  # Full file for variable tracing (Phase 33)
        }

        # Enrich with metadata extractors (streaming, vision, tools, batch)
        metadata = _extract_call_metadata(call_text)
        finding.update(metadata)

        findings.append(finding)

    return findings


# --- File List Operations ---
def scan_files_for_llm_calls(files: list[dict]) -> list[dict]:
    """Scan a list of normalized files for LLM SDK calls.

    Args:
        files: List of {"path": str, "content": str} dicts (normalizer contract).

    Returns:
        Flattened list of all finding dicts across all files.
    """
    all_findings: list[dict] = []
    for file_dict in files:
        path = file_dict.get("path", "")
        content = file_dict.get("content", "")
        if not path or not content:
            continue
        try:
            findings = detect_llm_calls(path, content)
            all_findings.extend(findings)
        except Exception as e:
            logger.warning("Error scanning %s: %s", path, e)
            continue
    # REC-01: deterministic recall extensions (framework idioms, local wrappers)
    try:
        from erabot._engine.detection_extensions import apply_extensions
        all_findings.extend(apply_extensions(files, all_findings))
    except Exception as e:  # never let extensions break the primary scan
        logger.warning("detection extensions failed (non-fatal): %s", e)
    return all_findings


def prioritize_files(files: list[dict], max_files: int = 50) -> list[dict]:
    """Prioritize files by LLM call density and return top N.

    Per D-07: max 50 files per scan. Files are sorted by LLM call count
    (descending) so that the most LLM-heavy files are scanned first.
    Files with 0 LLM calls are included at the end (Gemini may still
    find optimization patterns in helper/utility code).

    Args:
        files: List of {"path": str, "content": str} dicts.
        max_files: Maximum number of files to return (default 50).

    Returns:
        Subset of files sorted by LLM call density, capped at max_files.
    """
    if len(files) <= max_files:
        # Still sort by density even if under the cap
        scored = []
        for f in files:
            count = len(detect_llm_calls(f["path"], f["content"]))
            scored.append((count, f))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [f for _, f in scored]

    # Quick-scan all files to count LLM calls
    scored: list[tuple[int, dict]] = []
    for f in files:
        try:
            count = len(detect_llm_calls(f["path"], f["content"]))
        except Exception:
            count = 0
        scored.append((count, f))

    # Sort by call count descending
    scored.sort(key=lambda x: x[0], reverse=True)

    return [f for _, f in scored[:max_files]]


# --- Scan-Level Warnings ---

# Human-readable names for extensions that the normalizer accepts but the
# structural scanner does NOT cover (no tree-sitter patterns defined).
# Keyed by lowercase extension → display name.
_UNSUPPORTED_EXTENSION_LANGUAGE: dict[str, str] = {
    ".go":   "Go",
    ".java": "Java",
    ".rb":   "Ruby",
    ".cs":   "C#",
    ".rs":   "Rust",
}

# Canonical display names for what IS supported (sourced from _EXTENSION_REGISTRY)
_SUPPORTED_LANGUAGE_DISPLAY = "Python, JavaScript, and TypeScript"


def compute_scan_warnings(files: list[dict]) -> list[str]:
    """Compute scan-level warnings for a file list.

    Returns:
        List of human-readable warning strings covering:
        - Languages present in the scan that have no structural scanner patterns.
        - Pricing staleness (imported from config to avoid circular deps).

    The warnings list is attached to ScanAnalysisResult so it reaches the
    API response and (via the worker) is persisted on ScanJob.warnings.
    """
    warnings: list[str] = []

    # --- Unsupported-language warning ---
    # Tally files by unsupported language.
    lang_counts: dict[str, int] = {}
    for file_dict in files:
        path = file_dict.get("path", "")
        if not path:
            continue
        ext = os.path.splitext(path)[1].lower()
        lang = _UNSUPPORTED_EXTENSION_LANGUAGE.get(ext)
        if lang:
            lang_counts[lang] = lang_counts.get(lang, 0) + 1

    for lang, count in sorted(lang_counts.items()):
        noun = "file" if count == 1 else "files"
        warnings.append(
            f"{count} {noun} in {lang} were not structurally analyzed — "
            f"erabot's scanner currently supports {_SUPPORTED_LANGUAGE_DISPLAY}. "
            f"Cost findings for {lang} rely on the AI pass only and may be incomplete."
        )

    # --- Pricing staleness warning ---
    try:
        from erabot._engine._pricing import pricing_staleness_days, PRICING_STALENESS_WARN_DAYS
        days = pricing_staleness_days()
        if days > PRICING_STALENESS_WARN_DAYS:
            warnings.append(
                f"Model pricing was last verified {days} days ago; "
                "dollar figures may have drifted."
            )
    except Exception:
        pass  # Never let a config import failure break a scan

    return warnings
