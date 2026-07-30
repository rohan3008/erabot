# engine/history_model.py
"""Detect unbounded conversation-history growth: a running message list appended
each turn and re-sent whole, with no windowing/truncation/summary. Static heuristic."""
from __future__ import annotations
import re

_APPEND = re.compile(r"\b(\w+)\s*\.\s*append\s*\(|\b(\w+)\s*\+=\s*\[")
_HISTORY_NAME = re.compile(r"(messages|history|conversation|chat_history|memory)", re.I)
_WINDOW = re.compile(r"\[\s*-?\d+\s*:\s*\]?|\[-\d+:\]|truncat|summar|window|\.pop\(0\)|deque\(")


def detect_history_growth(code: str) -> dict:
    code = code or ""
    # a history-named list is appended to (or += [...])
    appended = None
    for m in _APPEND.finditer(code):
        name = m.group(1) or m.group(2) or ""
        if _HISTORY_NAME.search(name):
            appended = name
            break
    if not appended:
        return {"history_growth": False, "evidence": ""}
    # ...and that same list is passed into a call, with NO windowing/truncation nearby
    resent = re.search(rf"\bmessages\s*=\s*{re.escape(appended)}\b|\(\s*[^)]*\b{re.escape(appended)}\b",
                       code) or re.search(rf"messages\s*=\s*{re.escape(appended)}", code)
    windowed = bool(_WINDOW.search(code))
    grows = bool(appended) and not windowed
    return {"history_growth": grows, "evidence": appended if grows else ""}
