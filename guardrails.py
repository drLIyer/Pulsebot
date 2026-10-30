"""
Input and output safety guardrails for Pulsebot.

Input:  regex pattern list → optional Haiku LLM check for edge cases
Output: empty-response guard + injection-echo detection

Config:
  GUARDRAILS_ENABLED  true/false   (default: true)
  GUARDRAILS_LLM      true/false   (default: false — regex-only is fast)
"""
import os
import re
from typing import Optional

GUARDRAILS_ENABLED = os.getenv("GUARDRAILS_ENABLED", "true").lower() == "true"
GUARDRAILS_LLM     = os.getenv("GUARDRAILS_LLM",     "false").lower() == "true"

_INPUT_BLOCKS = [re.compile(p, re.I | re.S) for p in [
    r"\b(how\s+to|make|build|create|instructions?\s+for)\b.{0,60}\b(bomb|weapon|explosive|poison)\b",
    r"\b(how\s+to|instructions?\s+for)\b.{0,60}\b(kill|murder|assault|attack)\b.{0,40}\b(person|people|human)\b",
    r"\b(child|minor|underage).{0,30}\b(nude|naked|explicit|sexual|porn)\b",
    r"\bprompt\s+inject",
    r"\bignore\s+(all\s+)?(previous|prior|above|your)\s+instructions\b",
    r"\bdo\s+anything\s+now\b",
    r"\bjailbreak\b",
    r"\bnew\s+(persona|identity|instructions)\b.{0,30}\byou\s+(are|must|will)\b",
]]

_OUTPUT_INJECTION = [
    "ignore previous instructions",
    "ignore your instructions",
    "you are now",
    "new persona",
    "disregard your system",
]


def check_input(query: str) -> tuple[bool, Optional[str]]:
    """Returns (is_safe, reason). is_safe=False → reject with 400."""
    if not GUARDRAILS_ENABLED:
        return True, None
    if len(query.strip()) < 2:
        return False, "Query too short."
    for pat in _INPUT_BLOCKS:
        if pat.search(query):
            return False, "This type of request cannot be processed."
    if GUARDRAILS_LLM:
        return _llm_check(query)
    return True, None


def _llm_check(query: str) -> tuple[bool, Optional[str]]:
    try:
        import anthropic as _ant
        msg = _ant.Anthropic().messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=10,
            messages=[{"role": "user", "content": (
                "Is this query safe and appropriate for a newsletter chatbot? "
                "Reply with only 'safe' or 'unsafe'.\n\n"
                f"Query: {query[:500]}"
            )}],
        )
        if "unsafe" in msg.content[0].text.lower():
            return False, "This request was flagged as inappropriate."
    except Exception:
        pass  # fail open on LLM errors
    return True, None


def check_output(response: str) -> tuple[bool, Optional[str]]:
    """Returns (is_safe, reason). is_safe=False → drop the response."""
    if not GUARDRAILS_ENABLED:
        return True, None
    if not response.strip():
        return False, "Empty response from model."
    low = response.lower()
    for sig in _OUTPUT_INJECTION:
        if sig in low:
            return False, "Response flagged by output safety check."
    return True, None
