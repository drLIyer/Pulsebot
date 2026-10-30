"""
Prompt registry for Pulsebot.

PROMPT_VERSION env var controls which template is active.
Set to a comma-separated list for A/B testing (random per request, logged in traces).

  PROMPT_VERSION=v1         # always v1 (default)
  PROMPT_VERSION=v1,v2      # 50/50 A/B split
"""
import os
import random

_VERSION = os.getenv("PROMPT_VERSION", "v1")

REGISTRY: dict[str, dict] = {
    "v1": {
        "description": "Concise and conversational",
        "system": (
            "You are a knowledgeable assistant with access to {author}'s full LinkedIn newsletter archive. "
            "Answer the user's question using the article excerpts provided below. "
            "Cite article titles when referencing specific content. "
            "Be concise, accurate, and conversational. "
            "If the excerpts don't contain enough information to answer fully, say so honestly.\n\n"
            "ARTICLE EXCERPTS:\n{context}"
        ),
    },
    "v2": {
        "description": "Strict citation enforcement with explicit rules",
        "system": (
            "You are an expert on {author}'s LinkedIn newsletter archive. "
            "Help readers extract maximum value from the content.\n\n"
            "Rules:\n"
            "- Answer ONLY from the provided excerpts. Never invent or infer facts.\n"
            "- Cite the article title for every specific claim you make.\n"
            "- If the excerpts don't cover the question, say so clearly and concisely.\n"
            "- Be specific and actionable — avoid vague generalities.\n\n"
            "ARTICLE EXCERPTS:\n{context}"
        ),
    },
}


def get_system_prompt(author: str, context: str, confidence_pct: float = 100.0) -> tuple[str, str]:
    """
    Returns (system_prompt, version_id).
    Picks randomly from PROMPT_VERSION candidates for A/B testing.
    """
    candidates = [v.strip() for v in _VERSION.split(",") if v.strip()]
    version = random.choice(candidates) if candidates else "v1"

    template = REGISTRY.get(version, REGISTRY["v1"])["system"]
    system   = template.format(author=author, context=context)

    if confidence_pct < 50:
        system += (
            f"\n\nNOTE: Retrieval confidence for this question is low ({confidence_pct:.0f}%). "
            "The newsletter may not cover this topic. "
            "If you cannot answer with confidence from the excerpts, tell the reader explicitly "
            "and remind them to verify with primary sources."
        )

    return system, version


def list_versions() -> dict[str, str]:
    return {k: v["description"] for k, v in REGISTRY.items()}
