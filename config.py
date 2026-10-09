"""
Pulsebot — Central configuration
Edit this file to change limits and prices. No other file needs to change.
"""

import os

# ── Tiers ──────────────────────────────────────────────────────────────────────
# Valid values: "free" | "pro" | "power"

TIERS = ["free", "pro", "power"]

# ── Daily question limits per subscriber ──────────────────────────────────────
# Set to 0 for unlimited.

DAILY_QUESTIONS = {
    "free":  int(os.getenv("FREE_DAILY_QUESTIONS",  "10")),
    "pro":   int(os.getenv("PRO_DAILY_QUESTIONS",   "100")),
    "power": int(os.getenv("POWER_DAILY_QUESTIONS", "0")),   # 0 = unlimited
}

# ── RAG: top-k source chunks returned per query ───────────────────────────────

TOP_K = {
    "free":  int(os.getenv("FREE_TOP_K",  "5")),
    "pro":   int(os.getenv("PRO_TOP_K",   "5")),
    "power": int(os.getenv("POWER_TOP_K", "8")),
}

# ── Conversation history turns (user+assistant pairs kept) ────────────────────
# Set to 0 to disable history (each question is independent).

HISTORY_TURNS = {
    "free":  int(os.getenv("FREE_HISTORY_TURNS",  "0")),
    "pro":   int(os.getenv("PRO_HISTORY_TURNS",   "6")),
    "power": int(os.getenv("POWER_HISTORY_TURNS", "12")),
}

# ── Stripe prices ─────────────────────────────────────────────────────────────
# Fill in Stripe Price IDs once you create them in the Stripe dashboard.
# Monthly prices shown to users in the upgrade prompt.
# Set price to 0 to hide that tier from the upgrade UI.

STRIPE_PRICES = {
    "pro": {
        "price_id":    os.getenv("STRIPE_PRO_PRICE_ID",   ""),   # e.g. price_1ABC...
        "monthly_usd": int(os.getenv("STRIPE_PRO_USD",    "0")), # e.g. 29
        "label":       "Pro",
        "description": "100 questions/day, conversation memory",
    },
    "power": {
        "price_id":    os.getenv("STRIPE_POWER_PRICE_ID", ""),
        "monthly_usd": int(os.getenv("STRIPE_POWER_USD",  "0")),
        "label":       "Power",
        "description": "Unlimited questions, extended memory, priority",
    },
}

# ── Stripe keys (set in .env, never hardcode) ─────────────────────────────────

STRIPE_SECRET_KEY      = os.getenv("STRIPE_SECRET_KEY",      "")
STRIPE_WEBHOOK_SECRET  = os.getenv("STRIPE_WEBHOOK_SECRET",  "")
STRIPE_SUCCESS_URL     = os.getenv("STRIPE_SUCCESS_URL",     "")  # must be a public HTTPS URL in production
STRIPE_CANCEL_URL      = os.getenv("STRIPE_CANCEL_URL",      "")  # must be a public HTTPS URL in production

# ── Onboarding rate limit ─────────────────────────────────────────────────────
# Max new onboarding jobs per IP per day. Set to 0 to disable.

ONBOARD_DAILY_LIMIT = int(os.getenv("ONBOARD_DAILY_LIMIT", "5"))

# ── Models ────────────────────────────────────────────────────────────────────
# Default model per tier. Set MODEL to override all tiers at once.
_global_model = os.getenv("MODEL", "")

MODELS = {
    "free":  os.getenv("FREE_MODEL",  _global_model or "claude-sonnet-5-5"),
    "pro":   os.getenv("PRO_MODEL",   _global_model or "claude-sonnet-5-5"),
    "power": os.getenv("POWER_MODEL", _global_model or "claude-opus-5-5"),
}

# Models users can choose from in the chat UI. Empty list hides the picker.
AVAILABLE_MODELS = [
    m.strip()
    for m in os.getenv("AVAILABLE_MODELS", "claude-sonnet-5-5,claude-opus-5-5").split(",")
    if m.strip()
]

# ── Author-level billing ──────────────────────────────────────────────────────
# If True: the AUTHOR pays and all their subscribers get paid-tier access.
# If False: each SUBSCRIBER pays individually.
# MVP default: author pays.

AUTHOR_PAYS = os.getenv("AUTHOR_PAYS", "true").lower() == "true"

# ── Auto-refresh ──────────────────────────────────────────────────────────────
# How often (in hours) to re-check indexed authors for new articles. 0 = disabled.
AUTO_REFRESH_HOURS = int(os.getenv("AUTO_REFRESH_HOURS", "24"))

# ── LinkedIn OAuth ─────────────────────────────────────────────────────────────
# Register an app at https://www.linkedin.com/developers/apps and request
# the "Sign In with LinkedIn using OpenID Connect" product.
# Set LINKEDIN_REDIRECT_URI to:
#   local:      http://localhost:8000/auth/linkedin/callback
#   production: https://yourdomain.com/auth/linkedin/callback

LINKEDIN_CLIENT_ID     = os.getenv("LINKEDIN_CLIENT_ID",     "")
LINKEDIN_CLIENT_SECRET = os.getenv("LINKEDIN_CLIENT_SECRET", "")
LINKEDIN_REDIRECT_URI  = os.getenv("LINKEDIN_REDIRECT_URI",  "http://localhost:8000/auth/linkedin/callback")
SESSION_TTL_SECONDS    = int(os.getenv("SESSION_TTL_SECONDS", str(7 * 24 * 3600)))  # 7 days
