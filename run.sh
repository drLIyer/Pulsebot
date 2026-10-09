#!/bin/bash
# Pulsebot — Production launcher
# Usage: bash run.sh
#
# Prerequisites:
#   1. cp .env.example .env  — then fill in ANTHROPIC_API_KEY and ADMIN_KEY
#   2. pip install -r requirements.txt  (or activate .venv first)

set -e
cd "$(dirname "$0")"

# ── Load .env ──────────────────────────────────────────────────────────────────
if [ -f .env ]; then
  set -a && source .env && set +a
else
  echo "[error] No .env file found. Copy .env.example → .env and fill in your credentials."
  exit 1
fi

# ── Validate required vars ─────────────────────────────────────────────────────
if [ -z "${ANTHROPIC_API_KEY}" ] && [ -z "${ANTHROPIC_AUTH_TOKEN}" ]; then
  echo "[error] ANTHROPIC_API_KEY is not set. Add it to .env"
  exit 1
fi

if [ "${ADMIN_KEY:-changeme}" = "changeme" ]; then
  echo "[warn]  ADMIN_KEY is still 'changeme' — set a strong key in .env before exposing this server."
fi

# ── Defaults ───────────────────────────────────────────────────────────────────
export PORT="${PORT:-8000}"
export MODEL="${MODEL:-claude-sonnet-5-5}"
export DB_PATH="${DB_PATH:-./pulsebot.db}"
WORKERS="${WORKERS:-1}"

# Note: with WORKERS > 1, the in-memory job tracker (_jobs) is per-process.
# Keep WORKERS=1 unless you move job state to the DB.

# ── Resolve uvicorn ────────────────────────────────────────────────────────────
UVICORN=".venv/bin/uvicorn"
[ -f "$UVICORN" ] || UVICORN="uvicorn"

echo ""
echo "  ⚡ Pulsebot"
echo "  ─────────────────────────────────"
echo "  Onboarding:  http://localhost:$PORT/"
echo "  Admin:       http://localhost:$PORT/admin/dashboard"
echo "  API docs:    http://localhost:$PORT/docs"
echo "  Model:       ${MODEL}"
echo "  Workers:     ${WORKERS}"
echo ""
echo "  Press Ctrl+C to stop."
echo ""

exec "$UVICORN" api:app \
  --host 0.0.0.0 \
  --port "$PORT" \
  --workers "$WORKERS" \
  --log-level info
