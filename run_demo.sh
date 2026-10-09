#!/bin/bash
# Pulsebot — Demo launcher
# Usage: bash run_demo.sh

set -e
cd "$(dirname "$0")"

# ── Load credentials ───────────────────────────────────────────────────────────
# Copy .env.example → .env and fill in your ANTHROPIC_API_KEY (and optionally MODEL).
if [ -f .env ]; then
  set -a && source .env && set +a
  echo "[demo] Loaded .env, model=${MODEL:-claude-sonnet-5-5}"
fi

export MODEL="${MODEL:-claude-sonnet-5-5}"

# ── Config ─────────────────────────────────────────────────────────────────────
export PORT="${PORT:-8000}"
export ADMIN_KEY="${ADMIN_KEY:-demo-admin}"
export DB_PATH="./pulsebot_demo.db"

# ── Check index exists ─────────────────────────────────────────────────────────
if [ ! -f ./indexes/lenny-rachitsky/index.faiss ]; then
  echo "[demo] Building index first…"
  .venv/bin/python indexer.py --author lenny-rachitsky
fi

# ── Start server ───────────────────────────────────────────────────────────────
echo ""
echo "  ⚡ Pulsebot demo"
echo "  ─────────────────────────────────"
echo "  Onboarding:  http://localhost:$PORT/"
echo "  Demo chat:   http://localhost:$PORT/c/lenny-rachitsky"
echo "  Admin:       http://localhost:$PORT/admin/dashboard  (key: ${ADMIN_KEY:-demo-admin})"
echo "  API docs:    http://localhost:$PORT/docs"
echo ""
echo "  Press Ctrl+C to stop."
echo ""

# Open browser after 2s
(sleep 2 && open "http://localhost:$PORT/c/lenny-rachitsky") &

.venv/bin/uvicorn api:app --host 0.0.0.0 --port "$PORT" --log-level warning
