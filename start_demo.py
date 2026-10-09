#!/usr/bin/env python3
"""Start the Pulsebot demo server, loading credentials from ~/lennys-rag/.env"""
import os
import sys
import subprocess
from pathlib import Path

HERE = Path(__file__).parent

# Load .env from project root (copy .env.example → .env and fill in your key)
env_path = HERE / ".env"
if env_path.exists():
    with open(env_path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, val = line.partition("=")
            val = val.strip().strip('"').strip("'")
            os.environ.setdefault(key.strip(), val)

# Demo defaults
os.environ.setdefault("MODEL",      "claude-sonnet-5-5")
os.environ.setdefault("DB_PATH",    str(HERE / "pulsebot_demo.db"))
os.environ.setdefault("ADMIN_KEY",  "demo-admin")

port = int(os.environ.get("PORT", "8000"))

print(f"\n  ⚡ Pulsebot demo")
print(f"  ─────────────────────────────────")
print(f"  Onboarding:  http://localhost:{port}/")
print(f"  Demo chat:   http://localhost:{port}/c/lenny-rachitsky")
print(f"  Admin:       http://localhost:{port}/admin/dashboard  (key: {os.environ.get('ADMIN_KEY')})")
print(f"  API docs:    http://localhost:{port}/docs")
print(f"  Model:       {os.environ.get('MODEL')}")
print(f"\n  Press Ctrl+C to stop.\n")

# Prefer .venv uvicorn; fall back to PATH uvicorn
venv_uvicorn = HERE / ".venv" / "bin" / "uvicorn"
uvicorn_bin = str(venv_uvicorn) if venv_uvicorn.exists() else "uvicorn"

sys.exit(subprocess.call(
    [uvicorn_bin, "api:app", "--host", "0.0.0.0", "--port", str(port), "--log-level", "warning"],
    cwd=str(HERE),
    env=os.environ.copy(),
))
