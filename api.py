#!/usr/bin/env python3
"""
Pulsebot — Multi-tenant FastAPI Backend

Endpoints:
  GET  /health
  GET  /authors
  GET  /authors/{author}
  POST /chat/{author}              — SSE chat stream
  POST /upgrade/{author}           — start Stripe checkout
  GET  /upgrade/success|cancel     — Stripe redirect pages
  POST /stripe/webhook             — Stripe event handler
  POST /admin/index/{author}       — trigger re-index
  GET  /admin/usage

SSE event format:
  data: {"type": "sources", "sources": [{title, url, score}, ...]}
  data: {"type": "token",   "text": "..."}
  data: {"type": "done",    "usage": {input_tokens, output_tokens}}
  data: {"type": "error",   "message": "..."}
  data: {"type": "limit",   "tier": "free", "daily_limit": 10, "upgrade_url": "..."}

Run:
  ANTHROPIC_API_KEY=sk-... uvicorn api:app --reload --port 8000
"""

import os
import re
import json
import time
import hmac
import asyncio
import logging
import secrets
from collections import defaultdict
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional, AsyncIterator
from urllib.parse import urlencode

import aiosqlite
import anthropic
import httpx
from fastapi import FastAPI, HTTPException, Request, Header
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, FileResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from indexer import load_index, get_model
from models import Chunk
import faiss
import numpy as np
import config as cfg

# ── Config ─────────────────────────────────────────────────────────────────────

INDEX_DIR       = os.getenv("INDEX_DIR",   "./indexes")
DATA_DIR        = os.getenv("DATA_DIR",    "./data")
DB_PATH         = os.getenv("DB_PATH",     "./pulsebot.db")
ADMIN_KEY       = os.getenv("ADMIN_KEY",   "changeme")
ALLOWED_ORIGINS = [o.strip() for o in os.getenv("ALLOWED_ORIGINS", "*").split(",")]

# Support Align enterprise proxy: set ANTHROPIC_BASE_URL + ANTHROPIC_AUTH_TOKEN
_ANTHROPIC_BASE_URL = os.getenv("ANTHROPIC_BASE_URL", "")
_ANTHROPIC_AUTH_TOKEN = os.getenv("ANTHROPIC_AUTH_TOKEN", "")


def _make_anthropic_client() -> anthropic.AsyncAnthropic:
    kwargs: dict = {}
    if _ANTHROPIC_AUTH_TOKEN:
        kwargs["api_key"] = _ANTHROPIC_AUTH_TOKEN
    if _ANTHROPIC_BASE_URL:
        kwargs["base_url"] = _ANTHROPIC_BASE_URL
    return anthropic.AsyncAnthropic(**kwargs)

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("pulsebot")


def _check_admin_key(provided: str | None) -> bool:
    if not ADMIN_KEY or not provided:
        return False
    return hmac.compare_digest(provided, ADMIN_KEY)


def _real_ip(request: Request) -> str:
    xff = request.headers.get("X-Forwarded-For", "")
    if xff:
        return xff.split(",")[0].strip()
    xri = request.headers.get("X-Real-IP", "")
    if xri:
        return xri.strip()
    return request.client.host if request.client else "unknown"


def _sanitize_handle(raw: str) -> str:
    """Strip everything that isn't alphanumeric, hyphen, or underscore."""
    handle = re.sub(r"[^a-z0-9\-_]", "", raw.lower())
    if not handle:
        raise ValueError("Empty handle after sanitization")
    return handle

# ── LinkedIn OAuth ─────────────────────────────────────────────────────────────

SESSION_COOKIE         = "pb_session"
LINKEDIN_AUTH_URL      = "https://www.linkedin.com/oauth/v2/authorization"
LINKEDIN_TOKEN_URL     = "https://www.linkedin.com/oauth/v2/accessToken"
LINKEDIN_USERINFO_URL  = "https://api.linkedin.com/v2/userinfo"

# In-memory state store for CSRF protection (WORKERS=1 assumed)
_oauth_states: dict[str, dict] = {}


async def _create_session(linkedin_id: str, name: str, email: str, picture: str) -> str:
    session_id = secrets.token_urlsafe(32)
    now = time.time()
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute(
            """INSERT INTO sessions (id, linkedin_id, name, email, picture, created_at, expires_at)
               VALUES (?,?,?,?,?,?,?)""",
            (session_id, linkedin_id, name, email, picture, now, now + cfg.SESSION_TTL_SECONDS),
        )
        await db.commit()
    return session_id


async def _get_session(session_id: str) -> Optional[dict]:
    async with aiosqlite.connect(DB_PATH) as db:
        async with db.execute(
            "SELECT linkedin_id, name, email, picture, expires_at FROM sessions WHERE id=?",
            (session_id,),
        ) as cur:
            row = await cur.fetchone()
    if not row or row[4] < time.time():
        return None
    return {"linkedin_id": row[0], "name": row[1], "email": row[2], "picture": row[3]}


async def _delete_session(session_id: str) -> None:
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute("DELETE FROM sessions WHERE id=?", (session_id,))
        await db.commit()


def _is_secure_context() -> bool:
    return "localhost" not in cfg.LINKEDIN_REDIRECT_URI and "127.0.0.1" not in cfg.LINKEDIN_REDIRECT_URI

# ── App ────────────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app_: FastAPI):
    async with aiosqlite.connect(DB_PATH) as db:
        await db.executescript("""
            CREATE TABLE IF NOT EXISTS usage (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                ts          REAL    NOT NULL,
                author      TEXT    NOT NULL,
                ip          TEXT    NOT NULL,
                input_tok   INTEGER DEFAULT 0,
                output_tok  INTEGER DEFAULT 0
            );

            -- Per-author tier (AUTHOR_PAYS=true: one row per author)
            CREATE TABLE IF NOT EXISTS author_tiers (
                author          TEXT PRIMARY KEY,
                tier            TEXT NOT NULL DEFAULT 'free',
                stripe_customer TEXT,
                stripe_sub_id   TEXT,
                updated_at      REAL NOT NULL DEFAULT 0
            );

            -- Per-subscriber daily question counter
            CREATE TABLE IF NOT EXISTS daily_questions (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                date        TEXT    NOT NULL,   -- YYYY-MM-DD UTC
                author      TEXT    NOT NULL,
                ip          TEXT    NOT NULL,
                count       INTEGER NOT NULL DEFAULT 0,
                UNIQUE(date, author, ip)
            );

            -- Persistent onboarding job history
            CREATE TABLE IF NOT EXISTS onboard_jobs (
                id              INTEGER PRIMARY KEY AUTOINCREMENT,
                author          TEXT    NOT NULL,
                display_name    TEXT,
                status          TEXT    NOT NULL,
                started_at      REAL,
                finished_at     REAL,
                articles_found  INTEGER DEFAULT 0,
                articles_fetched INTEGER DEFAULT 0,
                chunks          INTEGER DEFAULT 0,
                error           TEXT
            );

            -- Per-article/URL fetch errors during onboarding
            CREATE TABLE IF NOT EXISTS onboard_errors (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                ts          REAL    NOT NULL,
                author      TEXT    NOT NULL,
                stage       TEXT    NOT NULL,   -- discovering | fetching | indexing
                url         TEXT,
                message     TEXT    NOT NULL
            );

            -- LinkedIn OAuth sessions
            CREATE TABLE IF NOT EXISTS sessions (
                id          TEXT    PRIMARY KEY,
                linkedin_id TEXT    NOT NULL,
                name        TEXT,
                email       TEXT,
                picture     TEXT,
                created_at  REAL    NOT NULL,
                expires_at  REAL    NOT NULL
            );
        """)
        await db.commit()
        # Authenticate with HuggingFace Hub to avoid anonymous rate limits
        _hf_token = os.getenv("HF_TOKEN", "")
        if _hf_token:
            try:
                from huggingface_hub import login as _hf_login
                _hf_login(token=_hf_token, add_to_git_credential=False)
                log.info("Authenticated with HuggingFace Hub")
            except Exception as _hf_err:
                log.warning(f"HF login failed: {_hf_err}")

        # Mark jobs that were in-flight when the server last stopped
        await db.execute(
            """UPDATE onboard_jobs SET status='error', error='Server restarted while job was running',
               finished_at=? WHERE status IN ('queued','fetching','indexing','discovering')""",
            (time.time(),),
        )
        await db.commit()

        # Migration: add newsletter_url column if upgrading from an older schema
        try:
            await db.execute("ALTER TABLE onboard_jobs ADD COLUMN newsletter_url TEXT DEFAULT ''")
            await db.commit()
        except Exception:
            pass  # column already exists

    # Start background auto-refresh loop
    asyncio.create_task(_auto_refresh_loop())
    yield


app = FastAPI(title="Pulsebot API", version="0.1.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Tier helpers ───────────────────────────────────────────────────────────────

async def _get_author_tier(author: str) -> str:
    async with aiosqlite.connect(DB_PATH) as db:
        async with db.execute(
            "SELECT tier FROM author_tiers WHERE author = ?", (author,)
        ) as cur:
            row = await cur.fetchone()
    return row[0] if row else "free"


async def _set_author_tier(author: str, tier: str, stripe_customer: str = "", stripe_sub_id: str = "") -> None:
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute(
            """INSERT INTO author_tiers (author, tier, stripe_customer, stripe_sub_id, updated_at)
               VALUES (?,?,?,?,?)
               ON CONFLICT(author) DO UPDATE SET
                 tier=excluded.tier, stripe_customer=excluded.stripe_customer,
                 stripe_sub_id=excluded.stripe_sub_id, updated_at=excluded.updated_at""",
            (author, tier, stripe_customer, stripe_sub_id, time.time()),
        )
        await db.commit()


async def _check_daily_limit(author: str, ip: str, tier: str) -> tuple[bool, int, int]:
    """
    Returns (allowed, used_today, daily_limit).
    Increments counter only if allowed.
    """
    import datetime
    limit = cfg.DAILY_QUESTIONS[tier]
    if limit == 0:
        return True, 0, 0  # unlimited

    today = datetime.datetime.utcnow().strftime("%Y-%m-%d")

    # Use author tier when AUTHOR_PAYS; otherwise per-subscriber (ip)
    key_ip = "shared" if cfg.AUTHOR_PAYS else ip

    async with aiosqlite.connect(DB_PATH) as db:
        async with db.execute(
            "SELECT count FROM daily_questions WHERE date=? AND author=? AND ip=?",
            (today, author, key_ip),
        ) as cur:
            row = await cur.fetchone()

        used = row[0] if row else 0
        if used >= limit:
            return False, used, limit

        if row:
            await db.execute(
                "UPDATE daily_questions SET count=count+1 WHERE date=? AND author=? AND ip=?",
                (today, author, key_ip),
            )
        else:
            await db.execute(
                "INSERT INTO daily_questions (date, author, ip, count) VALUES (?,?,?,1)",
                (today, author, key_ip),
            )
        await db.commit()

    return True, used + 1, limit

# ── Index cache (lazy-loaded per author) ──────────────────────────────────────

_index_cache: dict[str, tuple[faiss.Index, list[Chunk]]] = {}

def _get_index(author: str) -> tuple[Optional[faiss.Index], list[Chunk]]:
    if author not in _index_cache:
        idx, chunks = load_index(INDEX_DIR, author)
        if idx is not None:
            _index_cache[author] = (idx, chunks)
        else:
            return None, []
    return _index_cache[author]

def _invalidate_cache(author: str) -> None:
    _index_cache.pop(author, None)

def _list_indexed_authors() -> list[str]:
    idx_root = Path(INDEX_DIR)
    if not idx_root.exists():
        return []
    return [
        d.name for d in sorted(idx_root.iterdir())
        if d.is_dir() and (d / "index.faiss").exists()
    ]

# ── Database ───────────────────────────────────────────────────────────────────

async def log_usage(author: str, ip: str, input_tok: int, output_tok: int) -> None:
    try:
        async with aiosqlite.connect(DB_PATH) as db:
            await db.execute(
                "INSERT INTO usage (ts, author, ip, input_tok, output_tok) VALUES (?,?,?,?,?)",
                (time.time(), author, ip, input_tok, output_tok),
            )
            await db.commit()
    except Exception as e:
        log.warning(f"usage log error: {e}")


async def _persist_job(job: dict) -> None:
    """Upsert the final state of an onboarding job to the DB."""
    try:
        async with aiosqlite.connect(DB_PATH) as db:
            await db.execute(
                """INSERT INTO onboard_jobs
                   (author, display_name, status, started_at, finished_at,
                    articles_found, articles_fetched, chunks, error, newsletter_url)
                   VALUES (?,?,?,?,?,?,?,?,?,?)""",
                (
                    job.get("author", ""),
                    job.get("display_name", ""),
                    job.get("status", ""),
                    job.get("started_at"),
                    job.get("finished_at"),
                    job.get("articles_found", 0),
                    job.get("articles_fetched", 0),
                    job.get("chunks", 0),
                    job.get("error"),
                    job.get("newsletter_url", ""),
                ),
            )
            await db.commit()
    except Exception as e:
        log.warning(f"persist_job error: {e}")


async def _log_onboard_error(author: str, stage: str, message: str, url: str = "") -> None:
    try:
        async with aiosqlite.connect(DB_PATH) as db:
            await db.execute(
                "INSERT INTO onboard_errors (ts, author, stage, url, message) VALUES (?,?,?,?,?)",
                (time.time(), author, stage, url, message),
            )
            await db.commit()
    except Exception as e:
        log.warning(f"log_onboard_error: {e}")

# ── RAG search ─────────────────────────────────────────────────────────────────

def _rag_search(query: str, author: str, top_k: int = 5) -> list[dict]:
    idx, chunks = _get_index(author)
    if idx is None:
        return []
    model = get_model()
    q_emb = model.encode([query], convert_to_numpy=True).astype(np.float32)
    faiss.normalize_L2(q_emb)
    scores, indices = idx.search(q_emb, top_k)
    results = []
    for score, i in zip(scores[0], indices[0]):
        if i < 0:
            continue
        c = chunks[i]
        results.append({
            "score": float(score),
            "title": c.title,
            "url": c.url,
            "date": c.date,
            "chunk_index": c.chunk_index,
            "text": c.text,
        })
    return results

def _build_context(hits: list[dict]) -> str:
    parts = []
    for i, h in enumerate(hits, 1):
        date_str = f" ({h['date']})" if h.get("date") else ""
        parts.append(
            f"[Article {i}: {h['title']}{date_str}]\n{h['text']}"
        )
    return "\n\n---\n\n".join(parts)

# ── Claude streaming ───────────────────────────────────────────────────────────

async def _stream_claude(
    author: str,
    query: str,
    context: str,
    history: list[dict],
    use_thinking: bool = False,
    model: str = "",
) -> AsyncIterator[tuple[str, dict]]:
    """
    Yields (event_type, payload) tuples.
    event_type: "token" | "done" | "error"
    """
    client = _make_anthropic_client()

    system = (
        f"You are a knowledgeable assistant with access to {author}'s full LinkedIn newsletter archive. "
        "Answer the user's question using the article excerpts provided below. "
        "Cite article titles when referencing specific content. "
        "Be concise, accurate, and conversational. "
        "If the excerpts don't contain enough information to answer fully, say so honestly.\n\n"
        f"ARTICLE EXCERPTS:\n{context}"
    )

    messages = list(history)
    messages.append({"role": "user", "content": query})

    stream_kwargs: dict = dict(
        model=model or cfg.MODELS.get("free", "claude-sonnet-5-5"),
        max_tokens=2048,
        system=system,
        messages=messages,
    )
    if use_thinking:
        stream_kwargs["thinking"] = {"type": "enabled", "budget_tokens": 8000}

    try:
        async with client.messages.stream(**stream_kwargs) as stream:
            async for text in stream.text_stream:
                yield "token", {"text": text}
            final = await stream.get_final_message()
            usage = final.usage
            yield "done", {
                "usage": {
                    "input_tokens": usage.input_tokens,
                    "output_tokens": usage.output_tokens,
                }
            }
    except anthropic.APIError as e:
        yield "error", {"message": str(e)}

# ── Request / response models ──────────────────────────────────────────────────

class ChatRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=2000)
    history: list[dict] = Field(default_factory=list)
    model: Optional[str] = Field(default=None, max_length=100)

# ── Endpoints ──────────────────────────────────────────────────────────────────

_MODEL_META = {
    "claude-sonnet-5-5": {"label": "Sonnet 5.5", "description": "Fast · lower cost"},
    "claude-opus-5-5":   {"label": "Opus 5.5",   "description": "Most capable"},
}

@app.get("/health")
async def health():
    return {"status": "ok", "models": cfg.MODELS}


@app.get("/models")
async def list_models():
    return {
        "models": [
            {"id": m, **_MODEL_META.get(m, {"label": m, "description": ""})}
            for m in cfg.AVAILABLE_MODELS
        ]
    }


@app.get("/authors")
async def list_authors():
    authors = _list_indexed_authors()
    return {"authors": authors, "count": len(authors)}


@app.get("/authors/{author}")
async def author_info(author: str):
    idx, chunks = _get_index(author)
    if idx is None:
        raise HTTPException(404, f"No index found for author '{author}'")

    # Count unique articles
    titles = {c.title for c in chunks}
    data_dir = Path(DATA_DIR) / author
    md_files = list(data_dir.glob("*.md")) if data_dir.exists() else []

    return {
        "author": author,
        "articles": len(titles),
        "chunks": idx.ntotal,
        "markdown_files": len(md_files),
    }


@app.post("/chat/{author}")
async def chat(author: str, body: ChatRequest, request: Request):
    ip = _real_ip(request)

    # Check index exists
    idx, _ = _get_index(author)
    if idx is None:
        raise HTTPException(404, f"No newsletter index found for author '{author}'")

    # Resolve tier (author-pays: tier follows author; else per-subscriber would need auth)
    tier = await _get_author_tier(author) if cfg.AUTHOR_PAYS else "free"

    # Daily question limit check
    allowed, used, limit = await _check_daily_limit(author, ip, tier)
    if not allowed:
        upgrade_url = f"/upgrade/{author}"
        raise HTTPException(
            429,
            detail={
                "type": "limit",
                "tier": tier,
                "used": used,
                "daily_limit": limit,
                "upgrade_url": upgrade_url,
                "message": f"Daily limit of {limit} questions reached. Upgrade for more.",
            },
        )

    # Resolve model: request override (validated) > tier default
    if body.model:
        if body.model not in cfg.AVAILABLE_MODELS:
            raise HTTPException(400, f"Model '{body.model}' is not available")
        chat_model = body.model
    else:
        chat_model = cfg.MODELS.get(tier, cfg.MODELS["free"])

    # Tier-specific settings
    top_k        = cfg.TOP_K[tier]
    history_turns = cfg.HISTORY_TURNS[tier]
    trimmed_history = list(body.history[-(history_turns * 2):]) if history_turns > 0 else []

    # RAG retrieval
    hits = _rag_search(body.query, author, top_k=top_k)
    context = _build_context(hits)
    sources = [
        {"title": h["title"], "url": h["url"], "score": round(h["score"], 3), "date": h["date"]}
        for h in hits
    ]

    async def event_stream() -> AsyncIterator[str]:
        yield f"data: {json.dumps({'type': 'sources', 'sources': sources})}\n\n"

        input_tok = output_tok = 0
        async for event_type, payload in _stream_claude(
            author, body.query, context, trimmed_history,
            use_thinking=(tier == "power"),
            model=chat_model,
        ):
            if event_type == "token":
                yield f"data: {json.dumps({'type': 'token', 'text': payload['text']})}\n\n"
            elif event_type == "done":
                usage = payload.get("usage", {})
                input_tok = usage.get("input_tokens", 0)
                output_tok = usage.get("output_tokens", 0)
                yield f"data: {json.dumps({'type': 'done', 'usage': usage})}\n\n"
                asyncio.create_task(log_usage(author, ip, input_tok, output_tok))
            elif event_type == "error":
                yield f"data: {json.dumps({'type': 'error', 'message': payload['message']})}\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")


@app.post("/admin/index/{author}")
async def trigger_index(author: str, x_admin_key: str = Header(None)):
    if not _check_admin_key(x_admin_key):
        raise HTTPException(403, "Invalid admin key")
    author = _sanitize_handle(author)

    data_path = Path(DATA_DIR) / author
    if not data_path.exists():
        raise HTTPException(404, f"No data directory for author '{author}'")

    # Run indexer in background thread (CPU-bound)
    import asyncio
    from indexer import build_index

    def _rebuild():
        idx, chunks = build_index(author, DATA_DIR, INDEX_DIR, rebuild=False)
        _invalidate_cache(author)
        return idx.ntotal, len(chunks)

    loop = asyncio.get_running_loop()
    ntotal, nchunks = await loop.run_in_executor(None, _rebuild)
    _get_index(author)  # warm the cache

    return {"author": author, "vectors": ntotal, "chunks": nchunks, "status": "indexed"}


# ── Usage stats (admin) ────────────────────────────────────────────────────────

@app.get("/admin/usage")
async def usage_stats(x_admin_key: str = Header(None), days: int = 7):
    if not _check_admin_key(x_admin_key):
        raise HTTPException(403, "Invalid admin key")

    since = time.time() - days * 86400
    async with aiosqlite.connect(DB_PATH) as db:
        async with db.execute(
            """SELECT author, COUNT(*) as requests,
                      SUM(input_tok) as input_tok, SUM(output_tok) as output_tok
               FROM usage WHERE ts > ? GROUP BY author ORDER BY requests DESC""",
            (since,),
        ) as cur:
            rows = await cur.fetchall()

    return {
        "days": days,
        "by_author": [
            {"author": r[0], "requests": r[1], "input_tokens": r[2], "output_tokens": r[3]}
            for r in rows
        ],
    }


# ── Stripe ────────────────────────────────────────────────────────────────────

class UpgradeRequest(BaseModel):
    tier: str = Field(..., pattern="^(pro|power)$")
    email: Optional[str] = None  # pre-fill Stripe checkout


@app.post("/upgrade/{author}")
async def start_upgrade(author: str, body: UpgradeRequest):
    """Create a Stripe checkout session and return the redirect URL."""
    if not cfg.STRIPE_SECRET_KEY:
        raise HTTPException(503, "Payments not configured yet. Contact the author.")

    for label, url in (("STRIPE_SUCCESS_URL", cfg.STRIPE_SUCCESS_URL), ("STRIPE_CANCEL_URL", cfg.STRIPE_CANCEL_URL)):
        if not url or "localhost" in url or "127.0.0.1" in url:
            raise HTTPException(503, f"{label} must be set to a public HTTPS URL in production")

    price_cfg = cfg.STRIPE_PRICES.get(body.tier)
    if not price_cfg or not price_cfg["price_id"]:
        raise HTTPException(400, f"No Stripe price configured for tier '{body.tier}'")

    try:
        import stripe
        stripe.api_key = cfg.STRIPE_SECRET_KEY
        session = stripe.checkout.Session.create(
            mode="subscription",
            line_items=[{"price": price_cfg["price_id"], "quantity": 1}],
            success_url=cfg.STRIPE_SUCCESS_URL + f"?author={author}&tier={body.tier}",
            cancel_url=cfg.STRIPE_CANCEL_URL + f"?author={author}",
            customer_email=body.email or None,
            metadata={"author": author, "tier": body.tier},
        )
        return {"checkout_url": session.url}
    except Exception as e:
        log.exception("Stripe checkout error")
        raise HTTPException(500, f"Payment setup failed: {e}")


@app.post("/stripe/webhook")
async def stripe_webhook(request: Request):
    """Handle Stripe subscription events to update author tier."""
    if not cfg.STRIPE_WEBHOOK_SECRET:
        raise HTTPException(503, "Webhook not configured")

    payload = await request.body()
    sig = request.headers.get("stripe-signature", "")

    try:
        import stripe
        stripe.api_key = cfg.STRIPE_SECRET_KEY
        event = stripe.Webhook.construct_event(payload, sig, cfg.STRIPE_WEBHOOK_SECRET)
    except Exception as e:
        raise HTTPException(400, f"Webhook signature invalid: {e}")

    ev_type = event["type"]
    data = event["data"]["object"]

    if ev_type in ("customer.subscription.created", "customer.subscription.updated"):
        author   = data.get("metadata", {}).get("author", "")
        tier     = data.get("metadata", {}).get("tier", "pro")
        customer = data.get("customer", "")
        sub_id   = data.get("id", "")
        status   = data.get("status", "")
        if author and status in ("active", "trialing"):
            await _set_author_tier(author, tier, customer, sub_id)
            log.info(f"Upgraded {author} → {tier}")

    elif ev_type == "customer.subscription.deleted":
        author = data.get("metadata", {}).get("author", "")
        if author:
            await _set_author_tier(author, "free")
            log.info(f"Downgraded {author} → free (subscription cancelled)")

    return {"received": True}


@app.get("/upgrade/success")
async def upgrade_success(author: str = "", tier: str = ""):
    return FileResponse(STATIC_DIR / "upgrade_success.html")


@app.get("/upgrade/cancel")
async def upgrade_cancel(author: str = ""):
    return FileResponse(STATIC_DIR / "onboard.html")


@app.get("/tiers")
async def list_tiers():
    """Return tier config for the frontend upgrade modal."""
    return {
        "tiers": [
            {
                "id": "free",
                "label": "Free",
                "daily_questions": cfg.DAILY_QUESTIONS["free"],
                "history_turns":   cfg.HISTORY_TURNS["free"],
                "top_k":           cfg.TOP_K["free"],
                "price_usd":       0,
            },
            {
                "id": "pro",
                "label": cfg.STRIPE_PRICES["pro"]["label"],
                "description": cfg.STRIPE_PRICES["pro"]["description"],
                "daily_questions": cfg.DAILY_QUESTIONS["pro"],
                "history_turns":   cfg.HISTORY_TURNS["pro"],
                "top_k":           cfg.TOP_K["pro"],
                "price_usd":       cfg.STRIPE_PRICES["pro"]["monthly_usd"],
                "available":       bool(cfg.STRIPE_PRICES["pro"]["price_id"]),
            },
            {
                "id": "power",
                "label": cfg.STRIPE_PRICES["power"]["label"],
                "description": cfg.STRIPE_PRICES["power"]["description"],
                "daily_questions": cfg.DAILY_QUESTIONS["power"],
                "history_turns":   cfg.HISTORY_TURNS["power"],
                "top_k":           cfg.TOP_K["power"],
                "price_usd":       cfg.STRIPE_PRICES["power"]["monthly_usd"],
                "available":       bool(cfg.STRIPE_PRICES["power"]["price_id"]),
            },
        ]
    }


# ── Onboarding rate limit ─────────────────────────────────────────────────────

async def _check_onboard_rate_limit(ip: str) -> bool:
    """Return True and increment counter if the IP is within the daily onboard limit."""
    if cfg.ONBOARD_DAILY_LIMIT == 0:
        return True
    import datetime
    today = datetime.datetime.utcnow().strftime("%Y-%m-%d")
    async with aiosqlite.connect(DB_PATH) as db:
        async with db.execute(
            "SELECT count FROM daily_questions WHERE date=? AND author='__onboard__' AND ip=?",
            (today, ip),
        ) as cur:
            row = await cur.fetchone()
        used = row[0] if row else 0
        if used >= cfg.ONBOARD_DAILY_LIMIT:
            return False
        if row:
            await db.execute(
                "UPDATE daily_questions SET count=count+1 WHERE date=? AND author='__onboard__' AND ip=?",
                (today, ip),
            )
        else:
            await db.execute(
                "INSERT INTO daily_questions (date, author, ip, count) VALUES (?,?,?,1)",
                (today, "__onboard__", ip),
            )
        await db.commit()
    return True


# ── Onboarding job management ──────────────────────────────────────────────────

# job_id = author handle
# status: queued | fetching | indexing | done | error
_jobs: dict[str, dict] = {}
_onboard_semaphore = asyncio.Semaphore(int(os.getenv("MAX_ONBOARD_JOBS", "2")))


async def _do_onboard(
    author: str,
    newsletter_url: str = "",
    display_name: str = "",
    max_articles: int = 200,
) -> None:
    """Core discovery → fetch → index pipeline. Caller must set _jobs[author] first."""
    from fetcher import (fetch_article, save_as_markdown,
                         _urls_from_listing_page, _urls_from_search)
    from indexer import build_index
    import time as _time

    async with _onboard_semaphore:
        loop = asyncio.get_running_loop()
        try:
            _jobs[author]["status"] = "discovering"
            candidate_urls: list[str] = []

            if newsletter_url:
                urls = await loop.run_in_executor(
                    None, lambda: _urls_from_listing_page(newsletter_url, max_articles=max_articles)
                )
                candidate_urls.extend(urls)

            if len(candidate_urls) < 5 and author:
                search_urls = await loop.run_in_executor(
                    None, lambda: _urls_from_search(author, max_results=max_articles)
                )
                seen = set(candidate_urls)
                for u in search_urls:
                    if u not in seen:
                        candidate_urls.append(u)
                        seen.add(u)

            candidate_urls = candidate_urls[:max_articles]
            _jobs[author]["articles_found"] = len(candidate_urls)

            if not candidate_urls:
                msg = "No articles found. Try providing a newsletter URL or check the handle."
                _jobs[author]["status"] = "error"
                _jobs[author]["error"] = msg
                asyncio.create_task(_log_onboard_error(author, "discovering", msg))
                asyncio.create_task(_persist_job(_jobs[author]))
                return

            _jobs[author]["status"] = "fetching"
            articles: list[dict] = []
            fetch_errors: list[tuple[str, str]] = []

            def _fetch_all():
                for i, url in enumerate(candidate_urls):
                    try:
                        art = fetch_article(url)
                        if art:
                            articles.append(art)
                        else:
                            fetch_errors.append((url, "No content extracted"))
                    except Exception as fetch_exc:
                        fetch_errors.append((url, str(fetch_exc)))
                    _jobs[author]["articles_fetched"] = len(articles)
                    if i < len(candidate_urls) - 1:
                        _time.sleep(0.3)

            await loop.run_in_executor(None, _fetch_all)

            for url, err_msg in fetch_errors:
                asyncio.create_task(_log_onboard_error(author, "fetching", err_msg, url))
            if fetch_errors:
                _jobs[author]["fetch_errors"] = len(fetch_errors)

            if not articles:
                msg = f"Could not extract content from any of {len(candidate_urls)} articles."
                _jobs[author]["status"] = "error"
                _jobs[author]["error"] = msg
                asyncio.create_task(_persist_job(_jobs[author]))
                return

            await loop.run_in_executor(
                None, lambda: save_as_markdown(articles, DATA_DIR, author)
            )

            _jobs[author]["status"] = "indexing"
            idx, chunks = await loop.run_in_executor(
                None, lambda: build_index(author, DATA_DIR, INDEX_DIR, rebuild=True)
            )
            _invalidate_cache(author)
            _jobs[author]["chunks"] = len(chunks)
            _jobs[author]["status"] = "done"
            _jobs[author]["finished_at"] = time.time()
            asyncio.create_task(_persist_job(_jobs[author]))

        except Exception as e:
            log.exception(f"Onboard job failed for {author}")
            _jobs[author]["status"] = "error"
            _jobs[author]["error"] = str(e)
            asyncio.create_task(_log_onboard_error(author, _jobs[author].get("status", "unknown"), str(e)))
            asyncio.create_task(_persist_job(_jobs[author]))


async def _auto_refresh_loop() -> None:
    """Background task: re-index authors periodically to pick up new articles."""
    refresh_hours = cfg.AUTO_REFRESH_HOURS
    if refresh_hours <= 0:
        log.info("Auto-refresh disabled (AUTO_REFRESH_HOURS=0)")
        return

    await asyncio.sleep(120)  # brief delay after startup
    log.info(f"Auto-refresh started — checking every hour, refresh threshold: {refresh_hours}h")

    while True:
        try:
            cutoff = time.time() - refresh_hours * 3600
            async with aiosqlite.connect(DB_PATH) as db:
                async with db.execute(
                    """SELECT author,
                              MAX(newsletter_url) AS newsletter_url,
                              MAX(display_name)   AS display_name
                       FROM onboard_jobs
                       WHERE status = 'done' AND finished_at < ?
                       GROUP BY author""",
                    (cutoff,),
                ) as cur:
                    due = await cur.fetchall()

            for author, newsletter_url, display_name in due:
                if author in _jobs and _jobs[author].get("status") in (
                    "queued", "fetching", "indexing", "discovering"
                ):
                    continue
                log.info(f"Auto-refresh: re-indexing {author}")
                _jobs[author] = {
                    "author": author,
                    "status": "queued",
                    "display_name": display_name or author,
                    "newsletter_url": newsletter_url or "",
                    "started_at": time.time(),
                    "articles_found": 0,
                    "articles_fetched": 0,
                    "chunks": 0,
                    "error": None,
                }
                asyncio.create_task(_do_onboard(
                    author,
                    newsletter_url=newsletter_url or "",
                    display_name=display_name or author,
                ))
        except Exception as exc:
            log.warning(f"Auto-refresh loop error: {exc}")

        await asyncio.sleep(3600)


class OnboardRequest(BaseModel):
    author: str = Field(..., min_length=1, max_length=100)
    newsletter_url: Optional[str] = None
    display_name: Optional[str] = None
    max_articles: int = Field(default=20, ge=1, le=500)


def _extract_handle(raw: str) -> str:
    """Extract LinkedIn handle from a full URL or return the raw value."""
    raw = raw.strip().rstrip("/")
    m = re.search(r"linkedin\.com/in/([^/?&#]+)", raw, re.IGNORECASE)
    handle = m.group(1).lower() if m else raw.lower().replace(" ", "-")
    # Strip anything that isn't alphanumeric, hyphen, or underscore (prevents path traversal)
    return re.sub(r"[^a-z0-9\-_]", "", handle)


@app.post("/onboard")
async def start_onboard(body: OnboardRequest, request: Request):
    ip = _real_ip(request)
    if not await _check_onboard_rate_limit(ip):
        raise HTTPException(429, f"Too many onboarding requests from this IP. Try again tomorrow.")
    author = _extract_handle(body.author)
    if author in _jobs and _jobs[author].get("status") in ("queued", "fetching", "indexing"):
        return {"author": author, "status": _jobs[author]["status"], "message": "Job already running"}

    _jobs[author] = {
        "author": author,
        "status": "queued",
        "display_name": body.display_name or author,
        "newsletter_url": body.newsletter_url or "",
        "started_at": time.time(),
        "articles_found": 0,
        "articles_fetched": 0,
        "chunks": 0,
        "error": None,
    }
    asyncio.create_task(_do_onboard(
        author,
        newsletter_url=body.newsletter_url or "",
        display_name=body.display_name or author,
        max_articles=body.max_articles,
    ))
    return {"author": author, "status": "queued"}


@app.get("/admin/jobs")
async def list_jobs(x_admin_key: str = Header(None)):
    if not _check_admin_key(x_admin_key):
        raise HTTPException(403, "Invalid admin key")
    return {"jobs": {k: v for k, v in _jobs.items()}}


@app.get("/admin/errors")
async def list_errors(x_admin_key: str = Header(None), limit: int = 50):
    if not _check_admin_key(x_admin_key):
        raise HTTPException(403, "Invalid admin key")

    async with aiosqlite.connect(DB_PATH) as db:
        async with db.execute(
            """SELECT ts, author, stage, url, message
               FROM onboard_errors ORDER BY ts DESC LIMIT ?""",
            (limit,),
        ) as cur:
            rows = await cur.fetchall()

        async with db.execute(
            """SELECT author, display_name, status, started_at, finished_at,
                      articles_found, articles_fetched, chunks, error
               FROM onboard_jobs ORDER BY started_at DESC LIMIT ?""",
            (limit,),
        ) as cur:
            job_rows = await cur.fetchall()

    return {
        "errors": [
            {"ts": r[0], "author": r[1], "stage": r[2], "url": r[3], "message": r[4]}
            for r in rows
        ],
        "job_history": [
            {
                "author": r[0], "display_name": r[1], "status": r[2],
                "started_at": r[3], "finished_at": r[4],
                "articles_found": r[5], "articles_fetched": r[6],
                "chunks": r[7], "error": r[8],
            }
            for r in job_rows
        ],
    }


@app.get("/onboard/{author}/status")
async def onboard_status(author: str):
    job = _jobs.get(author)
    if not job:
        # Fall back to persisted job history (survives server restarts)
        async with aiosqlite.connect(DB_PATH) as db:
            async with db.execute(
                """SELECT status, display_name, started_at, finished_at,
                          articles_found, articles_fetched, chunks, error
                   FROM onboard_jobs WHERE author=? ORDER BY started_at DESC LIMIT 1""",
                (author,),
            ) as cur:
                row = await cur.fetchone()
        if row:
            return {
                "author": author,
                "status": row[0], "display_name": row[1],
                "started_at": row[2], "finished_at": row[3],
                "articles_found": row[4], "articles_fetched": row[5],
                "chunks": row[6], "error": row[7],
            }
        # Check if pre-indexed without a recorded job
        idx, chunks = _get_index(author)
        if idx is not None:
            return {"author": author, "status": "done", "chunks": idx.ntotal,
                    "articles_fetched": len({c.title for c in chunks})}
        raise HTTPException(404, f"No onboarding job found for '{author}'")
    return {"author": author, **job}


# ── LinkedIn OAuth endpoints ───────────────────────────────────────────────────

@app.get("/auth/linkedin")
async def linkedin_auth(next: str = "/"):
    if not cfg.LINKEDIN_CLIENT_ID:
        raise HTTPException(503, "LinkedIn OAuth not configured. Set LINKEDIN_CLIENT_ID and LINKEDIN_CLIENT_SECRET in .env")

    # Prune stale states before adding a new one
    now = time.time()
    stale = [k for k, v in _oauth_states.items() if now - v["created_at"] > 600]
    for k in stale:
        del _oauth_states[k]

    state = secrets.token_urlsafe(16)
    _oauth_states[state] = {"created_at": now, "next_url": next if next.startswith("/") else "/"}

    url = LINKEDIN_AUTH_URL + "?" + urlencode({
        "response_type": "code",
        "client_id": cfg.LINKEDIN_CLIENT_ID,
        "redirect_uri": cfg.LINKEDIN_REDIRECT_URI,
        "scope": "openid profile email",
        "state": state,
    })
    return RedirectResponse(url)


@app.get("/auth/linkedin/callback")
async def linkedin_callback(
    code: Optional[str] = None,
    state: Optional[str] = None,
    error: Optional[str] = None,
    error_description: Optional[str] = None,
):
    if error:
        desc = error_description or error
        return RedirectResponse(f"/?auth_error={desc}")

    if not code or not state:
        raise HTTPException(400, "Missing code or state parameter")

    state_data = _oauth_states.pop(state, None)
    if not state_data:
        raise HTTPException(400, "Invalid or expired OAuth state. Please try signing in again.")
    if time.time() - state_data["created_at"] > 600:
        raise HTTPException(400, "OAuth state expired. Please try signing in again.")

    async with httpx.AsyncClient(timeout=15.0) as client:
        token_resp = await client.post(LINKEDIN_TOKEN_URL, data={
            "grant_type": "authorization_code",
            "code": code,
            "redirect_uri": cfg.LINKEDIN_REDIRECT_URI,
            "client_id": cfg.LINKEDIN_CLIENT_ID,
            "client_secret": cfg.LINKEDIN_CLIENT_SECRET,
        }, headers={"Content-Type": "application/x-www-form-urlencoded"})

        if token_resp.status_code != 200:
            log.warning(f"LinkedIn token exchange failed: {token_resp.status_code} {token_resp.text}")
            raise HTTPException(502, "LinkedIn token exchange failed. Please try again.")

        access_token = token_resp.json().get("access_token", "")
        if not access_token:
            raise HTTPException(502, "No access token in LinkedIn response")

        userinfo_resp = await client.get(
            LINKEDIN_USERINFO_URL,
            headers={"Authorization": f"Bearer {access_token}"},
        )
        if userinfo_resp.status_code != 200:
            log.warning(f"LinkedIn userinfo failed: {userinfo_resp.status_code} {userinfo_resp.text}")
            raise HTTPException(502, "Failed to fetch LinkedIn profile")

        profile = userinfo_resp.json()

    linkedin_id = profile.get("sub", "")
    name        = profile.get("name", "")
    email       = profile.get("email", "")
    picture     = profile.get("picture", "")

    session_id = await _create_session(linkedin_id, name, email, picture)

    next_url = state_data.get("next_url", "/")
    response = RedirectResponse(next_url, status_code=303)
    response.set_cookie(
        SESSION_COOKIE,
        session_id,
        max_age=cfg.SESSION_TTL_SECONDS,
        httponly=True,
        samesite="lax",
        secure=_is_secure_context(),
    )
    return response


@app.get("/auth/me")
async def auth_me(request: Request):
    session_id = request.cookies.get(SESSION_COOKIE)
    user = await _get_session(session_id) if session_id else None
    if not user:
        return {"authenticated": False}
    return {"authenticated": True, **user}


@app.post("/auth/logout")
async def auth_logout(request: Request):
    session_id = request.cookies.get(SESSION_COOKIE)
    if session_id:
        await _delete_session(session_id)
    response = RedirectResponse("/", status_code=303)
    response.delete_cookie(SESSION_COOKIE, samesite="lax")
    return response


# ── Static UI pages ────────────────────────────────────────────────────────────

STATIC_DIR = Path(__file__).parent / "static"
STATIC_DIR.mkdir(exist_ok=True)


@app.get("/admin/dashboard")
async def admin_dashboard():
    return FileResponse(STATIC_DIR / "admin.html")


@app.get("/")
async def root():
    return FileResponse(STATIC_DIR / "onboard.html")


@app.get("/c/{author}")
async def chat_page(author: str):
    return FileResponse(STATIC_DIR / "chat.html")


app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
