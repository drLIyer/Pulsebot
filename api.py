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
  data: {"type": "sources",    "sources": [{title, url, score}, ...]}
  data: {"type": "confidence", "score": 0.0-1.0, "low": bool}
  data: {"type": "context",    "text": "..."}   # only when include_context=true in request
  data: {"type": "token",      "text": "..."}
  data: {"type": "done",       "usage": {input_tokens, output_tokens}}
  data: {"type": "error",      "message": "..."}
  data: {"type": "limit",      "tier": "free", "daily_limit": 10, "upgrade_url": "..."}

Run:
  ANTHROPIC_API_KEY=sk-... uvicorn api:app --reload --port 8000
"""

import math
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

from indexer import load_index, get_model, load_bm25_index, tokenize
from sentence_transformers import CrossEncoder
from models import Chunk
import faiss
import numpy as np
import config as cfg
import prompts as _prompts
import cache as _cache_mod
import guardrails as _guardrails

# ── Config ─────────────────────────────────────────────────────────────────────

INDEX_DIR       = os.getenv("INDEX_DIR",   "./indexes")
DATA_DIR        = os.getenv("DATA_DIR",    "./data")
DB_PATH         = os.getenv("DB_PATH",     "./pulsebot.db")
ADMIN_KEY       = os.getenv("ADMIN_KEY",   "changeme")
ALLOWED_ORIGINS = [o.strip() for o in os.getenv("ALLOWED_ORIGINS", "*").split(",")]

RERANKER_MODEL    = os.getenv("RERANKER_MODEL",  "cross-encoder/ms-marco-MiniLM-L-6-v2")
HYBRID_CANDIDATES = int(os.getenv("HYBRID_CANDIDATES", "20"))

RETRY_MAX      = int(os.getenv("RETRY_MAX",      "2"))
RETRY_BACKOFF  = [float(x) for x in os.getenv("RETRY_BACKOFF",  "1,3").split(",")]
FALLBACK_MODEL = os.getenv("FALLBACK_MODEL",  "claude-sonnet-5-5")
STREAM_TIMEOUT = int(os.getenv("STREAM_TIMEOUT", "90"))

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

            -- Per-request traces for observability and prompt debugging
            CREATE TABLE IF NOT EXISTS traces (
                id             TEXT PRIMARY KEY,
                author         TEXT NOT NULL,
                ip             TEXT,
                query          TEXT,
                answer         TEXT,
                confidence     REAL,
                prompt_version TEXT,
                model          TEXT,
                input_tokens   INTEGER DEFAULT 0,
                output_tokens  INTEGER DEFAULT 0,
                ttft_ms        INTEGER DEFAULT 0,
                total_ms       INTEGER DEFAULT 0,
                cache_hit      INTEGER DEFAULT 0,
                error          TEXT,
                created_at     TEXT DEFAULT (datetime('now'))
            );

            -- Reader feedback (thumbs up/down) linked to traces
            CREATE TABLE IF NOT EXISTS feedback (
                id             TEXT PRIMARY KEY,
                trace_id       TEXT,
                author         TEXT NOT NULL,
                query          TEXT,
                answer_preview TEXT,
                rating         INTEGER NOT NULL,   -- 1 = up, -1 = down
                note           TEXT,
                created_at     TEXT DEFAULT (datetime('now'))
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
_bm25_cache:  dict[str, object] = {}
_reranker = None
_sem_cache = _cache_mod._cache


def _get_index(author: str) -> tuple[Optional[faiss.Index], list[Chunk]]:
    if author not in _index_cache:
        idx, chunks = load_index(INDEX_DIR, author)
        if idx is not None:
            _index_cache[author] = (idx, chunks)
        else:
            return None, []
    return _index_cache[author]


def _get_bm25(author: str):
    if author not in _bm25_cache:
        bm25 = load_bm25_index(INDEX_DIR, author)
        if bm25 is not None:
            _bm25_cache[author] = bm25
    return _bm25_cache.get(author)


def _get_reranker():
    global _reranker
    if _reranker is None:
        log.info(f"Loading reranker {RERANKER_MODEL}…")
        _reranker = CrossEncoder(RERANKER_MODEL)
    return _reranker


def _rrf(ranked_lists: list[list[int]], k: int = 60) -> list[tuple[int, float]]:
    """Reciprocal Rank Fusion — merges multiple ranked lists without score normalisation."""
    scores: dict[int, float] = {}
    for ranked in ranked_lists:
        for rank, doc_id in enumerate(ranked, start=1):
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank)
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)


def _invalidate_cache(author: str) -> None:
    _index_cache.pop(author, None)
    _bm25_cache.pop(author, None)
    _sem_cache.invalidate(author)

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


_COST_PER_MTok = {
    "claude-haiku-4-5-20251001": {"input": 0.80,  "output": 4.00},
    "claude-sonnet-5-5":         {"input": 3.00,  "output": 15.00},
    "claude-opus-5-5":           {"input": 15.00, "output": 75.00},
}


def _compute_cost(model: str, input_tokens: int, output_tokens: int) -> float:
    prices = _COST_PER_MTok.get(model, {"input": 3.00, "output": 15.00})
    return (input_tokens * prices["input"] + output_tokens * prices["output"]) / 1_000_000


async def _log_trace(
    trace_id:      str,
    author:        str,
    ip:            str,
    query:         str,
    answer:        str,
    confidence:    float,
    prompt_version: str,
    model:         str,
    input_tokens:  int,
    output_tokens: int,
    ttft_ms:       int,
    total_ms:      int,
    cache_hit:     bool = False,
    error:         str  = "",
) -> None:
    try:
        async with aiosqlite.connect(DB_PATH) as db:
            await db.execute(
                """INSERT INTO traces
                   (id, author, ip, query, answer, confidence, prompt_version, model,
                    input_tokens, output_tokens, ttft_ms, total_ms, cache_hit, error)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                (trace_id, author, ip, query, answer[:2000], confidence,
                 prompt_version, model, input_tokens, output_tokens,
                 ttft_ms, total_ms, int(cache_hit), error),
            )
            await db.commit()
    except Exception as e:
        log.warning(f"trace log error: {e}")


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
    if idx is None or not chunks:
        return []

    candidates = min(HYBRID_CANDIDATES, len(chunks))

    # ── Vector search ──────────────────────────────────────────────────────────
    model = get_model()
    q_emb = model.encode([query], convert_to_numpy=True).astype(np.float32)
    faiss.normalize_L2(q_emb)
    _, vec_indices = idx.search(q_emb, candidates)
    vec_ranked = [int(i) for i in vec_indices[0] if i >= 0]

    # ── BM25 search + RRF fusion ───────────────────────────────────────────────
    bm25 = _get_bm25(author)
    if bm25 is not None:
        bm25_scores = bm25.get_scores(tokenize(query))
        bm25_ranked = sorted(range(len(chunks)), key=lambda i: bm25_scores[i], reverse=True)[:candidates]
        candidate_ids = [doc_id for doc_id, _ in _rrf([vec_ranked, bm25_ranked])][:candidates]
    else:
        candidate_ids = vec_ranked

    # ── Cross-encoder reranking ────────────────────────────────────────────────
    reranker = _get_reranker()
    pairs = [[query, chunks[i].text] for i in candidate_ids]
    rerank_scores = reranker.predict(pairs)
    reranked = sorted(zip(candidate_ids, rerank_scores), key=lambda x: x[1], reverse=True)

    return [
        {
            "score": float(score),
            "title": chunks[i].title,
            "url": chunks[i].url,
            "date": chunks[i].date,
            "chunk_index": chunks[i].chunk_index,
            "text": chunks[i].text,
        }
        for i, score in reranked[:top_k]
    ]

def _retrieval_confidence(hits: list[dict]) -> float:
    """Sigmoid of the top cross-encoder score → [0, 1].  0 when no hits."""
    if not hits:
        return 0.0
    return 1.0 / (1.0 + math.exp(-hits[0]["score"]))


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
    confidence_pct: float = 100.0,
) -> AsyncIterator[tuple[str, dict]]:
    """
    Yields (event_type, payload) tuples: "token" | "done" | "error"
    Retries up to RETRY_MAX times on 5xx / connection errors (before first token).
    Switches to FALLBACK_MODEL on the final retry.
    Enforces STREAM_TIMEOUT seconds total.
    """
    system, prompt_version = _prompts.get_system_prompt(author, context, confidence_pct)

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

    client  = _make_anthropic_client()
    started = False

    for attempt in range(RETRY_MAX + 1):
        try:
            async with asyncio.timeout(STREAM_TIMEOUT):
                async with client.messages.stream(**stream_kwargs) as stream:
                    async for text in stream.text_stream:
                        started = True
                        yield "token", {"text": text}
                    final = await stream.get_final_message()
                    usage = final.usage
                    yield "done", {
                        "usage": {
                            "input_tokens":  usage.input_tokens,
                            "output_tokens": usage.output_tokens,
                            "model":         stream_kwargs["model"],
                        },
                        "prompt_version": prompt_version,
                    }
                    return
        except asyncio.TimeoutError as exc:
            if started or attempt >= RETRY_MAX:
                yield "error", {"message": "Response timed out"}
                return
            log.warning(f"Stream timeout attempt {attempt + 1}/{RETRY_MAX + 1}")
            err: Exception = exc
        except anthropic.APIStatusError as exc:
            if started or exc.status_code < 500 or attempt >= RETRY_MAX:
                yield "error", {"message": str(exc)}
                return
            log.warning(f"API {exc.status_code} attempt {attempt + 1}/{RETRY_MAX + 1}")
            err = exc
        except anthropic.APIConnectionError as exc:
            if started or attempt >= RETRY_MAX:
                yield "error", {"message": str(exc)}
                return
            log.warning(f"Connection error attempt {attempt + 1}/{RETRY_MAX + 1}")
            err = exc

        backoff = RETRY_BACKOFF[min(attempt, len(RETRY_BACKOFF) - 1)]
        await asyncio.sleep(backoff)

        if attempt == RETRY_MAX - 1 and not use_thinking:
            log.warning(f"Switching to fallback model {FALLBACK_MODEL}")
            stream_kwargs["model"] = FALLBACK_MODEL

    yield "error", {"message": f"Request failed after {RETRY_MAX + 1} attempts"}

# ── Request / response models ──────────────────────────────────────────────────

class ChatRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=2000)
    history: list[dict] = Field(default_factory=list)
    model: Optional[str] = Field(default=None, max_length=100)
    include_context: bool = Field(default=False)  # evals: include retrieved chunks in stream

# ── Endpoints ──────────────────────────────────────────────────────────────────

_MODEL_META = {
    "claude-sonnet-5-5": {"label": "Sonnet 5.5", "description": "Fast · lower cost"},
    "claude-opus-5-5":   {"label": "Opus 5.5",   "description": "Most capable"},
}

@app.get("/health")
async def health():
    return {
        "status": "ok",
        "models": cfg.MODELS,
        "cache": {
            "enabled":  _cache_mod.CACHE_ENABLED,
            "size":     _sem_cache.size,
            "hit_rate": _sem_cache.hit_rate,
        },
        "prompt_versions": _prompts.list_versions(),
    }


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

    # Guardrails: input check
    safe, block_reason = _guardrails.check_input(body.query)
    if not safe:
        raise HTTPException(400, {"type": "blocked", "message": block_reason})

    # Check index exists
    idx, _ = _get_index(author)
    if idx is None:
        raise HTTPException(404, f"No newsletter index found for author '{author}'")

    # Resolve tier
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

    # Resolve model
    if body.model:
        if body.model not in cfg.AVAILABLE_MODELS:
            raise HTTPException(400, f"Model '{body.model}' is not available")
        chat_model = body.model
    else:
        chat_model = cfg.MODELS.get(tier, cfg.MODELS["free"])

    # Tier-specific settings
    top_k           = cfg.TOP_K[tier]
    history_turns   = cfg.HISTORY_TURNS[tier]
    trimmed_history = list(body.history[-(history_turns * 2):]) if history_turns > 0 else []

    # Semantic cache lookup (skip when caller wants chunks for evals)
    trace_id   = secrets.token_hex(8)
    t_start    = time.monotonic()
    cache_hit  = False

    if not body.include_context:
        cached = _sem_cache.get(body.query, author)
        if cached is not None:
            cache_hit = True
            cached_sources     = cached.get("sources", [])
            cached_answer      = cached.get("answer", "")
            cached_confidence  = cached.get("confidence", 1.0)
            cached_conf_pct    = cached_confidence * 100

            async def cached_stream() -> AsyncIterator[str]:
                yield f"data: {json.dumps({'type': 'sources', 'sources': cached_sources})}\n\n"
                yield f"data: {json.dumps({'type': 'confidence', 'score': round(cached_confidence, 3), 'low': cached_confidence < 0.5})}\n\n"
                yield f"data: {json.dumps({'type': 'cache', 'similarity': cached.get('cache_similarity', 1.0)})}\n\n"
                yield f"data: {json.dumps({'type': 'token', 'text': cached_answer})}\n\n"
                yield f"data: {json.dumps({'type': 'done', 'usage': {'input_tokens': 0, 'output_tokens': 0, 'model': 'cache'}})}\n\n"

            total_ms = (time.monotonic() - t_start) * 1000
            asyncio.create_task(_log_trace(
                trace_id, author, ip, body.query, cached_answer, cached_conf_pct,
                "cache", "cache", 0, 0, 0.0, total_ms, cache_hit=True,
            ))
            return StreamingResponse(cached_stream(), media_type="text/event-stream")

    # RAG retrieval
    hits       = _rag_search(body.query, author, top_k=top_k)
    context    = _build_context(hits)
    sources    = [
        {"title": h["title"], "url": h["url"], "score": round(h["score"], 3), "date": h["date"]}
        for h in hits
    ]
    confidence     = _retrieval_confidence(hits)
    confidence_pct = confidence * 100

    async def event_stream() -> AsyncIterator[str]:
        yield f"data: {json.dumps({'type': 'sources', 'sources': sources})}\n\n"
        yield f"data: {json.dumps({'type': 'confidence', 'score': round(confidence, 3), 'low': confidence < 0.5})}\n\n"
        if body.include_context:
            chunk_payloads = [{"title": h["title"], "url": h["url"], "text": h["text"]} for h in hits]
            yield f"data: {json.dumps({'type': 'context', 'text': context, 'chunks': chunk_payloads})}\n\n"

        answer_parts: list[str] = []
        input_tok = output_tok = 0
        prompt_version = "v1"
        ttft_ms   = 0.0
        error_msg = ""
        first_token = True

        async for event_type, payload in _stream_claude(
            author, body.query, context, trimmed_history,
            use_thinking=(tier == "power"),
            model=chat_model,
            confidence_pct=confidence_pct,
        ):
            if event_type == "token":
                if first_token:
                    ttft_ms    = (time.monotonic() - t_start) * 1000
                    first_token = False
                answer_parts.append(payload["text"])
                yield f"data: {json.dumps({'type': 'token', 'text': payload['text']})}\n\n"
            elif event_type == "done":
                usage          = payload.get("usage", {})
                input_tok      = usage.get("input_tokens", 0)
                output_tok     = usage.get("output_tokens", 0)
                prompt_version = payload.get("prompt_version", "v1")
                yield f"data: {json.dumps({'type': 'done', 'usage': usage})}\n\n"
                asyncio.create_task(log_usage(author, ip, input_tok, output_tok))
            elif event_type == "error":
                error_msg = payload["message"]
                yield f"data: {json.dumps({'type': 'error', 'message': error_msg})}\n\n"

        # Post-stream: output guardrail, cache store, trace
        answer   = "".join(answer_parts)
        total_ms = (time.monotonic() - t_start) * 1000

        out_safe, out_reason = _guardrails.check_output(answer)
        if not out_safe and answer:
            yield f"data: {json.dumps({'type': 'error', 'message': out_reason})}\n\n"
            answer = ""

        if answer and not body.include_context:
            _sem_cache.put(body.query, author, answer, sources, confidence)

        asyncio.create_task(_log_trace(
            trace_id, author, ip, body.query, answer, confidence_pct,
            prompt_version, chat_model, input_tok, output_tok,
            ttft_ms, total_ms, cache_hit=False, error=error_msg,
        ))

    return StreamingResponse(event_stream(), media_type="text/event-stream")


# ── Feedback & Admin Endpoints ─────────────────────────────────────────────────

class FeedbackRequest(BaseModel):
    trace_id: str = Field(..., min_length=1, max_length=64)
    rating:   int = Field(..., ge=1, le=5)
    note:     Optional[str] = Field(default=None, max_length=1000)


@app.post("/feedback/{author}")
async def submit_feedback(author: str, body: FeedbackRequest, request: Request):
    ip = _real_ip(request)
    try:
        async with aiosqlite.connect(DB_PATH) as db:
            await db.execute(
                "INSERT INTO feedback (trace_id, author, ip, rating, note) VALUES (?, ?, ?, ?, ?)",
                (body.trace_id, author, ip, body.rating, body.note or ""),
            )
            await db.commit()
    except Exception as e:
        log.warning(f"feedback write error: {e}")
        raise HTTPException(500, "Failed to record feedback")
    return {"ok": True}


@app.get("/admin/traces")
async def get_traces(
    author: Optional[str] = None,
    limit:  int = 50,
    x_admin_key: str = Header(None),
):
    if not _check_admin_key(x_admin_key):
        raise HTTPException(403, "Invalid admin key")
    limit = min(max(limit, 1), 500)
    async with aiosqlite.connect(DB_PATH) as db:
        db.row_factory = aiosqlite.Row
        if author:
            rows = await db.execute_fetchall(
                "SELECT * FROM traces WHERE author=? ORDER BY ts DESC LIMIT ?",
                (author, limit),
            )
        else:
            rows = await db.execute_fetchall(
                "SELECT * FROM traces ORDER BY ts DESC LIMIT ?", (limit,)
            )
    return {"traces": [dict(r) for r in rows]}


@app.get("/admin/metrics")
async def get_metrics(
    author: Optional[str] = None,
    x_admin_key: str = Header(None),
):
    if not _check_admin_key(x_admin_key):
        raise HTTPException(403, "Invalid admin key")

    async with aiosqlite.connect(DB_PATH) as db:
        db.row_factory = aiosqlite.Row
        where  = "WHERE author=?" if author else ""
        params = (author,) if author else ()

        rows = await db.execute_fetchall(
            f"SELECT confidence, total_ms, input_tokens, output_tokens, model, "
            f"cache_hit, error FROM traces {where} ORDER BY ts DESC LIMIT 1000",
            params,
        )

    rows = [dict(r) for r in rows]
    if not rows:
        return {"metrics": {}, "message": "No traces yet"}

    conf_vals    = [r["confidence"] for r in rows if r["confidence"] is not None]
    lat_vals     = [r["total_ms"]   for r in rows if r["total_ms"]   is not None]
    cache_hits   = sum(1 for r in rows if r["cache_hit"])
    errors       = sum(1 for r in rows if r.get("error"))
    total        = len(rows)

    def pct(data, p):
        if not data:
            return None
        s = sorted(data)
        idx = int(len(s) * p / 100)
        return round(s[min(idx, len(s) - 1)], 1)

    total_cost = sum(
        _compute_cost(r["model"] or "", r["input_tokens"] or 0, r["output_tokens"] or 0)
        for r in rows
    )

    conf_dist = {"lt50": 0, "50_75": 0, "gte75": 0}
    for c in conf_vals:
        if c < 50:
            conf_dist["lt50"] += 1
        elif c < 75:
            conf_dist["50_75"] += 1
        else:
            conf_dist["gte75"] += 1

    return {
        "total_queries": total,
        "cache_hit_rate": round(cache_hits / total, 3) if total else 0,
        "error_rate":     round(errors   / total, 3) if total else 0,
        "confidence": {
            "mean":       round(sum(conf_vals) / len(conf_vals), 1) if conf_vals else None,
            "distribution": conf_dist,
        },
        "latency_ms": {
            "p50": pct(lat_vals, 50),
            "p95": pct(lat_vals, 95),
            "p99": pct(lat_vals, 99),
        },
        "cost_usd": {"total": round(total_cost, 4), "per_query": round(total_cost / total, 5) if total else 0},
        "cache": {
            "size":     _sem_cache.size,
            "hit_rate": _sem_cache.hit_rate,
        },
        "prompt_versions": _prompts.list_versions(),
    }


@app.post("/admin/feedback/golden")
async def export_golden(
    author: str,
    min_rating: int = 1,
    x_admin_key: str = Header(None),
):
    """Export low-rated answers to a golden.json file for evals regression."""
    if not _check_admin_key(x_admin_key):
        raise HTTPException(403, "Invalid admin key")

    async with aiosqlite.connect(DB_PATH) as db:
        db.row_factory = aiosqlite.Row
        rows = await db.execute_fetchall(
            """SELECT t.query, t.answer, f.rating, f.note
               FROM feedback f
               JOIN traces t ON t.trace_id = f.trace_id
               WHERE f.author=? AND f.rating <= ?
               ORDER BY f.rating ASC LIMIT 200""",
            (author, min_rating),
        )

    items = [dict(r) for r in rows]
    golden = [
        {
            "question": r["query"],
            "expected_answer": r["answer"],
            "note": r["note"] or "",
            "rating": r["rating"],
        }
        for r in items
    ]

    out_dir  = Path("evals") / author
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "golden.json"
    out_path.write_text(json.dumps(golden, indent=2))
    return {"ok": True, "path": str(out_path), "count": len(golden)}


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
