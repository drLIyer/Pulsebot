# Pulsebot

Turn any LinkedIn newsletter into a chatbot. Authors onboard their newsletter via a self-serve form; readers can then search and chat with the full article archive.

## How it works

1. **Onboard** — author submits their LinkedIn profile URL (or signs in with LinkedIn). Pulsebot discovers and fetches their articles, builds a FAISS vector index.
2. **Chat** — readers visit `/c/<author-handle>` and ask questions. The backend runs RAG search over the index and streams a Claude-powered answer with cited sources.
3. **Auto-refresh** — every 24 hours (configurable), Pulsebot re-crawls each indexed author and adds any new articles to the index automatically.
4. **Tiers** — free readers get 10 questions/day; Pro and Power tiers (via Stripe) unlock more.
5. **Chrome extension** — readers can chat with any indexed newsletter from a toolbar popup without leaving their current tab.

## Quick start

```bash
# 1. Clone and create a virtualenv
git clone https://github.com/your-username/pulsebot
cd pulsebot
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 2. Set credentials (copy .env.example to .env, then edit)
# At minimum: ANTHROPIC_API_KEY and ADMIN_KEY
```

**Production** (requires `.env`, validates credentials, logs at INFO level):
```bash
bash run.sh
```

**Demo** (pre-indexed newsletter, opens browser automatically):
```bash
bash run_demo.sh
```

Open http://localhost:8000 to onboard a newsletter, or http://localhost:8000/c/your-handle once an author has been indexed.

---

## Chrome extension

The `chrome-extension/` directory contains a Manifest V3 browser extension. Once installed and pointed at your Pulsebot server, readers can chat with any indexed newsletter from a toolbar popup — from any tab, not just LinkedIn.

**To load it:**
1. Go to `chrome://extensions`
2. Enable **Developer mode** (top right)
3. Click **Load unpacked** → select the `chrome-extension/` folder
4. On first install, the settings page opens automatically

**Settings (click ⚙ in the popup):**
- **Server URL** — your Pulsebot server (e.g. `https://your-pulsebot.com`)
- **Default author handle** — pre-loads this author whenever you open the popup

The extension works from any browser tab — no need to be on LinkedIn.

---

## LinkedIn OAuth for authors

Authors can sign in with their LinkedIn account during onboarding. This pre-fills their handle, display name, and profile photo, which then appears in the chat header for their readers.

**To enable:**
1. Create a LinkedIn app at https://www.linkedin.com/developers/apps
2. Add the **Sign In with LinkedIn using OpenID Connect** product
3. Set the OAuth redirect URL to `{APP_BASE_URL}/auth/linkedin/callback`
4. Add `LINKEDIN_CLIENT_ID`, `LINKEDIN_CLIENT_SECRET`, `APP_BASE_URL`, and `SESSION_SECRET` to `.env`

---

## Environment variables

### Required

| Variable | Description |
|---|---|
| `ANTHROPIC_API_KEY` | Anthropic API key |
| `ADMIN_KEY` | Admin dashboard password — **change this in production** |

### Models

| Variable | Default | Description |
|---|---|---|
| `FREE_MODEL` | `claude-sonnet-5-5` | Model for free-tier chats |
| `PRO_MODEL` | `claude-sonnet-5-5` | Model for pro-tier chats |
| `POWER_MODEL` | `claude-opus-5-5` | Model for power-tier chats |
| `MODEL` | — | Override all tiers at once |
| `AVAILABLE_MODELS` | `claude-sonnet-5-5,claude-opus-5-5` | Models shown in the chat UI picker (comma-separated) |

### Server & infra

| Variable | Default | Description |
|---|---|---|
| `PORT` | `8000` | Server port |
| `DB_PATH` | `./pulsebot.db` | SQLite database path |
| `ALLOWED_ORIGINS` | `*` | Comma-separated CORS origins |
| `MAX_ONBOARD_JOBS` | `2` | Max concurrent onboarding jobs |
| `ONBOARD_DAILY_LIMIT` | `5` | Max new onboarding jobs per IP per day (0 = off) |
| `AUTO_REFRESH_HOURS` | `24` | How often to re-index authors for new articles (0 = disabled) |
| `HF_TOKEN` | — | HuggingFace token — avoids rate limits on embedding model downloads |

### LinkedIn OAuth (author onboarding)

| Variable | Description |
|---|---|
| `LINKEDIN_CLIENT_ID` | LinkedIn app client ID |
| `LINKEDIN_CLIENT_SECRET` | LinkedIn app client secret |
| `APP_BASE_URL` | Public base URL, e.g. `https://your-pulsebot.com` |
| `SESSION_SECRET` | Secret for signing session cookies — **required in production** |

### Stripe (optional)

| Variable | Description |
|---|---|
| `STRIPE_SECRET_KEY` | Stripe secret key |
| `STRIPE_WEBHOOK_SECRET` | Stripe webhook signing secret |
| `STRIPE_PRO_PRICE_ID` | Stripe Price ID for Pro tier |
| `STRIPE_POWER_PRICE_ID` | Stripe Price ID for Power tier |
| `STRIPE_SUCCESS_URL` | Public HTTPS URL for post-checkout redirect |
| `STRIPE_CANCEL_URL` | Public HTTPS URL if user cancels checkout |

See `config.py` for per-tier question limits (`FREE_DAILY_QUESTIONS`, `PRO_DAILY_QUESTIONS`, etc.).

---

## Project structure

```
api.py              FastAPI backend — chat, onboarding, auth, Stripe, admin
config.py           Tier limits, model config, Stripe config
fetcher.py          LinkedIn article discovery and scraping
indexer.py          FAISS vector index builder
models.py           Chunk dataclass
static/
  onboard.html      Author self-serve onboarding page  (GET /)
  chat.html         Reader chat UI                     (GET /c/<author>)
  admin.html        Admin dashboard                    (GET /admin/dashboard)
chrome-extension/   Manifest V3 browser extension
run.sh              Production launcher (validates .env)
run_demo.sh         Demo launcher (pre-indexed author, opens browser)
```

---

## API endpoints

| Method | Path | Description |
|---|---|---|
| `GET` | `/` | Author onboarding page |
| `GET` | `/c/{author}` | Reader chat page |
| `POST` | `/onboard` | Start an onboarding job |
| `GET` | `/onboard/{author}/status` | Poll job progress |
| `POST` | `/chat/{author}` | SSE chat stream |
| `GET` | `/authors/{author}` | Author metadata + branding |
| `GET` | `/models` | Available Claude models |
| `GET` | `/health` | Server health + feature flags |
| `GET` | `/auth/linkedin` | Start LinkedIn OAuth (author onboarding) |
| `GET` | `/auth/linkedin/callback` | LinkedIn OAuth callback |
| `GET` | `/auth/me` | Current session info |
| `GET` | `/auth/logout` | Clear session |
| `POST` | `/upgrade/{author}` | Start Stripe checkout |
| `GET` | `/admin/dashboard` | Admin UI |
| `GET` | `/admin/usage` | Usage stats (requires `x-admin-key`) |
| `GET` | `/admin/jobs` | Live job list (requires `x-admin-key`) |
| `POST` | `/admin/index/{author}` | Trigger manual re-index (requires `x-admin-key`) |

---

## Deployment notes

- **Workers**: Keep `WORKERS=1` (the default). The in-memory job tracker (`_jobs`) is per-process — multiple workers give each request a different view of live job state. Move job state fully to the DB before scaling.
- **Database**: SQLite works fine up to ~100 concurrent users. The `MAX_ONBOARD_JOBS` semaphore prevents write contention during indexing.
- **CORS**: Set `ALLOWED_ORIGINS` to your frontend domain in production.
- **SESSION_SECRET**: Must be set in production. If omitted, a random secret is generated at startup and sessions are invalidated on every restart.
- **Stripe**: Live keys require `STRIPE_SUCCESS_URL` / `STRIPE_CANCEL_URL` to be public HTTPS URLs — the server validates this at checkout time.
- **Auto-refresh**: Checks once per hour, re-indexing authors whose last index is older than `AUTO_REFRESH_HOURS`. Only new articles are fetched; existing markdown files are skipped. Set to `0` to disable.
- **Data and indexes**: `data/` and `indexes/` are excluded from git. Re-index on deploy via `POST /admin/index/{author}` or by re-running the onboarding flow.
