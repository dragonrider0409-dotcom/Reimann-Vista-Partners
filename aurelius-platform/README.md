# Aurelius

Quantitative risk models behind one API, with an analyst agent (an open Llama model) that calls them as tools.
Invite-only: there is no public sign-up. The owner signs in as admin and adds people by email.

## Deploy on Vercel (no laptop needed)

Everything below happens in a browser.

1. **Put the files in a GitHub repository.** github.com > New repository > *Add file* > *Upload files*.
   Upload `server.py`, `requirements.txt`, `vercel.json`, `.gitignore` and the `web` folder (with `index.html` inside it). Keep them at the top level of the repository, not inside another folder. `server.py` is the server: without it you only get the web page, and sign-in fails with 404.
2. **Import it into Vercel.** vercel.com > *Add New* > *Project* > pick the repository. Vercel detects FastAPI from `server.py`.
3. **Add a database.** In the project: *Storage* (or Marketplace) > *Neon Postgres* > connect it to the project.
   This sets `DATABASE_URL`. Vercel has no persistent disk, so without it nothing you save would survive.
4. **Get a model key.** console.groq.com > API Keys (see "The model" below for alternatives).
5. **Set environment variables** (Project > Settings > Environment Variables):

   | Variable | Value |
   |---|---|
   | `AURELIUS_BOOTSTRAP_EMAIL` | your admin email |
   | `AURELIUS_BOOTSTRAP_PASSWORD` | a first password, 12+ characters. You choose your own at first sign-in |
   | `AURELIUS_LLM_API_KEY` | the key from step 4 |
   | `AURELIUS_DATA` | `yfinance`, or `synthetic` for a demo with simulated prices |

6. **Deploy** (or *Redeploy*: variables only apply to new deployments). Open `https://<your-project>.vercel.app/v1/health`.
   You want `"status":"ok"` and `"storage":"postgres"`.
7. **Sign in** at the site with the bootstrap email and password. You are asked to choose a new password. Then
   *Workspace > Admin > Issue sign-in* to add people.

Lost your admin password? Set `AURELIUS_BOOTSTRAP_PASSWORD` to a new value and redeploy. A changed value is applied once;
an unchanged one never overwrites the password you chose.

**Plan.** Vercel's free Hobby plan is restricted to non-commercial personal use. Serving firms needs Pro.

## If sign-in says 404

The web page loaded but the server did not. Check, in this order:

1. Open `https://<your-project>.vercel.app/v1/health`. JSON means the server is running (tell me what it says). Vercel's own "404 NOT_FOUND" page means it is not.
2. In GitHub, the top level of the repository must list `server.py`, `requirements.txt` and `web`. If you see a single folder instead, either move the files up or set Vercel > Settings > General > *Root Directory* to that folder.
3. Vercel > Settings > Build and Development > *Framework Preset*: FastAPI (or let it auto-detect). Then *Deployments > Redeploy*.
4. The deployment's page should list a Python function. If its build log shows an error, that error is the cause.

## The model (Llama)

The agent talks to any OpenAI-compatible chat endpoint. Llama's weights are free, but running a 70B model needs a GPU,
which Vercel does not provide, so you use one of:

| Option | Settings | Notes |
|---|---|---|
| Groq (default) | `AURELIUS_LLM_API_KEY` only | Fast, cheap per token. The free tier's per-minute token cap is too small for this agent (each call carries about 3,500 tokens of tool definitions), so expect to use the paid developer tier. |
| OpenRouter / Together / Fireworks | `AURELIUS_LLM_BASE_URL`, `AURELIUS_LLM_API_KEY`, `AURELIUS_LLM_MODEL` | Many Llama variants. |
| Your own GPU server (vLLM, Ollama) | `AURELIUS_LLM_BASE_URL=https://your-host/v1` | Data never leaves infrastructure you control. Costs GPU hours, not tokens. Best answer for firms that cannot send data out. |

Default model: `llama-3.3-70b-versatile`. Open models follow instructions less reliably than the largest proprietary ones:
they can call the wrong tool or mis-explain a result. Every answer shows the tool calls and raw results above it so a
reader can check the numbers.

## Run locally

```
pip install -r requirements.txt
python server.py create-user you@firm.com --role admin
export AURELIUS_LLM_API_KEY=...            # or AURELIUS_LLM_BASE_URL=http://localhost:11434/v1 for Ollama
AURELIUS_DEV=1 python server.py serve      # drop AURELIUS_DEV behind HTTPS
```

`AURELIUS_DATA=synthetic` runs offline with simulated prices (clearly labelled in the UI). SQLite is used locally;
set `DATABASE_URL` to use Postgres.

## Admins

*Workspace > Admin*: issue a sign-in (one-time temporary password; the person must choose their own), make admin/analyst,
disable/enable, reset password, revoke keys. You cannot disable or demote yourself, and the last active admin cannot be
removed. Admins see account details only, never portfolios or conversations. API keys cannot create or change accounts.

## Environment

| Variable | Purpose |
|---|---|
| `DATABASE_URL` | Postgres connection string (`POSTGRES_URL` also works). If unset, SQLite is used (not on Vercel) |
| `AURELIUS_DB` | SQLite path |
| `AURELIUS_BOOTSTRAP_EMAIL` / `_PASSWORD` | Create or reset the owner's admin sign-in |
| `AURELIUS_DATA` | `yfinance` (default) or `synthetic` |
| `AURELIUS_LLM_API_KEY` | Model endpoint key (`GROQ_API_KEY`, `OPENROUTER_API_KEY` also read) |
| `AURELIUS_LLM_BASE_URL` | Default `https://api.groq.com/openai/v1` |
| `AURELIUS_LLM_MODEL` | Default `llama-3.3-70b-versatile` |
| `AURELIUS_DEV` | `1` allows non-HTTPS cookies for local use only |
| `AURELIUS_SESSION_HOURS` | Session lifetime (default 12) |
| `AURELIUS_AGENT_PER_HOUR` | Agent requests per user per hour |
| `AURELIUS_TRUST_PROXY` | Trust forwarded-IP headers (automatic on Vercel) |

## Before real firms use it

- Yahoo Finance data is for personal use, and Yahoo often throttles cloud IP ranges, so yfinance on Vercel may fail
  intermittently. A licensed feed is needed commercially; `Provider` in `server.py` is the one place to swap it.
- The agent sends messages and tool results to whichever model endpoint you configure.
- Not built: SSO, two-factor sign-in, application-level encryption at rest, per-firm tenancy, independent security review.
- Never commit `.env` or a database. `python server.py selftest` checks the engines.
