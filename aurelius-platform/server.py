#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AURELIUS
Risk, regime, forecasting, validation and execution models behind one API,
with an agent that answers questions by calling them.

One file on purpose. Sections, in order:
  1. Configuration
  2. Quant engines (pure NumPy, no ML framework)
  3. Market data providers (yfinance, plus an offline synthetic provider for demos and tests)
  4. Model registry: one definition drives the REST API, the docs site and the agent's tools
  5. Database, accounts, sessions, API keys, audit log
  6. Agent (an open Llama model with tool use via any OpenAI-compatible endpoint, streamed to the browser)
  7. HTTP routes and the command line

Quick start (local)
  pip install -r requirements.txt
  python server.py create-user you@firm.com --role admin     # prompts for a password (12+ characters)
  export AURELIUS_LLM_API_KEY=...                             # a Groq/OpenRouter/Together key; omit for a local Ollama (see below)
  AURELIUS_DEV=1 python server.py serve                       # http://127.0.0.1:8000

  No network or no key? AURELIUS_DATA=synthetic runs everything on clearly labelled simulated prices.

Quick start (Vercel): see README.md. Needs DATABASE_URL (Postgres), AURELIUS_BOOTSTRAP_EMAIL/_PASSWORD and a model key.

Environment
  DATABASE_URL              Postgres connection string (Neon etc.)   if unset, SQLite is used (not on Vercel)
  AURELIUS_DB               SQLite file                              default ./aurelius.db
  AURELIUS_BOOTSTRAP_EMAIL / _PASSWORD   create or reset the owner's admin sign-in at start-up. Applied once per
                            distinct pair; you must choose your own password at first sign-in
  AURELIUS_DATA             yfinance | synthetic                     default yfinance
  AURELIUS_LLM_API_KEY      key for the model endpoint (GROQ_API_KEY and OPENROUTER_API_KEY also work)
  AURELIUS_LLM_BASE_URL     OpenAI-compatible endpoint               default https://api.groq.com/openai/v1
                            e.g. https://openrouter.ai/api/v1, https://api.together.xyz/v1, http://localhost:11434/v1 (Ollama)
  AURELIUS_LLM_MODEL        model id at that endpoint                default llama-3.3-70b-versatile
  AURELIUS_DEV              1 = allow cookies over plain http        default off (cookies are Secure)
  AURELIUS_SESSION_HOURS    session lifetime                         default 12
  AURELIUS_AGENT_PER_HOUR   agent messages per user per hour         default 60
  AURELIUS_TRUST_PROXY      1 = honour X-Forwarded-For (automatic on Vercel)

Production notes
  * Serve over HTTPS. Cookies are Secure unless AURELIUS_DEV=1.
  * yfinance reads Yahoo Finance's public endpoints. Yahoo's terms limit that data to personal use, and Yahoo
    often throttles cloud IP ranges. It is fine for building and demos. Serving paying firms needs a licensed
    feed; the Provider class below is the single place to swap one in.
  * The agent sends the user's message and the tool results to whichever model endpoint you configure. Llama weights
    are open, but a hosted endpoint (Groq, OpenRouter...) is still a third party. For data that cannot leave your
    control, host the model yourself (vLLM or Ollama on a GPU server) and point AURELIUS_LLM_BASE_URL at it.
"""
import argparse
import base64
import hashlib
import hmac
import json
import logging
import math
import os
import re
import secrets
import sqlite3
import sys
import threading
import time
import zlib
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from statistics import NormalDist
from typing import Callable, Iterator, Literal, Optional

import numpy as np
from pydantic import BaseModel, Field, ValidationError, field_validator, model_validator

log = logging.getLogger("aurelius")
ROOT = os.path.dirname(os.path.abspath(__file__))

# ============================================================================
# 1. CONFIGURATION
# ============================================================================
CFG = {
    "db": os.environ.get("AURELIUS_DB", os.path.join(ROOT, "aurelius.db")),
    "db_url": os.environ.get("DATABASE_URL") or os.environ.get("POSTGRES_URL") or "",
    "vercel": bool(os.environ.get("VERCEL")),
    "data": os.environ.get("AURELIUS_DATA", "yfinance").lower(),
    "llm_base": os.environ.get("AURELIUS_LLM_BASE_URL", "").rstrip("/"),
    "llm_key": os.environ.get("AURELIUS_LLM_API_KEY") or os.environ.get("GROQ_API_KEY") or os.environ.get("OPENROUTER_API_KEY") or "",
    "model": os.environ.get("AURELIUS_LLM_MODEL", "llama-3.3-70b-versatile"),
    "dev": os.environ.get("AURELIUS_DEV") == "1",
    "session_hours": int(os.environ.get("AURELIUS_SESSION_HOURS", "12")),
    "agent_per_hour": int(os.environ.get("AURELIUS_AGENT_PER_HOUR", "60")),
    "trust_proxy": os.environ.get("AURELIUS_TRUST_PROXY") == "1" or bool(os.environ.get("VERCEL")),
    "agent_max_steps": 8,
    "agent_max_tokens": 2048,
}
COOKIE = "aurelius_session"
VERSION = "0.2.0"


class ModelError(Exception):
    """A request the models cannot answer (bad inputs, missing portfolio). Shown to the caller."""


class DataError(Exception):
    """Market data could not be fetched or was unusable."""


# ============================================================================
# 2. QUANT ENGINES
# ============================================================================
_N = NormalDist()
Phi, phi, PhiInv = _N.cdf, _N.pdf, _N.inv_cdf


def moments(x):
    x = np.asarray(x, float)
    m = x.mean()
    d = x - m
    s2 = float((d * d).mean())
    if s2 <= 0:
        return {"mean": float(m), "sd": 0.0, "skew": 0.0, "kurt": 3.0}
    return {"mean": float(m), "sd": math.sqrt(s2), "skew": float((d ** 3).mean() / s2 ** 1.5),
            "kurt": float((d ** 4).mean() / s2 ** 2)}


# ---- covariance ------------------------------------------------------------
def lw_delta(R):
    """Ledoit-Wolf shrinkage intensity on standardised returns, target = identity (zero correlation)."""
    T, N = R.shape
    sd = R.std(0) + 1e-12
    Z = (R - R.mean(0)) / sd
    C = Z.T @ Z / T
    d2 = float(((C - np.eye(N)) ** 2).sum() / N)
    b2 = float((np.sum(np.sum(Z * Z, 1) ** 2) - T * np.sum(C * C)) / (T * T * N))
    b2 = min(max(b2, 0.0), d2)
    return b2 / d2 if d2 > 0 else 0.0


def shrink_offdiag(S, delta):
    out = S * (1 - delta)
    np.fill_diagonal(out, np.diag(S))
    return out


def ewma_cov(R, lam=0.94):
    m = R.mean(0)
    S = np.cov(R, rowvar=False).reshape(R.shape[1], R.shape[1])
    X = R - m
    for r in X:
        S = lam * S + (1 - lam) * np.outer(r, r)
    return S


def build_cov(R, estimator):
    S = np.atleast_2d(np.cov(R, rowvar=False))
    if estimator == "sample":
        return S, 0.0
    d = lw_delta(R) if R.shape[1] > 1 else 0.0
    if estimator == "shrunk":
        return shrink_offdiag(S, d), d
    return shrink_offdiag(ewma_cov(R), d), d


# ---- risk ------------------------------------------------------------------
def cf_z(z, skew, exkurt):
    """Cornish-Fisher adjusted loss quantile for a return distribution with the given skew and excess kurtosis.
    Negative return skew (a fat left tail) raises the loss quantile, hence the minus sign on the skew term."""
    return z - (z * z - 1) * skew / 6 + (z ** 3 - 3 * z) * exkurt / 24 - (2 * z ** 3 - 5 * z) * skew * skew / 36


def kupiec_pvalue(x, n, p):
    """Kupiec proportion-of-failures test. Returns the chi-square(1) p-value."""
    if n <= 0:
        return None
    ph = x / n

    def ll(q):
        a = (n - x) * math.log(1 - q) if (n - x) > 0 else 0.0
        b = x * math.log(q) if x > 0 else 0.0
        return a + b
    lr = max(0.0, -2 * (ll(p) - ll(ph))) if 0 < p < 1 else 0.0
    return math.erfc(math.sqrt(lr / 2))


def risk_report(R, w, estimator, conf, h):
    S, delta = build_cov(R, estimator)
    Sw = S @ w
    var = float(w @ Sw)
    sig = math.sqrt(max(var, 1e-18))
    rp = R @ w
    mo = moments(rp)
    sq = math.sqrt(h)
    z = PhiInv(conf)
    ek = mo["kurt"] - 3
    var_cf = max(cf_z(z, mo["skew"], ek), 0.0) * sig * sq
    q = float(np.quantile(rp, 1 - conf))
    tail = float(rp[rp <= q].mean()) if (rp <= q).any() else q
    exc = int((rp < -z * sig).sum())
    n = len(rp)
    sd_i = np.sqrt(np.diag(S))
    return {
        "delta": delta, "sigma": sig, "sigma_ann": sig * math.sqrt(252), "skew": mo["skew"], "exkurt": ek,
        "worst": float(rp.min()), "best": float(rp.max()),
        "var": {"parametric": z * sig * sq, "cornish_fisher": var_cf, "historical": -q * sq},
        "cvar": {"parametric": sig * sq * phi(z) / (1 - conf), "historical": -tail * sq},
        "backtest": {"exceedances": exc, "expected": (1 - conf) * n, "n": n, "kupiec_p": kupiec_pvalue(exc, n, 1 - conf)},
        "rc": (w * Sw / var if var > 0 else np.zeros_like(w)), "sd_i": sd_i, "cov": S,
    }


def stress_conditional(S, shocks):
    """Gaussian conditional expectation of every asset given shocks on a subset: E[r | r_S = s] = S[:,S] S[S,S]^-1 s."""
    idx = list(shocks)
    s = np.array([shocks[i] for i in idx], float)
    SS = S[np.ix_(idx, idx)] + 1e-12 * np.eye(len(idx))
    imp = S[:, idx] @ np.linalg.solve(SS, s)
    for k, i in enumerate(idx):
        imp[i] = s[k]
    return imp


# ---- regimes: Gaussian hidden Markov model ---------------------------------
def hmm_fit(x, K, max_iter=150):
    x = np.asarray(x, float)
    T = len(x)
    sd0 = float(x.std(ddof=1))
    sc = np.array([0.6, 1.6]) if K == 2 else np.array([0.5, 1.0, 2.0])
    floor = 0.2 * sd0
    mu = np.zeros(K)
    sg = sc * sd0
    A = np.full((K, K), 0.05 / (K - 1))
    np.fill_diagonal(A, 0.95)
    pi = np.full(K, 1.0 / K)

    def estep(mu, sg, A, pi):
        z = (x[:, None] - mu[None, :]) / sg[None, :]
        B = np.exp(-0.5 * z * z) / (sg[None, :] * math.sqrt(2 * math.pi)) + 1e-300
        alpha = np.empty((T, K))
        c = np.empty(T)
        a = pi * B[0]
        c[0] = max(a.sum(), 1e-300)
        alpha[0] = a / c[0]
        for t in range(1, T):
            a = (alpha[t - 1] @ A) * B[t]
            c[t] = max(a.sum(), 1e-300)
            alpha[t] = a / c[t]
        beta = np.ones((T, K))
        for t in range(T - 2, -1, -1):
            beta[t] = (A @ (B[t + 1] * beta[t + 1])) / c[t + 1]
        gamma = alpha * beta
        gamma /= gamma.sum(1, keepdims=True)
        xi = A * (alpha[:-1].T @ (B[1:] * beta[1:] / c[1:, None]))
        return float(np.log(c).sum()), alpha, gamma, xi

    ll_old, it = -np.inf, 0
    for it in range(1, max_iter + 1):
        ll, alpha, gamma, xi = estep(mu, sg, A, pi)
        gs = gamma.sum(0)
        mu = (gamma * x[:, None]).sum(0) / gs
        sg = np.maximum(np.sqrt((gamma * (x[:, None] - mu[None, :]) ** 2).sum(0) / gs), floor)
        A = xi / xi.sum(1, keepdims=True)
        pi = gamma[0].copy()
        if abs(ll - ll_old) < 1e-6:
            break
        ll_old = ll
    ll, alpha, gamma, _ = estep(mu, sg, A, pi)
    order = np.argsort(sg)
    A2 = A[np.ix_(order, order)]
    post, filt = gamma[:, order], alpha[:, order]
    path = post.argmax(1)
    labels = ["Calm", "Stress"] if K == 2 else ["Calm", "Stress", "Crisis"]
    return {"K": K, "mu": mu[order], "sg": sg[order], "A": A2, "post": post, "filt": filt, "path": path,
            "labels": labels, "ll": ll, "iters": it, "share": np.array([(path == k).mean() for k in range(K)]),
            "dur": 1.0 / (1.0 - np.diag(A2))}


# ---- volatility: GARCH(1,1) with variance targeting ------------------------
def _garch_nll(e2, a, b):
    v0 = e2.mean()
    om = v0 * (1 - a - b)
    h = np.full_like(a, v0)
    s = np.zeros_like(a)
    with np.errstate(all="ignore"):  # invalid (a, b) candidates are masked below
        for t in range(len(e2)):
            s += np.log(h) + e2[t] / h
            h = om + a * e2[t] + b * h
    return np.where((a >= 1e-4) & (b >= 0) & (a + b <= 0.9995), 0.5 * s, 1e18)


def garch_fit(x):
    x = np.asarray(x, float)
    m = x.mean()
    e = x - m
    e2 = e * e
    v0 = float(e2.mean())
    aa, bb = np.meshgrid(np.arange(0.02, 0.2001, 0.02), np.arange(0.6, 0.9801, 0.02))
    a, b = aa.ravel(), bb.ravel()
    v = _garch_nll(e2, a, b)
    i = int(v.argmin())
    ba, bb_, bv = float(a[i]), float(b[i]), float(v[i])
    st = 0.02
    d = np.array([[1, 0], [-1, 0], [0, 1], [0, -1], [1, 1], [-1, -1], [1, -1], [-1, 1]], float)
    for _ in range(100):
        if st <= 1e-5:
            break
        ca, cb = ba + st * d[:, 0], bb_ + st * d[:, 1]
        v = _garch_nll(e2, ca, cb)
        j = int(v.argmin())
        if v[j] < bv - 1e-10:
            ba, bb_, bv = float(ca[j]), float(cb[j]), float(v[j])
        else:
            st /= 2
    om = v0 * (1 - ba - bb_)
    h = np.empty(len(x) + 1)
    h[0] = v0
    for t in range(len(x)):
        h[t + 1] = om + ba * e2[t] + bb_ * h[t]
    pers = ba + bb_
    return {"alpha": ba, "beta": bb_, "omega": om, "m": float(m), "v0": v0, "h": h, "h_next": float(h[-1]),
            "persistence": pers, "half_life": math.log(0.5) / math.log(pers) if 0 < pers < 1 else float("inf"),
            "ll": -bv}


def garch_vol_path(g, H):
    return np.array([g["v0"] + g["persistence"] ** (k - 1) * (g["h_next"] - g["v0"]) for k in range(1, H + 1)])


def garch_fan(g, H, M=4000, seed=7, df=6):
    rng = np.random.default_rng(seed)
    h = np.full(M, g["h_next"])
    c = np.zeros(M)
    out = np.empty((H, M))
    scale = math.sqrt((df - 2) / df)
    for d in range(H):
        eps = np.sqrt(h) * rng.standard_t(df, M) * scale
        c = c + eps
        out[d] = c
        h = g["omega"] + g["alpha"] * eps * eps + g["beta"] * h
    return np.sort(out, axis=1)


def garch_calibration(x, win=250, df=6, seed=7):
    x = np.asarray(x, float)
    g = garch_fit(x[:-win])
    e = x - g["m"]
    e2 = e * e
    h = np.empty(len(x) + 1)
    h[0] = g["v0"]
    for t in range(len(x)):
        h[t + 1] = g["omega"] + g["alpha"] * e2[t] + g["beta"] * h[t]
    zs = e[-win:] / np.sqrt(h[len(x) - win:len(x)])
    rng = np.random.default_rng(seed)
    ref = np.sort(rng.standard_t(df, 20000) * math.sqrt((df - 2) / df))
    rows = []
    for q in (0.5, 0.8, 0.95):
        lo, hi = np.quantile(ref, (1 - q) / 2), np.quantile(ref, 1 - (1 - q) / 2)
        zn = PhiInv(1 - (1 - q) / 2)
        rows.append({"nominal": q, "fat_tailed_coverage": float(((zs >= lo) & (zs <= hi)).mean()),
                     "normal_coverage": float((np.abs(zs) <= zn).mean())})
    return rows


# ---- validation: deflated Sharpe ratio -------------------------------------
def deflated_sharpe(sr, n, K, sr_std, skew=0.0, kurt=3.0):
    """Bailey and Lopez de Prado. sr and sr_std are per-period (not annualised). kurt is raw kurtosis."""
    g = 0.5772156649
    sr0 = sr_std * ((1 - g) * PhiInv(1 - 1 / K) + g * PhiInv(1 - 1 / (K * math.e))) if K > 1 else 0.0
    den = math.sqrt(max(1e-12, 1 - skew * sr + (kurt - 1) / 4 * sr * sr))
    return sr0, Phi((sr - sr0) * math.sqrt(n - 1) / den)


# ---- execution: Almgren-Chriss with linear temporary impact ----------------
def exec_plan(Q, sigma, c, spread_bps, T, u, n=120):
    """Q = order as a fraction of daily volume, sigma = daily volatility, c = impact coefficient,
    T = horizon in trading days, u = urgency (kappa * T). Returns cost and risk in basis points."""
    k = u / T
    dt = T / n
    t = np.arange(n + 1) * dt
    x = 1 - t / T if u < 1e-6 else (np.exp(-k * t) - np.exp(-k * (2 * T - t))) / (1 - np.exp(-2 * k * T))
    dx = x[:-1] - x[1:]
    temp = float((dx * dx / dt).sum())
    xm = (x[:-1] + x[1:]) / 2
    return {"t": t, "x": x, "impact_bps": 1e4 * c * sigma * Q * temp, "spread_bps": spread_bps,
            "cost_bps": 1e4 * c * sigma * Q * temp + spread_bps,
            "risk_bps": 1e4 * sigma * math.sqrt(float((xm * xm).sum() * dt)),
            "peak_participation": float(Q * (dx / dt).max())}


# ============================================================================
# 3. MARKET DATA
# ============================================================================
TICKER_RE = re.compile(r"^[A-Z0-9][A-Z0-9.\-=^]{0,14}$")


def norm_ticker(t):
    t = str(t).strip().upper()
    if not TICKER_RE.match(t):
        raise ValueError(f"'{t}' is not a valid ticker symbol")
    return t


@dataclass
class Panel:
    tickers: list
    dates: list
    close: np.ndarray
    volume: np.ndarray
    source: str
    asof: str
    dropped: dict = field(default_factory=dict)
    notes: list = field(default_factory=list)

    @property
    def returns(self):
        return self.close[1:] / self.close[:-1] - 1.0

    def meta(self):
        return {"source": self.source, "asof": self.asof, "price_basis": "daily adjusted close, simple returns",
                "window": {"start": self.dates[0], "end": self.dates[-1], "n_returns": len(self.dates) - 1},
                "tickers": self.tickers, "dropped": self.dropped, "notes": self.notes}


_yf_lock = threading.Lock()


class Provider:
    name = "base"
    ttl = 900

    def __init__(self):
        self._cache = {}
        self._lock = threading.Lock()

    def panel(self, tickers, lookback_days):
        tickers = [norm_ticker(t) for t in tickers]
        if not tickers:
            raise ModelError("no tickers given")
        key = (tuple(sorted(tickers)), int(lookback_days))
        with self._lock:
            hit = self._cache.get(key)
            if hit and time.time() - hit[0] < self.ttl:
                return self._order(hit[1], tickers)
        p = self._fetch(sorted(set(tickers)), int(lookback_days))
        with self._lock:
            if len(self._cache) > 200:
                self._cache.clear()
            self._cache[key] = (time.time(), p)
        return self._order(p, tickers)

    @staticmethod
    def _order(p, tickers):
        keep = [t for t in dict.fromkeys(tickers) if t in p.tickers]
        idx = [p.tickers.index(t) for t in keep]
        return Panel(keep, p.dates, p.close[:, idx], p.volume[:, idx], p.source, p.asof,
                     {t: r for t, r in p.dropped.items() if t in tickers}, p.notes)

    def _fetch(self, tickers, lookback_days):
        raise NotImplementedError


class YFinanceProvider(Provider):
    name = "yfinance"

    def _fetch(self, tickers, lookback_days):
        try:
            import yfinance as yf
        except ImportError as e:
            raise DataError("yfinance is not installed (pip install yfinance)") from e
        if CFG["vercel"]:  # the deployment filesystem is read-only except /tmp
            try:
                yf.set_tz_cache_location("/tmp/py-yfinance")
            except Exception:
                pass
        start = (datetime.now(timezone.utc) - timedelta(days=int(lookback_days * 1.6) + 10)).date().isoformat()
        with _yf_lock:
            try:
                df = yf.download(tickers if len(tickers) > 1 else tickers[0], start=start, auto_adjust=True,
                                 progress=False, threads=False, group_by="column")
            except Exception as e:  # network, rate limit, parse
                raise DataError(f"yfinance request failed: {type(e).__name__}: {e}") from e
        if df is None or len(df) == 0:
            raise DataError("yfinance returned no data. The network may be blocked, Yahoo may be rate limiting, "
                            "or none of the tickers exist.")
        cols = df.columns
        if getattr(cols, "nlevels", 1) > 1:
            close, vol = df["Close"], df["Volume"]
        else:
            close, vol = df[["Close"]], df[["Volume"]]
            close.columns = vol.columns = [tickers[0]]
        if close.ndim == 1:
            close, vol = close.to_frame(tickers[0]), vol.to_frame(tickers[0])
        dropped, keep = {}, []
        for t in tickers:
            if t not in close.columns or close[t].dropna().empty:
                dropped[t] = "no price history returned"
            else:
                keep.append(t)
        if not keep:
            raise DataError("none of the requested tickers returned price history: " + ", ".join(tickers))
        sub = close[keep].dropna(how="any")
        vsub = vol[keep].reindex(sub.index).fillna(0.0)
        sub, vsub = sub.iloc[-(lookback_days + 1):], vsub.iloc[-(lookback_days + 1):]
        if len(sub) < 61:
            raise DataError(f"only {len(sub)} overlapping daily observations across {', '.join(keep)}; at least 61 are needed")
        notes = []
        if len(sub) < lookback_days + 1:
            notes.append(f"requested {lookback_days} days of returns, got {len(sub) - 1} after aligning the tickers on common trading dates")
        return Panel(keep, [d.strftime("%Y-%m-%d") for d in sub.index], sub.to_numpy(float), vsub.to_numpy(float),
                     "yfinance (Yahoo Finance public data)", datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
                     dropped, notes)


class SyntheticProvider(Provider):
    """Deterministic simulated prices. Any ticker is accepted. For demos, tests and offline development only."""
    name = "synthetic"
    N_DAYS = 2700

    def _market(self):
        rng = np.random.default_rng(12345)
        P = np.array([[.987, .011, .002], [.035, .945, .020], [.015, .065, .920]])
        mult, drift = np.array([0.75, 1.6, 3.0]), np.array([0.0006, -0.0004, -0.0030])
        s, out = 0, np.empty(self.N_DAYS)
        for t in range(self.N_DAYS):
            s = int(rng.choice(3, p=P[s]))
            out[t] = drift[s] + 0.0085 * mult[s] * rng.standard_t(5) * math.sqrt(3 / 5)
        return out

    def _fetch(self, tickers, lookback_days):
        n = min(lookback_days + 1, self.N_DAYS)
        mkt = self._market()[-n + 1:]
        end = np.busday_offset(np.datetime64(datetime.now(timezone.utc).date(), "D"), 0, roll="backward")
        dates = np.busday_offset(end, np.arange(-n + 1, 1), roll="backward")
        closes, vols = [], []
        for t in tickers:
            rng = np.random.default_rng(zlib.crc32(t.encode()))
            beta, idio = 0.5 + rng.random(), 0.008 + 0.014 * rng.random()
            r = 0.0003 + beta * mkt + idio * rng.standard_t(6, n - 1) * math.sqrt(4 / 6)
            p = (20 + 380 * rng.random()) * np.concatenate([[1.0], np.cumprod(1 + r)])
            closes.append(p)
            vols.append(np.exp(rng.normal(15.5, 0.35, n)))
        return Panel(tickers, [str(d) for d in dates], np.column_stack(closes), np.column_stack(vols),
                     "SYNTHETIC (simulated prices, not market data)", datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
                     {}, ["Simulated data. Set AURELIUS_DATA=yfinance for Yahoo Finance data."])


_provider = None


def get_provider():
    global _provider
    want = CFG["data"]
    if _provider is None or _provider.name != want:
        _provider = SyntheticProvider() if want == "synthetic" else YFinanceProvider()
    return _provider


# ============================================================================
# 4. MODEL REGISTRY
#    One definition feeds POST /v1/models/{name}, the docs site (/v1/catalog) and the agent's tool list.
#    Results are decimal fractions (0.0177 = 1.77%). Every result carries a `data` block naming the source.
# ============================================================================
def clean(o, sig=6):
    if isinstance(o, (bool, np.bool_)):
        return bool(o)
    if isinstance(o, (float, np.floating)):
        x = float(o)
        return float(f"{x:.{sig}g}") if math.isfinite(x) else None
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.ndarray):
        return clean(o.tolist(), sig)
    if isinstance(o, dict):
        return {str(k): clean(v, sig) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [clean(v, sig) for v in o]
    return o


def thin(arr, n=240):
    arr = list(arr)
    if len(arr) <= n:
        return arr
    idx = np.unique(np.linspace(0, len(arr) - 1, n).round().astype(int))
    return [arr[i] for i in idx]


@dataclass
class Ctx:
    user_id: int
    provider: Provider


@dataclass
class Model:
    name: str
    title: str
    summary: str
    input: type
    fn: Callable
    method: list
    limits: list
    example: dict
    agent: bool = True


REGISTRY = {}


def register(name, title, summary, input, method, limits, example, agent=True):
    def deco(fn):
        REGISTRY[name] = Model(name, title, summary, input, fn, method, limits, example, agent)
        return fn
    return deco


def clean_weights(v):
    if not v or len(v) > 25:
        raise ValueError("weights must hold between 1 and 25 tickers")
    out = {norm_ticker(k): float(x) for k, x in v.items()}
    if any(abs(x) > 5 for x in out.values()):
        raise ValueError("a single weight above 500% looks like a units mistake; weights are fractions")
    if all(x == 0 for x in out.values()):
        raise ValueError("all weights are zero")
    return out


class BookIn(BaseModel):
    portfolio: str | None = Field(None, description="Name of one of the caller's saved portfolios. Give this or weights, not both.")
    weights: dict[str, float] | None = Field(None, description="Map of ticker to portfolio weight as a fraction (0.25 means 25%). Weights need not sum to 1; the remainder is treated as cash. Give this or portfolio.")
    lookback_days: int = Field(756, ge=60, le=2520, description="Trading days of history to use (756 is about three years).")

    @field_validator("weights")
    @classmethod
    def _w(cls, v):
        return None if v is None else clean_weights(v)

    @model_validator(mode="after")
    def _one(self):
        if (self.portfolio is None) == (self.weights is None):
            raise ValueError("give exactly one of 'portfolio' (a saved name) or 'weights' (a ticker map)")
        return self


def load_book(inp, ctx):
    nav, pname = None, None
    if inp.portfolio is not None:
        p = db_get_portfolio(ctx.user_id, inp.portfolio)
        if not p:
            raise ModelError(f"no saved portfolio named '{inp.portfolio}'")
        weights, nav, pname = p["weights"], p["nav_usd"], p["name"]
    else:
        weights = inp.weights
    tickers = list(weights)
    panel = ctx.provider.panel(tickers, inp.lookback_days)
    missing = [t for t in tickers if t not in panel.tickers]
    if missing:
        raise ModelError("no usable price history for " + ", ".join(missing) +
                         ". Positions are never dropped silently; fix the ticker or remove the position.")
    R = panel.returns[:, [panel.tickers.index(t) for t in tickers]]
    w = np.array([weights[t] for t in tickers], float)
    return panel, tickers, R, w, nav, pname


# ---- market_summary --------------------------------------------------------
class MarketSummaryIn(BaseModel):
    tickers: list[str] = Field(..., min_length=1, max_length=25, description="Ticker symbols, e.g. ['SPY','TLT','GLD'].")
    lookback_days: int = Field(252, ge=60, le=2520, description="Trading days of history.")

    @field_validator("tickers")
    @classmethod
    def _t(cls, v):
        return list(dict.fromkeys(norm_ticker(t) for t in v))


@register("market_summary", "Market summary",
          "Price and volatility summary for up to 25 tickers: last close, total return, annualised volatility, maximum drawdown and average dollar volume.",
          MarketSummaryIn,
          ["Daily adjusted closes, aligned on the dates every ticker traded.",
           "Annualised volatility is the daily standard deviation times the square root of 252.",
           "Maximum drawdown is the largest peak-to-trough fall of the adjusted close inside the window.",
           "Average dollar volume is the mean of close times volume over the last 20 sessions."],
          ["Adjusted close is not a tradable price history.", "Dollar volume uses the adjusted close, so it is approximate."],
          {"tickers": ["SPY", "TLT", "GLD"], "lookback_days": 252})
def m_market_summary(inp, ctx):
    p = ctx.provider.panel(inp.tickers, inp.lookback_days)
    R = p.returns
    rows = []
    for j, t in enumerate(p.tickers):
        c = p.close[:, j]
        dd = float((c / np.maximum.accumulate(c) - 1).min())
        rows.append({"ticker": t, "last_close": c[-1], "total_return": c[-1] / c[0] - 1,
                     "volatility_annualised": R[:, j].std(ddof=1) * math.sqrt(252), "max_drawdown": dd,
                     "avg_dollar_volume_20d": float((p.close[-20:, j] * p.volume[-20:, j]).mean())})
    return {"assets": rows, "data": p.meta()}


# ---- portfolio_risk --------------------------------------------------------
class RiskIn(BookIn):
    confidence: float = Field(0.99, ge=0.9, le=0.999, description="VaR confidence level, e.g. 0.99.")
    horizon_days: int = Field(1, ge=1, le=60, description="Holding period in trading days. Multi-day figures scale by the square root of the horizon.")
    estimator: Literal["sample", "shrunk", "dynamic"] = Field("dynamic", description="Covariance estimator: sample, shrunk (Ledoit-Wolf), or dynamic (EWMA 0.94 plus shrinkage).")
    nav_usd: float | None = Field(None, gt=0, description="Portfolio value in US dollars. Defaults to the saved portfolio's NAV, or 1,000,000.")


@register("portfolio_risk", "Portfolio risk",
          "Volatility, Value-at-Risk (parametric, Cornish-Fisher, historical), CVaR, an in-sample VaR backtest and each position's share of risk for a portfolio given as weights or a saved name.",
          RiskIn,
          ["Covariance: sample, Ledoit-Wolf shrinkage toward zero correlation, or EWMA (lambda 0.94) then shrinkage.",
           "Parametric VaR = z x sigma x sqrt(h). CVaR = sigma x phi(z) / (1 - c) x sqrt(h). Mean return is set to zero.",
           "Cornish-Fisher VaR corrects the normal quantile for the portfolio's skew and excess kurtosis.",
           "Historical VaR and CVaR are the empirical quantile and tail mean of the window's daily portfolio returns, scaled by sqrt(h).",
           "Risk contribution of asset i is w_i (Sigma w)_i / (w' Sigma w); contributions sum to 1.",
           "The VaR backtest counts days the 1-day parametric VaR was exceeded and applies the Kupiec proportion-of-failures test."],
          ["Weights are held constant over the window (no rebalancing, no trading).",
           "The VaR backtest is in-sample: sigma is estimated from the same window it is tested on.",
           "Multi-day figures assume independent days.", "Historical measures cannot see events outside the window."],
          {"weights": {"SPY": 0.5, "TLT": 0.3, "GLD": 0.2}, "confidence": 0.99, "horizon_days": 1, "estimator": "dynamic", "nav_usd": 50000000})
def m_portfolio_risk(inp, ctx):
    panel, tickers, R, w, nav, pname = load_book(inp, ctx)
    nav = inp.nav_usd or nav or 1_000_000.0
    r = risk_report(R, w, inp.estimator, inp.confidence, inp.horizon_days)
    u = lambda x: {"fraction": x, "usd": x * nav}
    rp = R @ w
    return {
        "portfolio": pname, "nav_usd": nav, "gross_exposure": float(np.abs(w).sum()), "net_exposure": float(w.sum()),
        "estimator": inp.estimator, "shrinkage_intensity": r["delta"], "confidence": inp.confidence, "horizon_days": inp.horizon_days,
        "volatility": {"daily": r["sigma"], "annualised": r["sigma_ann"]},
        "distribution": {"skew": r["skew"], "excess_kurtosis": r["exkurt"], "worst_day": r["worst"], "best_day": r["best"]},
        "var": {k: u(v) for k, v in r["var"].items()},
        "cvar": {k: u(v) for k, v in r["cvar"].items()},
        "var_backtest_in_sample": r["backtest"],
        "risk_contribution": [{"ticker": t, "weight": w[i], "share_of_variance": r["rc"][i],
                               "standalone_volatility_annualised": r["sd_i"][i] * math.sqrt(252)} for i, t in enumerate(tickers)],
        "data": panel.meta(),
        "_viz": {"type": "bars", "title": "Share of portfolio variance",
                 "rows": [{"label": t, "value": r["rc"][i], "note": f"{w[i] * 100:.0f}% weight"} for i, t in enumerate(tickers)]},
    }


# ---- regime_detection ------------------------------------------------------
class RegimeIn(BookIn):
    lookback_days: int = Field(1260, ge=250, le=2520, description="Trading days of history (at least 250; 1260 is about five years).")
    states: int = Field(3, ge=2, le=3, description="Number of hidden volatility states: 2 (calm, stress) or 3 (calm, stress, crisis).")


@register("regime_detection", "Regime detection",
          "Fits a hidden Markov model to a portfolio's daily returns and reports which volatility regime it is in now, how long regimes last and the chance of leaving calm.",
          RegimeIn,
          ["Gaussian hidden Markov model with 2 or 3 states, fitted by Baum-Welch (EM) with scaled forward-backward passes.",
           "States are sorted by volatility and named Calm, Stress, Crisis.",
           "Filtered probabilities use only data up to each day, so the latest value is what a live system would see.",
           "Expected duration of a state is 1 / (1 - p_ii), from the transition matrix."],
          ["EM finds a local optimum; the starting values are fixed so results are repeatable, not guaranteed global.",
           "Regimes are statistical labels for volatility, not economic events.",
           "State means are noisy and should not be traded on."],
          {"weights": {"SPY": 0.6, "TLT": 0.4}, "states": 3, "lookback_days": 1260})
def m_regime(inp, ctx):
    panel, tickers, R, w, nav, pname = load_book(inp, ctx)
    rp = R @ w
    hm = hmm_fit(rp, inp.states)
    last = hm["filt"][-1]
    cur = int(last.argmax())
    nxt = last @ hm["A"]
    path = hm["path"]
    run = 1
    while run < len(path) and path[-run - 1] == path[-1]:
        run += 1
    z = PhiInv(0.99)
    cum = np.cumsum(rp)
    dates = panel.dates[1:]
    return {
        "portfolio": pname,
        "current_regime": hm["labels"][cur], "filtered_probabilities_today": dict(zip(hm["labels"], last)),
        "days_in_current_regime": run, "current_regime_since": dates[len(dates) - run],
        "probability_next_day": dict(zip(hm["labels"], nxt)),
        "states": [{"label": hm["labels"][k], "daily_volatility": hm["sg"][k], "annualised_volatility": hm["sg"][k] * math.sqrt(252),
                    "mean_daily_return": hm["mu"][k], "share_of_time": hm["share"][k], "expected_duration_days": hm["dur"][k],
                    "one_day_99pct_var_fraction": z * hm["sg"][k]} for k in range(hm["K"])],
        "transition_matrix": {"rows_are_today": hm["labels"], "columns_are_tomorrow": hm["labels"], "values": hm["A"]},
        "fit": {"log_likelihood": hm["ll"], "iterations": hm["iters"], "n_observations": len(rp)},
        "data": panel.meta(),
        "_viz": {"type": "regime", "labels": hm["labels"], "dates": thin(dates), "cum": thin(cum * 100),
                 "state": thin(path.tolist())},
    }


# ---- volatility_forecast ---------------------------------------------------
class ForecastIn(BookIn):
    lookback_days: int = Field(1260, ge=250, le=2520, description="Trading days of history (at least 250).")
    horizon_days: int = Field(10, ge=1, le=60, description="Forecast horizon in trading days.")
    quantiles: list[float] = Field([0.05, 0.25, 0.5, 0.75, 0.95], max_length=9, description="Quantiles of cumulative return to report.")

    @field_validator("quantiles")
    @classmethod
    def _q(cls, v):
        if any(not 0.001 <= q <= 0.999 for q in v):
            raise ValueError("quantiles must lie between 0.001 and 0.999")
        return sorted(set(v))


@register("volatility_forecast", "Volatility forecast",
          "GARCH(1,1) volatility forecast and a Monte Carlo distribution of cumulative return over a horizon, with a walk-forward check of how often the bands held.",
          ForecastIn,
          ["GARCH(1,1) with variance targeting, fitted by Gaussian quasi-maximum likelihood (grid search then pattern search).",
           "Return paths: 4,000 simulated paths with Student-t(6) shocks, fixed seed 7 so results repeat exactly.",
           "Drift is set to zero: the bands describe spread and tail weight, not direction.",
           "Calibration refits on all but the last 250 days, then counts how often realised standardised returns fell inside the nominal bands."],
          ["GARCH reacts to past volatility; it does not anticipate scheduled events.",
           "Coverage on 250 days carries roughly 3 points of sampling noise.",
           "Cumulative return is the sum of daily simple returns."],
          {"weights": {"SPY": 1.0}, "horizon_days": 10})
def m_forecast(inp, ctx):
    panel, tickers, R, w, nav, pname = load_book(inp, ctx)
    rp = R @ w
    g = garch_fit(rp)
    H = inp.horizon_days
    fan = garch_fan(g, H)
    vp = garch_vol_path(g, H)
    q = {str(p): float(np.quantile(fan[-1], p)) for p in inp.quantiles}
    cal = garch_calibration(rp, 250) if len(rp) >= 500 else None
    qs = [0.05, 0.25, 0.5, 0.75, 0.95]
    return {
        "portfolio": pname, "horizon_days": H, "drift_assumption": "zero",
        "volatility": {"next_day_annualised": math.sqrt(g["h_next"] * 252), "long_run_annualised": math.sqrt(g["v0"] * 252),
                       "horizon_average_annualised": math.sqrt(vp.mean() * 252)},
        "cumulative_return_quantiles": q,
        "garch": {"alpha": g["alpha"], "beta": g["beta"], "persistence": g["persistence"], "shock_half_life_days": g["half_life"]},
        "calibration_last_250_days": cal, "simulation": {"paths": fan.shape[1], "seed": 7, "shock_distribution": "Student-t, 6 degrees of freedom"},
        "data": panel.meta(),
        "_viz": {"type": "fan", "x": list(range(1, H + 1)), "bands": {str(p): [float(np.quantile(fan[d], p)) * 100 for d in range(H)] for p in qs}},
    }


# ---- stress_test -----------------------------------------------------------
class StressIn(BookIn):
    shocks: dict[str, float] = Field(..., description="Map of ticker to instantaneous return shock as a fraction, e.g. {'SPY': -0.20}. Tickers need not be in the portfolio.")

    @field_validator("shocks")
    @classmethod
    def _s(cls, v):
        if not v or len(v) > 10:
            raise ValueError("give between 1 and 10 shocks")
        out = {norm_ticker(k): float(x) for k, x in v.items()}
        if any(abs(x) > 0.9 for x in out.values()):
            raise ValueError("shocks are fractions between -0.9 and 0.9 (-0.2 means a 20% fall)")
        return out


@register("stress_test", "Stress test",
          "Shock one or more assets and propagate the move to every holding through the estimated covariance, then report the portfolio's profit and loss.",
          StressIn,
          ["Uses the EWMA-plus-shrinkage covariance of the portfolio and shocked assets.",
           "Implied moves are the Gaussian conditional expectation E[r | r_shocked = s] = Sigma[:,S] Sigma[S,S]^-1 s.",
           "Portfolio P&L is the weighted sum of implied moves; shocked assets move by exactly their shock."],
          ["Linear and symmetric: it assumes correlations seen in normal times hold in a crash, which they often do not.",
           "No second-order effects (convexity, option payoffs, margin calls, liquidity)."],
          {"weights": {"SPY": 0.5, "TLT": 0.3, "GLD": 0.2}, "shocks": {"SPY": -0.2}})
def m_stress(inp, ctx):
    if inp.portfolio is not None:
        p = db_get_portfolio(ctx.user_id, inp.portfolio)
        if not p:
            raise ModelError(f"no saved portfolio named '{inp.portfolio}'")
        weights, nav, pname = p["weights"], p["nav_usd"], p["name"]
    else:
        weights, nav, pname = inp.weights, None, None
    universe = list(dict.fromkeys(list(weights) + list(inp.shocks)))
    panel = ctx.provider.panel(universe, inp.lookback_days)
    miss = [t for t in universe if t not in panel.tickers]
    if miss:
        raise ModelError("no usable price history for " + ", ".join(miss))
    R = panel.returns[:, [panel.tickers.index(t) for t in universe]]
    S, _ = build_cov(R, "dynamic")
    imp = stress_conditional(S, {universe.index(t): s for t, s in inp.shocks.items()})
    w = np.array([weights.get(t, 0.0) for t in universe])
    pnl = float(w @ imp)
    nav = nav or 1_000_000.0
    return {
        "portfolio": pname, "shocks": inp.shocks, "portfolio_pnl": {"fraction": pnl, "usd_on_1m_nav": pnl * 1_000_000, "nav_usd_used": nav, "usd": pnl * nav},
        "implied_moves": [{"ticker": t, "weight": w[i], "implied_return": imp[i], "contribution_to_pnl": w[i] * imp[i],
                           "shocked": t in inp.shocks} for i, t in enumerate(universe)],
        "method": "Gaussian conditional expectation under the dynamic covariance",
        "data": panel.meta(),
        "_viz": {"type": "bars", "title": "Implied return by asset (%)",
                 "rows": [{"label": t, "value": float(imp[i]), "note": f"{imp[i] * 100:.1f}%"} for i, t in enumerate(universe)], "signed": True},
    }


# ---- asset_correlation -----------------------------------------------------
class CorrIn(BaseModel):
    tickers: list[str] = Field(..., min_length=2, max_length=25, description="Between 2 and 25 ticker symbols.")
    lookback_days: int = Field(252, ge=60, le=2520, description="Trading days of history.")

    @field_validator("tickers")
    @classmethod
    def _t(cls, v):
        v = list(dict.fromkeys(norm_ticker(t) for t in v))
        if len(v) < 2:
            raise ValueError("need at least two distinct tickers")
        return v


@register("asset_correlation", "Correlation",
          "Pairwise return correlations for 2 to 25 tickers, with the most and least correlated pairs.",
          CorrIn,
          ["Pearson correlation of daily simple returns over the window."],
          ["Correlations move with regime; a calm-period figure understates stress-period co-movement."],
          {"tickers": ["SPY", "QQQ", "TLT", "GLD"], "lookback_days": 252})
def m_corr(inp, ctx):
    p = ctx.provider.panel(inp.tickers, inp.lookback_days)
    if len(p.tickers) < 2:
        raise ModelError("fewer than two tickers returned usable history")
    C = np.corrcoef(p.returns, rowvar=False)
    n = len(p.tickers)
    pairs = sorted(({"a": p.tickers[i], "b": p.tickers[j], "correlation": C[i, j]} for i in range(n) for j in range(i + 1, n)),
                   key=lambda d: d["correlation"])
    return {"tickers": p.tickers, "matrix": C, "average_pair_correlation": float(C[np.triu_indices(n, 1)].mean()),
            "least_correlated": pairs[:3], "most_correlated": pairs[::-1][:3], "data": p.meta()}


# ---- deflated_sharpe -------------------------------------------------------
class DSRIn(BaseModel):
    sharpe_annualised: float = Field(..., description="Annualised Sharpe ratio of the best strategy found.")
    n_obs: int = Field(..., ge=10, le=1_000_000, description="Number of return observations behind that Sharpe ratio.")
    n_trials: int = Field(1, ge=1, le=1_000_000, description="How many strategies or parameter sets were tried to find it.")
    trial_sharpe_std_annualised: float = Field(0.5, gt=0, le=10, description="Standard deviation of annualised Sharpe ratios across the trials.")
    skew: float = Field(0.0, ge=-10, le=10, description="Skewness of the strategy's returns.")
    kurtosis: float = Field(3.0, ge=1, le=500, description="Raw (not excess) kurtosis of the strategy's returns; 3 is normal.")
    periods_per_year: int = Field(252, ge=1, le=365 * 24, description="Observations per year (252 for daily).")


@register("deflated_sharpe", "Deflated Sharpe ratio",
          "Probability that a strategy's Sharpe ratio is real after accounting for how many strategies were tried, and for skew and fat tails.",
          DSRIn,
          ["Expected best Sharpe ratio of K skill-free trials: SR0 = sd x [(1 - g) Phi^-1(1 - 1/K) + g Phi^-1(1 - 1/(K e))], g = Euler-Mascheroni constant.",
           "DSR = Phi[(SR - SR0) sqrt(n - 1) / sqrt(1 - skew x SR + (kurt - 1)/4 x SR^2)], with SR per period.",
           "With one trial SR0 is zero and DSR is the probabilistic Sharpe ratio against zero."],
          ["Needs an honest count of trials and their dispersion. Undercounting trials overstates DSR.",
           "Assumes independent trials; correlated trials make the benchmark too high."],
          {"sharpe_annualised": 1.1, "n_obs": 1295, "n_trials": 100, "trial_sharpe_std_annualised": 0.55})
def m_dsr(inp, ctx):
    a = math.sqrt(inp.periods_per_year)
    sr0, d = deflated_sharpe(inp.sharpe_annualised / a, inp.n_obs, inp.n_trials, inp.trial_sharpe_std_annualised / a, inp.skew, inp.kurtosis)
    return {"luck_benchmark_sharpe_annualised": sr0 * a, "deflated_sharpe_probability": d, "passes_95_percent_bar": d >= 0.95,
            "inputs": inp.model_dump(),
            "reading": "Probability that the true Sharpe ratio exceeds what the best of n_trials skill-free strategies would show by chance."}


# ---- execution_cost --------------------------------------------------------
class ExecIn(BaseModel):
    ticker: str = Field(..., description="Ticker of the stock or ETF being traded.")
    order_usd: float = Field(..., gt=0, description="Order size in US dollars.")
    horizon_days: float = Field(1.0, ge=0.05, le=20, description="Time allowed to complete the order, in trading days.")
    urgency: float = Field(2.0, ge=0, le=10, description="0 trades at an even pace; higher trades earlier to cut timing risk at the price of more impact.")
    impact_coefficient: float = Field(0.5, gt=0, le=3, description="Linear temporary impact: price moves by this multiple of daily volatility per unit of participation.")
    half_spread_bps: float = Field(2.0, ge=0, le=200, description="Half the bid-ask spread in basis points.")
    adv_usd: float | None = Field(None, gt=0, description="Override average daily dollar volume. Defaults to the 20-day average from market data.")
    daily_vol: float | None = Field(None, gt=0, lt=1, description="Override daily volatility as a fraction. Defaults to the 60-day standard deviation.")

    @field_validator("ticker")
    @classmethod
    def _t(cls, v):
        return norm_ticker(v)


@register("execution_cost", "Execution cost",
          "Expected market-impact cost, timing risk and an optimal trading schedule for working a large order over a horizon.",
          ExecIn,
          ["Almgren-Chriss with linear temporary impact. Schedule x(t) = sinh(k(T - t)) / sinh(kT), k = urgency / T.",
           "Impact cost = c x sigma x Q x sum(dx^2 / dt), with Q the order as a fraction of daily volume.",
           "Timing risk = sigma x sqrt(integral of x(t)^2 dt). Cost at 95% = expected cost + 1.645 x risk.",
           "Daily volume and volatility come from market data unless overridden."],
          ["The impact coefficient is a placeholder: fit it to your own fills before relying on it.",
           "Linear impact understates cost for orders that are a large share of daily volume.",
           "No permanent impact, no intraday volume curve, no venue effects."],
          {"ticker": "AAPL", "order_usd": 25000000, "horizon_days": 1, "urgency": 2})
def m_exec(inp, ctx):
    adv, vol, meta = inp.adv_usd, inp.daily_vol, None
    if adv is None or vol is None:
        p = ctx.provider.panel([inp.ticker], 120)
        if inp.ticker not in p.tickers:
            raise ModelError(f"no price history for {inp.ticker}")
        meta = p.meta()
        adv = adv or float((p.close[-20:, 0] * p.volume[-20:, 0]).mean())
        vol = vol or float(p.returns[-60:, 0].std(ddof=1))
    if adv <= 0:
        raise ModelError("average daily dollar volume is zero; supply adv_usd")
    Q = inp.order_usd / adv
    run = lambda u: exec_plan(Q, vol, inp.impact_coefficient, inp.half_spread_bps, inp.horizon_days, u)
    cur = run(inp.urgency)
    row = lambda name, r: {"schedule": name, "expected_cost_bps": r["cost_bps"], "timing_risk_bps": r["risk_bps"],
                           "cost_at_95_bps": r["cost_bps"] + 1.645 * r["risk_bps"], "peak_participation": r["peak_participation"]}
    warns = []
    if Q > 0.25:
        warns.append(f"Order is {Q:.0%} of average daily volume; linear impact is a weak assumption at this size.")
    if inp.horizon_days * 1.0 < Q / 0.3:
        warns.append("Horizon implies participation above 30% of volume.")
    return {"ticker": inp.ticker, "order_usd": inp.order_usd, "adv_usd_used": adv, "daily_volatility_used": vol, "order_share_of_daily_volume": Q,
            "expected_cost_bps": cur["cost_bps"], "impact_bps": cur["impact_bps"], "spread_bps": cur["spread_bps"], "timing_risk_bps": cur["risk_bps"],
            "cost_at_95_bps": cur["cost_bps"] + 1.645 * cur["risk_bps"], "expected_cost_usd": cur["cost_bps"] / 1e4 * inp.order_usd,
            "peak_participation": cur["peak_participation"],
            "comparison": [row("even pace (urgency 0)", run(0)), row(f"chosen (urgency {inp.urgency:g})", cur), row("front-loaded (urgency 8)", run(8))],
            "square_root_law_cross_check_bps": 0.8 * vol * math.sqrt(Q) * 1e4 + inp.half_spread_bps,
            "schedule": [{"t_days": float(cur["t"][i]), "fraction_remaining": float(cur["x"][i])} for i in range(0, 121, 10)],
            "warnings": warns, "data": meta}


# ---- saved portfolios (per user) -------------------------------------------
class NoIn(BaseModel):
    pass


class NameIn(BaseModel):
    name: str = Field(..., min_length=1, max_length=60, description="Name of a saved portfolio.")


@register("list_portfolios", "List portfolios", "List the caller's saved portfolios (name, NAV, number of positions). Use to find a portfolio the user refers to by name.",
          NoIn, ["Reads the caller's own saved portfolios. Nothing is shared between accounts."], ["Read only."], {}, agent=True)
def m_list_portfolios(inp, ctx):
    return {"portfolios": [{"name": p["name"], "nav_usd": p["nav_usd"], "positions": len(p["weights"]), "updated_at": p["updated_at"]}
                           for p in db_list_portfolios(ctx.user_id)]}


@register("get_portfolio", "Get portfolio", "Return the weights and NAV of one saved portfolio.",
          NameIn, ["Reads the caller's own saved portfolio."], ["Read only."], {"name": "Core Book"})
def m_get_portfolio(inp, ctx):
    p = db_get_portfolio(ctx.user_id, inp.name)
    if not p:
        raise ModelError(f"no saved portfolio named '{inp.name}'")
    return {"name": p["name"], "nav_usd": p["nav_usd"], "weights": p["weights"], "updated_at": p["updated_at"]}


def run_model(name, raw, ctx):
    m = REGISTRY.get(name)
    if not m:
        raise KeyError(name)
    inp = m.input.model_validate(raw or {})
    out = m.fn(inp, ctx)
    return clean(out)


# ============================================================================
# 5. DATABASE, ACCOUNTS, SESSIONS, API KEYS, AUDIT
# ============================================================================
_tl = threading.local()


class Row(dict):
    """A result row that reads by column name or by position, so one set of queries serves both databases."""
    __slots__ = ()

    def __getitem__(self, k):
        return list(self.values())[k] if isinstance(k, int) else dict.__getitem__(self, k)


def _sqlite_row(cur, row):
    return Row(zip([d[0] for d in cur.description], row))


def _pg_row(cur):
    names = [d.name for d in cur.description] if cur.description else []
    return lambda values: Row(zip(names, values))


try:
    import psycopg
    IntegrityError = (sqlite3.IntegrityError, psycopg.errors.IntegrityError)
except ImportError:  # SQLite-only installs do not need it
    psycopg = None
    IntegrityError = (sqlite3.IntegrityError,)


class DB:
    """The small slice of the sqlite3 connection API this server uses, over SQLite or Postgres.
    Queries are written with ? placeholders; they are translated for Postgres."""

    def __init__(self):
        self.pg = bool(CFG["db_url"])
        self.depth = 0
        self._tx = []
        self.con = self._connect()

    def _connect(self):
        if self.pg:
            if psycopg is None:
                raise RuntimeError("DATABASE_URL is set but psycopg is not installed (pip install 'psycopg[binary]')")
            return psycopg.connect(CFG["db_url"], autocommit=True, row_factory=_pg_row, prepare_threshold=None, connect_timeout=10)
        if CFG["vercel"]:
            raise RuntimeError("DATABASE_URL is not set. Vercel has no persistent disk: add a Postgres database (Neon) to the project.")
        con = sqlite3.connect(CFG["db"], timeout=15)
        con.row_factory = _sqlite_row
        con.execute("PRAGMA journal_mode=WAL")
        con.execute("PRAGMA foreign_keys=ON")
        return con

    def execute(self, sql, params=()):
        if not self.pg:
            return self.con.execute(sql, params)
        sql = sql.replace("?", "%s")
        try:
            return self.con.execute(sql, params)
        except psycopg.OperationalError:
            # a pooled connection the server closed while idle: reconnect once, but never in the middle of a transaction
            if self.depth:
                raise
            try:
                self.con.close()
            except Exception:
                pass
            self.con = self._connect()
            return self.con.execute(sql, params)

    def __enter__(self):
        self.depth += 1
        if self.pg:
            tx = self.con.transaction()
            tx.__enter__()
            self._tx.append(tx)
        return self

    def __exit__(self, et, ev, tb):
        self.depth -= 1
        if self.pg:
            return self._tx.pop().__exit__(et, ev, tb)
        return self.con.__exit__(et, ev, tb)


def db():
    d = getattr(_tl, "db", None)
    key = (CFG["db_url"], CFG["db"])
    if d is None or getattr(_tl, "key", None) != key or (d.pg and d.con.closed):
        d = DB()
        _tl.db, _tl.key = d, key
    return d


def now():
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


_SCHEMA = """
CREATE TABLE IF NOT EXISTS users(id {PK}, email TEXT UNIQUE NOT NULL, name TEXT NOT NULL DEFAULT '',
    pw_hash TEXT NOT NULL, role TEXT NOT NULL DEFAULT 'analyst', disabled INTEGER NOT NULL DEFAULT 0, created_at TEXT NOT NULL,
    must_change INTEGER NOT NULL DEFAULT 0);
CREATE TABLE IF NOT EXISTS sessions(token_hash TEXT PRIMARY KEY, user_id {INT} NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    created_at TEXT NOT NULL, expires_at TEXT NOT NULL, ip TEXT, ua TEXT);
CREATE TABLE IF NOT EXISTS api_keys(id {PK}, user_id {INT} NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    name TEXT NOT NULL, prefix TEXT NOT NULL, key_hash TEXT UNIQUE NOT NULL, created_at TEXT NOT NULL, last_used TEXT, revoked INTEGER NOT NULL DEFAULT 0);
CREATE TABLE IF NOT EXISTS portfolios(id {PK}, user_id {INT} NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    name TEXT NOT NULL, weights TEXT NOT NULL, nav_usd {REAL} NOT NULL, updated_at TEXT NOT NULL, UNIQUE(user_id, name));
CREATE TABLE IF NOT EXISTS conversations(id TEXT PRIMARY KEY, user_id {INT} NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    title TEXT NOT NULL, messages TEXT NOT NULL, created_at TEXT NOT NULL, updated_at TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS audit(id {PK}, ts TEXT NOT NULL, user_id {INT}, event TEXT NOT NULL, detail TEXT NOT NULL DEFAULT '');
CREATE INDEX IF NOT EXISTS audit_user ON audit(user_id, id);
CREATE TABLE IF NOT EXISTS login_fails(k TEXT NOT NULL, ts {REAL} NOT NULL);
CREATE INDEX IF NOT EXISTS login_fails_k ON login_fails(k, ts);
CREATE TABLE IF NOT EXISTS agent_hits(user_id {INT} NOT NULL, ts {REAL} NOT NULL);
CREATE INDEX IF NOT EXISTS agent_hits_u ON agent_hits(user_id, ts);
CREATE TABLE IF NOT EXISTS meta(k TEXT PRIMARY KEY, v TEXT NOT NULL);
"""


def init_db():
    d = db()
    sql = _SCHEMA.format(PK="BIGSERIAL PRIMARY KEY" if d.pg else "INTEGER PRIMARY KEY", INT="BIGINT" if d.pg else "INTEGER",
                         REAL="DOUBLE PRECISION" if d.pg else "REAL")
    with d as c:
        for stmt in [x.strip() for x in sql.split(";") if x.strip()]:
            c.execute(stmt)
        if not d.pg:
            cols = {r["name"] for r in c.execute("PRAGMA table_info(users)")}
            if "must_change" not in cols:  # databases created before admin-issued accounts existed
                c.execute("ALTER TABLE users ADD COLUMN must_change INTEGER NOT NULL DEFAULT 0")


def audit(user_id, event, detail=""):
    try:
        with db() as c:
            c.execute("INSERT INTO audit(ts,user_id,event,detail) VALUES(?,?,?,?)", (now(), user_id, event, detail[:300]))
    except Exception:  # an audit failure must not break a request
        log.exception("audit write failed")


def _b64(b):
    return base64.b64encode(b).decode()


def hash_password(pw):
    salt = os.urandom(16)
    dk = hashlib.scrypt(pw.encode(), salt=salt, n=2 ** 14, r=8, p=1, dklen=32)
    return f"scrypt$14$8$1${_b64(salt)}${_b64(dk)}"


def verify_password(pw, stored):
    try:
        _, n, r, p, salt, dk = stored.split("$")
        got = hashlib.scrypt(pw.encode(), salt=base64.b64decode(salt), n=2 ** int(n), r=int(r), p=int(p), dklen=32)
        return hmac.compare_digest(got, base64.b64decode(dk))
    except Exception:
        return False


_DUMMY = hash_password("not-a-real-password-timing-equaliser")
sha = lambda s: hashlib.sha256(s.encode()).hexdigest()


def check_password_policy(pw):
    if len(pw) < 12:
        raise ValueError("password must be at least 12 characters")
    if pw.lower() == pw or not any(ch.isdigit() for ch in pw):
        if len(set(pw)) < 6:
            raise ValueError("password is too repetitive")


def create_user(email, password, name="", role="analyst", must_change=False):
    email = email.strip().lower()
    if not re.match(r"^[^@\s]+@[^@\s]+\.[^@\s]+$", email):
        raise ValueError("not a valid email address")
    if role not in ("admin", "analyst"):
        raise ValueError("role must be admin or analyst")
    check_password_policy(password)
    with db() as c:
        c.execute("INSERT INTO users(email,name,pw_hash,role,created_at,must_change) VALUES(?,?,?,?,?,?)",
                  (email, name.strip(), hash_password(password), role, now(), 1 if must_change else 0))


def user_by_email(email):
    r = db().execute("SELECT * FROM users WHERE email=?", (email.strip().lower(),)).fetchone()
    return dict(r) if r else None


def new_session(user_id, ip, ua):
    tok = secrets.token_urlsafe(32)
    exp = (datetime.now(timezone.utc) + timedelta(hours=CFG["session_hours"])).strftime("%Y-%m-%dT%H:%M:%SZ")
    with db() as c:
        c.execute("DELETE FROM sessions WHERE expires_at < ?", (now(),))
        c.execute("INSERT INTO sessions VALUES(?,?,?,?,?,?)", (sha(tok), user_id, now(), exp, ip, (ua or "")[:200]))
    return tok


def new_api_key(user_id, name):
    key = "aur_" + secrets.token_urlsafe(32)
    with db() as c:
        row = c.execute("INSERT INTO api_keys(user_id,name,prefix,key_hash,created_at) VALUES(?,?,?,?,?) RETURNING id", (user_id, name.strip()[:60], key[:10], sha(key), now())).fetchone()
    return row["id"], key


# ---- portfolios ------------------------------------------------------------
def db_list_portfolios(uid):
    return [{"name": r["name"], "weights": json.loads(r["weights"]), "nav_usd": r["nav_usd"], "updated_at": r["updated_at"]}
            for r in db().execute("SELECT * FROM portfolios WHERE user_id=? ORDER BY name", (uid,))]


def db_get_portfolio(uid, name):
    r = db().execute("SELECT * FROM portfolios WHERE user_id=? AND lower(name)=lower(?)", (uid, name)).fetchone()
    return {"name": r["name"], "weights": json.loads(r["weights"]), "nav_usd": r["nav_usd"], "updated_at": r["updated_at"]} if r else None


# ---- login throttle (in the database: serverless instances share no memory) ----
def _fk(k):
    return ("p:" + "|".join(k)) if isinstance(k, tuple) else ("i:" + k)


def throttled(keys):
    cutoff = time.time() - 900
    for k in keys:
        n = db().execute("SELECT COUNT(*) AS n FROM login_fails WHERE k=? AND ts>?", (_fk(k), cutoff)).fetchone()["n"]
        if n >= (5 if isinstance(k, tuple) else 25):
            return True
    return False


def note_fail(keys):
    t = time.time()
    with db() as c:
        c.execute("DELETE FROM login_fails WHERE ts<?", (t - 900,))
        for k in keys:
            c.execute("INSERT INTO login_fails(k,ts) VALUES(?,?)", (_fk(k), t))


# ============================================================================
# 6. AGENT
# ============================================================================
SYSTEM_PROMPT = """You are Aurelius, a quantitative analyst assistant for portfolio managers and risk teams. You answer by calling the Aurelius models, which are your tools.

Rules
1. Every number you state about a market, a portfolio or a model must come from a tool result in this conversation. Never quote prices, volatilities, correlations or any statistic from memory. If you cannot get the number from a tool, say so.
2. Choose the tool that answers the question and call it. If the user names a saved portfolio, use that name; if they are vague about which portfolio, call list_portfolios. If you lack tickers or weights, ask one short question.
3. Tool results are decimal fractions (0.0177 means 1.77%). Convert to percentages and dollars when you write.
4. When you explain a result, say what was computed, give the figures that matter, say how to read them, and state the limits that bear on the conclusion: data source and window (from the result's data block), the estimator, and any warning the tool returned. Data marked SYNTHETIC is simulated and must be called out as such.
5. You describe risk. You do not give personalised investment advice or tell anyone what to buy or sell. If asked, explain what the numbers say and what they do not.
6. Be concise. Use a short table when comparing more than three figures. Use the terms of art the audience knows (VaR, CVaR, Kupiec, GARCH, half-life) without defining the basics.
7. If a tool returns an error, report what failed and what you tried. Do not invent a result. You may retry once with corrected inputs.
8. You cannot place trades, change portfolios or reach anything outside the tools.
9. Use the function-calling interface for every tool call. Never write a function call, JSON or tool syntax in your reply text. Call a tool with only the parameters you actually have; leave the others out rather than passing null or placeholders."""


class AgentUnavailable(Exception):
    pass


GROQ_BASE = "https://api.groq.com/openai/v1"


def llm_base():
    return CFG["llm_base"] or GROQ_BASE


def llm_enabled():
    """True when a key is set, or when a base URL is set explicitly (a local Ollama or vLLM server needs no key)."""
    return bool(CFG["llm_key"] or CFG["llm_base"])


def llm_provider():
    from urllib.parse import urlparse
    return urlparse(llm_base()).hostname or "unknown"


class LLMError(Exception):
    def __init__(self, message, status=None, code=""):
        super().__init__(message)
        self.status, self.code = status, code


def chat_completion(payload):
    """One call to an OpenAI-compatible /chat/completions endpoint (Groq, OpenRouter, Together, Fireworks, Ollama, vLLM...)."""
    import httpx
    headers = {"Content-Type": "application/json"}
    if CFG["llm_key"]:
        headers["Authorization"] = "Bearer " + CFG["llm_key"]
    try:
        r = httpx.post(llm_base() + "/chat/completions", json=payload, headers=headers, timeout=httpx.Timeout(120.0, connect=10.0))
    except httpx.HTTPError as e:
        raise LLMError(f"could not reach the model server ({type(e).__name__})")
    if r.status_code >= 400:
        code = ""
        try:
            err = r.json().get("error", {})
            code = str(err.get("code") or err.get("type") or "")
        except Exception:
            pass
        raise LLMError(f"model server returned HTTP {r.status_code}", r.status_code, code)
    return r.json()


def _inline_refs(node, defs):
    """Resolve $ref/$defs and drop titles: small open models follow flat schemas more reliably, and it saves tokens."""
    if isinstance(node, dict):
        if "$ref" in node:
            return _inline_refs(defs[node["$ref"].split("/")[-1]], defs)
        return {k: _inline_refs(v, defs) for k, v in node.items() if k not in ("title", "$defs")}
    if isinstance(node, list):
        return [_inline_refs(x, defs) for x in node]
    return node


def agent_tools():
    out = []
    for m in REGISTRY.values():
        if m.agent:
            schema = m.input.model_json_schema()
            schema = _inline_refs(schema, schema.get("$defs", {}))
            out.append({"type": "function", "function": {"name": m.name, "description": m.summary, "parameters": schema}})
    return out


def to_chat_messages(system, messages):
    """Stored conversations keep one neutral block format; translate to OpenAI chat messages at call time."""
    out = [{"role": "system", "content": system}]
    answered = {b["tool_use_id"] for m in messages if m["role"] == "user" and not isinstance(m["content"], str)
                for b in m["content"] if b.get("type") == "tool_result"}
    for m in messages:
        c = m["content"]
        if m["role"] == "user":
            if isinstance(c, str):
                out.append({"role": "user", "content": c})
            else:
                for b in c:
                    if b.get("type") == "tool_result":
                        out.append({"role": "tool", "tool_call_id": b["tool_use_id"], "content": b["content"]})
        else:
            text = "\n\n".join(b["text"] for b in c if b["type"] == "text")
            calls = [{"id": b["id"], "type": "function", "function": {"name": b["name"], "arguments": json.dumps(b["input"], separators=(",", ":"))}}
                     for b in c if b["type"] == "tool_use" and b["id"] in answered]  # a call whose result was never stored (dropped connection) would be rejected
            if not text and not calls:
                continue
            msg = {"role": "assistant", "content": text or None}
            if calls:
                msg["tool_calls"] = calls
            out.append(msg)
    return out


def _parse_args(raw):
    """Open models sometimes emit empty, null or malformed argument strings. Return (dict, error)."""
    if raw is None or (isinstance(raw, str) and not raw.strip()):
        return {}, None
    if isinstance(raw, dict):
        return raw, None
    try:
        v = json.loads(raw)
    except (TypeError, ValueError):
        return {}, "arguments were not valid JSON"
    if v is None:
        return {}, None
    if not isinstance(v, dict):
        return {}, "arguments must be a JSON object"
    return v, None


def llm_turn(system, messages, tools):
    """Run one model call. Returns (blocks, stop_reason, usage). Retries once when the server rejects a malformed tool call."""
    payload = {"model": CFG["model"], "messages": to_chat_messages(system, messages), "tools": tools, "tool_choice": "auto",
               "max_tokens": CFG["agent_max_tokens"], "temperature": 0.1}
    for attempt in (0, 1):
        try:
            data = chat_completion(payload)
            break
        except LLMError as e:
            transient = e.status in (429, 500, 502, 503, 504) or e.code == "tool_use_failed"
            if attempt == 0 and transient:
                time.sleep(1.5 if e.status == 429 else 0.2)
                continue
            raise
    choice = (data.get("choices") or [{}])[0]
    msg = choice.get("message") or {}
    blocks = []
    if (msg.get("content") or "").strip():
        blocks.append({"type": "text", "text": msg["content"]})
    for i, tc in enumerate(msg.get("tool_calls") or []):
        fn = tc.get("function") or {}
        args, err = _parse_args(fn.get("arguments"))
        blocks.append({"type": "tool_use", "id": tc.get("id") or f"call_{uuid_hex()}_{i}", "name": fn.get("name", ""), "input": args,
                       **({"parse_error": err} if err else {})})
    fin = choice.get("finish_reason")
    stop = "tool_use" if any(b["type"] == "tool_use" for b in blocks) else "max_tokens" if fin == "length" else "end_turn"
    u = data.get("usage") or {}
    return blocks, stop, {"input_tokens": u.get("prompt_tokens", 0) or 0, "output_tokens": u.get("completion_tokens", 0) or 0}


def agent_rate_ok(uid):
    t = time.time()
    with db() as c:
        c.execute("DELETE FROM agent_hits WHERE ts<?", (t - 3600,))
        n = c.execute("SELECT COUNT(*) AS n FROM agent_hits WHERE user_id=?", (uid,)).fetchone()["n"]
        if n >= CFG["agent_per_hour"]:
            return False
        c.execute("INSERT INTO agent_hits(user_id,ts) VALUES(?,?)", (uid, t))
    return True


def sse(event, data):
    return f"event: {event}\ndata: {json.dumps(data, separators=(',', ':'), default=str)}\n\n"


def agent_stream(user, text, conv_id=None) -> Iterator[str]:
    """Run one user turn. Yields server-sent events: conversation, text, tool_call, tool_result, error, done."""
    uid = user["id"]
    if not llm_enabled():
        yield sse("error", {"message": "The agent is not enabled on this server (no model endpoint configured). The models are still available through the API and the Models page."})
        return
    if not agent_rate_ok(uid):
        yield sse("error", {"message": f"Rate limit: {CFG['agent_per_hour']} agent messages per hour. Use the Models page or API meanwhile."})
        return
    messages, title = [], text.strip().replace("\n", " ")[:60]
    if conv_id:
        row = db().execute("SELECT * FROM conversations WHERE id=? AND user_id=?", (conv_id, uid)).fetchone()
        if not row:
            yield sse("error", {"message": "Conversation not found."})
            return
        messages, title = json.loads(row["messages"]), row["title"]
        if len(messages) > 60:
            yield sse("error", {"message": "This conversation is long. Start a new one to keep answers sharp."})
            return
    else:
        conv_id = uuid_hex()
        with db() as c:
            c.execute("INSERT INTO conversations VALUES(?,?,?,?,?,?)", (conv_id, uid, title, "[]", now(), now()))
    yield sse("conversation", {"id": conv_id, "title": title})
    messages.append({"role": "user", "content": text})
    ctx = Ctx(uid, get_provider())
    system = SYSTEM_PROMPT + f"\n\nToday is {datetime.now(timezone.utc).date().isoformat()}. Market data provider: {ctx.provider.name}."
    tools = agent_tools()
    used, usage = [], {"input_tokens": 0, "output_tokens": 0}
    try:
        for step in range(CFG["agent_max_steps"]):
            blocks, stop, u = llm_turn(system, messages, tools)
            usage["input_tokens"] += u["input_tokens"]
            usage["output_tokens"] += u["output_tokens"]
            if not blocks:
                yield sse("text", {"text": "(The model returned an empty reply. Try rephrasing the question.)"})
                break
            messages.append({"role": "assistant", "content": [{k: v for k, v in b.items() if k != "parse_error"} for b in blocks]})
            calls = [b for b in blocks if b["type"] == "tool_use"]
            for b in blocks:
                if b["type"] == "text" and b["text"].strip():
                    yield sse("text", {"text": b["text"]})
            if stop != "tool_use" or not calls:
                if stop == "max_tokens":
                    yield sse("text", {"text": "\n\n(The answer was cut off at the length limit. Ask me to continue.)"})
                break
            results = []
            for b in calls:
                yield sse("tool_call", {"id": b["id"], "name": b["name"], "input": b["input"]})
                t0, ok, viz = time.time(), True, None
                try:
                    if b["name"] not in REGISTRY or not REGISTRY[b["name"]].agent:
                        raise ModelError(f"unknown tool '{b['name']}'")
                    if b.get("parse_error"):
                        raise ModelError(f"tool call rejected: {b['parse_error']}. Call it again with a valid JSON object.")
                    res = run_model(b["name"], b["input"], ctx)
                    viz = res.pop("_viz", None)
                except ValidationError as e:
                    ok, res = False, {"error": "invalid input", "details": json.loads(e.json(include_url=False))}
                except (ModelError, DataError) as e:
                    ok, res = False, {"error": str(e)}
                except Exception as e:  # a model bug must not end the conversation
                    log.exception("tool %s failed", b["name"])
                    ok, res = False, {"error": f"internal error in {b['name']}: {type(e).__name__}"}
                used.append(b["name"])
                body = json.dumps(res, separators=(",", ":"))
                if len(body) > 9000:
                    body = body[:9000] + '..."truncated"'
                yield sse("tool_result", {"id": b["id"], "name": b["name"], "ok": ok, "ms": int((time.time() - t0) * 1000), "result": res, "viz": viz})
                results.append({"type": "tool_result", "tool_use_id": b["id"], "content": body, **({} if ok else {"is_error": True})})
            messages.append({"role": "user", "content": results})
        else:
            yield sse("text", {"text": "\n\n(Stopped after the maximum number of tool steps. Ask a narrower question to continue.)"})
    except AgentUnavailable as e:
        yield sse("error", {"message": str(e)})
    except Exception as e:
        log.exception("agent failure")
        if isinstance(e, LLMError):
            hint = {401: "Check AURELIUS_LLM_API_KEY.", 403: "Check AURELIUS_LLM_API_KEY.", 404: "Check AURELIUS_LLM_MODEL and AURELIUS_LLM_BASE_URL.",
                    413: "The request was too large for this model's limits.", 429: "The model provider's rate limit was hit. Try again in a minute."}.get(e.status, "Try again in a moment.")
            yield sse("error", {"message": f"The model call failed: {e} ({llm_provider()}). {hint}"})
        else:
            yield sse("error", {"message": f"The model call failed ({type(e).__name__})."})
    finally:
        with db() as c:
            c.execute("UPDATE conversations SET messages=?, updated_at=? WHERE id=? AND user_id=?", (json.dumps(messages), now(), conv_id, uid))
        audit(uid, "agent_message", f"conv={conv_id} tools={','.join(used)}")
    yield sse("done", {"conversation_id": conv_id, "usage": usage, "tools_used": used})


def uuid_hex():
    return secrets.token_hex(8)


def ui_messages(messages):
    """Convert stored API messages into the simple shape the website renders."""
    out, calls = [], {}
    for m in messages:
        c = m["content"]
        if m["role"] == "user":
            if isinstance(c, str):
                out.append({"role": "user", "text": c})
            else:
                for b in c:
                    if b.get("type") == "tool_result" and b["tool_use_id"] in calls:
                        calls[b["tool_use_id"]]["result"] = b["content"]
                        calls[b["tool_use_id"]]["ok"] = not b.get("is_error")
        else:
            text = "\n\n".join(b["text"] for b in c if b["type"] == "text")
            tools = [{"id": b["id"], "name": b["name"], "input": b["input"]} for b in c if b["type"] == "tool_use"]
            for t in tools:
                calls[t["id"]] = t
            out.append({"role": "assistant", "text": text, "tools": tools})
    return out


# ============================================================================
# 7. HTTP ROUTES AND COMMAND LINE
# ============================================================================
from fastapi import Body, Depends, FastAPI, HTTPException, Request, Response  # noqa: E402
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse  # noqa: E402


def bootstrap_admin():
    """Create or reset the owner's admin sign-in from environment variables, so no command line is needed (e.g. on Vercel).
    Applies once per distinct email+password pair: change AURELIUS_BOOTSTRAP_PASSWORD and redeploy to recover a lost admin password."""
    email, pw = (os.environ.get("AURELIUS_BOOTSTRAP_EMAIL") or "").strip().lower(), os.environ.get("AURELIUS_BOOTSTRAP_PASSWORD") or ""
    if not email or not pw:
        return
    fp = sha(email + "\n" + pw)
    seen = db().execute("SELECT v FROM meta WHERE k='bootstrap'").fetchone()
    if seen and seen["v"] == fp:
        return
    u = user_by_email(email)
    try:
        if u is None:
            create_user(email, pw, "Admin", "admin", must_change=True)
        else:
            check_password_policy(pw)
            with db() as c:
                c.execute("UPDATE users SET pw_hash=?, role='admin', disabled=0, must_change=1 WHERE id=?", (hash_password(pw), u["id"]))
                c.execute("DELETE FROM sessions WHERE user_id=?", (u["id"],))
        with db() as c:
            c.execute("DELETE FROM meta WHERE k='bootstrap'")
            c.execute("INSERT INTO meta(k,v) VALUES('bootstrap',?)", (fp,))
        audit(None, "bootstrap_admin", email)
        log.warning("bootstrap: admin sign-in applied for %s. You will be asked to choose your own password at first sign-in.", email)
    except (ValueError, *IntegrityError) as e:
        log.error("bootstrap admin not applied: %s", e)


@asynccontextmanager
async def lifespan(_app):
    try:
        init_db()
        bootstrap_admin()
    except Exception:  # keep serving so /v1/health can say what is wrong
        log.exception("startup database setup failed")
    yield


app = FastAPI(title="Aurelius", version=VERSION, docs_url=None, redoc_url=None, openapi_url=None, lifespan=lifespan)


def client_ip(request: Request):
    if CFG["trust_proxy"]:
        if CFG["vercel"]:  # Vercel sets these itself and discards any client-supplied value
            for h in ("x-vercel-forwarded-for", "x-real-ip"):
                if request.headers.get(h):
                    return request.headers[h].split(",")[0].strip()
        xf = request.headers.get("x-forwarded-for", "")
        if xf:
            return xf.split(",")[0].strip()
    return request.client.host if request.client else "unknown"


@app.middleware("http")
async def security_headers(request: Request, call_next):
    resp = await call_next(request)
    resp.headers.setdefault("X-Content-Type-Options", "nosniff")
    resp.headers.setdefault("X-Frame-Options", "DENY")
    resp.headers.setdefault("Referrer-Policy", "no-referrer")
    resp.headers.setdefault("Permissions-Policy", "camera=(), microphone=(), geolocation=()")
    if request.url.path.startswith("/v1/") and not request.url.path.startswith("/v1/catalog"):
        resp.headers["Cache-Control"] = "no-store"
    return resp


def current_user(request: Request):
    authz = request.headers.get("authorization", "")
    if authz.lower().startswith("bearer "):
        key = authz[7:].strip()
        r = db().execute("SELECT k.id AS kid, k.revoked, u.* FROM api_keys k JOIN users u ON u.id=k.user_id WHERE k.key_hash=?", (sha(key),)).fetchone()
        if not r or r["revoked"] or r["disabled"]:
            raise HTTPException(401, "invalid API key")
        with db() as c:
            c.execute("UPDATE api_keys SET last_used=? WHERE id=?", (now(), r["kid"]))
        if r["must_change"]:
            raise HTTPException(403, "password change required before this key can be used")
        return {"id": r["id"], "email": r["email"], "name": r["name"], "role": r["role"], "via": "key", "must_change": 0}
    tok = request.cookies.get(COOKIE)
    if not tok:
        raise HTTPException(401, "not signed in")
    r = db().execute("SELECT u.*, s.expires_at FROM sessions s JOIN users u ON u.id=s.user_id WHERE s.token_hash=?", (sha(tok),)).fetchone()
    if not r or r["expires_at"] < now() or r["disabled"]:
        raise HTTPException(401, "session expired")
    if request.method not in ("GET", "HEAD", "OPTIONS") and request.headers.get("x-requested-with") != "aurelius":
        raise HTTPException(403, "missing X-Requested-With header")
    if r["must_change"] and request.url.path not in ("/v1/me", "/v1/auth/password", "/v1/auth/logout"):
        raise HTTPException(403, "password change required")
    return {"id": r["id"], "email": r["email"], "name": r["name"], "role": r["role"], "via": "session", "must_change": r["must_change"]}


# ---- public ----------------------------------------------------------------
_index_cache = {}


@app.get("/", response_class=HTMLResponse)
def index():
    path = os.path.join(ROOT, "index.html")
    if CFG["dev"] or "html" not in _index_cache:
        try:
            _index_cache["html"] = open(path, encoding="utf-8").read()
        except OSError:
            return HTMLResponse("index.html was not found next to server.py", status_code=503)
    nonce = secrets.token_urlsafe(16)
    csp = ("default-src 'none'; script-src 'nonce-%s'; style-src 'nonce-%s' https://fonts.googleapis.com; style-src-attr 'unsafe-inline'; "
           "font-src https://fonts.gstatic.com; img-src 'self' data:; connect-src 'self'; base-uri 'none'; form-action 'self'; frame-ancestors 'none'") % (nonce, nonce)
    return HTMLResponse(_index_cache["html"].replace("__NONCE__", nonce), headers={"Content-Security-Policy": csp, "Cache-Control": "no-store"})


@app.get("/v1/health")
def health():
    try:
        db().execute("SELECT 1").fetchone()
        stored, err = "postgres" if CFG["db_url"] else "sqlite", ""
    except Exception as e:
        log.error("health: database check failed: %s", e)
        stored, err = "unavailable", "The database is not reachable or not configured."
    return {"status": "ok" if not err else "degraded", "version": VERSION, "data_provider": get_provider().name, "storage": stored,
            "agent_enabled": llm_enabled(), "agent_model": CFG["model"] if llm_enabled() else None, "agent_provider": llm_provider() if llm_enabled() else None,
            **({"error": err} if err else {})}


@app.get("/v1/catalog")
def catalog():
    return {"version": VERSION, "conventions": [
        "Returns, losses, volatilities and probabilities are decimal fractions: 0.0177 means 1.77%.",
        "Every market-data result carries a `data` block: source, as-of time, window, tickers used and any dropped.",
        "Weights are fractions of NAV. Weights that sum to less than 1 leave the remainder in cash.",
        "Authenticate with `Authorization: Bearer aur_...` (API key) or the browser session."],
        "models": [{"name": m.name, "title": m.title, "summary": m.summary, "method": m.method, "limits": m.limits, "example": m.example,
                    "agent_tool": m.agent, "input_schema": {k: v for k, v in m.input.model_json_schema().items() if k != "title"}}
                   for m in REGISTRY.values()]}


# ---- auth ------------------------------------------------------------------
class LoginIn(BaseModel):
    email: str = Field(..., max_length=200)
    password: str = Field(..., max_length=500)


@app.post("/v1/auth/login")
def login(body: LoginIn, request: Request, response: Response):
    ip, email = client_ip(request), body.email.strip().lower()
    keys = [(ip, email), ip]
    if throttled(keys):
        raise HTTPException(429, "Too many failed attempts. Wait 15 minutes and try again.")
    u = user_by_email(email)
    ok = verify_password(body.password, u["pw_hash"] if u else _DUMMY)
    if not (u and ok and not u["disabled"]):
        note_fail(keys)
        audit(u["id"] if u else None, "login_failed", f"{email} from {ip}")
        raise HTTPException(401, "Email or password is incorrect.")
    tok = new_session(u["id"], ip, request.headers.get("user-agent"))
    response.set_cookie(COOKIE, tok, httponly=True, samesite="strict", secure=not CFG["dev"], max_age=CFG["session_hours"] * 3600, path="/")
    audit(u["id"], "login", ip)
    return {"email": u["email"], "name": u["name"], "role": u["role"], "must_change": bool(u["must_change"])}


@app.post("/v1/auth/logout")
def logout(request: Request, response: Response, user=Depends(current_user)):
    tok = request.cookies.get(COOKIE)
    if tok:
        with db() as c:
            c.execute("DELETE FROM sessions WHERE token_hash=?", (sha(tok),))
    response.delete_cookie(COOKIE, path="/")
    audit(user["id"], "logout")
    return {"ok": True}


@app.get("/v1/me")
def me(user=Depends(current_user)):
    return {**{k: user[k] for k in ("email", "name", "role", "via")}, "must_change": bool(user["must_change"])}


class PwIn(BaseModel):
    current: str = Field(..., max_length=500)
    new: str = Field(..., max_length=500)


@app.post("/v1/auth/password")
def change_password(body: PwIn, request: Request, user=Depends(current_user)):
    row = db().execute("SELECT pw_hash FROM users WHERE id=?", (user["id"],)).fetchone()
    if not verify_password(body.current, row["pw_hash"]):
        raise HTTPException(403, "Current password is incorrect.")
    try:
        check_password_policy(body.new)
    except ValueError as e:
        raise HTTPException(422, str(e))
    keep = sha(request.cookies.get(COOKIE, ""))
    with db() as c:
        c.execute("UPDATE users SET pw_hash=?, must_change=0 WHERE id=?", (hash_password(body.new), user["id"]))
        c.execute("DELETE FROM sessions WHERE user_id=? AND token_hash<>?", (user["id"], keep))
    audit(user["id"], "password_changed")
    return {"ok": True}


# ---- admin: issue and manage sign-ins ---------------------------------------
def require_admin(user=Depends(current_user)):
    # Browser sessions only: an API key can never create or change accounts.
    if user["role"] != "admin" or user["via"] != "session":
        raise HTTPException(403, "admin access required")
    return user


_PW_ALPHABET = "ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz23456789"


def temp_password():
    while True:
        pw = "".join(secrets.choice(_PW_ALPHABET) for _ in range(16))
        if any(c.isdigit() for c in pw) and any(c.isupper() for c in pw) and any(c.islower() for c in pw):
            return pw


def _user_row(r):
    return {"id": r["id"], "email": r["email"], "name": r["name"], "role": r["role"], "disabled": bool(r["disabled"]),
            "must_change": bool(r["must_change"]), "created_at": r["created_at"], "last_login": r["last_login"],
            "keys": r["keys"], "portfolios": r["portfolios"]}


_USERS_Q = """SELECT u.*, (SELECT MAX(ts) FROM audit a WHERE a.user_id=u.id AND a.event='login') AS last_login,
              (SELECT COUNT(*) FROM api_keys k WHERE k.user_id=u.id AND k.revoked=0) AS keys,
              (SELECT COUNT(*) FROM portfolios p WHERE p.user_id=u.id) AS portfolios FROM users u """


def _active_admins(c, excluding=None):
    q = "SELECT COUNT(*) FROM users WHERE role='admin' AND disabled=0" + (" AND id<>?" if excluding else "")
    return c.execute(q, (excluding,) if excluding else ()).fetchone()[0]


class NewUserIn(BaseModel):
    email: str = Field(..., max_length=200)
    name: str = Field("", max_length=100)
    role: str = "analyst"


class UserPatchIn(BaseModel):
    name: Optional[str] = Field(None, max_length=100)
    role: Optional[str] = None
    disabled: Optional[bool] = None


@app.get("/v1/admin/users")
def admin_users(user=Depends(require_admin)):
    # Account metadata only. Portfolio contents and conversations are never exposed to admins.
    return [_user_row(r) for r in db().execute(_USERS_Q + "ORDER BY u.id")]


@app.post("/v1/admin/users")
def admin_create_user(body: NewUserIn, user=Depends(require_admin)):
    pw = temp_password()
    try:
        create_user(body.email, pw, body.name, body.role, must_change=True)
    except ValueError as e:
        raise HTTPException(422, str(e))
    except IntegrityError:
        raise HTTPException(409, "an account with that email already exists")
    audit(user["id"], "admin_user_created", f"{body.email.strip().lower()} as {body.role}")
    return {"email": body.email.strip().lower(), "role": body.role, "temporary_password": pw,
            "note": "Shown once. The person must set their own password at first sign-in."}


def _target(uid):
    t = db().execute("SELECT * FROM users WHERE id=?", (uid,)).fetchone()
    if not t:
        raise HTTPException(404, "no such user")
    return t


@app.patch("/v1/admin/users/{uid}")
def admin_update_user(uid: int, body: UserPatchIn, user=Depends(require_admin)):
    t = _target(uid)
    if body.role is not None and body.role not in ("admin", "analyst"):
        raise HTTPException(422, "role must be admin or analyst")
    if uid == user["id"] and (body.disabled or (body.role and body.role != "admin")):
        raise HTTPException(409, "you cannot disable or demote your own account")
    with db() as c:
        if t["role"] == "admin" and not t["disabled"] and (body.disabled or (body.role and body.role != "admin")) and _active_admins(c, uid) == 0:
            raise HTTPException(409, "at least one active admin must remain")
        if body.name is not None:
            c.execute("UPDATE users SET name=? WHERE id=?", (body.name.strip(), uid))
        if body.role is not None and body.role != t["role"]:
            c.execute("UPDATE users SET role=? WHERE id=?", (body.role, uid))
            c.execute("DELETE FROM sessions WHERE user_id=?", (uid,))  # new role applies on next sign-in
            audit(user["id"], "admin_role_changed", f"{t['email']}: {t['role']} -> {body.role}")
        if body.disabled is not None and bool(t["disabled"]) != body.disabled:
            c.execute("UPDATE users SET disabled=? WHERE id=?", (1 if body.disabled else 0, uid))
            if body.disabled:
                c.execute("DELETE FROM sessions WHERE user_id=?", (uid,))
            audit(user["id"], "admin_user_disabled" if body.disabled else "admin_user_enabled", t["email"])
    return _user_row(db().execute(_USERS_Q + "WHERE u.id=?", (uid,)).fetchone())


@app.post("/v1/admin/users/{uid}/reset-password")
def admin_reset_password(uid: int, user=Depends(require_admin)):
    t = _target(uid)
    pw = temp_password()
    with db() as c:
        c.execute("UPDATE users SET pw_hash=?, must_change=1 WHERE id=?", (hash_password(pw), uid))
        c.execute("DELETE FROM sessions WHERE user_id=?", (uid,))
    audit(user["id"], "admin_password_reset", t["email"])
    return {"email": t["email"], "temporary_password": pw, "note": "Shown once. All their sessions were ended."}


@app.post("/v1/admin/users/{uid}/revoke-keys")
def admin_revoke_keys(uid: int, user=Depends(require_admin)):
    t = _target(uid)
    with db() as c:
        n = c.execute("UPDATE api_keys SET revoked=1 WHERE user_id=? AND revoked=0", (uid,)).rowcount
    audit(user["id"], "admin_keys_revoked", f"{t['email']} ({n})")
    return {"revoked": n}


# ---- API keys --------------------------------------------------------------
class KeyIn(BaseModel):
    name: str = Field(..., min_length=1, max_length=60)


@app.get("/v1/keys")
def keys_list(user=Depends(current_user)):
    return [dict(r) for r in db().execute("SELECT id,name,prefix,created_at,last_used FROM api_keys WHERE user_id=? AND revoked=0 ORDER BY id DESC", (user["id"],))]


@app.post("/v1/keys")
def keys_create(body: KeyIn, user=Depends(current_user)):
    if user["via"] == "key":
        raise HTTPException(403, "API keys cannot mint API keys. Sign in to the website.")
    kid, key = new_api_key(user["id"], body.name)
    audit(user["id"], "key_created", body.name)
    return {"id": kid, "name": body.name, "key": key, "note": "Copy this now. It is shown once and only a hash is stored."}


@app.delete("/v1/keys/{kid}")
def keys_revoke(kid: int, user=Depends(current_user)):
    with db() as c:
        n = c.execute("UPDATE api_keys SET revoked=1 WHERE id=? AND user_id=?", (kid, user["id"])).rowcount
    if not n:
        raise HTTPException(404, "key not found")
    audit(user["id"], "key_revoked", str(kid))
    return {"ok": True}


# ---- portfolios ------------------------------------------------------------
NAME_RE = re.compile(r"^[\w .&()\-]{1,60}$")


class PortfolioIn(BaseModel):
    weights: dict[str, float]
    nav_usd: float = Field(1_000_000, gt=0, le=1e13)

    @field_validator("weights")
    @classmethod
    def _w(cls, v):
        return clean_weights(v)


@app.get("/v1/portfolios")
def portfolios_list(user=Depends(current_user)):
    return db_list_portfolios(user["id"])


@app.put("/v1/portfolios/{name}")
def portfolios_put(name: str, body: PortfolioIn, user=Depends(current_user)):
    if not NAME_RE.match(name):
        raise HTTPException(422, "name must be 1-60 characters: letters, digits, spaces and . & ( ) -")
    with db() as c:
        c.execute("""INSERT INTO portfolios(user_id,name,weights,nav_usd,updated_at) VALUES(?,?,?,?,?)
                     ON CONFLICT(user_id,name) DO UPDATE SET weights=excluded.weights, nav_usd=excluded.nav_usd, updated_at=excluded.updated_at""",
                  (user["id"], name, json.dumps(body.weights), body.nav_usd, now()))
    audit(user["id"], "portfolio_saved")
    return db_get_portfolio(user["id"], name)


@app.delete("/v1/portfolios/{name}")
def portfolios_del(name: str, user=Depends(current_user)):
    with db() as c:
        n = c.execute("DELETE FROM portfolios WHERE user_id=? AND lower(name)=lower(?)", (user["id"], name)).rowcount
    if not n:
        raise HTTPException(404, "portfolio not found")
    audit(user["id"], "portfolio_deleted")
    return {"ok": True}


# ---- models ----------------------------------------------------------------
@app.post("/v1/models/{name}")
def models_run(name: str, body: dict = Body(default={}), user=Depends(current_user)):
    if name not in REGISTRY:
        raise HTTPException(404, f"unknown model '{name}'")
    t0 = time.time()
    try:
        res = run_model(name, body, Ctx(user["id"], get_provider()))
    except ValidationError as e:
        raise HTTPException(422, json.loads(e.json(include_url=False)))
    except ModelError as e:
        raise HTTPException(400, str(e))
    except DataError as e:
        raise HTTPException(502, str(e))
    except Exception:
        log.exception("model %s failed", name)
        raise HTTPException(500, "internal error while running the model")
    res.pop("_viz", None)
    audit(user["id"], "model_call", f"{name} via {user['via']}")
    return {"model": name, "elapsed_ms": int((time.time() - t0) * 1000), "result": res}


# ---- agent -----------------------------------------------------------------
class ChatIn(BaseModel):
    message: str = Field(..., min_length=1, max_length=4000)
    conversation_id: str | None = Field(None, max_length=40)


@app.post("/v1/agent/chat")
def agent_chat(body: ChatIn, user=Depends(current_user)):
    return StreamingResponse(agent_stream(user, body.message, body.conversation_id), media_type="text/event-stream",
                             headers={"Cache-Control": "no-store", "X-Accel-Buffering": "no"})


@app.get("/v1/conversations")
def conv_list(user=Depends(current_user)):
    return [dict(r) for r in db().execute("SELECT id,title,updated_at FROM conversations WHERE user_id=? ORDER BY updated_at DESC LIMIT 100", (user["id"],))]


@app.get("/v1/conversations/{cid}")
def conv_get(cid: str, user=Depends(current_user)):
    r = db().execute("SELECT * FROM conversations WHERE id=? AND user_id=?", (cid, user["id"])).fetchone()
    if not r:
        raise HTTPException(404, "conversation not found")
    return {"id": r["id"], "title": r["title"], "messages": ui_messages(json.loads(r["messages"]))}


@app.delete("/v1/conversations/{cid}")
def conv_del(cid: str, user=Depends(current_user)):
    with db() as c:
        n = c.execute("DELETE FROM conversations WHERE id=? AND user_id=?", (cid, user["id"])).rowcount
    if not n:
        raise HTTPException(404, "conversation not found")
    return {"ok": True}


@app.get("/v1/audit")
def audit_list(user=Depends(current_user)):
    q = "SELECT a.ts, a.event, a.detail, u.email FROM audit a LEFT JOIN users u ON u.id=a.user_id "
    rows = db().execute(q + ("ORDER BY a.id DESC LIMIT 200" if user["role"] == "admin" else "WHERE a.user_id=? ORDER BY a.id DESC LIMIT 200"),
                        () if user["role"] == "admin" else (user["id"],))
    return [dict(r) for r in rows]


# ---- command line ----------------------------------------------------------
def selftest():
    """Engine sanity checks. No network, no database."""
    ok = True

    def check(name, cond, info=""):
        nonlocal ok
        ok &= bool(cond)
        print(("PASS " if cond else "FAIL ") + name + (f"  {info}" if info else ""))
    p = SyntheticProvider().panel(["SPY", "TLT", "GLD", "QQQ"], 1260)
    R = p.returns
    d = lw_delta(R)
    check("Ledoit-Wolf intensity in [0,1]", 0 <= d <= 1, f"{d:.4f}")
    for est in ("sample", "shrunk", "dynamic"):
        S, _ = build_cov(R, est)
        check(f"{est} covariance is positive definite", np.linalg.eigvalsh(S).min() > 0)
    w = np.array([.4, .3, .2, .1])
    r = risk_report(R, w, "sample", 0.99, 1)
    check("risk contributions sum to 1", abs(r["rc"].sum() - 1) < 1e-9)
    check("Cornish-Fisher raises VaR when left-skewed", cf_z(2.326, -0.5, 0) > 2.326 > cf_z(2.326, 0.5, 0))
    check("Kupiec p-value is 1 at the expected failure count", kupiec_pvalue(10, 1000, 0.01) > 0.99)
    check("Kupiec rejects 4x too many failures", kupiec_pvalue(40, 1000, 0.01) < 0.001)
    rng = np.random.default_rng(1)
    st = np.zeros(1500, int)
    for t in range(1, 1500):
        st[t] = st[t - 1] if rng.random() < 0.97 else 1 - st[t - 1]
    x = np.where(st == 0, 0.006, 0.02) * rng.standard_normal(1500)
    hm = hmm_fit(x, 2)
    agree = max((hm["path"] == st).mean(), (hm["path"] != st).mean())
    check("HMM recovers two volatility regimes", agree > 0.85, f"agreement {agree:.2f}")
    check("HMM transition rows sum to 1", np.allclose(hm["A"].sum(1), 1))
    g = garch_fit(x)
    check("GARCH parameters stationary", 0 < g["persistence"] < 1, f"persistence {g['persistence']:.3f}")
    s1, d1 = deflated_sharpe(1.0 / math.sqrt(252), 1000, 1, 0.5 / math.sqrt(252))
    s2, d2 = deflated_sharpe(1.0 / math.sqrt(252), 1000, 200, 0.5 / math.sqrt(252))
    check("more trials lower the deflated Sharpe ratio", d2 < d1 and s2 > s1, f"{d1:.3f} -> {d2:.3f}")
    e0 = exec_plan(0.1, 0.018, 0.5, 2.0, 1.0, 0.0)
    check("even-pace cost matches closed form", abs(e0["cost_bps"] - (1e4 * 0.5 * 0.018 * 0.1 / 1.0 + 2.0)) < 1e-6, f"{e0['cost_bps']:.3f} bps")
    e4 = exec_plan(0.1, 0.018, 0.5, 2.0, 1.0, 4.0)
    check("urgency cuts timing risk and raises impact", e4["risk_bps"] < e0["risk_bps"] and e4["impact_bps"] > e0["impact_bps"])
    check("schedule ends fully traded", abs(e4["x"][-1]) < 1e-9 and abs(e4["x"][0] - 1) < 1e-9)
    imp = stress_conditional(np.cov(R, rowvar=False), {0: -0.1})
    check("shocked asset moves by its shock", abs(imp[0] + 0.1) < 1e-12)
    print("ALL PASSED" if ok else "SOME CHECKS FAILED")
    return 0 if ok else 1


def _prompt_password(args):
    pw = os.environ.get("AURELIUS_NEW_PASSWORD")
    if pw:
        return pw
    import getpass
    a, b = getpass.getpass("Password (12+ characters): "), getpass.getpass("Repeat: ")
    if a != b:
        sys.exit("Passwords do not match.")
    return a


def main(argv=None):
    ap = argparse.ArgumentParser(prog="server.py", description="Aurelius quantitative API, agent and website.")
    sub = ap.add_subparsers(dest="cmd")
    s = sub.add_parser("serve", help="run the server")
    s.add_argument("--host", default="127.0.0.1")
    s.add_argument("--port", type=int, default=8000)
    c = sub.add_parser("create-user", help="add an account (there is no public sign-up)")
    c.add_argument("email")
    c.add_argument("--name", default="")
    c.add_argument("--role", default="analyst", choices=["admin", "analyst"])
    sub.add_parser("list-users")
    for n in ("set-password", "disable-user", "enable-user"):
        sub.add_parser(n).add_argument("email")
    sub.add_parser("selftest", help="run engine sanity checks")
    a = ap.parse_args(argv)
    cmd = a.cmd or "serve"
    if cmd == "selftest":
        return selftest()
    init_db()
    if cmd == "create-user":
        try:
            create_user(a.email, _prompt_password(a), a.name, a.role)
        except (ValueError, *IntegrityError) as e:
            sys.exit(f"Could not create user: {e}")
        print(f"Created {a.role} {a.email.lower()}")
    elif cmd == "list-users":
        for r in db().execute("SELECT email,name,role,disabled,created_at FROM users ORDER BY id"):
            print(f"{r['email']:<36} {r['role']:<8} {'DISABLED' if r['disabled'] else 'active':<9} {r['created_at']}  {r['name']}")
    elif cmd == "set-password":
        u = user_by_email(a.email)
        if not u:
            sys.exit("No such user.")
        pw = _prompt_password(a)
        try:
            check_password_policy(pw)
        except ValueError as e:
            sys.exit(str(e))
        with db() as cn:
            cn.execute("UPDATE users SET pw_hash=? WHERE id=?", (hash_password(pw), u["id"]))
            cn.execute("DELETE FROM sessions WHERE user_id=?", (u["id"],))
        print("Password updated and sessions ended.")
    elif cmd in ("disable-user", "enable-user"):
        flag = 1 if cmd == "disable-user" else 0
        with db() as cn:
            n = cn.execute("UPDATE users SET disabled=? WHERE email=?", (flag, a.email.lower())).rowcount
            if flag:
                cn.execute("DELETE FROM sessions WHERE user_id=(SELECT id FROM users WHERE email=?)", (a.email.lower(),))
        print("Done." if n else "No such user.")
    else:
        import uvicorn
        if not CFG["dev"]:
            print("Cookies are Secure: serve over HTTPS, or set AURELIUS_DEV=1 for plain http on localhost.")
        print(f"Aurelius {VERSION} · data={CFG['data']} · agent={(CFG['model'] + ' via ' + llm_provider()) if llm_enabled() else 'off (set AURELIUS_LLM_API_KEY)'} · http://{a.host}:{a.port}")
        uvicorn.run(app, host=a.host, port=a.port, proxy_headers=CFG["trust_proxy"], forwarded_allow_ips="*" if CFG["trust_proxy"] else None, log_level="info")
    return 0


if __name__ == "__main__":
    sys.exit(main())
