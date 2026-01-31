# app.py

# toolbox
import os
import json
import time
import threading
from typing import Dict, List, Set, Tuple, Optional

import requests
import ccxt
import pandas as pd
import numpy as np

from dash import Dash, dcc, html, Input, Output, State, no_update, ctx, ALL
from dash.exceptions import PreventUpdate

from db import list_favorites, add_favorite, remove_favorite
from auth_ui import login_card, register_card, auth_shell, register_auth_callbacks

import plotly.graph_objects as go
from plotly.subplots import make_subplots

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.pipeline import make_pipeline


MAINTENANCE = os.getenv("MAINTENANCE_MODE", "1") == "1"

# Control panel
UI_REFRESH_MS = 1200
TICKERS_REFRESH_MS = 12000

MAX_BARS_ALL = 6000
FETCH_PAGE_SIZE = 1000

COINGECKO_CACHE_FILE = ".coingecko_coinlist_cache.json"
COINGECKO_CACHE_MAX_AGE_SEC = 24 * 60 * 60

DEFAULT_MOVER_QUOTE = "USDT"
TOP_MOVERS_N = 12

FAV_TF = "15m"
FAV_LIMIT = 800
FAV_POLL_SEC = 8
FAV_RETRAIN_SEC = 15 * 60

PROB_THR = 0.52
STRAT_PROX_PCT = 0.40  # %

BT_LEVERAGE = 4

BT_TP_PNL = 0.15                           # +15% PnL
BT_TP_PRICE_MOVE = BT_TP_PNL / BT_LEVERAGE # underlying move (+3.75%)

BT_SL_PNL = -0.10                          # -10% PnL
BT_SL_PRICE_MOVE = abs(BT_SL_PNL) / BT_LEVERAGE # underlying move (-2.50%)

BT_MAX_TRADES_TO_DRAW = 80
BT_CACHE: Dict[Tuple[str, str, str, int], dict] = {}  # (symbol, tf, mode, lim) -> {"trades":[...], "n":..., "buys":[...]}

FAV_POS_FILE = ".fav_positions.json"


# Exchange (do NOT init in maintenance)
exchange = None
if not MAINTENANCE:
    exchange = ccxt.binance({"enableRateLimit": True})


# --------------------------
# CoinGecko symbol->names cache
# --------------------------
def load_coingecko_symbol_to_names() -> Dict[str, List[str]]:
    now = time.time()
    if os.path.exists(COINGECKO_CACHE_FILE):
        try:
            with open(COINGECKO_CACHE_FILE, "r", encoding="utf-8") as f:
                payload = json.load(f)
            ts = float(payload.get("timestamp", 0))
            if now - ts < COINGECKO_CACHE_MAX_AGE_SEC:
                data = payload.get("data", {})
                return {k: list(v) for k, v in data.items()}
        except Exception:
            pass

    url = "https://api.coingecko.com/api/v3/coins/list"
    r = requests.get(url, timeout=30)
    r.raise_for_status()
    coins = r.json()

    symbol_to_names: Dict[str, List[str]] = {}
    for c in coins:
        sym = (c.get("symbol") or "").strip().upper()
        name = (c.get("name") or "").strip()
        if not sym or not name:
            continue
        symbol_to_names.setdefault(sym, [])
        if name not in symbol_to_names[sym]:
            symbol_to_names[sym].append(name)

    try:
        with open(COINGECKO_CACHE_FILE, "w", encoding="utf-8") as f:
            json.dump({"timestamp": now, "data": symbol_to_names}, f, ensure_ascii=False)
    except Exception:
        pass

    return symbol_to_names


COINGECKO_SYMBOL_TO_NAMES = load_coingecko_symbol_to_names()


def pick_display_name(symbol: str) -> str:
    majors = {
        "BTC": "Bitcoin",
        "ETH": "Ethereum",
        "BNB": "BNB",
        "SOL": "Solana",
        "XRP": "XRP",
        "ADA": "Cardano",
        "DOGE": "Dogecoin",
        "DOT": "Polkadot",
        "LINK": "Chainlink",
        "AVAX": "Avalanche",
    }
    s = (symbol or "").upper()
    if s in majors:
        return majors[s]
    names = COINGECKO_SYMBOL_TO_NAMES.get(s, [])
    if not names:
        return s if s else "—"
    return sorted(names, key=lambda x: (len(x), x.lower()))[0]


# --------------------------
# Markets / Base options
# --------------------------
def load_markets_assets() -> Tuple[dict, Set[str], Dict[str, Set[str]]]:
    if MAINTENANCE or exchange is None:
        return {}, set(), {}
    markets = exchange.load_markets()
    base_assets: Set[str] = set()
    base_to_quotes: Dict[str, Set[str]] = {}

    for sym, m in markets.items():
        if not m.get("active", True):
            continue
        base = m.get("base")
        quote = m.get("quote")
        if not base or not quote:
            continue
        base = base.upper()
        quote = quote.upper()
        base_assets.add(base)
        base_to_quotes.setdefault(base, set()).add(quote)

    return markets, base_assets, base_to_quotes


# FIX #1: define build_base_options BEFORE using it
def build_base_options(base_assets: Set[str]) -> List[dict]:
    out = []
    for base in sorted(base_assets):
        names = COINGECKO_SYMBOL_TO_NAMES.get(base, [])
        primary = pick_display_name(base)
        label = f"{primary} ({base})" if primary != base else base
        search_blob = " ".join([base] + names + [primary]).lower()
        out.append({"label": label, "value": base, "search_blob": search_blob})
    return out


MARKETS, BASE_ASSETS, BASE_TO_QUOTES = {}, set(), {}
BASE_OPTIONS: List[dict] = []

if not MAINTENANCE:
    MARKETS, BASE_ASSETS, BASE_TO_QUOTES = load_markets_assets()
    BASE_OPTIONS = build_base_options(BASE_ASSETS)

DEFAULT_BASE = "BTC" if "BTC" in BASE_TO_QUOTES else (BASE_OPTIONS[0]["value"] if BASE_OPTIONS else "BTC")
DEFAULT_QUOTE = "USDT" if "USDT" in BASE_TO_QUOTES.get(DEFAULT_BASE, set()) else (
    sorted(list(BASE_TO_QUOTES.get(DEFAULT_BASE, {"USDT"})))[0]
)


# --------------------------
# Indicators + ML features
# --------------------------
def add_ema(df: pd.DataFrame, period: int) -> pd.DataFrame:
    df[f"EMA_{period}"] = df["close"].ewm(span=period, adjust=False).mean()
    return df


def add_macd(df: pd.DataFrame) -> pd.DataFrame:
    fast, slow, signal = 12, 26, 9
    ema_fast = df["close"].ewm(span=fast, adjust=False).mean()
    ema_slow = df["close"].ewm(span=slow, adjust=False).mean()
    df["MACD"] = ema_fast - ema_slow
    df["MACD_SIGNAL"] = df["MACD"].ewm(span=signal, adjust=False).mean()
    df["MACD_HIST"] = df["MACD"] - df["MACD_SIGNAL"]
    return df


def add_rsi(df: pd.DataFrame, length: int = 14) -> pd.DataFrame:
    out = df.copy()
    delta = out["close"].diff()
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)

    avg_gain = gain.ewm(alpha=1 / length, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / length, adjust=False).mean()

    rs = avg_gain / avg_loss.replace(0, np.nan)
    out["rsi"] = 100 - (100 / (1 + rs))
    return out


def add_atr(df: pd.DataFrame, length: int = 14) -> pd.DataFrame:
    out = df.copy()
    high = out["high"]
    low = out["low"]
    close = out["close"]
    prev_close = close.shift(1)

    tr = pd.concat(
        [(high - low).abs(), (high - prev_close).abs(), (low - prev_close).abs()],
        axis=1,
    ).max(axis=1)

    out["atr"] = tr.ewm(alpha=1 / length, adjust=False).mean()
    return out


def fib_levels(df: pd.DataFrame, lookback: int = 200) -> Dict[str, float]:
    w = df.tail(lookback)
    hi = float(w["high"].max())
    lo = float(w["low"].min())
    rng = hi - lo
    if rng <= 0:
        return {"0": lo, "0.382": lo, "0.5": lo, "0.618": lo, "1": hi}
    return {
        "0": lo,
        "0.382": hi - 0.382 * rng,
        "0.5": hi - 0.5 * rng,
        "0.618": hi - 0.618 * rng,
        "1": hi,
    }


def make_features(df: pd.DataFrame, rsi_len=14, lookback=200, prox_pct=0.40):
    out = df.copy()
    out = add_rsi(out, rsi_len)
    out = add_ema(out, 50)
    out = add_ema(out, 200)
    out = add_atr(out, 14)
    out = add_macd(out)

    out["ret"] = np.log(out["close"]).diff()
    out["ema_spread"] = out["EMA_50"] - out["EMA_200"]
    out["ema_cross_up"] = (
        (out["EMA_50"] > out["EMA_200"]) & (out["EMA_50"].shift(1) <= out["EMA_200"].shift(1))
    ).astype(int)
    out["ema_cross_down"] = (
        (out["EMA_50"] < out["EMA_200"]) & (out["EMA_50"].shift(1) >= out["EMA_200"].shift(1))
    ).astype(int)

    out["macd_edge"] = out["MACD"] - out["MACD_SIGNAL"]
    out["vol_10"] = out["ret"].rolling(10).std()
    out["atr_norm"] = out["atr"] / out["close"]

    lvls = fib_levels(out, lookback=lookback)
    fib_vals = np.array(list(lvls.values()), dtype=float)
    prices = out["close"].values.reshape(-1, 1)
    denom = np.maximum(prices, 1e-9)

    out["fib_rel_dist"] = np.min(np.abs(prices - fib_vals.reshape(1, -1)) / denom, axis=1)
    out["near_fib"] = (out["fib_rel_dist"] <= (prox_pct / 100.0)).astype(int)

    out["y"] = (out["ret"].shift(-1) > 0).astype(int)

    feature_cols = [
        "rsi",
        "ema_spread",
        "ema_cross_up",
        "ema_cross_down",
        "macd_edge",
        "vol_10",
        "atr_norm",
        "fib_rel_dist",
        "near_fib",
    ]
    return out, feature_cols


def fit_prob_model(df: pd.DataFrame, feature_cols, C=1.0):
    base = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=1000, class_weight="balanced", C=C),
    )
    model = CalibratedClassifierCV(base, method="sigmoid", cv=3)
    X, y = df[feature_cols].values, df["y"].values
    model.fit(X, y)
    return model


def compute_buy_candidates(df: pd.DataFrame) -> List[dict]:
    if df is None or df.empty or len(df) < 260:
        return []

    feat_df, feats = make_features(df, rsi_len=14, lookback=200, prox_pct=STRAT_PROX_PCT)
    feat_df = feat_df.dropna().copy()

    model = None
    if len(feat_df) > 350:
        try:
            model = fit_prob_model(feat_df, feats)
        except Exception:
            model = None

    feat_now, _ = make_features(df, rsi_len=14, lookback=200, prox_pct=STRAT_PROX_PCT)
    feat_now = feat_now.dropna().copy()
    feat_now["prob_up"] = 0.5

    if model is not None and feats and not feat_now.empty:
        try:
            X = feat_now[feats].values
            feat_now["prob_up"] = model.predict_proba(X)[:, 1]
        except Exception:
            pass

    trend_up = feat_now["EMA_50"] > feat_now["EMA_200"]
    rsi_ok = feat_now["rsi"] < 50
    near_ok = feat_now["near_fib"] == 1
    macd_ok = feat_now["MACD"] > feat_now["MACD_SIGNAL"]
    p_ok = feat_now["prob_up"] > PROB_THR

    mask = trend_up & rsi_ok & near_ok & macd_ok & p_ok

    buys = []
    for _, r in feat_now.loc[mask].iterrows():
        buys.append({"ts": r["ts"], "price": float(r["close"]), "prob_up": float(r["prob_up"])})
    return buys[-300:]


# --------------------------
# Backtest
# --------------------------
def backtest_buy_tp_sl_only(
    df: pd.DataFrame,
    prob_thr: float,
    leverage: int,
    tp_pnl: float,
    sl_pnl: float,
    rsi_len=14,
    lookback=200,
    prox_pct=0.40,
    max_trades: int = 80
) -> List[dict]:
    if df is None or df.empty or len(df) < 260:
        return []

    tp_move = float(tp_pnl) / float(leverage)
    sl_move = abs(float(sl_pnl)) / float(leverage)
    if tp_move <= 0 or sl_move <= 0:
        return []

    feat_df, feats = make_features(df, rsi_len=rsi_len, lookback=lookback, prox_pct=prox_pct)
    feat_df = feat_df.dropna().copy()

    model = None
    if len(feat_df) > 350:
        try:
            model = fit_prob_model(feat_df, feats)
        except Exception:
            model = None

    prob_series = {}
    if model is not None and feats:
        try:
            feat_now, _ = make_features(df, rsi_len=rsi_len, lookback=lookback, prox_pct=prox_pct)
            feat_now = feat_now.dropna()
            if not feat_now.empty:
                X = feat_now[feats].values
                probs = model.predict_proba(X)[:, 1]
                for ts, p in zip(feat_now["ts"].values, probs):
                    prob_series[pd.to_datetime(ts)] = float(p)
        except Exception:
            prob_series = {}

    w = df.copy()
    w = add_rsi(w, rsi_len)
    w = add_ema(w, 50)
    w = add_ema(w, 200)
    w = add_macd(w)

    lvls = fib_levels(w, lookback=lookback)
    fib_vals = np.array(list(lvls.values()), dtype=float)
    prices = w["close"].values.reshape(-1, 1)
    denom = np.maximum(prices, 1e-9)
    fib_rel = np.min(np.abs(prices - fib_vals.reshape(1, -1)) / denom, axis=1)
    near_fib = fib_rel <= (prox_pct / 100.0)

    trades: List[dict] = []
    in_pos = False
    entry_i = None
    entry_price = None
    tp_price = None
    sl_price = None

    for i in range(2, len(w)):
        if len(trades) >= max_trades:
            break

        ts_i = pd.to_datetime(w["ts"].iloc[i])
        p = float(prob_series.get(ts_i, 0.5))

        trend_up = bool(w["EMA_50"].iloc[i] > w["EMA_200"].iloc[i])
        rsi_ok = float(w["rsi"].iloc[i]) < 50
        macd_ok = float(w["MACD"].iloc[i]) > float(w["MACD_SIGNAL"].iloc[i])
        near_ok = bool(near_fib[i])

        buy_now = (trend_up and rsi_ok and near_ok and macd_ok and (p > prob_thr))

        if not in_pos:
            if buy_now:
                entry_i = i
                entry_price = float(w["close"].iloc[i])
                tp_price = entry_price * (1.0 + tp_move)
                sl_price = entry_price * (1.0 - sl_move)
                in_pos = True
        else:
            hi = float(w["high"].iloc[i])
            lo = float(w["low"].iloc[i])

            hit_tp = (tp_price is not None and hi >= tp_price)
            hit_sl = (sl_price is not None and lo <= sl_price)

            if hit_sl:
                exit_i = i
                exit_price = float(sl_price)
                trades.append(
                    {
                        "entry_i": int(entry_i),
                        "entry_ts": str(w["ts"].iloc[entry_i]),
                        "entry_price": float(entry_price),
                        "exit_i": int(exit_i),
                        "exit_ts": str(w["ts"].iloc[exit_i]),
                        "exit_price": float(exit_price),
                        "tp_price": float(tp_price),
                        "sl_price": float(sl_price),
                        "result": "SL",
                        "bars": int(exit_i - entry_i),
                    }
                )
                in_pos = False
                entry_i = None
                entry_price = None
                tp_price = None
                sl_price = None
            elif hit_tp:
                exit_i = i
                exit_price = float(tp_price)
                trades.append(
                    {
                        "entry_i": int(entry_i),
                        "entry_ts": str(w["ts"].iloc[entry_i]),
                        "entry_price": float(entry_price),
                        "exit_i": int(exit_i),
                        "exit_ts": str(w["ts"].iloc[exit_i]),
                        "exit_price": float(exit_price),
                        "tp_price": float(tp_price),
                        "sl_price": float(sl_price),
                        "result": "TP",
                        "bars": int(exit_i - entry_i),
                    }
                )
                in_pos = False
                entry_i = None
                entry_price = None
                tp_price = None
                sl_price = None

    return trades


# --------------------------
# Formatting
# --------------------------
def format_price(p: float) -> str:
    try:
        if p != p:
            return "—"
    except Exception:
        return "—"
    if p >= 1000:
        return f"{p:,.2f}"
    if p >= 1:
        return f"{p:,.4f}"
    if p >= 0.01:
        return f"{p:,.6f}"
    return f"{p:.8f}"


def pct_str(x: float) -> str:
    sign = "+" if x >= 0 else ""
    return f"{sign}{x:.2f}%"


# --------------------------
# Live state / caches
# --------------------------
LIVE_LOCK = threading.Lock()
LIVE_DF: Optional[pd.DataFrame] = None
LIVE_LAST_POLL: Dict[Tuple[str, str], float] = {}
LIVE_POLL_MIN_SEC = 2.0

SHAPES_MEM: Dict[str, List[dict]] = {}
HIST_CACHE: Dict[Tuple[str, str, str, int], pd.DataFrame] = {}


def fetch_recent_history(symbol: str, timeframe: str, limit: int) -> pd.DataFrame:
    # FIX guard: maintenance or no exchange
    if MAINTENANCE or exchange is None:
        return pd.DataFrame(columns=["ts", "open", "high", "low", "close", "volume"])
    ohlcv = exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
    df = pd.DataFrame(ohlcv, columns=["ts", "open", "high", "low", "close", "volume"])
    df["ts"] = pd.to_datetime(df["ts"], unit="ms", utc=True).dt.tz_convert(None)
    return df


def fetch_max_available_history(symbol: str, timeframe: str, max_bars: int) -> pd.DataFrame:
    if MAINTENANCE or exchange is None:
        return pd.DataFrame(columns=["ts", "open", "high", "low", "close", "volume"])

    tf_map = {
        "1m": 60_000,
        "5m": 5 * 60_000,
        "15m": 15 * 60_000,
        "1h": 60 * 60_000,
        "4h": 4 * 60 * 60_000,
        "1d": 24 * 60 * 60_000,
    }
    step_ms = tf_map.get(timeframe, 60_000)

    now_ms = int(time.time() * 1000)
    probe_since = now_ms - (2000 * 24 * 60 * 60 * 1000)

    all_rows = []
    since = probe_since

    while True:
        chunk = exchange.fetch_ohlcv(symbol, timeframe=timeframe, since=since, limit=FETCH_PAGE_SIZE)
        if not chunk:
            break

        all_rows.extend(chunk)
        if len(all_rows) >= max_bars:
            all_rows = all_rows[-max_bars:]
            break

        last_ts = chunk[-1][0]
        since = last_ts + step_ms
        if len(chunk) < FETCH_PAGE_SIZE:
            break

    df = pd.DataFrame(all_rows, columns=["ts", "open", "high", "low", "close", "volume"])
    if df.empty:
        return pd.DataFrame(columns=["ts", "open", "high", "low", "close", "volume"])
    df["ts"] = pd.to_datetime(df["ts"], unit="ms", utc=True).dt.tz_convert(None)
    df = df.drop_duplicates(subset=["ts"]).sort_values("ts").reset_index(drop=True)
    return df


def fetch_tickers_24h() -> List[dict]:
    if MAINTENANCE or exchange is None:
        return []
    try:
        tickers = exchange.fetch_tickers()
    except Exception:
        return []

    rows = []
    for sym, t in tickers.items():
        if not isinstance(sym, str) or "/" not in sym:
            continue
        _, quote = sym.split("/", 1)
        if quote.upper() != DEFAULT_MOVER_QUOTE:
            continue

        pct = t.get("percentage", None)
        last = t.get("last", None)
        if pct is None or last is None:
            continue

        try:
            rows.append({"symbol": sym.upper(), "pct": float(pct), "last": float(last)})
        except Exception:
            continue

    return rows


# --------------------------
# Favorites background engine
# --------------------------
FAV_LOCK = threading.Lock()
FAV_WORKERS: Dict[str, dict] = {}
FAV_SIGNALS: Dict[str, dict] = {}
FAV_MODELS: Dict[str, dict] = {}

FAV_POS: Dict[str, dict] = {}


def load_fav_pos():
    global FAV_POS
    try:
        if os.path.exists(FAV_POS_FILE):
            with open(FAV_POS_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, dict):
                FAV_POS = {k.upper(): v for k, v in data.items()}
    except Exception:
        pass


def save_fav_pos():
    try:
        with open(FAV_POS_FILE, "w", encoding="utf-8") as f:
            json.dump(FAV_POS, f, ensure_ascii=False, indent=2)
    except Exception:
        pass


load_fav_pos()


def fav_worker_loop(pair: str):
    # FIX guard
    if MAINTENANCE or exchange is None:
        return

    tf = FAV_TF
    limit = FAV_LIMIT

    # FIX #3: safe get to avoid KeyError
    with FAV_LOCK:
        w = FAV_WORKERS.get(pair)
        if not w:
            return
        stop_event = w["stop"]

    while not stop_event.is_set():
        try:
            df = fetch_recent_history(pair, tf, limit=limit)

            if df is None or df.empty or len(df) < 260:
                time.sleep(FAV_POLL_SEC)
                continue

            now = time.time()

            with FAV_LOCK:
                bundle = FAV_MODELS.get(pair)

            need_fit = (
                bundle is None
                or bundle.get("model") is None
                or (now - bundle.get("last_fit", 0) > FAV_RETRAIN_SEC)
            )

            if need_fit:
                feat_df, feats = make_features(df, rsi_len=14, lookback=200, prox_pct=STRAT_PROX_PCT)
                feat_df = feat_df.dropna().copy()
                if len(feat_df) > 300:
                    model = fit_prob_model(feat_df, feats)
                    with FAV_LOCK:
                        FAV_MODELS[pair] = {"model": model, "feats": feats, "last_fit": now}
                else:
                    with FAV_LOCK:
                        FAV_MODELS[pair] = {"model": None, "feats": None, "last_fit": now}

            with FAV_LOCK:
                model = (FAV_MODELS.get(pair) or {}).get("model")
                feats = (FAV_MODELS.get(pair) or {}).get("feats")

            prob_up = None
            if model is not None and feats:
                feat_now, _ = make_features(df, rsi_len=14, lookback=200, prox_pct=STRAT_PROX_PCT)
                feat_now = feat_now.dropna()
                if not feat_now.empty:
                    X_last = feat_now[feats].iloc[[-1]].values
                    prob_up = float(model.predict_proba(X_last)[:, 1][0])

            wdf = df.copy()
            wdf = add_rsi(wdf, 14)
            wdf = add_ema(wdf, 50)
            wdf = add_ema(wdf, 200)
            wdf = add_macd(wdf)

            lvls = fib_levels(wdf, lookback=200)
            fib_vals = np.array(list(lvls.values()), dtype=float)
            prices = wdf["close"].values.reshape(-1, 1)
            denom = np.maximum(prices, 1e-9)
            fib_rel = np.min(np.abs(prices - fib_vals.reshape(1, -1)) / denom, axis=1)
            near_fib = fib_rel <= (STRAT_PROX_PCT / 100.0)

            last = wdf.iloc[-1]
            ts = last["ts"]
            price = float(last["close"])

            p = 0.5 if prob_up is None else float(prob_up)
            trend_up = bool(last["EMA_50"] > last["EMA_200"])
            rsi_ok = float(last["rsi"]) < 50
            macd_ok = float(last["MACD"]) > float(last["MACD_SIGNAL"])
            near_ok = bool(near_fib[-1])

            signal = "BUY" if (trend_up and rsi_ok and near_ok and macd_ok and (p > PROB_THR)) else ""

            with FAV_LOCK:
                pos = FAV_POS.get(pair, {"in_pos": False})

            # ISO time helps sorting
            ts_iso = pd.to_datetime(ts).isoformat()

            if pos.get("in_pos"):
                tp = float(pos.get("tp", 0) or 0)
                sl = float(pos.get("sl", 0) or 0)
                hi = float(wdf["high"].iloc[-1])
                lo = float(wdf["low"].iloc[-1])

                if sl > 0 and lo <= sl:
                    item = {"pair": pair, "time": ts_iso, "signal": "SL", "price": float(sl), "prob_up": prob_up}
                    with FAV_LOCK:
                        FAV_SIGNALS[pair] = item
                        FAV_POS[pair] = {"in_pos": False}
                        save_fav_pos()
                elif tp > 0 and hi >= tp:
                    item = {"pair": pair, "time": ts_iso, "signal": "TP", "price": float(tp), "prob_up": prob_up}
                    with FAV_LOCK:
                        FAV_SIGNALS[pair] = item
                        FAV_POS[pair] = {"in_pos": False}
                        save_fav_pos()
            else:
                if signal == "BUY":
                    entry = float(price)
                    tp = entry * (1.0 + (BT_TP_PNL / BT_LEVERAGE))
                    sl = entry * (1.0 - (abs(BT_SL_PNL) / BT_LEVERAGE))
                    item = {"pair": pair, "time": ts_iso, "signal": "BUY", "price": entry, "prob_up": prob_up}
                    with FAV_LOCK:
                        FAV_SIGNALS[pair] = item
                        FAV_POS[pair] = {"in_pos": True, "entry": entry, "tp": tp, "sl": sl, "entry_time": ts_iso}
                        save_fav_pos()

        except Exception as e:
            with FAV_LOCK:
                FAV_SIGNALS[pair] = {
                    "pair": pair, "time": "", "signal": "ERROR",
                    "price": None, "prob_up": None, "err": str(e)
                }
        time.sleep(FAV_POLL_SEC)


def start_fav_worker(pair: str):
    # FIX guard
    if MAINTENANCE or exchange is None:
        return
    pair = pair.upper()
    with FAV_LOCK:
        if pair in FAV_WORKERS:
            return
        ev = threading.Event()
        th = threading.Thread(target=fav_worker_loop, args=(pair,), daemon=True)
        FAV_WORKERS[pair] = {"stop": ev, "thread": th}
        th.start()


def stop_fav_worker(pair: str):
    pair = pair.upper()
    with FAV_LOCK:
        w = FAV_WORKERS.get(pair)
        if not w:
            return
        w["stop"].set()
        del FAV_WORKERS[pair]
        FAV_MODELS.pop(pair, None)
        FAV_SIGNALS.pop(pair, None)
        save_fav_pos()


# --------------------------
# UI helpers / app init
# --------------------------
app = Dash(__name__, suppress_callback_exceptions=True)
server = app.server
app.title = "Fluxor"


def under_construction_layout():
    css = """
<style>
@keyframes drift {
  0%   { transform: translate3d(0,0,0) scale(1); }
  100% { transform: translate3d(-4%, -6%, 0) scale(1.02); }
}

@keyframes twinkle {
  0%, 100% { opacity: 0.25; }
  50%      { opacity: 0.85; }
}

@keyframes pulseBorder {
  0%, 100% {
    box-shadow: 0 0 18px rgba(80,120,255,0.22), 0 0 40px rgba(80,120,255,0.12);
    border-color: rgba(120,160,255,0.35);
  }
  50% {
    box-shadow: 0 0 28px rgba(120,180,255,0.32), 0 0 70px rgba(120,180,255,0.18);
    border-color: rgba(160,210,255,0.55);
  }
}

.space-layer {
  position: absolute;
  inset: -20%;
  pointer-events: none;
  z-index: 0;
  animation: drift 18s linear infinite alternate;
}

/* Stars layer 1 */
.stars1 {
  background:
    radial-gradient(1px 1px at 20% 30%, rgba(255,255,255,0.70) 50%, transparent 51%),
    radial-gradient(1px 1px at 70% 80%, rgba(255,255,255,0.55) 50%, transparent 51%),
    radial-gradient(1px 1px at 40% 60%, rgba(255,255,255,0.45) 50%, transparent 51%),
    radial-gradient(1px 1px at 90% 20%, rgba(255,255,255,0.60) 50%, transparent 51%),
    radial-gradient(2px 2px at 10% 90%, rgba(255,255,255,0.35) 50%, transparent 51%),
    radial-gradient(1200px 700px at 50% 20%, rgba(80,120,255,0.18), rgba(0,0,0,0)),
    radial-gradient(1000px 600px at 50% 80%, rgba(60,90,255,0.12), rgba(0,0,0,0));
  filter: blur(0px);
  opacity: 0.75;
}

/* Stars layer 2 (twinkle) */
.stars2 {
  background:
    radial-gradient(1px 1px at 15% 75%, rgba(255,255,255,0.60) 50%, transparent 51%),
    radial-gradient(1px 1px at 55% 25%, rgba(255,255,255,0.45) 50%, transparent 51%),
    radial-gradient(1px 1px at 85% 55%, rgba(255,255,255,0.50) 50%, transparent 51%),
    radial-gradient(2px 2px at 35% 15%, rgba(255,255,255,0.35) 50%, transparent 51%),
    radial-gradient(2px 2px at 75% 90%, rgba(255,255,255,0.30) 50%, transparent 51%);
  opacity: 0.55;
  animation: twinkle 3.6s ease-in-out infinite;
}

.center-card {
  position: relative;
  z-index: 2;
  max-width: 760px;
  padding: 36px 32px;
  border-radius: 22px;
  background: rgba(10,15,30,0.55);
  border: 1px solid rgba(120,160,255,0.35);
  backdrop-filter: blur(10px);
  animation: pulseBorder 3.2s ease-in-out infinite;
}

.subtitle {
  font-size: 16px;
  opacity: 0.9;
  line-height: 1.55;
}

.accent {
  color: #9fb7ff;
  font-weight: 900;
}

@media (max-width: 520px) {
  .center-card { padding: 28px 18px; }
}
</style>
"""

    return html.Div(
        style={
            "minHeight": "100vh",
            "display": "flex",
            "alignItems": "center",
            "justifyContent": "center",
            "padding": "40px",
            "color": "white",
            "fontFamily": "ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, Arial",
            "textAlign": "center",
            "position": "relative",
            "overflow": "hidden",
            "background": "linear-gradient(180deg, #040814, #02040a)",
        },
        children=[
            # ✅ CSS animations (inline, no assets) — FIXED
            dcc.Markdown(css, dangerously_allow_html=True),

            # 🌌 animated background layers
            html.Div(className="space-layer stars1"),
            html.Div(className="space-layer stars2"),

            # ✅ content card
            html.Div(
                className="center-card",
                children=[
                    html.H1(
                        "Fluxor",
                        style={
                            "margin": "0 0 14px 0",
                            "fontWeight": 900,
                            "letterSpacing": "0.4px",
                        },
                    ),
                    html.Div(
                        "Remember, the crypto market can be highly volatile and unpredictable.",
                        className="subtitle",
                    ),
                    html.Div(
                        [
                            html.Span("Educate yourself", className="accent"),
                            " so you can make informed decisions.",
                        ],
                        className="subtitle",
                        style={"marginTop": "10px"},
                    ),
                    html.Div(style={"height": "18px"}),
                    html.Div(
                        "Under Construction",
                        style={
                            "fontSize": "14px",
                            "opacity": 0.75,
                            "letterSpacing": "1px",
                        },
                    ),
                ],
            ),
        ],
    )



# FIX #2: do NOT leave an empty else; app.index_string is safe to set always
app.index_string = """
<!DOCTYPE html>
<html>
  <head>
    <meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">

    {%metas%}
    <title>{%title%}</title>
    {%favicon%}
    {%css%}
    <style>
      html, body {
        height: 100%;
        margin: 0;
        padding: 0;
        background: #070B14;
        color: #E6E6E6;
        font-family: ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, Arial;
        overflow-y: auto;
        overflow-x: hidden;
        box-sizing: border-box;
      }
      ::-webkit-scrollbar { width: 10px; height: 10px; }
      ::-webkit-scrollbar-thumb { background: rgba(255,255,255,0.16); border-radius: 10px; }
      ::-webkit-scrollbar-track { background: rgba(255,255,255,0.04); }

    @keyframes popIn {
      0%   { opacity: 0; transform: translateY(10px) scale(0.98); }
      100% { opacity: 1; transform: translateY(0) scale(1); }
    }
    @keyframes logoIn {
      0%   { opacity: 0; transform: translateY(-6px) scale(0.96); filter: blur(1px); }
      100% { opacity: 1; transform: translateY(0) scale(1); filter: blur(0px); }
    }

@keyframes floatUp {
  0%   { transform: translateY(0) translateX(0); opacity: 0; }
  10%  { opacity: 0.35; }
  100% { transform: translateY(-120vh) translateX(40px); opacity: 0; }
}

.bubbles-layer {
  position: absolute;
  inset: 0;
  overflow: hidden;
  z-index: 0;
  pointer-events: none;
}

.bubble {
  position: absolute;
  bottom: -140px;
  border-radius: 999px;
  background: radial-gradient(
    circle at 30% 30%,
    rgba(62,180,255,0.35),
    rgba(39,76,255,0.14),
    rgba(39,76,255,0.0)
  );
  filter: blur(2px);
  animation-name: floatUp;
  animation-timing-function: linear;
  animation-iteration-count: infinite;
  will-change: transform, opacity;
}

@media (max-width: 900px) {
  .app-wrap { padding: 10px !important; }
  .app-row { flex-direction: column !important; }
  .sidebar { width: 100% !important; position: relative !important; top: auto !important; }
  .main { width: 100% !important; min-width: 0 !important; }
  .graph-lg { height: 48vh !important; }
  .graph-md { height: 30vh !important; }
  .graph-sm { height: 30vh !important; }
  .movers-grid { grid-template-columns: repeat(2, minmax(0, 1fr)) !important; }
}

@media (max-width: 520px) {
  .graph-lg { height: 44vh !important; }
  .graph-md { height: 28vh !important; }
  .graph-sm { height: 28vh !important; }
  .movers-grid { grid-template-columns: repeat(1, minmax(0, 1fr)) !important; }
}

* { box-sizing: border-box; }

html, body {
  width: 100%;
  max-width: 100%;
  overflow-x: hidden;
}

.js-plotly-plot, .plot-container, .svg-container {
  max-width: 100% !important;
}

@media (max-width: 900px) {
  .app-wrap { padding: 10px !important; }
  .app-row { flex-direction: column !important; flex-wrap: nowrap !important; min-width: 0 !important; }
  .sidebar { width: 100% !important; max-width: 100% !important; position: relative !important; top: auto !important; padding: 12px !important; }
  .main { width: 100% !important; max-width: 100% !important; min-width: 0 !important; overflow: hidden !important; }
  .graph-lg { height: 46vh !important; }
  .graph-md { height: 34vh !important; }
  .graph-sm { height: 34vh !important; }
  .movers-grid { grid-template-columns: repeat(2, minmax(0, 1fr)) !important; }
  .sidebar-collapsed { display: none !important; }
  .mobile-topbar { display: flex !important; }
}

@media (max-width: 520px) {
  .movers-grid { grid-template-columns: repeat(1, minmax(0, 1fr)) !important; }
  .graph-lg { height: 44vh !important; }
  .graph-md { height: 32vh !important; }
  .graph-sm { height: 32vh !important; }
}

.mobile-topbar { display: none; }

</style>
</head>
<body>
    {%app_entry%}
    <footer>
    {%config%}{%scripts%}{%renderer%}
    </footer>
</body>
</html>
"""


def dark_layout(title: str, uirev: str):
    return dict(
        title=title,
        uirevision=uirev,
        paper_bgcolor="#070B14",
        plot_bgcolor="#070B14",
        font=dict(color="#E6E6E6"),
        xaxis=dict(gridcolor="rgba(255,255,255,0.06)", zeroline=False),
        yaxis=dict(gridcolor="rgba(255,255,255,0.06)", zeroline=False),
        legend=dict(bgcolor="rgba(0,0,0,0)"),
        margin=dict(l=52, r=18, t=55, b=35),
    )


def message_figure(title: str, message: str, uirev: str):
    fig = go.Figure()
    fig.update_layout(**dark_layout(title, uirev))
    fig.add_annotation(
        text=message,
        x=0.5, y=0.5,
        xref="paper", yref="paper",
        showarrow=False,
        font=dict(size=14, color="#E6E6E6"),
        align="center",
    )
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False)
    return fig


def mover_btn(sym: str, last: Optional[float], pct: Optional[float], btn_id: dict, badge_text: Optional[str] = None):
    pct_val = 0.0 if pct is None else float(pct)
    last_val = float("nan") if last is None else float(last)

    pct_color = "rgba(35,205,130,0.18)" if pct_val >= 0 else "rgba(255,70,70,0.18)"
    right = badge_text if badge_text is not None else pct_str(pct_val)

    return html.Button(
        id=btn_id,
        n_clicks=0,
        style={
            "width": "100%",
            "textAlign": "left",
            "padding": "10px 10px",
            "borderRadius": "12px",
            "border": "1px solid rgba(255,255,255,0.10)",
            "backgroundColor": "rgba(7,11,20,0.65)",
            "color": "#E6E6E6",
            "cursor": "pointer",
        },
        children=[
            html.Div(
                style={"display": "flex", "justifyContent": "space-between", "alignItems": "center", "gap": "10px"},
                children=[
                    html.Div(
                        children=[
                            html.Div(sym, style={"fontWeight": 900, "fontSize": "13px"}),
                            html.Div(
                                format_price(last_val) if last is not None else "",
                                style={"opacity": 0.85, "fontSize": "12px", "marginTop": "2px"},
                            ),
                        ]
                    ),
                    html.Div(
                        right,
                        style={
                            "padding": "6px 10px",
                            "borderRadius": "999px",
                            "backgroundColor": "rgba(39,76,255,0.18)" if badge_text else pct_color,
                            "border": "1px solid rgba(255,255,255,0.10)",
                            "fontSize": "12px",
                            "fontWeight": 900,
                        },
                    ),
                ],
            ),
        ],
    )


def sig_style_and_badge(sig: str):
    sig = (sig or "").upper()
    if sig == "BUY":
        return ("BUY", "rgba(35,205,130,0.22)", "rgba(35,205,130,0.55)")
    if sig == "SELL":
        return ("SELL", "rgba(255,70,70,0.18)", "rgba(255,70,70,0.55)")
    if sig == "TP":
        return ("TP", "rgba(170,90,255,0.18)", "rgba(170,90,255,0.55)")
    if sig == "SL":
        return ("SL", "rgba(255,120,0,0.18)", "rgba(255,120,0,0.55)")
    if sig == "IN":
        return ("IN", "rgba(39,76,255,0.18)", "rgba(39,76,255,0.55)")
    if sig == "RUN":
        return ("RUN", "rgba(255,255,255,0.10)", "rgba(255,255,255,0.18)")
    if sig == "ERROR":
        return ("ERR", "rgba(255,185,70,0.18)", "rgba(255,185,70,0.55)")
    return ("", None, None)


# --------------------------
# Layout (Dashboard)
# --------------------------
main_dashboard_layout = html.Div(
    className="app-wrap",
    style={
        "minHeight": "100vh",
        "width": "100%",
        "background": "radial-gradient(1200px 700px at 15% 10%, rgba(39,76,255,0.14), rgba(7,11,20,0)) , #070B14",
        "padding": "14px 14px 18px 14px",
        "boxSizing": "border-box",
    },
    children=[
        dcc.Interval(id="ui_interval", interval=UI_REFRESH_MS, n_intervals=0),
        dcc.Interval(id="tickers_interval", interval=TICKERS_REFRESH_MS, n_intervals=0),
        dcc.Interval(id="signals_interval", interval=1500, n_intervals=0),

        dcc.Store(id="ui_mobile", data={"sidebar_open": True, "tab": "tab_price"}),
        dcc.Store(id="shape_store", data={"symbol": "", "shapes": []}),
        dcc.Store(id="movers_store", data={"rows": []}),
        dcc.Store(id="pair_pick_store", data={"pair": None}),

        html.Div(
            className="app-row",
            style={"display": "flex", "gap": "12px", "alignItems": "flex-start", "flexWrap": "wrap", "minWidth": 0},
            children=[
                # Sidebar
                html.Div(
                    className="sidebar",
                    style={
                        "width": "360px",
                        "maxWidth": "100%",
                        "background": "linear-gradient(180deg, rgba(14,24,48,0.92), rgba(8,12,24,0.92))",
                        "border": "1px solid rgba(255,255,255,0.10)",
                        "borderRadius": "16px",
                        "padding": "14px",
                        "position": "sticky",
                        "top": "12px",
                        "backdropFilter": "blur(10px)",
                        "boxSizing": "border-box",
                    },
                    children=[
                        html.Div("Επιλογές", style={"fontSize": "16px", "fontWeight": 900}),
                        html.Div(style={"height": "10px"}),

                        # Watchlist
                        html.Div(
                            style={
                                "padding": "10px 12px",
                                "borderRadius": "14px",
                                "border": "1px solid rgba(255,255,255,0.10)",
                                "backgroundColor": "rgba(7,11,20,0.55)",
                            },
                            children=[
                                html.Div(
                                    style={"display": "flex", "justifyContent": "space-between", "alignItems": "center"},
                                    children=[
                                        html.Div("Watchlist", style={"fontWeight": 900}),
                                        html.Button(
                                            "＋ Add current",
                                            id="add_to_watchlist",
                                            n_clicks=0,
                                            style={
                                                "padding": "6px 10px",
                                                "borderRadius": "999px",
                                                "border": "1px solid rgba(255,255,255,0.12)",
                                                "backgroundColor": "rgba(7,11,20,0.85)",
                                                "color": "#E6E6E6",
                                                "cursor": "pointer",
                                                "fontSize": "12px",
                                            },
                                        ),
                                    ],
                                ),
                                html.Div(style={"height": "8px"}),
                                html.Div(id="watchlist_view", style={"display": "grid", "gap": "6px"}),
                                html.Div(style={"height": "8px"}),
                                html.Div(
                                    f"Favorites signals run in background @ {FAV_TF} (server-side).",
                                    style={"opacity": 0.75, "fontSize": "12px"},
                                ),
                            ],
                        ),

                        html.Div(style={"height": "12px"}),

                        # Search
                        html.Label("Search crypto (ethereum, solana, pepe…)"),
                        dcc.Input(
                            id="base_search",
                            type="text",
                            placeholder="Γράψε όνομα ή σύμβολο και πάτα Enter",
                            debounce=True,
                            style={
                                "width": "100%",
                                "padding": "10px 12px",
                                "borderRadius": "12px",
                                "border": "1px solid rgba(255,255,255,0.14)",
                                "backgroundColor": "rgba(7,11,20,0.9)",
                                "color": "#E6E6E6",
                                "outline": "none",
                                "marginTop": "8px",
                                "boxSizing": "border-box",
                            },
                        ),

                        html.Div(style={"height": "12px"}),

                        html.Label("Crypto (BASE)"),
                        dcc.Dropdown(
                            id="base_asset",
                            options=[{"label": o["label"], "value": o["value"]} for o in BASE_OPTIONS],
                            value=DEFAULT_BASE,
                            clearable=False,
                            searchable=True,
                            style={"color": "#111"},
                        ),

                        html.Div(style={"height": "12px"}),

                        html.Label("Quote (όλα τα διαθέσιμα στο Binance)"),
                        dcc.Dropdown(
                            id="quote_asset",
                            options=[{"label": q, "value": q} for q in sorted(BASE_TO_QUOTES.get(DEFAULT_BASE, []))],
                            value=DEFAULT_QUOTE,
                            clearable=False,
                            searchable=True,
                            style={"color": "#111"},
                        ),

                        html.Div(style={"height": "12px"}),

                        html.Label("Timeframe"),
                        dcc.Dropdown(
                            id="timeframe",
                            options=[{"label": tf, "value": tf} for tf in ["1m", "5m", "15m", "1h", "4h", "1d"]],
                            value="15m",
                            clearable=False,
                            style={"color": "#111"},
                        ),

                        html.Div(style={"height": "12px"}),

                        html.Label("History mode"),
                        dcc.Dropdown(
                            id="history_mode",
                            options=[
                                {"label": "Recent (limit)", "value": "recent"},
                                {"label": f"Max available (cap {MAX_BARS_ALL})", "value": "all"},
                            ],
                            value="recent",
                            clearable=False,
                            style={"color": "#111"},
                        ),

                        html.Div(style={"height": "12px"}),

                        html.Label("Candles (limit)"),
                        dcc.Dropdown(
                            id="limit",
                            options=[{"label": str(n), "value": n} for n in [200, 300, 500, 800, 1000]],
                            value=500,
                            clearable=False,
                            style={"color": "#111"},
                        ),

                        html.Div(style={"height": "12px"}),

                        html.Label("EMA period"),
                        dcc.Dropdown(
                            id="ema_period",
                            options=[{"label": str(n), "value": n} for n in [9, 20, 50, 100, 200]],
                            value=50,
                            clearable=False,
                            style={"color": "#111"},
                        ),

                        html.Div(style={"height": "12px"}),

                        html.Button(
                            "Clear drawings (this pair)",
                            id="clear_shapes",
                            n_clicks=0,
                            style={
                                "width": "100%",
                                "padding": "10px 12px",
                                "borderRadius": "12px",
                                "border": "1px solid rgba(255,255,255,0.14)",
                                "backgroundColor": "rgba(7,11,20,0.9)",
                                "color": "#E6E6E6",
                                "cursor": "pointer",
                            },
                        ),

                        html.Div(style={"height": "12px"}),

                        html.Div(
                            id="selected_symbol_view",
                            style={
                                "padding": "10px 12px",
                                "borderRadius": "12px",
                                "border": "1px solid rgba(255,255,255,0.12)",
                                "backgroundColor": "rgba(7,11,20,0.75)",
                                "fontSize": "13px",
                                "opacity": 0.95,
                                "whiteSpace": "pre-line",
                                "lineHeight": "1.35",
                            },
                        ),
                    ],
                ),

                # Main
                html.Div(
                    className="main",
                    style={"flex": "1", "minWidth": 0, "maxWidth": "100%", "overflow": "hidden"},
                    children=[
                        html.Div(
                            style={"display": "flex", "justifyContent": "space-between", "alignItems": "center", "marginBottom": "10px"},
                            children=[
                                html.Div("Crypto Dashboard", style={"fontSize": "22px", "fontWeight": 900}),
                                html.Button(
                                    "Logout",
                                    id="btn_logout",
                                    n_clicks=0,
                                    style={
                                        "padding": "10px 12px",
                                        "borderRadius": "12px",
                                        "border": "1px solid rgba(255,255,255,0.14)",
                                        "backgroundColor": "rgba(7,11,20,0.85)",
                                        "color": "#E6E6E6",
                                        "cursor": "pointer",
                                        "fontWeight": 900,
                                        "fontSize": "12px",
                                    },
                                ),
                            ],
                        ),

                        dcc.Graph(
                            id="price_chart",
                            className="graph-lg",
                            figure=message_figure("Loading…", "Fetching data…", "init"),
                            style={"height": "46vh", "width": "100%"},
                            config={
                                "displayModeBar": True,
                                "scrollZoom": True,
                                "displaylogo": False,
                                "responsive": True,
                                "modeBarButtonsToAdd": ["drawline", "drawopenpath", "drawrect", "drawcircle", "eraseshape"],
                                "modeBarButtonsToRemove": ["lasso2d", "select2d"],
                            },
                        ),
                        dcc.Graph(
                            id="ema_chart",
                            className="graph-md",
                            figure=message_figure("Loading…", "Fetching data…", "init"),
                            style={"height": "46vh", "width": "100%"},
                            config={"displayModeBar": True, "scrollZoom": True, "displaylogo": False, "responsive": True},
                        ),
                        dcc.Graph(
                            id="macd_chart",
                            figure=message_figure("Loading…", "Fetching data…", "init"),
                            style={"height": "46vh", "width": "100%"},
                            config={"displayModeBar": True, "scrollZoom": True, "displaylogo": False, "responsive": True},
                        ),

                        # Movers
                        html.Div(
                            style={
                                "marginTop": "12px",
                                "padding": "12px",
                                "borderRadius": "16px",
                                "border": "1px solid rgba(255,255,255,0.10)",
                                "background": "linear-gradient(180deg, rgba(14,24,48,0.60), rgba(8,12,24,0.60))",
                            },
                            children=[
                                html.Div(
                                    style={"display": "flex", "justifyContent": "space-between", "alignItems": "center"},
                                    children=[
                                        html.Div("Top movers (24h)", style={"fontWeight": 900}),
                                        html.Div(
                                            style={"display": "flex", "gap": "8px", "alignItems": "center"},
                                            children=[
                                                html.Div(f"Quote: {DEFAULT_MOVER_QUOTE}", style={"fontSize": "12px", "opacity": 0.85}),
                                                dcc.Dropdown(
                                                    id="movers_mode",
                                                    options=[{"label": "Gainers", "value": "gainers"}, {"label": "Losers", "value": "losers"}],
                                                    value="gainers",
                                                    clearable=False,
                                                    style={"width": "160px", "color": "#111"},
                                                ),
                                            ],
                                        ),
                                    ],
                                ),
                                html.Div(style={"height": "10px"}),
                                html.Div(id="movers_view", className="movers-grid", style={"display": "grid", "gridTemplateColumns": "repeat(3, minmax(0, 1fr))", "gap": "8px"}),
                            ],
                        ),

                        # Signals (favorites)
                        html.Div(
                            style={
                                "marginTop": "12px",
                                "padding": "12px",
                                "borderRadius": "16px",
                                "border": "1px solid rgba(255,255,255,0.10)",
                                "background": "linear-gradient(180deg, rgba(14,24,48,0.60), rgba(8,12,24,0.60))",
                            },
                            children=[
                                html.Div("Signals (favorites)", style={"fontWeight": 900}),
                                html.Div(style={"height": "10px"}),
                                html.Div(
                                    f"Rule+ML server-side @ {FAV_TF}. Retrain every {int(FAV_RETRAIN_SEC/60)} min, polling every {FAV_POLL_SEC}s. one-position-at-a-time. TP(+{int(BT_TP_PNL*100)}%@{BT_LEVERAGE}x) / SL({int(BT_SL_PNL*100)}%@{BT_LEVERAGE}x).",
                                    style={"opacity": 0.75, "fontSize": "12px", "marginBottom": "10px"},
                                ),
                                html.Div(id="signals_view", style={"display": "grid", "gap": "8px"}),
                            ],
                        ),
                    ],
                ),
            ],
        ),
    ],
)


# --------------------------
# Top-level routing (Auth -> App)
# --------------------------
app.layout = html.Div(
    children=[
        dcc.Store(id="session", data={"logged_in": False, "email": None}),
        dcc.Store(id="view", data={"page": "login"}),
        dcc.Store(id="watchlist_store", data={"pairs": []}),
        dcc.Store(id="watchlist_loaded", data=False),
        dcc.Store(id="signals_store", data={"items": []}),
        html.Div(id="page_container"),
    ]
)

app.validation_layout = html.Div(
    [
        dcc.Store(id="session"),
        dcc.Store(id="view"),
        dcc.Store(id="watchlist_store"),
        dcc.Store(id="signals_store"),
        dcc.Interval(id="watchlist_boot"),
        html.Div(id="watchlist_debug"),
        html.Div(id="page_container"),
        auth_shell(login_card()),
        auth_shell(register_card()),
        main_dashboard_layout,
    ]
)


# --------------------------
# Callbacks
# --------------------------
@app.callback(
    Output("watchlist_store", "data", allow_duplicate=True),
    Output("watchlist_loaded", "data"),
    Input("ui_interval", "n_intervals", allow_optional=True),
    State("session", "data"),
    State("watchlist_loaded", "data"),
    prevent_initial_call=True,
)
def load_watchlist_after_dashboard(_n, session, loaded):
    session = session or {}
    if not session.get("logged_in"):
        return {"pairs": []}, False

    if loaded:
        raise PreventUpdate

    email = session.get("email")
    if not email:
        return {"pairs": []}, False

    pairs = list_favorites(email)
    return {"pairs": pairs[:120]}, True


@app.callback(
    Output("watchlist_loaded", "data", allow_duplicate=True),
    Input("session", "data"),
    prevent_initial_call=True,
)
def reset_loaded_on_logout(session):
    session = session or {}
    if not session.get("logged_in"):
        return False
    raise PreventUpdate


@app.callback(
    Output("page_container", "children"),
    Input("session", "data"),
    Input("view", "data"),
)
def route_pages(session, view):
    session = session or {"logged_in": False, "email": None}
    view = view or {"page": "login"}

    if session.get("logged_in"):
        return main_dashboard_layout

    page = (view.get("page") or "login").lower()
    if page == "register":
        return auth_shell(register_card())
    return auth_shell(login_card())


register_auth_callbacks(app)


@app.callback(
    Output("watchlist_view", "children"),
    Input("watchlist_store", "data", allow_optional=True),
    Input("signals_store", "data", allow_optional=True),
)
def render_watchlist(wstore, sstore):
    pairs = (wstore or {}).get("pairs", [])
    if not pairs:
        return html.Div("No favorites yet.", style={"opacity": 0.8, "fontSize": "12px"})

    sig_items = (sstore or {}).get("items", [])
    pair_to_sig = {}
    for it in sig_items:
        p = (it.get("pair") or "").upper()
        if p and p not in pair_to_sig:
            pair_to_sig[p] = (it.get("signal") or "").upper()

    with FAV_LOCK:
        running = set(FAV_WORKERS.keys())
        pos_copy = {k: dict(v) for k, v in FAV_POS.items()}

    rows = []
    for p in pairs:
        p_up = p.upper()

        sig = pair_to_sig.get(p_up, "")
        if not sig and p_up in running:
            sig = "RUN"

        pos = pos_copy.get(p_up)
        if pos and pos.get("in_pos"):
            sig = "IN"

        badge_text, bg_hi, border_hi = sig_style_and_badge(sig)

        btn_style = {
            "flex": "1",
            "padding": "10px 10px",
            "borderRadius": "12px",
            "border": "1px solid rgba(255,255,255,0.10)",
            "backgroundColor": "rgba(7,11,20,0.65)",
            "color": "#E6E6E6",
            "cursor": "pointer",
            "textAlign": "left",
            "fontWeight": 900,
            "display": "flex",
            "alignItems": "center",
            "justifyContent": "space-between",
            "gap": "10px",
        }
        if badge_text:
            btn_style["backgroundColor"] = bg_hi
            btn_style["border"] = f"1px solid {border_hi}"

        rows.append(
            html.Div(
                style={"display": "flex", "gap": "8px", "alignItems": "center"},
                children=[
                    html.Button(
                        id={"type": "watch_pair", "pair": p_up},
                        n_clicks=0,
                        style=btn_style,
                        children=[
                            html.Span(p_up),
                            html.Span(
                                badge_text if badge_text else "",
                                style={
                                    "padding": "5px 10px",
                                    "borderRadius": "999px",
                                    "border": "1px solid rgba(255,255,255,0.14)",
                                    "backgroundColor": "rgba(0,0,0,0.20)",
                                    "fontSize": "12px",
                                    "fontWeight": 900,
                                    "opacity": 0.95,
                                    "display": "inline-block" if badge_text else "none",
                                },
                            ),
                        ],
                    ),
                    html.Button(
                        "✕",
                        id={"type": "watch_remove", "pair": p_up},
                        n_clicks=0,
                        title="Remove from watchlist",
                        style={
                            "width": "42px",
                            "padding": "10px 0",
                            "borderRadius": "12px",
                            "border": "1px solid rgba(255,255,255,0.10)",
                            "backgroundColor": "rgba(255,70,70,0.12)",
                            "color": "#E6E6E6",
                            "cursor": "pointer",
                            "fontWeight": 900,
                        },
                    ),
                ],
            )
        )
    return rows


@app.callback(
    Output("movers_store", "data"),
    Input("tickers_interval", "n_intervals", allow_optional=True),
)
def refresh_movers(_):
    if _ is None:
        raise PreventUpdate
    return {"rows": fetch_tickers_24h()}


@app.callback(
    Output("movers_view", "children"),
    Input("movers_store", "data", allow_optional=True),
    Input("movers_mode", "value", allow_optional=True),
)
def render_movers(mstore, mode):
    rows = (mstore or {}).get("rows", [])
    if not rows:
        return html.Div("Loading movers…", style={"opacity": 0.8, "fontSize": "12px"})

    if mode == "losers":
        rows_sorted = sorted(rows, key=lambda x: x["pct"])[:TOP_MOVERS_N]
    else:
        rows_sorted = sorted(rows, key=lambda x: x["pct"], reverse=True)[:TOP_MOVERS_N]

    return [mover_btn(r["symbol"], r["last"], r["pct"], {"type": "mover_pair", "pair": r["symbol"]}) for r in rows_sorted]


@app.callback(
    Output("watchlist_store", "data"),
    Input("session", "data"),
    Input("add_to_watchlist", "n_clicks", allow_optional=True),
    Input({"type": "watch_remove", "pair": ALL}, "n_clicks", allow_optional=True),
    State("watchlist_store", "data"),
    State("base_asset", "value", allow_optional=True),
    State("quote_asset", "value", allow_optional=True),
    prevent_initial_call=True,
)
def watchlist_controller(session, n_add, n_remove_list, wstore, base, quote):
    trig = ctx.triggered_id

    session = session or {}
    email = session.get("email")
    logged_in = bool(session.get("logged_in"))

    if not logged_in or not email:
        return {"pairs": []}

    if trig == "session":
        pairs = list_favorites(email)
        return {"pairs": pairs[:120]}

    if not ctx.triggered or not ctx.triggered[0].get("value"):
        return no_update

    wstore = wstore or {"pairs": []}
    pairs = [p.upper() for p in wstore.get("pairs", [])]

    if trig == "add_to_watchlist":
        base = base or DEFAULT_BASE
        quote = quote or DEFAULT_QUOTE
        pair = f"{base}/{quote}".upper()

        add_favorite(email, pair)
        if pair not in pairs:
            pairs = [pair] + pairs
        return {"pairs": pairs[:120]}

    if isinstance(trig, dict) and trig.get("type") == "watch_remove":
        pair = (trig.get("pair") or "").upper()
        if pair:
            remove_favorite(email, pair)
            pairs = [p for p in pairs if p != pair]
        return {"pairs": pairs}

    return no_update


@app.callback(
    Output("pair_pick_store", "data"),
    Input({"type": "mover_pair", "pair": ALL}, "n_clicks", allow_optional=True),
    Input({"type": "watch_pair", "pair": ALL}, "n_clicks", allow_optional=True),
    prevent_initial_call=True,
)
def select_pair_from_lists(_, __):
    if not ctx.triggered:
        return no_update
    if not ctx.triggered[0].get("value"):
        return no_update

    trig = ctx.triggered_id
    if not trig or "pair" not in trig:
        return no_update

    pair = trig["pair"]
    if "/" not in pair:
        return no_update
    return {"pair": pair.upper()}


@app.callback(
    Output("base_asset", "options"),
    Input("base_search", "value", allow_optional=True),
    State("base_asset", "value", allow_optional=True),
)
def filter_base_options(search_value, current_base):
    base_map = {o["value"]: o for o in BASE_OPTIONS}

    if not search_value:
        opts = [{"label": o["label"], "value": o["value"]} for o in BASE_OPTIONS]
    else:
        q = search_value.strip().lower()
        filtered = [o for o in BASE_OPTIONS if q in o["search_blob"]]
        if not filtered:
            filtered = BASE_OPTIONS
        opts = [{"label": o["label"], "value": o["value"]} for o in filtered]

    if current_base and current_base in base_map and all(x["value"] != current_base for x in opts):
        opts = [{"label": base_map[current_base]["label"], "value": current_base}] + opts

    return opts


@app.callback(
    Output("pair_pick_store", "data", allow_duplicate=True),
    Input("base_search", "n_submit", allow_optional=True),
    State("base_search", "value", allow_optional=True),
    State("quote_asset", "value", allow_optional=True),
    prevent_initial_call=True,
)
def pick_from_search_enter(_n, text, current_quote):
    if not text:
        return no_update
    q = text.strip().lower()
    if not q:
        return no_update

    filtered = [o for o in BASE_OPTIONS if q in o["search_blob"]]
    if not filtered:
        return no_update

    base = filtered[0]["value"]
    quote = (current_quote or "USDT").upper()
    return {"pair": f"{base}/{quote}".upper()}


@app.callback(
    Output("base_asset", "value"),
    Output("quote_asset", "options"),
    Output("quote_asset", "value"),
    Input("pair_pick_store", "data", allow_optional=True),
    Input("base_asset", "value", allow_optional=True),
    State("quote_asset", "value", allow_optional=True),
)
def unified_pair_and_quote(pair_store, base_value, current_quote):
    if not base_value:
        base_value = DEFAULT_BASE

    picked_pair = (pair_store or {}).get("pair")
    preferred_quote = None

    if picked_pair and "/" in picked_pair:
        picked_base, picked_quote = picked_pair.split("/", 1)
        base_value = (picked_base or DEFAULT_BASE).upper()
        preferred_quote = (picked_quote or "").upper() or None

    quotes = sorted(list(BASE_TO_QUOTES.get(base_value, [])))
    opts = [{"label": q, "value": q} for q in quotes]

    if not quotes:
        return base_value, [], None

    if preferred_quote and preferred_quote in quotes:
        chosen = preferred_quote
    elif current_quote and current_quote in quotes:
        chosen = current_quote
    elif "USDT" in quotes:
        chosen = "USDT"
    else:
        chosen = quotes[0]

    return base_value, opts, chosen


@app.callback(
    Output("signals_store", "data"),
    Input("watchlist_store", "data", allow_optional=True),
    State("signals_store", "data"),
)
def manage_fav_workers(wstore, sstore):
    pairs = [p.upper() for p in (wstore or {}).get("pairs", [])]
    want = set(pairs)

    with FAV_LOCK:
        running = set(FAV_WORKERS.keys())

    for p in sorted(want - running):
        if p in MARKETS:
            start_fav_worker(p)

    for p in sorted(running - want):
        stop_fav_worker(p)

    return sstore or {"items": []}


@app.callback(
    Output("signals_store", "data", allow_duplicate=True),
    Input("signals_interval", "n_intervals", allow_optional=True),
    prevent_initial_call=True,
)
def refresh_signals_store(_):
    if _ is None:
        raise PreventUpdate
    with FAV_LOCK:
        items = list(FAV_SIGNALS.values())

    items = sorted(items, key=lambda x: x.get("time", ""), reverse=True)[:50]
    return {"items": items}


@app.callback(
    Output("signals_view", "children"),
    Input("signals_store", "data", allow_optional=True),
)
def render_signals(sstore):
    items = (sstore or {}).get("items", [])
    if not items:
        return html.Div("No signals yet (favorites running in background).", style={"opacity": 0.8, "fontSize": "12px"})

    cards = []
    for it in items:
        pair = it.get("pair", "")
        sig = (it.get("signal", "") or "").upper()
        price = it.get("price", None)
        prob = it.get("prob_up", None)
        t = it.get("time", "")

        if sig == "ERROR":
            err = it.get("err", "Unknown error")
            cards.append(
                html.Div(
                    style={
                        "padding": "10px 12px",
                        "borderRadius": "14px",
                        "border": "1px solid rgba(255,255,255,0.10)",
                        "backgroundColor": "rgba(255,70,70,0.10)",
                    },
                    children=[
                        html.Div(f"{pair} | ERROR", style={"fontWeight": 900}),
                        html.Div(err, style={"opacity": 0.85, "fontSize": "12px", "marginTop": "6px"}),
                    ],
                )
            )
            continue

        right = f"prob_up: {prob:.3f}" if isinstance(prob, (int, float)) and not np.isnan(prob) else "prob_up: —"
        line1 = f"{pair}  |  {sig}"
        line2 = f"{t}"
        line3 = f"price: {format_price(float(price)) if price is not None else '—'}  |  {right}"

        cards.append(
            html.Div(
                style={
                    "padding": "10px 12px",
                    "borderRadius": "14px",
                    "border": "1px solid rgba(255,255,255,0.10)",
                    "backgroundColor": "rgba(7,11,20,0.55)",
                },
                children=[
                    html.Div(line1, style={"fontWeight": 900}),
                    html.Div(line2, style={"opacity": 0.8, "fontSize": "12px", "marginTop": "3px"}),
                    html.Div(line3, style={"opacity": 0.9, "fontSize": "12px", "marginTop": "6px"}),
                ],
            )
        )

    return cards


@app.callback(
    Output("shape_store", "data"),
    Input("base_asset", "value", allow_optional=True),
    Input("quote_asset", "value", allow_optional=True),
    Input("timeframe", "value", allow_optional=True),
    Input("history_mode", "value", allow_optional=True),
    Input("limit", "value", allow_optional=True),
)
def init_or_restart(base, quote, timeframe, history_mode, limit):
    global LIVE_DF

    base = base or DEFAULT_BASE
    quote = quote or DEFAULT_QUOTE

    symbol = f"{base}/{quote}"
    mode = history_mode or "recent"
    lim = int(limit)

    if symbol not in MARKETS:
        with LIVE_LOCK:
            LIVE_DF = None
        return {"symbol": symbol, "shapes": SHAPES_MEM.get(symbol, [])}

    try:
        cache_key = (symbol, timeframe, mode, lim)
        if cache_key in HIST_CACHE:
            df = HIST_CACHE[cache_key].copy()
        else:
            if mode == "all":
                df = fetch_max_available_history(symbol, timeframe, max_bars=MAX_BARS_ALL)
            else:
                df = fetch_recent_history(symbol, timeframe, limit=lim)
            HIST_CACHE[cache_key] = df.copy()

        with LIVE_LOCK:
            LIVE_DF = df.copy()

        return {"symbol": symbol, "shapes": SHAPES_MEM.get(symbol, [])}

    except Exception as e:
        with LIVE_LOCK:
            LIVE_DF = None
        return {"symbol": symbol, "shapes": SHAPES_MEM.get(symbol, []), "error": str(e)}


@app.callback(
    Output("shape_store", "data", allow_duplicate=True),
    Input("price_chart", "relayoutData", allow_optional=True),
    State("shape_store", "data"),
    prevent_initial_call=True,
)
def persist_shapes(relayout, store):
    if not relayout or not store:
        return no_update
    if "shapes" in relayout and isinstance(relayout["shapes"], list):
        store["shapes"] = relayout["shapes"]
        sym = store.get("symbol", "")
        if sym:
            SHAPES_MEM[sym] = store["shapes"]
        return store
    return no_update


@app.callback(
    Output("shape_store", "data", allow_duplicate=True),
    Input("clear_shapes", "n_clicks", allow_optional=True),
    State("shape_store", "data"),
    prevent_initial_call=True,
)
def clear_shapes(_, store):
    if not store:
        return no_update
    store["shapes"] = []
    sym = store.get("symbol", "")
    if sym:
        SHAPES_MEM[sym] = []
    return store


@app.callback(
    Output("price_chart", "figure"),
    Output("ema_chart", "figure"),
    Output("macd_chart", "figure"),
    Output("selected_symbol_view", "children"),
    Input("ui_interval", "n_intervals", allow_optional=True),
    State("base_asset", "value", allow_optional=True),
    State("quote_asset", "value", allow_optional=True),
    State("timeframe", "value", allow_optional=True),
    State("history_mode", "value", allow_optional=True),
    State("limit", "value", allow_optional=True),
    State("ema_period", "value", allow_optional=True),
    State("shape_store", "data", allow_optional=True),
)
def render_live(_, base, quote, timeframe, history_mode, limit, ema_period, shape_store):
    if _ is None:
        raise PreventUpdate

    global LIVE_DF

    base = base or DEFAULT_BASE
    quote = quote or DEFAULT_QUOTE
    symbol = f"{base}/{quote}"

    uirev = f"{symbol}-{timeframe}"
    mode = history_mode or "recent"
    lim = int(limit)

    if symbol in MARKETS and mode != "all":
        now = time.time()
        key = (symbol, timeframe)
        last_poll = LIVE_LAST_POLL.get(key, 0.0)
        if now - last_poll >= LIVE_POLL_MIN_SEC:
            try:
                df_new = fetch_recent_history(symbol, timeframe, limit=lim)
                with LIVE_LOCK:
                    LIVE_DF = df_new.copy()
                LIVE_LAST_POLL[key] = now
            except Exception:
                pass

    with LIVE_LOCK:
        df = None if LIVE_DF is None else LIVE_DF.copy()

    if df is None or df.empty:
        fig = message_figure(symbol, "No data. (Check terminal for errors/rate limits).", uirev)
        return fig, fig, fig, f"⚠️ No data for {symbol}"

    if mode != "all":
        df = df.tail(lim).reset_index(drop=True)
    else:
        df = df.reset_index(drop=True)

    df = add_ema(df, int(ema_period))
    df = add_macd(df)

    shapes = (shape_store or {}).get("shapes", [])

    last_price = float(df["close"].iloc[-1])
    title_main = f"{symbol} — {format_price(last_price)}"

    bt_key = (symbol, timeframe, mode, lim)
    if bt_key in BT_CACHE:
        trades = BT_CACHE[bt_key].get("trades", [])
        buy_candidates = BT_CACHE[bt_key].get("buys", [])
    else:
        trades = backtest_buy_tp_sl_only(
            df=df,
            prob_thr=PROB_THR,
            leverage=BT_LEVERAGE,
            tp_pnl=BT_TP_PNL,
            sl_pnl=BT_SL_PNL,
            rsi_len=14,
            lookback=200,
            prox_pct=STRAT_PROX_PCT,
            max_trades=BT_MAX_TRADES_TO_DRAW
        )
        buy_candidates = compute_buy_candidates(df)
        BT_CACHE[bt_key] = {"trades": trades, "n": len(trades), "buys": buy_candidates}

    fig_price = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.05,
        row_heights=[0.72, 0.28],
        specs=[[{"type": "candlestick"}], [{"type": "bar"}]],
    )
    fig_price.add_trace(
        go.Candlestick(
            x=df["ts"],
            open=df["open"], high=df["high"], low=df["low"], close=df["close"],
            name="Price",
        ),
        row=1, col=1
    )
    fig_price.add_trace(
        go.Bar(x=df["ts"], y=df["volume"], name="Volume", opacity=0.85),
        row=2, col=1
    )

    if buy_candidates:
        bx = [b["ts"] for b in buy_candidates]
        by = [b["price"] for b in buy_candidates]
        fig_price.add_trace(
            go.Scatter(
                x=bx, y=by,
                mode="markers",
                name="BUY candidates (history)",
                marker=dict(symbol="circle", size=5),
                hovertemplate="BUY candidate<br>%{x}<br>price=%{y}<extra></extra>",
                opacity=0.65,
            ),
            row=1, col=1
        )

    entry_x, entry_y = [], []
    tp_x, tp_y = [], []
    sl_x, sl_y = [], []

    if trades:
        for tr in trades:
            ei = tr["entry_i"]
            exi = tr["exit_i"]
            xi = df["ts"].iloc[ei]
            xj = df["ts"].iloc[exi]
            entry = float(tr["entry_price"])
            tp = float(tr.get("tp_price", np.nan))
            sl = float(tr.get("sl_price", np.nan))
            exitp = float(tr["exit_price"])
            res = (tr.get("result") or "").upper()

            entry_x.append(xi); entry_y.append(entry)

            if res == "TP":
                tp_x.append(xj); tp_y.append(exitp)
            elif res == "SL":
                sl_x.append(xj); sl_y.append(exitp)

            if np.isfinite(tp):
                fig_price.add_trace(
                    go.Scatter(x=[xi, xj], y=[tp, tp], mode="lines", opacity=0.28, hoverinfo="skip", showlegend=False),
                    row=1, col=1
                )
                fig_price.add_annotation(
                    x=xj, y=tp, xref="x", yref="y",
                    text="TP", showarrow=True, arrowhead=2, ax=12, ay=-18,
                    font=dict(size=11, color="#E6E6E6"),
                    bgcolor="rgba(0,0,0,0.35)",
                    bordercolor="rgba(255,255,255,0.18)",
                    borderwidth=1,
                )

            if np.isfinite(sl):
                fig_price.add_trace(
                    go.Scatter(x=[xi, xj], y=[sl, sl], mode="lines", opacity=0.22, hoverinfo="skip", showlegend=False),
                    row=1, col=1
                )
                fig_price.add_annotation(
                    x=xj, y=sl, xref="x", yref="y",
                    text="SL", showarrow=True, arrowhead=2, ax=12, ay=18,
                    font=dict(size=11, color="#E6E6E6"),
                    bgcolor="rgba(0,0,0,0.35)",
                    bordercolor="rgba(255,255,255,0.18)",
                    borderwidth=1,
                )

        fig_price.add_trace(
            go.Scatter(
                x=entry_x, y=entry_y,
                mode="markers",
                name="BUY entries (trades)",
                marker=dict(symbol="circle", size=9),
                hovertemplate="BUY ENTRY<br>%{x}<br>entry=%{y}<extra></extra>",
            ),
            row=1, col=1
        )

        if tp_x:
            fig_price.add_trace(
                go.Scatter(
                    x=tp_x, y=tp_y,
                    mode="markers",
                    name=f"TP hit (+{int(BT_TP_PNL*100)}% @ {BT_LEVERAGE}x)",
                    marker=dict(symbol="triangle-down", size=10),
                    hovertemplate="TP HIT<br>%{x}<br>tp=%{y}<extra></extra>",
                ),
                row=1, col=1
            )

        if sl_x:
            fig_price.add_trace(
                go.Scatter(
                    x=sl_x, y=sl_y,
                    mode="markers",
                    name=f"SL hit ({int(BT_SL_PNL*100)}% @ {BT_LEVERAGE}x)",
                    marker=dict(symbol="x", size=10),
                    hovertemplate="SL HIT<br>%{x}<br>sl=%{y}<extra></extra>",
                ),
                row=1, col=1
            )

    fig_price.update_xaxes(rangeslider_visible=False, showspikes=True, spikemode="across", spikesnap="cursor")
    fig_price.update_yaxes(showspikes=True, spikemode="across", spikesnap="cursor")
    fig_price.update_layout(**dark_layout(title_main, uirev))
    fig_price.update_layout(dragmode="pan", hovermode="x unified")
    if shapes:
        fig_price.update_layout(shapes=shapes)

    fig_ema = go.Figure()
    fig_ema.add_trace(go.Scatter(x=df["ts"], y=df["close"], name="Close", mode="lines"))
    fig_ema.add_trace(go.Scatter(x=df["ts"], y=df[f"EMA_{int(ema_period)}"], name=f"EMA {int(ema_period)}", mode="lines"))
    fig_ema.update_layout(**dark_layout(title_main, uirev))
    fig_ema.update_layout(dragmode="pan", hovermode="x unified")

    fig_macd = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.05, row_heights=[0.60, 0.40])
    fig_macd.add_trace(go.Scatter(x=df["ts"], y=df["MACD"], name="MACD", mode="lines"), row=1, col=1)
    fig_macd.add_trace(go.Scatter(x=df["ts"], y=df["MACD_SIGNAL"], name="Signal", mode="lines"), row=1, col=1)
    fig_macd.add_trace(go.Bar(x=df["ts"], y=df["MACD_HIST"], name="Histogram", opacity=0.9), row=2, col=1)
    fig_macd.update_layout(**dark_layout(title_main, uirev))
    fig_macd.update_layout(dragmode="pan", hovermode="x unified")

    primary = pick_display_name(base)
    bt_info = (
        f"Overlay: TP +{int(BT_TP_PNL*100)}% @ {BT_LEVERAGE}x (move +{BT_TP_PRICE_MOVE*100:.2f}%) "
        f"| SL {int(BT_SL_PNL*100)}% @ {BT_LEVERAGE}x (move -{BT_SL_PRICE_MOVE*100:.2f}%)\n"
        f"Trades drawn: {len(trades)} | BUY candidates shown: {len(buy_candidates)}"
    )

    sidebar = (
        f"Επιλεγμένο: {primary} ({base}) / {quote}\n"
        f"Τιμή: {format_price(last_price)} | TF: {timeframe}\n"
        f"History: {'Max available' if mode=='all' else f'Recent ({lim})'}\n"
        f"Data: polling (no websockets) | refresh >= {LIVE_POLL_MIN_SEC:.1f}s\n"
        f"Drawings saved for pair: {symbol}\n"
        f"{bt_info}\n"
        f"Strategy: thr={PROB_THR}, prox={STRAT_PROX_PCT}% | one-position-at-a-time"
    )

    return fig_price, fig_ema, fig_macd, sidebar


# --------------------------
# Cleanup
# --------------------------
import atexit

@atexit.register
def _cleanup():
    with FAV_LOCK:
        running = list(FAV_WORKERS.keys())
    for p in running:
        stop_fav_worker(p)
    save_fav_pos()


# IMPORTANT: if maintenance => force only under construction page
if MAINTENANCE:
    app.layout = under_construction_layout()
    app.validation_layout = app.layout


if __name__ == "__main__":
    app.run(debug=False, host="0.0.0.0", port=8050)








