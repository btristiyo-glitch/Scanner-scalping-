import os
import csv
import time
import random
import logging
import requests
import pandas as pd
import yfinance as yf

from datetime import datetime, timedelta
from pathlib import Path
from ta.momentum import RSIIndicator
from ta.volatility import AverageTrueRange

# =========================
# CONFIG
# =========================
TOKEN = os.getenv("TELEGRAM_TOKEN")
CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")

CSV_FILE = "signals.csv"
ALERT_FILE = "alerts.csv"
STOCK_FILE = "stocks.txt"
SECTOR_FILE = "sectors.csv"
FAILED_FILE = "failed_tickers.txt"
ERROR_LOG = "scanner_errors.log"
FAILED_CACHE_FILE = "failed_cache.csv"

FAST_MODE = True

if FAST_MODE:
    REQUEST_DELAY = 0.6
    BATCH_SIZE = 6
    BATCH_PAUSE_SECONDS = 2.0
    MAX_RETRIES = 2
    USE_FUNDAMENTALS = False
    USE_INTRADAY = True
else:
    REQUEST_DELAY = 1.5
    BATCH_SIZE = 4
    BATCH_PAUSE_SECONDS = 4.0
    MAX_RETRIES = 3
    USE_FUNDAMENTALS = True
    USE_INTRADAY = True

FAILED_CACHE_TTL_HOURS = 24

MIN_DAILY_VALUE = 5_000_000_000
MIN_PRICE = 50

STRONG_ACCUMULATION = -1_000_000_000
MEDIUM_ACCUMULATION = -250_000_000
MEDIUM_DISTRIBUTION = 250_000_000
STRONG_DISTRIBUTION = 1_000_000_000

BLACKLIST = {
    "BBKP", "BNII", "FREN", "GIAA", "INAF", "MLPL", "WSKT", "XLAX", "ZATA"
}

logging.basicConfig(
    filename=ERROR_LOG,
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)

# =========================
# HELPERS
# =========================
def log_error(msg):
    try:
        logging.info(msg)
    except:
        pass

def send(msg, parse_mode="Markdown"):
    if not TOKEN or not CHAT_ID:
        print("Telegram token/chat id belum diset.")
        return

    msg = str(msg)
    for i in range(0, len(msg), 3500):
        chunk = msg[i:i + 3500]
        for attempt in range(3):
            try:
                requests.post(
                    f"https://api.telegram.org/bot{TOKEN}/sendMessage",
                    json={
                        "chat_id": CHAT_ID,
                        "text": chunk,
                        "disable_web_page_preview": True,
                        "parse_mode": parse_mode,
                    },
                    timeout=15,
                )
                break
            except Exception as e:
                log_error(f"Telegram send error attempt={attempt+1}: {e}")
                if attempt == 2:
                    print(f"Gagal kirim Telegram: {e}")
                time.sleep(2 + attempt)

def normalize_ticker(text):
    t = str(text).strip().upper()
    if not t or t.startswith("#"):
        return ""
    if t.endswith(".JK"):
        t = t[:-3]
    return t

def normalize_columns(df):
    if df is None or df.empty:
        return None

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]

    rename_map = {}
    for c in df.columns:
        cl = str(c).strip().lower()
        if cl in {"adj close", "open", "high", "low", "close", "volume"}:
            rename_map[c] = cl.title() if cl != "adj close" else "Close"

    if rename_map:
        df = df.rename(columns=rename_map)

    needed = {"Open", "High", "Low", "Close", "Volume"}
    if not needed.issubset(set(df.columns)):
        return None

    df = df.dropna(subset=["Close", "Volume"])
    if df.empty:
        return None

    return df

def sleep_jitter(base):
    time.sleep(base + random.uniform(0.05, 0.35))

# =========================
# FAILED CACHE
# =========================
def load_failed_cache():
    cache = {}
    p = Path(FAILED_CACHE_FILE)
    if not p.exists():
        return cache

    now = time.time()
    ttl_sec = FAILED_CACHE_TTL_HOURS * 3600

    try:
        with open(FAILED_CACHE_FILE, "r", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                ticker = row.get("ticker", "").strip().upper()
                ts = row.get("ts", "").strip()
                if not ticker or not ts:
                    continue
                try:
                    tsf = float(ts)
                except:
                    continue
                if now - tsf <= ttl_sec:
                    cache[ticker] = tsf
    except:
        return cache

    return cache

def save_failed_cache(ticker, reason):
    ticker = normalize_ticker(ticker)
    if not ticker:
        return

    file_exists = Path(FAILED_CACHE_FILE).exists()
    try:
        with open(FAILED_CACHE_FILE, "a", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            if not file_exists:
                w.writerow(["ticker", "ts", "reason"])
            w.writerow([ticker, time.time(), reason])
    except:
        pass

def save_failed_ticker(ticker, reason):
    try:
        with open(FAILED_FILE, "a", encoding="utf-8") as f:
            f.write(f"{ticker},{reason}\n")
    except:
        pass
    save_failed_cache(ticker, reason)
    log_error(f"FAILED {ticker}: {reason}")

# =========================
# LOAD LISTS
# =========================
def load_stock_list():
    if not os.path.exists(STOCK_FILE):
        print("stocks.txt tidak ditemukan.")
        return []

    stocks = []
    seen = set()
    with open(STOCK_FILE, "r", encoding="utf-8") as f:
        for line in f:
            t = normalize_ticker(line)
            if not t:
                continue
            if t in BLACKLIST:
                continue
            if t in seen:
                continue
            seen.add(t)
            stocks.append(f"{t}.JK")
    return stocks

def load_sector_map():
    sector_map = {}
    if os.path.exists(SECTOR_FILE):
        with open(SECTOR_FILE, "r", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                ticker = normalize_ticker(row.get("ticker", ""))
                sector = str(row.get("sector", "OTHER")).strip().upper() or "OTHER"
                if ticker:
                    sector_map[ticker] = sector
    return sector_map

# =========================
# FETCH DATA
# =========================
def fetch_data(symbol, period="6mo", interval="1d"):
    for attempt in range(MAX_RETRIES):
        try:
            df = yf.download(
                symbol,
                period=period,
                interval=interval,
                auto_adjust=True,
                progress=False,
                threads=False,
                group_by="column",
                timeout=20
            )
            df = normalize_columns(df)
            if df is None or len(df) < 30:
                raise ValueError("empty_or_invalid_dataframe")
            return df
        except Exception as e:
            log_error(f"fetch_data {symbol} attempt={attempt+1}: {e}")
            if attempt == MAX_RETRIES - 1:
                return None
            time.sleep((2 ** attempt) + random.uniform(0.2, 0.8))

def fetch_intraday(symbol):
    for attempt in range(MAX_RETRIES):
        try:
            df = yf.download(
                symbol,
                period="5d",
                interval="15m",
                auto_adjust=True,
                progress=False,
                threads=False,
                group_by="column",
                timeout=20
            )
            df = normalize_columns(df)
            if df is None or df.empty or len(df) < 10:
                raise ValueError("empty_or_invalid_intraday")

            daily = df.resample("D").agg({
                "Open": "first",
                "High": "max",
                "Low": "min",
                "Close": "last",
                "Volume": "sum"
            }).dropna()

            if len(daily) < 2:
                raise ValueError("insufficient_daily_intraday")

            open_today = float(daily["Open"].iloc[-1])
            close_yest = float(daily["Close"].iloc[-2])
            gap_pct = ((open_today - close_yest) / close_yest) * 100 if close_yest else 0
            recent = df.tail(16)
            support = float(recent["Low"].min())
            current_price = float(df["Close"].iloc[-1])

            return open_today, close_yest, gap_pct, support, current_price
        except Exception as e:
            log_error(f"fetch_intraday {symbol} attempt={attempt+1}: {e}")
            if attempt == MAX_RETRIES - 1:
                return None, None, None, None, None
            time.sleep((2 ** attempt) + random.uniform(0.2, 0.8))

def get_fundamentals(ticker):
    if not USE_FUNDAMENTALS:
        return None, None, None
    try:
        stock = yf.Ticker(ticker)
        info = stock.info or {}
        return info.get("priceToBook"), info.get("trailingPE"), info.get("marketCap")
    except Exception as e:
        log_error(f"get_fundamentals {ticker}: {e}")
        return None, None, None

# =========================
# FLOW
# =========================
def flow_score(net_flow_idr):
    score = 0
    label = "NEUTRAL"

    if net_flow_idr <= STRONG_ACCUMULATION:
        score += 35
        label = "ACCUMULATION STRONG"
    elif net_flow_idr <= MEDIUM_ACCUMULATION:
        score += 20
        label = "ACCUMULATION"
    elif net_flow_idr >= STRONG_DISTRIBUTION:
        score -= 30
        label = "DISTRIBUTION STRONG"
    elif net_flow_idr >= MEDIUM_DISTRIBUTION:
        score -= 15
        label = "DISTRIBUTION"

    return score, label

def get_flow_proxy(df):
    try:
        close = df["Close"].squeeze()
        volume = df["Volume"].squeeze()

        price = float(close.iloc[-1])
        vol_now = float(volume.iloc[-1])
        vol_avg20 = float(volume.tail(20).mean())

        if vol_avg20 <= 0 or price <= 0:
            return 0

        rvol = vol_now / vol_avg20
        traded_value_today = price * vol_now
        ema20 = float(close.ewm(span=20, adjust=False).mean().iloc[-1])
        ema50 = float(close.ewm(span=50, adjust=False).mean().iloc[-1])

        if rvol >= 5 and price > ema20:
            return -traded_value_today
        elif rvol >= 3 and price > ema20:
            return -traded_value_today * 0.5
        elif rvol >= 4 and price < ema20:
            return traded_value_today * 0.5
        elif rvol >= 2 and price < ema20:
            return traded_value_today
        elif price > ema20 > ema50:
            return -traded_value_today * 0.25
        elif price < ema20 < ema50:
            return traded_value_today * 0.25

        return 0
    except Exception as e:
        log_error(f"get_flow_proxy: {e}")
        return 0

# =========================
# SETUPS
# =========================
def level_entry(item, price, support, atr_val):
    if item["tipe"] == "breakout":
        entry_limit = max(support, price * 0.995) if support else price * 0.995
        atr_buffer = max(price * 0.01, atr_val * 0.8)
        stop = min(entry_limit - atr_buffer, price - atr_buffer)
        tp1 = entry_limit + max(atr_val * 1.5, price * 0.04)
        tp2 = entry_limit + max(atr_val * 2.5, price * 0.08)
        tp3 = entry_limit + max(atr_val * 3.5, price * 0.12)
    else:
        entry_limit = price * 0.995
        atr_buffer = max(price * 0.008, atr_val * 0.7)
        stop = price - max(atr_buffer, price * 0.03)
        tp1 = price + max(atr_val * 1.2, price * 0.03)
        tp2 = price + max(atr_val * 2.0, price * 0.06)
        tp3 = None

    stop = max(1, stop)
    tp1 = max(1, tp1)
    tp2 = max(1, tp2)
    return round(entry_limit, 0), round(stop, 0), round(tp1, 0), round(tp2, 0), (round(tp3, 0) if tp3 else None)

# =========================
# TELEGRAM
# =========================
def send_short_summary(mom_count, rev_count, fail_count, total_count):
    msg = (
        "📊 *SCAN SUMMARY*\n"
        f"• Total: {total_count}\n"
        f"• Momentum: {mom_count}\n"
        f"• Reversal: {rev_count}\n"
        f"• Fail load: {fail_count}\n"
    )
    send(msg)

def send_setups(items, kind="momentum"):
    if not items:
        return

    msg = "🔥 *BREAKOUT MOMENTUM*\n" if kind == "momentum" else "🔄 *REVERSAL EXTREME*\n"

    for item in items:
        tv = f"https://www.tradingview.com/chart/?symbol=IDX:{item['ticker']}"
        rr1 = round(
            (item["tp1"] - item["entry_limit"]) / (item["entry_limit"] - item["stop_loss"]),
            2
        ) if item["entry_limit"] > item["stop_loss"] else 0

        msg += (
            f"\n• *#{item['ticker']}* | S{item['score']}\n"
            f"  {item['flow_label']} | Rp{item['flow_idr']:,.0f}\n"
            f"  Entry Rp{item['entry_limit']:,.0f} | SL Rp{item['stop_loss']:,.0f}\n"
            f"  TP1 Rp{item['tp1']:,.0f} | TP2 Rp{item['tp2']:,.0f}\n"
            f"  RVOL {item['rvol']}x | RSI {item['rsi']} | R:R {rr1}\n"
            f"  [TV]({tv})"
        )
        if item.get("tp3"):
            msg += f"\n  TP3 Rp{item['tp3']:,.0f}"

        if len(msg) > 3300:
            break

    send(msg)

# =========================
# MAIN
# =========================
def morning_scan():
    print(f"\n=== SCAN {datetime.now().strftime('%H:%M')} ===")

    if not os.path.exists(CSV_FILE):
        with open(CSV_FILE, "w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow([
                "date", "time", "ticker", "score",
                "rsi", "price", "rvol", "gap_pct",
                "breakout", "atr_pct", "flow_label",
                "flow_idr", "entry_limit", "stop_loss",
                "tp1", "tp2", "tp3", "signal_type"
            ])

    stocks = load_stock_list()
    if not stocks:
        return

    sector_map = load_sector_map()
    failed_cache = load_failed_cache()

    results_momentum = []
    results_reversal = []
    failed_tickers = []
    all_setups = []

    for i, stock in enumerate(stocks, start=1):
        ticker_clean = stock.replace(".JK", "")

        if ticker_clean in failed_cache:
            failed_tickers.append(stock)
            continue

        print(f"Scanning {i}/{len(stocks)}: {stock}")
        df = fetch_data(stock, period="6mo")

        if df is None:
            failed_tickers.append(stock)
            save_failed_ticker(stock, "fetch_data_none")
            sleep_jitter(REQUEST_DELAY)
            continue

        try:
            close = df["Close"].squeeze()
            high = df["High"].squeeze()
            low = df["Low"].squeeze()
            volume = df["Volume"].squeeze()

            price = float(close.iloc[-1])
            if price <= 0:
                save_failed_ticker(stock, "price_le_0")
                sleep_jitter(REQUEST_DELAY)
                continue

            rsi = RSIIndicator(close, window=4).rsi()
            rsi_now = float(rsi.iloc[-1])
            rsi_prev = float(rsi.iloc[-2]) if len(rsi) > 1 else rsi_now

            ema20 = float(close.ewm(span=20, adjust=False).mean().iloc[-1])
            ema50 = float(close.ewm(span=50, adjust=False).mean().iloc[-1])

            avg_vol = float(volume.tail(20).mean())
            daily_value = price * float(volume.iloc[-1])

            if avg_vol <= 0 or daily_value <= 0:
                save_failed_ticker(stock, "bad_volume_or_value")
                sleep_jitter(REQUEST_DELAY)
                continue

            rvol = float(volume.iloc[-1]) / avg_vol

            atr = AverageTrueRange(high, low, close, window=14).average_true_range()
            atr_val = float(atr.iloc[-1])
            atr_pct = (atr_val / price) * 100 if price else 0

            prev_20_high = float(close.shift(1).tail(20).max())
            prev_20_low = float(close.shift(1).tail(20).min())
            breakout_high = price > prev_20_high
            range_pct = ((prev_20_high - prev_20_low) / prev_20_low) * 100 if prev_20_low else 0

            open_today, close_yest, gap_pct, support, current_price = (None, None, None, None, None)
            if USE_INTRADAY:
                open_today, close_yest, gap_pct, support, current_price = fetch_intraday(stock)

            pbv, per, mcap = get_fundamentals(stock)
            fundamental_flag = ""
            if pbv and per and mcap:
                if pbv < 2 and per < 25 and 500_000_000_000 < mcap < 50_000_000_000_000:
                    fundamental_flag = "FUND OK"

            sector = sector_map.get(ticker_clean, "OTHER")

            atr_threshold = max(2.5, min(8.0, (rvol * 0.9) + 1.5))
            if atr_pct < atr_threshold or daily_value < MIN_DAILY_VALUE or price < MIN_PRICE:
                sleep_jitter(REQUEST_DELAY)
                continue

            proxy_flow_idr = get_flow_proxy(df)
            flow_bonus, flow_label = flow_score(proxy_flow_idr)

            item = {
                "ticker": ticker_clean,
                "rsi": round(rsi_now, 1),
                "price": round(price, 0),
                "rvol": round(rvol, 2),
                "gap_pct": round(gap_pct, 2) if gap_pct is not None else 0,
                "breakout": breakout_high,
                "range_pct": round(range_pct, 1),
                "atr_pct": round(atr_pct, 2),
                "fundamental": fundamental_flag,
                "sector": sector,
                "daily_value": round(daily_value, 0),
                "open_today": round(open_today, 0) if open_today else 0,
                "support": round(support, 0) if support else 0,
                "flow_idr": round(proxy_flow_idr, 0),
                "flow_label": flow_label,
            }

            score_m = 0
            if rvol > 5:
                score_m += 35
            elif rvol > 3:
                score_m += 20
            elif rvol > 2:
                score_m += 10

            if gap_pct is not None and gap_pct > 3:
                score_m += min(30, gap_pct * 4)
            elif gap_pct is not None and gap_pct > 0:
                score_m += 8

            if breakout_high:
                score_m += 15

            if rsi_now > 50 and rsi_now > rsi_prev:
                score_m += 10
            elif rsi_now > 40 and rsi_now > rsi_prev:
                score_m += 5

            if 3 < range_pct < 25:
                score_m += 10
            elif range_pct >= 25:
                score_m += 5

            if price > ema20:
                score_m += 5
            if ema20 > ema50:
                score_m += 5

            score_m += flow_bonus
            item["score"] = round(score_m, 1)
            item["tipe"] = "breakout"

            strong_flow_breakout = (
                flow_label in ["ACCUMULATION STRONG", "ACCUMULATION"]
                and breakout_high
                and rvol >= 2.5
                and price > ema20
            )

            if score_m >= 40 and strong_flow_breakout:
                el, sl, tp1, tp2, tp3 = level_entry(item, price, support, atr_val)
                item["entry_limit"] = el
                item["stop_loss"] = sl
                item["tp1"] = tp1
                item["tp2"] = tp2
                item["tp3"] = tp3
                results_momentum.append(item)
                all_setups.append(item)

            score_r = 0
            if rsi_now < 22:
                score_r += 30
            elif rsi_now < 28:
                score_r += 20
            elif rsi_now < 35:
                score_r += 10

            score_r += max(0, min(20, (rsi_now - rsi_prev) * 5))
            score_r += min(15, rvol * 3)

            if price < ema20:
                score_r += 10
            if not breakout_high:
                score_r += 10
            if fundamental_flag:
                score_r += 5
            if flow_label in ["DISTRIBUTION STRONG", "DISTRIBUTION"]:
                score_r -= 10

            if score_r >= 25:
                item_rev = item.copy()
                item_rev["tipe"] = "reversal"
                item_rev["score"] = round(score_r, 1)

                el, sl, tp1, tp2, tp3 = level_entry(item_rev, price, support, atr_val)
                item_rev["entry_limit"] = el
                item_rev["stop_loss"] = sl
                item_rev["tp1"] = tp1
                item_rev["tp2"] = tp2
                item_rev["tp3"] = tp3

                results_reversal.append(item_rev)
                all_setups.append(item_rev)

        except Exception as e:
            failed_tickers.append(f"{stock}: {e}")
            save_failed_ticker(stock, str(e))
            sleep_jitter(REQUEST_DELAY)
            continue

        if i % BATCH_SIZE == 0:
            time.sleep(BATCH_PAUSE_SECONDS)
        else:
            sleep_jitter(REQUEST_DELAY)

    results_momentum = sorted(results_momentum, key=lambda x: x["score"], reverse=True)
    results_reversal = sorted(results_reversal, key=lambda x: x["score"], reverse=True)

    with open(ALERT_FILE, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["ticker", "entry_limit", "tp1", "tp2", "tp3", "stop_loss", "score", "tipe", "flow_label", "flow_idr"])
        for item in all_setups:
            w.writerow([
                item["ticker"], item["entry_limit"], item["tp1"],
                item["tp2"], item["tp3"], item["stop_loss"],
                item["score"], item["tipe"], item["flow_label"], item["flow_idr"]
            ])

    now = datetime.now()
    with open(CSV_FILE, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        for item in results_momentum[:10]:
            w.writerow([
                now.strftime("%Y-%m-%d"), now.strftime("%H:%M"),
                item["ticker"], item["score"], item["rsi"], item["price"],
                item["rvol"], item["gap_pct"], item["breakout"],
                item["atr_pct"], item["flow_label"], item["flow_idr"],
                item["entry_limit"], item["stop_loss"],
                item["tp1"], item["tp2"], item["tp3"], item["tipe"]
            ])

    send_short_summary(
        len(results_momentum),
        len(results_reversal),
        len(failed_tickers),
        len(stocks)
    )

    if results_momentum:
        send_setups(results_momentum[:8], kind="momentum")

    if results_reversal:
        send_setups(results_reversal[:6], kind="reversal")

    if failed_tickers:
        fail_msg = "⚠️ *FAILED TICKERS*\n"
        for ft in failed_tickers[:8]:
            fail_msg += f"• {ft}\n"
        if len(failed_tickers) > 8:
            fail_msg += f"• ...dan {len(failed_tickers) - 8} lainnya"
        send(fail_msg)

    print(f"\n✅ SCAN SELESAI — {datetime.now().strftime('%H:%M')}")
    print(f"Momentum: {len(results_momentum)} | Reversal: {len(results_reversal)}")

if __name__ == "__main__":
    print("=" * 50)
    print("NEUROBRO SCANNER - ACCUMULATION + DISTRIBUTION + FAST_MODE")
    print(f"Mulai: {datetime.now().strftime('%H:%M')}")
    print("=" * 50)
    morning_scan()
