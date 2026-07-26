#!/usr/bin/env python3
import os
import sys
import csv
import time
import math
import json
import random
import logging
import requests
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

# =========================
# CONFIG
# =========================
TICKER_FILE = "stocks.txt"
ALERT_FILE = "alert.csv"
LOG_FILE = "scanner.log"

TELEGRAM_BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN", "").strip()
TELEGRAM_CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID", "").strip()

REQUEST_DELAY = 0.15
MIN_DAILY_VALUE = 5_000_000_000
COOLDOWN_DAYS = 5

# trigger entry alert only when price is near/past entry
ENTRY_TRIGGER_BUFFER_PCT = 1.0  # alert if price within 1% of entry
ENTRY_ALERT_MODE = "entry"

# =========================
# LOGGER
# =========================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

# =========================
# SECTOR MAP
# =========================
sector_map = {
    "ADRO": "BATUBARA", "PTBA": "BATUBARA", "BUMI": "BATUBARA", "ITMG": "BATUBARA",
    "HRUM": "BATUBARA", "MYOH": "BATUBARA", "KKGI": "BATUBARA", "ARII": "BATUBARA",
    "GEMS": "BATUBARA", "TOBA": "BATUBARA", "BSSR": "BATUBARA", "DEWA": "BATUBARA",
    "BBCA": "BANK", "BBRI": "BANK", "BMRI": "BANK", "BNGA": "BANK",
    "BDMN": "BANK", "BNLI": "BANK", "NISP": "BANK", "BBTN": "BANK",
    "MEGA": "BANK", "MAYA": "BANK", "AGRO": "BANK",
    "TLKM": "INFRA", "EXCL": "INFRA", "ISAT": "INFRA", "TOWR": "INFRA",
    "MTEL": "INFRA", "TBIG": "INFRA", "CENT": "INFRA", "PENG": "INFRA",
    "ASII": "AUTOMOTIF", "AUTO": "AUTOMOTIF", "INDF": "KONSUMER", "ICBP": "KONSUMER",
    "MYOR": "KONSUMER", "CPIN": "KONSUMER", "JPFA": "KONSUMER", "MAIN": "KONSUMER",
    "UNVR": "KONSUMER", "HMSP": "KONSUMER", "GGRM": "KONSUMER",
    "PGAS": "ENERGI", "MEDC": "ENERGI", "AKRA": "ENERGI", "RAJA": "ENERGI",
    "ANTM": "MINERBA", "INCO": "MINERBA", "MDKA": "MINERBA", "ADMR": "MINERBA",
    "SMGR": "SEMEN", "INTP": "SEMEN", "SMBR": "SEMEN", "KLBF": "FARMASI",
    "KAEF": "FARMASI", "SIDO": "FARMASI", "WIKA": "KONSTRUKSI", "PTPP": "KONSTRUKSI",
    "ADHI": "KONSTRUKSI", "WSKT": "KONSTRUKSI", "BSDE": "PROPERTI",
    "CTRA": "PROPERTI", "PWON": "PROPERTI", "SMRA": "PROPERTI",
    "MAPI": "RITEL", "ERAA": "RITEL", "ACES": "RITEL", "RALS": "RITEL",
    "AMRT": "RITEL", "LPKR": "PROPERTI",
}

# =========================
# HELPERS
# =========================
def sleep_jitter(base=0.15):
    time.sleep(round(random.uniform(base * 0.5, base * 1.5), 3))

def load_ticker_list(file_path):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"{file_path} not found")
    with open(file_path, "r", encoding="utf-8") as f:
        return [line.strip().upper() for line in f if line.strip()]

def fetch_data(ticker, retries=2):
    url = f"https://query1.finance.yahoo.com/v8/finance/chart/{ticker}.JK"
    params = {"range": "6mo", "interval": "1d", "includePrePost": "false"}
    headers = {"User-Agent": "Mozilla/5.0"}
    for _ in range(retries):
        try:
            r = requests.get(url, params=params, headers=headers, timeout=20)
            if r.status_code == 200:
                return r.json()
        except Exception as e:
            logging.warning(f"fetch_data error {ticker}: {e}")
            sleep_jitter(0.3)
    return None

def extract_series(data):
    try:
        result = data.get("chart", {}).get("result", [None])[0]
        if not result:
            return None
        timestamps = result.get("timestamp", [])
        quotes = result.get("indicators", {}).get("quote", [{}])[0]
        adj = result.get("indicators", {}).get("adjclose", [{}])[0]
        df = pd.DataFrame({
            "open": quotes.get("open", []),
            "high": quotes.get("high", []),
            "low": quotes.get("low", []),
            "close": quotes.get("close", []),
            "volume": quotes.get("volume", []),
            "adjclose": adj.get("adjclose", [])
        }, index=pd.to_datetime(timestamps, unit="s"))
        df = df.dropna(subset=["close", "volume"])
        df = df[df["volume"] > 0]
        if len(df) < 30:
            return None
        return df
    except Exception as e:
        logging.warning(f"extract_series error: {e}")
        return None

def calc_rsi(series, period=14):
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(period).mean()
    avg_loss = loss.rolling(period).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))

def calc_atr(df, period=14):
    high, low, close = df["high"], df["low"], df["close"]
    tr = pd.concat([
        high - low,
        (high - close.shift()).abs(),
        (low - close.shift()).abs()
    ], axis=1).max(axis=1)
    return tr.rolling(period).mean()

def calc_support_resistance(df, lookback=20):
    recent = df.tail(lookback)
    return float(recent["low"].min()), float(recent["high"].max())

def td_sequential(close, period=4):
    if len(close) < period + 5:
        return 0
    score = 0
    for i in range(1, period + 1):
        if close.iloc[-i] > close.iloc[-i - period]:
            score += 1
        elif close.iloc[-i] < close.iloc[-i - period]:
            score -= 1
    return score

def get_flow_proxy(df):
    if df is None or len(df) < 10:
        return 0.0
    close = df["close"].values
    volume = df["volume"].values
    high = df["high"].values
    low = df["low"].values
    net = 0.0
    for i in range(1, min(len(df), 30)):
        tp = (high[i] + low[i] + close[i]) / 3.0
        if close[i] > close[i - 1]:
            net += tp * volume[i]
        elif close[i] < close[i - 1]:
            net -= tp * volume[i]
    return float(net)

def flow_score(net_flow_idr, avg_daily_value):
    score = 0
    label = "NEUTRAL"
    if avg_daily_value <= 0:
        return score, label
    flow_pct = (net_flow_idr / avg_daily_value) * 100
    if flow_pct <= -30:
        score += 35
        label = "ACCUMULATION STRONG"
    elif flow_pct <= -10:
        score += 20
        label = "ACCUMULATION"
    elif flow_pct >= 30:
        score -= 30
        label = "DISTRIBUTION STRONG"
    elif flow_pct >= 10:
        score -= 15
        label = "DISTRIBUTION"
    return score, label

def level_entry(signal_type, price, support, atr_val):
    if signal_type == "breakout":
        entry = max(support, price * 0.995) if support else price * 0.995
        atr_buffer = max(price * 0.01, atr_val * 0.8)
        stop = min(entry - atr_buffer, price - atr_buffer)
        tp1 = entry + min(atr_val * 1.5, price * 0.06)
        tp2 = entry + min(atr_val * 2.5, price * 0.10)
        tp3 = entry + min(atr_val * 3.5, price * 0.15)
    else:
        entry = price * 0.995
        atr_buffer = max(price * 0.008, atr_val * 0.7)
        stop = price - max(atr_buffer, price * 0.03)
        tp1 = price + min(atr_val * 1.2, price * 0.04)
        tp2 = price + min(atr_val * 2.0, price * 0.08)
        tp3 = None
    stop = max(1, stop)
    return round(entry, 0), round(stop, 0), round(tp1, 0), round(tp2, 0), (round(tp3, 0) if tp3 else None)

def check_cooldown(ticker, alert_file, days=COOLDOWN_DAYS):
    if not os.path.exists(alert_file):
        return False
    cutoff = datetime.now() - timedelta(days=days)
    try:
        with open(alert_file, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row.get("ticker") != ticker:
                    continue
                ts = row.get("timestamp", "")
                if not ts:
                    return True
                try:
                    dt = datetime.fromisoformat(ts)
                    if dt > cutoff:
                        return True
                except:
                    return True
    except:
        return False
    return False

# =========================
# TELEGRAM
# =========================
def telegram_send(text, markdown=True, retries=3):
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        logging.warning("Telegram env missing")
        return False
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    payload = {
        "chat_id": TELEGRAM_CHAT_ID,
        "text": text,
        "disable_web_page_preview": True,
    }
    if markdown:
        payload["parse_mode"] = "Markdown"
    for i in range(retries):
        try:
            r = requests.post(url, json=payload, timeout=15)
            if r.status_code == 200:
                return True
            logging.warning(f"Telegram failed {r.status_code}: {r.text[:200]}")
        except Exception as e:
            logging.warning(f"Telegram error attempt {i+1}: {e}")
            time.sleep(1.5)
    return False

def format_signal_message(item):
    tp3 = item.get("tp3")
    tp3_text = f"{tp3:,.0f}" if tp3 else "-"
    return (
        f"{'🟢' if item['type']=='breakout' else '🟣'} *{item['ticker']}* - {item['type'].upper()}\n"
        f"Sector: {item.get('sector','LAIN')}\n"
        f"Entry: {item['entry']:,.0f}\n"
        f"SL: {item['stop']:,.0f}\n"
        f"TP1: {item['tp1']:,.0f} | TP2: {item['tp2']:,.0f} | TP3: {tp3_text}\n"
        f"Score: {item.get('score_m', item.get('score_r', 0)):.1f}\n"
        f"RSI: {item['rsi']:.1f} | RVOL: {item['rvol']:.2f}x | Flow: {item['flow']}\n"
        f"Price: {item['price']:,.0f} | Gap: {item['chg_pct']:.2f}%\n"
        f"https://www.tradingview.com/chart/?symbol=IDX:{item['ticker']}"
    )

def format_plain_signal_message(item):
    tp3 = item.get("tp3")
    tp3_text = f"{tp3:,.0f}" if tp3 else "-"
    return (
        f"{item['ticker']} - {item['type'].upper()}\n"
        f"Sector: {item.get('sector','LAIN')}\n"
        f"Entry: {item['entry']:,.0f}\n"
        f"SL: {item['stop']:,.0f}\n"
        f"TP1: {item['tp1']:,.0f} | TP2: {item['tp2']:,.0f} | TP3: {tp3_text}\n"
        f"Score: {item.get('score_m', item.get('score_r', 0)):.1f}\n"
        f"RSI: {item['rsi']:.1f} | RVOL: {item['rvol']:.2f}x | Flow: {item['flow']}\n"
        f"Price: {item['price']:,.0f} | Gap: {item['chg_pct']:.2f}%\n"
        f"TradingView: IDX:{item['ticker']}"
    )

def send_entry_alert(item):
    msg = format_signal_message(item)
    ok = telegram_send(msg, markdown=True)
    if not ok:
        logging.warning(f"Markdown Telegram failed for {item['ticker']} - retry plain text")
        ok = telegram_send(format_plain_signal_message(item), markdown=False)
    return ok

def send_summary_alert(results_m, results_r, scanned):
    total = len(results_m) + len(results_r)
    text = (
        f"Scanner selesai - {datetime.now().strftime('%Y-%m-%d %H:%M WIB')}\n"
        f"Scanned: {scanned}\n"
        f"Breakout: {len(results_m)}\n"
        f"Reversal: {len(results_r)}\n"
        f"Total setup: {total}"
    )
    telegram_send(text, markdown=False)

# =========================
# SCAN
# =========================
def scan_market(tickers):
    results_m = []
    results_r = []
    scanned = 0

    for raw_ticker in tickers:
        scanned += 1
        ticker = raw_ticker.replace(".JK", "").strip()
        sleep_jitter(REQUEST_DELAY)

        if check_cooldown(ticker, ALERT_FILE):
            continue

        data = fetch_data(ticker)
        df = extract_series(data)
        if df is None or len(df) < 30:
            continue

        close = df["close"]
        volume = df["volume"]
        price = float(close.iloc[-1])
        prev_price = float(close.iloc[-2])
        chg_pct = ((price / prev_price) - 1) * 100 if prev_price > 0 else 0

        daily_value = float(price * volume.iloc[-1])
        avg_daily_value = float((close.tail(20) * volume.tail(20)).mean()) if len(close) >= 20 else daily_value
        if daily_value < MIN_DAILY_VALUE:
            continue

        atr_series = calc_atr(df)
        atr_val = float(atr_series.iloc[-1]) if pd.notna(atr_series.iloc[-1]) else price * 0.02
        rsi_series = calc_rsi(close)
        rsi_val = float(rsi_series.iloc[-1]) if pd.notna(rsi_series.iloc[-1]) else 50.0

        avg_vol_20 = float(volume.tail(20).mean()) if len(volume) >= 20 else float(volume.mean())
        rvol = float(volume.iloc[-1] / avg_vol_20) if avg_vol_20 > 0 else 1.0

        sma20 = float(close.tail(20).mean()) if len(close) >= 20 else price
        sma50 = float(close.tail(50).mean()) if len(close) >= 50 else price
        sma200 = float(close.tail(200).mean()) if len(close) >= 200 else price
        td_setup = td_sequential(close)
        support, resistance = calc_support_resistance(df)
        proxy_flow_idr = get_flow_proxy(df)
        flow_bonus, flow_label = flow_score(proxy_flow_idr, avg_daily_value)
        sector = sector_map.get(ticker, "LAIN")

        # -- SCORE M (BREAKOUT) --
        score_m = 0
        score_m += (atr_val / price) * 300
        score_m += rvol * 10
        score_m += flow_bonus
        score_m += min(chg_pct, 10.0) * 2.0
        if price > sma50 > sma200:
            score_m += 15
        if price > sma20:
            score_m += 5
        if td_setup >= 4:
            score_m += 10
        if rsi_val < 30 or rsi_val > 70:
            score_m -= 5

        # -- SCORE R (REVERSAL) --
        score_r = 0
        score_r += rvol * 5
        if rsi_val < 30:
            score_r += (30 - rsi_val) * 2
        score_r += abs(flow_bonus) * 0.5
        score_r += (atr_val / price) * 200
        if td_setup >= 6:
            score_r += 15
        if rvol > 1.5:
            score_r += 8

        flow_risky = flow_label in ["DISTRIBUTION", "DISTRIBUTION STRONG"]
        gap_pct = ((price / prev_price) - 1) * 100 if prev_price > 0 else 0

        # ---- BREAKOUT SIGNAL ----
        if not flow_risky and score_m >= 40 and 2 <= gap_pct <= 5:
            if any(x["ticker"] == ticker for x in results_m):
                continue
            entry, stop, tp1, tp2, tp3 = level_entry("breakout", price, support, atr_val)
            item = {
                "ticker": ticker, "type": "breakout", "sector": sector,
                "price": price, "prev": prev_price, "chg_pct": round(gap_pct, 2),
                "volume": int(volume.iloc[-1]), "value": int(daily_value),
                "rvol": round(rvol, 2), "atr_pct": round((atr_val / price) * 100, 2),
                "rsi": round(rsi_val, 1), "flow": flow_label,
                "score_m": round(score_m, 1),
                "entry": entry, "stop": stop, "tp1": tp1, "tp2": tp2, "tp3": tp3,
                "support": round(support, 0), "resistance": round(resistance, 0),
            }
            results_m.append(item)

        # ---- REVERSAL SIGNAL ----
        if score_r >= 25 and rsi_val < 30 and rvol > 1.5 and chg_pct < -2:
            if any(x["ticker"] == ticker for x in results_r):
                continue
            entry, stop, tp1, tp2, tp3 = level_entry("reversal", price, support, atr_val)
            item = {
                "ticker": ticker, "type": "reversal", "sector": sector,
                "price": price, "prev": prev_price, "chg_pct": round(chg_pct, 2),
                "volume": int(volume.iloc[-1]), "value": int(daily_value),
                "rvol": round(rvol, 2), "atr_pct": round((atr_val / price) * 100, 2),
                "rsi": round(rsi_val, 1), "flow": flow_label,
                "score_r": round(score_r, 1),
                "entry": entry, "stop": stop, "tp1": tp1, "tp2": tp2, "tp3": tp3 if tp3 else 0,
                "support": round(support, 0), "resistance": round(resistance, 0),
            }
            results_r.append(item)

        logging.info(
            f"{ticker}: gap={gap_pct:.2f}% score_m={score_m:.1f} score_r={score_r:.1f} flow={flow_label} rvol={rvol:.2f}"
        )

    results_m.sort(key=lambda x: x["score_m"], reverse=True)
    results_r.sort(key=lambda x: x["score_r"], reverse=True)
    return scanned, results_m[:15], results_r[:10]

def is_entry_triggered(item):
    price = float(item["price"])
    entry = float(item["entry"])
    signal_type = item["type"]
    if signal_type == "breakout":
        return price >= entry * (1 - ENTRY_TRIGGER_BUFFER_PCT / 100)
    else:
        return abs(price - entry) / entry * 100 <= ENTRY_TRIGGER_BUFFER_PCT

def save_alert_csv(results_m, results_r):
    rows = []
    for item in results_m + results_r:
        rows.append({
            **item,
            "timestamp": datetime.now().isoformat()
        })
    with open(ALERT_FILE, "w", newline="", encoding="utf-8") as f:
        fieldnames = [
            "ticker", "type", "sector", "price", "prev", "chg_pct", "volume", "value", "rvol",
            "atr_pct", "rsi", "flow", "score_m", "score_r", "entry", "stop", "tp1", "tp2",
            "tp3", "support", "resistance", "timestamp"
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)

def print_results(scanned, results_m, results_r):
    print("=" * 78)
    print(f"{'SCANNER IDX - FINAL':^78}")
    print("=" * 78)
    if results_m:
        print("\nBREAKOUT SIGNAL")
        print("-" * 78)
        for r in results_m[:10]:
            print(
                f"{r['ticker']:<8} score={r['score_m']:<6.1f} "
                f"rvol={r['rvol']:<5.2f} rsi={r['rsi']:<5.1f} "
                f"entry={r['entry']:<8,.0f} sl={r['stop']:<8,.0f} tp2={r['tp2']:<8,.0f} "
                f"flow={r['flow']} sector={r['sector']}"
            )
    if results_r:
        print("\nREVERSAL SIGNAL")
        print("-" * 78)
        for r in results_r[:10]:
            print(
                f"{r['ticker']:<8} score={r['score_r']:<6.1f} "
                f"rvol={r['rvol']:<5.2f} rsi={r['rsi']:<5.1f} "
                f"entry={r['entry']:<8,.0f} sl={r['stop']:<8,.0f} tp2={r['tp2']:<8,.0f} "
                f"flow={r['flow']} sector={r['sector']}"
            )
    print("\n" + "=" * 78)
    print(f"SCAN DONE - scanned={scanned} breakout={len(results_m)} reversal={len(results_r)}")
    print("=" * 78)

def main():
    print(f"SCANNER IDX FINAL - {datetime.now().strftime('%Y-%m-%d %H:%M WIB')}")
    tickers = load_ticker_list(TICKER_FILE)

    # Clear old alerts at start of scan
    if os.path.exists(ALERT_FILE):
        os.remove(ALERT_FILE)
        logging.info("Cleared old alert.csv")

    print(f"Loaded {len(tickers)} tickers from {TICKER_FILE}")

    scanned, results_m, results_r = scan_market(tickers)
    print_results(scanned, results_m, results_r)
    save_alert_csv(results_m, results_r)

    # 1) Entry alerts FIRST
    sent = 0
    for item in (results_m + results_r):
        if is_entry_triggered(item):
            ok = send_entry_alert(item)
            if ok:
                sent += 1
            time.sleep(1.2)

    # 2) Fallback if none triggered
    if sent == 0 and (results_m or results_r):
        fallback = (
            f"No entry alert sent.\n"
            f"Setup found but not in trigger zone yet.\n"
            f"Breakout: {len(results_m)} | Reversal: {len(results_r)}"
        )
        telegram_send(fallback, markdown=False)

    # 3) Summary LAST
    send_summary_alert(results_m, results_r, scanned)

    print(f"Telegram entry alerts sent: {sent}")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logging.exception(f"Fatal error: {e}")
        sys.exit(1)
