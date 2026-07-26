#!/usr/bin/env python3
import os
import sys
import csv
import time
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

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "").strip()

REQUEST_DELAY = 0.10
MIN_DAILY_VALUE = 1_000_000_000  # diturunin biar peluang alert lebih banyak
COOLDOWN_DAYS = 2

# Market open mode - fokus intraday Indonesia
MARKET_OPEN_MODE = True
OPEN_SESSION_START = "08:55"
OPEN_SESSION_END = "10:30"

# Entry trigger lebih longgar
ENTRY_TRIGGER_BUFFER_PCT = 3.0

# Filters yang lebih sering ngasih setup
BREAKOUT_MIN_SCORE = 28
REVERSAL_MIN_SCORE = 18

BREAKOUT_GAP_MIN = -1.0
BREAKOUT_GAP_MAX = 15.0

REVERSAL_RSI_MAX = 40
REVERSAL_RVOL_MIN = 1.1
REVERSAL_DROP_MAX = 1.0

# Telegram behavior
SEND_FALLBACK_MESSAGE = True
SEND_SUMMARY = True

# =========================
# LOGGER
# =========================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

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

def sleep_jitter(base=0.10):
    time.sleep(round(random.uniform(base * 0.5, base * 1.5), 3))

def in_open_session():
    now = datetime.now()
    if now.weekday() >= 5:
        return False
    hm = now.strftime("%H:%M")
    return OPEN_SESSION_START <= hm <= OPEN_SESSION_END

def load_ticker_list(file_path):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"{file_path} not found")
    with open(file_path, "r", encoding="utf-8") as f:
        return [line.strip().upper().replace(".JK", "") for line in f if line.strip()]

def fetch_data(ticker, retries=3):
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
            sleep_jitter(0.2)
    return None

def extract_series(data):
    try:
        if not data:
            return None
        chart = data.get("chart") or {}
        result_list = chart.get("result") or []
        if not result_list:
            return None
        result = result_list[0] or {}
        timestamps = result.get("timestamp") or []
        indicators = result.get("indicators") or {}
        quotes_list = indicators.get("quote") or []
        adj_list = indicators.get("adjclose") or []
        if not timestamps or not quotes_list:
            return None
        quotes = quotes_list[0] or {}
        adj = adj_list[0] if adj_list else {}
        df = pd.DataFrame({
            "open": quotes.get("open") or [],
            "high": quotes.get("high") or [],
            "low": quotes.get("low") or [],
            "close": quotes.get("close") or [],
            "volume": quotes.get("volume") or [],
            "adjclose": (adj.get("adjclose") if isinstance(adj, dict) else []) or []
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
    if flow_pct <= -15:
        score += 25
        label = "ACCUMULATION STRONG"
    elif flow_pct <= -5:
        score += 12
        label = "ACCUMULATION"
    elif flow_pct >= 20:
        score -= 20
        label = "DISTRIBUTION STRONG"
    elif flow_pct >= 8:
        score -= 10
        label = "DISTRIBUTION"
    return score, label

def level_entry(signal_type, price, support, atr_val):
    if signal_type == "breakout":
        entry = max(support, price * 0.995) if support else price * 0.995
        stop = max(1, min(entry - max(price * 0.01, atr_val * 0.7), price - max(price * 0.01, atr_val * 0.7)))
        tp1 = entry + min(atr_val * 1.0, price * 0.03)
        tp2 = entry + min(atr_val * 1.8, price * 0.06)
        tp3 = entry + min(atr_val * 2.6, price * 0.09)
    else:
        entry = price * 0.995
        stop = max(1, price - max(price * 0.02, atr_val * 0.8))
        tp1 = price + min(atr_val * 0.9, price * 0.025)
        tp2 = price + min(atr_val * 1.5, price * 0.05)
        tp3 = None
    return round(entry, 0), round(stop, 0), round(tp1, 0), round(tp2, 0), (round(tp3, 0) if tp3 else None)

def telegram_send(text, markdown=True, retries=3):
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        logging.warning("Telegram env missing")
        return False
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    payload = {"chat_id": TELEGRAM_CHAT_ID, "text": text, "disable_web_page_preview": True}
    if markdown:
        payload["parse_mode"] = "Markdown"
    for _ in range(retries):
        try:
            r = requests.post(url, json=payload, timeout=15)
            if r.status_code == 200:
                return True
            logging.warning(f"Telegram failed {r.status_code}: {r.text[:200]}")
        except Exception as e:
            logging.warning(f"Telegram error: {e}")
            time.sleep(1.2)
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

def send_entry_alert(item):
    msg = format_signal_message(item)
    ok = telegram_send(msg, markdown=True)
    if not ok:
        ok = telegram_send(msg.replace("*", ""), markdown=False)
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

def is_entry_triggered(item):
    price = float(item["price"])
    entry = float(item["entry"])
    signal_type = item["type"]
    if signal_type == "breakout":
        return price >= entry * (1 - ENTRY_TRIGGER_BUFFER_PCT / 100)
    return abs(price - entry) / entry * 100 <= ENTRY_TRIGGER_BUFFER_PCT

def scan_market(tickers):
    results_m = []
    results_r = []
    scanned = 0

    for raw_ticker in tickers:
        scanned += 1
        ticker = raw_ticker.replace(".JK", "").strip()
        sleep_jitter(REQUEST_DELAY)

        data = fetch_data(ticker)
        df = extract_series(data)
        if df is None or len(df) < 30:
            continue

        close = df["close"]
        volume = df["volume"]
        price = float(close.iloc[-1])
        prev_price = float(close.iloc[-2])
        gap_pct = ((price / prev_price) - 1) * 100 if prev_price > 0 else 0

        daily_value = float(price * volume.iloc[-1])
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
        flow_bonus, flow_label = flow_score(proxy_flow_idr, daily_value)
        sector = sector_map.get(ticker, "LAIN")

        score_m = 0
        score_m += (atr_val / price) * 220
        score_m += rvol * 12
        score_m += flow_bonus
        score_m += min(max(gap_pct, -2), 12) * 2.0
        if price > sma20:
            score_m += 8
        if price > sma50:
            score_m += 6
        if price > sma50 > sma200:
            score_m += 8
        if td_setup >= 3:
            score_m += 6
        if 48 <= rsi_val <= 78:
            score_m += 8

        score_r = 0
        score_r += rvol * 6
        if rsi_val < REVERSAL_RSI_MAX:
            score_r += (REVERSAL_RSI_MAX - rsi_val) * 2
        score_r += abs(flow_bonus) * 0.4
        score_r += (atr_val / price) * 150
        if td_setup >= 5:
            score_r += 10

        flow_risky = flow_label in ["DISTRIBUTION", "DISTRIBUTION STRONG"]

        if score_m >= BREAKOUT_MIN_SCORE and BREAKOUT_GAP_MIN <= gap_pct <= BREAKOUT_GAP_MAX and not flow_risky:
            entry, stop, tp1, tp2, tp3 = level_entry("breakout", price, support, atr_val)
            results_m.append({
                "ticker": ticker, "type": "breakout", "sector": sector, "price": price, "prev": prev_price,
                "chg_pct": round(gap_pct, 2), "volume": int(volume.iloc[-1]), "value": int(daily_value),
                "rvol": round(rvol, 2), "atr_pct": round((atr_val / price) * 100, 2), "rsi": round(rsi_val, 1),
                "flow": flow_label, "score_m": round(score_m, 1), "entry": entry, "stop": stop,
                "tp1": tp1, "tp2": tp2, "tp3": tp3, "support": round(support, 0), "resistance": round(resistance, 0)
            })

        if score_r >= REVERSAL_MIN_SCORE and rsi_val < REVERSAL_RSI_MAX and rvol > REVERSAL_RVOL_MIN and gap_pct <= REVERSAL_DROP_MAX:
            entry, stop, tp1, tp2, tp3 = level_entry("reversal", price, support, atr_val)
            results_r.append({
                "ticker": ticker, "type": "reversal", "sector": sector, "price": price, "prev": prev_price,
                "chg_pct": round(gap_pct, 2), "volume": int(volume.iloc[-1]), "value": int(daily_value),
                "rvol": round(rvol, 2), "atr_pct": round((atr_val / price) * 100, 2), "rsi": round(rsi_val, 1),
                "flow": flow_label, "score_r": round(score_r, 1), "entry": entry, "stop": stop,
                "tp1": tp1, "tp2": tp2, "tp3": tp3 if tp3 else 0, "support": round(support, 0), "resistance": round(resistance, 0)
            })

        logging.info(f"{ticker}: gap={gap_pct:.2f}% score_m={score_m:.1f} score_r={score_r:.1f} flow={flow_label} rvol={rvol:.2f}")

    results_m.sort(key=lambda x: x["score_m"], reverse=True)
    results_r.sort(key=lambda x: x["score_r"], reverse=True)
    return scanned, results_m[:20], results_r[:15]

def save_alert_csv(results_m, results_r):
    rows = []
    for item in results_m + results_r:
        rows.append({**item, "timestamp": datetime.now().isoformat()})
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
            print(f"{r['ticker']:<8} score={r['score_m']:<6.1f} rvol={r['rvol']:<5.2f} rsi={r['rsi']:<5.1f} entry={r['entry']:<8,.0f} sl={r['stop']:<8,.0f} tp2={r['tp2']:<8,.0f} flow={r['flow']} sector={r['sector']}")
    if results_r:
        print("\nREVERSAL SIGNAL")
        print("-" * 78)
        for r in results_r[:10]:
            print(f"{r['ticker']:<8} score={r['score_r']:<6.1f} rvol={r['rvol']:<5.2f} rsi={r['rsi']:<5.1f} entry={r['entry']:<8,.0f} sl={r['stop']:<8,.0f} tp2={r['tp2']:<8,.0f} flow={r['flow']} sector={r['sector']}")
    print("\n" + "=" * 78)
    print(f"SCAN DONE - scanned={scanned} breakout={len(results_m)} reversal={len(results_r)}")
    print("=" * 78)

def main():
    print(f"SCANNER IDX FINAL - {datetime.now().strftime('%Y-%m-%d %H:%M WIB')}")

    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        logging.warning("Telegram env missing - set GitHub Actions secrets")

    tickers = load_ticker_list(TICKER_FILE)
    if os.path.exists(ALERT_FILE):
        os.remove(ALERT_FILE)

    scanned, results_m, results_r = scan_market(tickers)
    print_results(scanned, results_m, results_r)
    save_alert_csv(results_m, results_r)

    sent = 0
    for item in (results_m + results_r):
        if is_entry_triggered(item):
            if send_entry_alert(item):
                sent += 1
            time.sleep(0.6)

    if sent == 0 and (results_m or results_r) and SEND_FALLBACK_MESSAGE:
        telegram_send(
            f"Setup ditemukan tapi belum masuk entry zone.\nBreakout: {len(results_m)} | Reversal: {len(results_r)}",
            markdown=False
        )

    if SEND_SUMMARY:
        send_summary_alert(results_m, results_r, scanned)

    print(f"Telegram entry alerts sent: {sent}")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logging.exception(f"Fatal error: {e}")
        sys.exit(1)
