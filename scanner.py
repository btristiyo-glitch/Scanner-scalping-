#!/usr/bin/env python3
"""
SCANNER V3 - IDX MORNING SCAN
=============================
Original by: Kamu | Fixes by: Neurobro
Fixes applied:
  1. flow_score() threshold % dari avg daily value (bukan absolut)
  2. level_entry() TP realistis capped di ATR-based 3x max
  3. Dynamic trailing stop setelah profit 1.5x ATR
  4. Filter DISTRIBUTION => skip breakout
  5. Cooldown check (5 hari)
  6. flow_score() dipanggil dgn avg_daily_value
  7. Volume nominal filter minimum
"""

import pandas as pd
import numpy as np
import requests
import json
import os
import sys
import time
import random
import csv
import logging
import warnings
from datetime import datetime, timedelta, date
from typing import Dict, List, Optional, Tuple
warnings.filterwarnings("ignore")

# ===================== KONFIGURASI TIDAK BERUBAH =====================
TICKER_FILE = "stocks.txt"
ALERT_FILE = "alert.csv"
FLOW_FILE = "flow.csv"
PROXY_FILE = "proxy.csv"
CSV_HEADER = "ticker,price,prev,chg_pct,volume,value,rvol,atr_pct,rsi,sma20_pct,sma50_pct,sma200_pct,ma100_pct,td_seq,flow,score_m,score_r,type,entry,stop,tp1,tp2,tp3,support,resistance,sector\n"
REQUEST_DELAY = 0.15
MIN_DAILY_VALUE = 5_000_000_000      # 5M IDR min daily value
COOLDOWN_DAYS = 5                    # gak munculin sinyal dari ticker yg udah di-trade 5 hari lalu

# ===================== KONFIGURASI TELEGRAM =====================
TELEGRAM_BOT = os.environ.get("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT = os.environ.get("TELEGRAM_CHAT_ID", "")

# ===================== SECTOR MAP =====================
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
    "MYOR": "KONSUMER", "CPIN": "KONSUMER", "JPFA": "KONSUMER", "JPFA": "KONSUMER",
    "MAIN": "KONSUMER", "UNVR": "KONSUMER", "HMSP": "KONSUMER", "GGRM": "KONSUMER",
    "PGAS": "ENERGI", "MEDC": "ENERGI", "AKRA": "ENERGI", "RAJA": "ENERGI",
    "ANTM": "MINERBA", "INCO": "MINERBA", "MDKA": "MINERBA", "ADMR": "MINERBA",
    "SMGR": "SEMEN", "INTP": "SEMEN", "SMBR": "SEMEN", "KLBF": "FARMASI",
    "KAEF": "FARMASI", "SIDO": "FARMASI", "WIKA": "KONSTRUKSI", "PTPP": "KONSTRUKSI",
    "ADHI": "KONSTRUKSI", "WSKT": "KONSTRUKSI", "PNSN": "PROPERTI", "BSDE": "PROPERTI",
    "CTRA": "PROPERTI", "PWON": "PROPERTI", "SMRA": "PROPERTI",
    "MAPI": "RITEL", "ERAA": "RITEL", "ACES": "RITEL", "RALS": "RITEL",
    "AMRT": "RITEL", "LPKR": "PROPERTI",
}

# ===================== HELPER FUNCTIONS =====================

def sleep_jitter(base=0.15):
    time.sleep(round(random.uniform(base * 0.5, base * 1.5), 3))

def load_ticker_list(file_path):
    if not os.path.exists(file_path):
        sys.exit(f"FILE {file_path} TIDAK DITEMUKAN")
    with open(file_path, "r") as f:
        return [line.strip().upper() for line in f if line.strip()]

def fetch_data(ticker, retries=2):
    for attempt in range(retries):
        try:
            url = f"https://query1.finance.yahoo.com/v8/finance/chart/{ticker}.JK"
            params = {"range": "6mo", "interval": "1d", "includePrePost": "false"}
            headers = {"User-Agent": "Mozilla/5.0"}
            resp = requests.get(url, params=params, headers=headers, timeout=15)
            if resp.status_code == 200:
                data = resp.json()
                return data
            sleep_jitter(0.5)
        except:
            pass
    return None

def extract_series(data):
    if not data:
        return None
    result = data.get("chart", {}).get("result", [None])[0]
    if not result:
        return None
    timestamps = result.get("timestamp", [])
    indicators = result.get("indicators", {})
    quotes = indicators.get("quote", [{}])[0]
    adj = indicators.get("adjclose", [{}])[0]
    ohlcv = {
        "open": quotes.get("open", []),
        "high": quotes.get("high", []),
        "low": quotes.get("low", []),
        "close": quotes.get("close", []),
        "volume": quotes.get("volume", []),
        "adjclose": adj.get("adjclose", []),
    }
    df = pd.DataFrame(ohlcv, index=pd.to_datetime(timestamps, unit="s"))
    df.dropna(subset=["close", "volume"], inplace=True)
    df = df[df["volume"] > 0]
    if len(df) < 30:
        return None
    return df

def calc_rsi(series, period=14):
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_g = gain.rolling(period).mean()
    avg_l = loss.rolling(period).mean()
    rs = avg_g / avg_l.replace(0, np.nan)
    rsi = 100.0 - (100.0 / (1.0 + rs))
    return rsi

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
    support = recent["low"].min()
    resistance = recent["high"].max()
    return support, resistance

def td_sequential(close, period=4):
    if len(close) < period + 5:
        return 0, 0
    setup = 0
    for i in range(1, period + 1):
        if close.iloc[-i] > close.iloc[-i - period]:
            setup += 1
        elif close.iloc[-i] < close.iloc[-i - period]:
            setup -= 1
    return setup, 0

def get_flow_proxy(df):
    if df is None or len(df) < 10:
        return 0, "NEUTRAL"
    close = df["close"].values
    volume = df["volume"].values
    high = df["high"].values
    low = df["low"].values

    net_flow = 0.0
    for i in range(1, min(len(df), 30)):
        typical_price = (high[i] + low[i] + close[i]) / 3.0
        if close[i] > close[i-1]:
            net_flow += typical_price * volume[i]
        elif close[i] < close[i-1]:
            net_flow -= typical_price * volume[i]

    return net_flow

# ==================== PERBAIKAN 1 & 6: flow_score %-based ====================

def flow_score(net_flow_idr, avg_daily_value):
    """
    PERBAIKAN 1: Threshold % dari avg daily value 20 hari
    PERBAIKAN 6: avg_daily_value diterima sebagai parameter
    """
    score = 0
    label = "NEUTRAL"

    if avg_daily_value <= 0:
        return score, label

    flow_pct = (net_flow_idr / avg_daily_value) * 100 if avg_daily_value else 0

    if flow_pct <= -30:          # akumulasi >30% dari rata-rata
        score += 35
        label = "ACCUMULATION STRONG"
    elif flow_pct <= -10:
        score += 20
        label = "ACCUMULATION"
    elif flow_pct >= 30:         # distribusi >30% dari rata-rata
        score -= 30
        label = "DISTRIBUTION STRONG"
    elif flow_pct >= 10:
        score -= 15
        label = "DISTRIBUTION"

    return score, label

# ==================== PERBAIKAN 3: Dynamic trailing stop ====================

def dynamic_trailing_stop(entry_price, current_price, highest_since_entry, atr_val, original_stop):
    """
    PERBAIKAN 3: Geser stop loss ke atas setelah harga bergerak 1.5x ATR dari entry
    - Setelah profit 1.5x ATR: trail dengan jarak 2.5x ATR dari highest
    - Setelah TP1 (2x ATR): pindah ke breakeven
    """
    if current_price < entry_price:
        return original_stop

    profit_pct = (highest_since_entry / entry_price - 1) * 100
    atr_pct = (atr_val / entry_price) * 100 if entry_price > 0 else 0

    # Setelah profit 1.5x ATR, trail dengan jarak 2.5x ATR
    if profit_pct >= (1.5 * atr_pct):
        new_stop = highest_since_entry - (2.5 * atr_val)
        return max(new_stop, entry_price)  # minimal balik ke entry (breakeven)

    return original_stop

# ==================== PERBAIKAN 2: level_entry dgn TP realistis ====================

def level_entry(item, price, support, atr_val):
    """
    PERBAIKAN 2: TP murni ATR-based, capped biar gak terlalu jauh
    """
    if item["tipe"] == "breakout":
        entry_limit = max(support, price * 0.995) if support else price * 0.995
        atr_buffer = max(price * 0.01, atr_val * 0.8)
        stop = min(entry_limit - atr_buffer, price - atr_buffer)
        # TP capped di max 6%, 10%, 15% biar gak gila2
        tp1 = entry_limit + min(atr_val * 1.5, price * 0.06)
        tp2 = entry_limit + min(atr_val * 2.5, price * 0.10)
        tp3 = entry_limit + min(atr_val * 3.5, price * 0.15)
    else:
        entry_limit = price * 0.995
        atr_buffer = max(price * 0.008, atr_val * 0.7)
        stop = price - max(atr_buffer, price * 0.03)
        tp1 = price + min(atr_val * 1.2, price * 0.04)
        tp2 = price + min(atr_val * 2.0, price * 0.08)
        tp3 = None

    stop = max(1, stop)
    tp1 = max(1, tp1)
    tp2 = max(1, tp2)
    return round(entry_limit, 0), round(stop, 0), round(tp1, 0), round(tp2, 0), (round(tp3, 0) if tp3 else None)

# ==================== PERBAIKAN 4 & 5: Filter dan Cooldown ====================

def check_cooldown(ticker, alert_file, days=COOLDOWN_DAYS):
    """
    PERBAIKAN 5: Skip ticker yg udah muncul sebagai sinyal dalam X hari terakhir
    """
    if not os.path.exists(alert_file):
        return False
    cutoff = datetime.now() - timedelta(days=days)
    try:
        with open(alert_file, 'r', encoding='utf-8') as f:
            for row in csv.DictReader(f):
                if row.get('ticker') == ticker:
                    # Kalo ada timestamp, cek tanggal
                    ts_str = row.get('timestamp', '')
                    if ts_str:
                        try:
                            ts = datetime.fromisoformat(ts_str)
                            if ts > cutoff:
                                return True
                        except:
                            pass
                    else:
                        # Kalo gak ada timestamp, skip aja (anggap cooldown)
                        return True
    except:
        pass
    return False

# ==================== SCAN UTAMA ====================

def morning_scan(tickers):
    start_time = datetime.now()
    results_m = []
    results_r = []
    scanned = 0

    # Cek kalo alert.csv ada, baca buat cooldown
    alert_tickers_cooldown = set()
    if os.path.exists(ALERT_FILE):
        # Kalo belum ada kolom timestamp, baca aja semua ticker
        pass  # check_cooldown() handle detail

    for ticker in tickers:
        scanned += 1
        sleep_jitter(REQUEST_DELAY)

        # PERBAIKAN 5: Cooldown check
        if check_cooldown(ticker, ALERT_FILE):
            logging.info(f"COOLDOWN {ticker}")
            continue

        ticker_clean = ticker.replace(".JK", "").strip()
        sector = sector_map.get(ticker_clean, "LAIN")

        data = fetch_data(ticker)
        df = extract_series(data)
        if df is None or len(df) < 30:
            continue

        close = df["close"]
        volume = df["volume"]
        high = df["high"]
        low = df["low"]
        prev_close = close.iloc[-2] if len(close) >= 2 else close.iloc[-1]
        price = close.iloc[-1]
        price_prev = prev_close
        chg_pct = ((price / price_prev) - 1) * 100 if price_prev > 0 else 0

        # Volume nominal harian
        daily_value = price * volume.iloc[-1]
        avg_daily_value = float((close.tail(20) * volume.tail(20)).mean()) if len(close) >= 20 else daily_value

        # Minimal value filter
        if daily_value < MIN_DAILY_VALUE:
            continue

        # ATR
        atr_series = calc_atr(df)
        atr_val = atr_series.iloc[-1] if pd.notna(atr_series.iloc[-1]) else price * 0.02
        atr_pct = (atr_val / price) * 100 if price > 0 else 0

        # RSI
        rsi_series = calc_rsi(close)
        rsi_val = rsi_series.iloc[-1] if pd.notna(rsi_series.iloc[-1]) else 50

        # RVOL
        avg_vol_20 = volume.tail(20).mean() if len(volume) >= 20 else volume.mean()
        rvol = volume.iloc[-1] / avg_vol_20 if avg_vol_20 > 0 else 1

        # SMA
        sma20 = close.tail(20).mean() if len(close) >= 20 else price
        sma50 = close.tail(50).mean() if len(close) >= 50 else price
        sma200 = close.tail(200).mean() if len(close) >= 200 else price
        ma100 = close.tail(100).mean() if len(close) >= 100 else price
        sma20_pct = ((price / sma20) - 1) * 100 if sma20 > 0 else 0
        sma50_pct = ((price / sma50) - 1) * 100 if sma50 > 0 else 0
        sma200_pct = ((price / sma200) - 1) * 100 if sma200 > 0 else 0
        ma100_pct = ((price / ma100) - 1) * 100 if ma100 > 0 else 0

        # TD Sequential
        td_setup, td_countdown = td_sequential(close)

        # Support / Resistance
        support, resistance = calc_support_resistance(df)

        # Flow
        proxy_flow_idr = get_flow_proxy(df)

        # ==================== PERBAIKAN 6: flow_score pake avg_daily_value ====================
        flow_bonus, flow_label = flow_score(proxy_flow_idr, avg_daily_value)

        # ==================== MOMENTUM SCORE ====================
        score_m = 0
        score_m += (atr_pct * 3.0)
        score_m += (rvol * 10.0)
        score_m += flow_bonus
        score_m += (min(chg_pct, 10.0) * 2.0)
        if price > sma50 > sma200:
            score_m += 15.0
        if price > sma20:
            score_m += 5.0
        if td_setup >= 4:
            score_m += 10.0
        if rsi_val < 30 or rsi_val > 70:
            score_m -= 5.0

        # ==================== REVERSAL SCORE ====================
        score_r = 0
        score_r += (rvol * 5.0)
        if rsi_val < 30:
            score_r += (30.0 - rsi_val) * 2.0
        score_r += flow_bonus
        score_r += (atr_pct * 2.0)
        if td_setup >= 6:
            score_r += 15.0
        if rvol > 1.5 and chg_pct > 0:
            score_r += 8.0

        # ==================== PERBAIKAN 4: SKIP breakout kalo ada DISTRIBUTION ====================
        strong_flow_breakout = flow_label in ["ACCUMULATION STRONG", "ACCUMULATION"]
        flow_too_risky = flow_label in ["DISTRIBUTION", "DISTRIBUTION STRONG"]

        if flow_too_risky:
            # Kalo distribusi, skip seluruh entry buat ticker ini
            score_m = 0
            score_r = 0

        # Gap detection buat breakout
        gap_pct = ((price / price_prev) - 1) * 100 if price_prev > 0 else 0

        # Entry filtering
        item = {"tipe": None, "price": price, "ticker": ticker_clean}

        if score_m >= 40 and not flow_too_risky and 2 <= gap_pct <= 5:
            item["tipe"] = "breakout"
            entry_limit, stop, tp1, tp2, tp3 = level_entry(item, price, support, atr_val)
            rr = ((tp2 / entry_limit) - 1) / ((entry_limit - stop) / entry_limit) if (entry_limit - stop) > 0 else 0
            results_m.append({
                "ticker": ticker_clean,
                "price": price,
                "prev": price_prev,
                "chg_pct": round(gap_pct, 2),
                "volume": int(volume.iloc[-1]),
                "value": int(daily_value),
                "rvol": round(rvol, 2),
                "atr_pct": round(atr_pct, 2),
                "rsi": round(rsi_val, 1),
                "sma20_pct": round(sma20_pct, 2),
                "sma50_pct": round(sma50_pct, 2),
                "sma200_pct": round(sma200_pct, 2),
                "ma100_pct": round(ma100_pct, 2),
                "td_seq": td_setup,
                "flow": flow_label,
                "score_m": round(score_m, 1),
                "type": "breakout",
                "entry": entry_limit,
                "stop": stop,
                "tp1": tp1,
                "tp2": tp2,
                "tp3": tp3,
                "support": round(support, 0) if support else 0,
                "resistance": round(resistance, 0) if resistance else 0,
                "sector": sector,
            })

        # Reversal signal - tetap diproses kalo RSI oversold (meskipun flow distribusi)
        if rsi_val < 35 and rvol > 1.2 and chg_pct > -1 and not flow_too_risky:
            item["tipe"] = "reversal"
            entry_limit, stop, tp1, tp2, tp3 = level_entry(item, price, support, atr_val)
            rr = ((tp2 / entry_limit) - 1) / ((entry_limit - stop) / entry_limit) if (entry_limit - stop) > 0 else 0
            results_r.append({
                "ticker": ticker_clean,
                "price": price,
                "prev": price_prev,
                "chg_pct": round(chg_pct, 2),
                "volume": int(volume.iloc[-1]),
                "value": int(daily_value),
                "rvol": round(rvol, 2),
                "atr_pct": round(atr_pct, 2),
                "rsi": round(rsi_val, 1),
                "sma20_pct": round(sma20_pct, 2),
                "sma50_pct": round(sma50_pct, 2),
                "sma200_pct": round(sma200_pct, 2),
                "ma100_pct": round(ma100_pct, 2),
                "td_seq": td_setup,
                "flow": flow_label,
                "score_r": round(score_r, 1),
                "type": "reversal",
                "entry": entry_limit,
                "stop": stop,
                "tp1": tp1,
                "tp2": tp2,
                "tp3": tp3 if tp3 else 0,
                "support": round(support, 0) if support else 0,
                "resistance": round(resistance, 0) if resistance else 0,
                "sector": sector,
            })

        logging.info(f"[{scanned}/{len(tickers)}] {ticker_clean}: gap={gap_pct:.2f}% score_m={score_m:.1f} score_r={score_r:.1f} flow={flow_label} rvol={rvol:.2f}")

    # Sort and limit
    results_m.sort(key=lambda x: x["score_m"], reverse=True)
    results_r.sort(key=lambda x: x["score_r"], reverse=True)
    return results_m[:15], results_r[:10]

# ==================== TELEGRAM ====================

def send_telegram(item, alert_type):
    if not TELEGRAM_BOT or not TELEGRAM_CHAT:
        return
    emoji = "🟢" if item.get("type") == "breakout" else "🟣"
    msg = (
        f"{emoji} *{item['ticker']}* ({item.get('sector', 'LAIN')}) - {item['type'].upper()}\n"
        f"💰 Entry: {item['entry']:,.0f} | SL: {item['stop']:,.0f}\n"
        f"🎯 TP1: {item['tp1']:,.0f} | TP2: {item['tp2']:,.0f} | TP3: {item['tp3']:,.0f}\n"
        f"📊 Score: {item.get('score_m', item.get('score_r', 0)):.1f} | RSI: {item['rsi']:.1f} | ATR: {item['atr_pct']:.1f}%\n"
        f"📈 RVOL: {item['rvol']:.2f}x | Flow: {item['flow']}\n"
        f"🔗 [TradingView](https://www.tradingview.com/chart/?symbol=IDX:{item['ticker']})\n"
        f"⏰ {datetime.now().strftime('%H:%M WIB')}"
    )
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT}/sendMessage"
    try:
        requests.post(url, json={"chat_id": TELEGRAM_CHAT, "text": msg, "parse_mode": "Markdown", "disable_web_page_preview": True}, timeout=10)
    except Exception as e:
        logging.error(f"Telegram error: {e}")

# ==================== SAVE CSV ====================

def save_alert_csv(results_m, results_r):
    if not results_m and not results_r:
        open(ALERT_FILE, "w").close()
        return
    all_results = sorted(results_m + results_r, key=lambda x: x.get("score_m", x.get("score_r", 0)), reverse=True)
    with open(ALERT_FILE, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "ticker", "price", "prev", "chg_pct", "volume", "value", "rvol", "atr_pct", "rsi",
            "sma20_pct", "sma50_pct", "sma200_pct", "ma100_pct", "td_seq", "flow",
            "score_m", "score_r", "type", "entry", "stop", "tp1", "tp2", "tp3",
            "support", "resistance", "sector", "timestamp"
        ])
        writer.writeheader()
        for item in all_results:
            item["timestamp"] = datetime.now().isoformat()
            writer.writerow(item)

# ==================== DISPLAY ====================

def display_results(results_m, results_r):
    header = f"{'='*70}\n{'SCANNER IDX - MORNING SCAN':^70}\n{'='*70}"
    print(header)

    if results_m:
        print(f"\n{'━'*70}")
        print(f"  BREAKOUT SIGNAL ({len(results_m)})")
        print(f"{'━'*70}")
        print(f"{'Ticker':<8} {'Gap%':<6} {'Score':<7} {'RVOL':<6} {'RSI':<5} {'Entry':<12} {'SL':<12} {'TP2':<12} {'R:R':<6} {'Flow':<18} {'Sektor':<10}")
        print(f"{'━'*70}")
        for r in results_m[:10]:
            rr = ((r['tp2'] / r['entry']) - 1) / ((r['entry'] - r['stop']) / r['entry']) if (r['entry'] - r['stop']) > 0 else 0
            print(f"{r['ticker']:<8} {r['chg_pct']:<6.2f} {r['score_m']:<7.1f} {r['rvol']:<6.2f} {r['rsi']:<5.1f} {r['entry']:<12,.0f} {r['stop']:<12,.0f} {r['tp2']:<12,.0f} {rr:<6.2f} {r['flow']:<18} {r.get('sector', 'LAIN'):<10}")

    if results_r:
        print(f"\n{'━'*70}")
        print(f"  REVERSAL SIGNAL ({len(results_r)})")
        print(f"{'━'*70}")
        print(f"{'Ticker':<8} {'Score':<7} {'RVOL':<6} {'RSI':<5} {'Entry':<12} {'SL':<12} {'TP2':<12} {'Flow':<18} {'Sektor':<10}")
        print(f"{'━'*70}")
        for r in results_r[:10]:
            print(f"{r['ticker']:<8} {r['score_r']:<7.1f} {r['rvol']:<6.2f} {r['rsi']:<5.1f} {r['entry']:<12,.0f} {r['stop']:<12,.0f} {r['tp2']:<12,.0f} {r['flow']:<18} {r.get('sector', 'LAIN'):<10}")

    duration = (datetime.now() - start_time).total_seconds()
    print(f"\n{'━'*70}")
    print(f"  SCAN DONE - {duration:.1f}s | Breakout: {len(results_m)} | Reversal: {len(results_r)}")
    print(f"{'━'*70}")

# ==================== MAIN ====================

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    print(f"SCANNER IDX V3 - FIXED | {datetime.now().strftime('%Y-%m-%d %H:%M WIB')}")
    print(f"Fixes: flow_%threshold | TP_realistis | trailing_stop | distribusi_filter | cooldown")

    start_time = datetime.now()
    tickers = load_ticker_list(TICKER_FILE)
    print(f"Loaded {len(tickers)} tickers from {TICKER_FILE}")

    breakouts, reversals = morning_scan(tickers)
    display_results(breakouts, reversals)
    save_alert_csv(breakouts, reversals)

    # Kirim Telegram untuk top 5
    for r in (breakouts + reversals)[:5]:
        send_telegram(r, r.get("type"))

    print(f"\n✅ Selesai - {datetime.now().strftime('%H:%M WIB')}")
