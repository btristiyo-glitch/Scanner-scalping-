import pandas as pd
import numpy as np
import yfinance as yf
import talib
import datetime
import os
import requests

# Settings
TICKERS_FILE = "stocks.txt"
SETUPS_FILE = "setups.csv"
MIN_GAP = 0.02  # minimal 2% gap
MAX_GAP = 0.05  # maksimal 5% gap
RVOL_THRESHOLD = 1.5  # relative volume threshold
RSI_PERIOD = 4
STOCH_K = 5
STOCH_D = 3
STOCH_SLOW = 3
ATR_PERIOD = 14
TRAILING_STOP_ATR_MULTIPLIER = 0.7  # trailing stop at 0.7*ATR

TELEGRAM_BOT_TOKEN = '8716504704:AAGepO7au5uRxlA1vGr0Vl4nx77NHf9DHAU'
  # ganti sesuai tokenmu
TELEGRAM_CHAT_ID = '6262086905'  # ganti sesuai chat id

# Telegram alert helper
def send_telegram_message(text):
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    payload = {
        "chat_id": TELEGRAM_CHAT_ID,
        "text": text,
        "parse_mode": "Markdown"
    }
    requests.post(url, data=payload)

# Ambil data intraday dengan multi timeframe, default 5m + 15m
def get_multi_tf_data(ticker, periods_intervals=[('1d', '5m'), ('5d', '15m')]):
    dfs = {}
    for period, interval in periods_intervals:
        df = yf.download(ticker, period=period, interval=interval, progress=False)
        if df.empty or len(df) < 20:
            return None
        df.dropna(inplace=True)
        dfs[f"{interval}"] = df
    return dfs

# Hitung indikator teknikal pada data multi timeframe
def calculate_multi_tf_indicators(dfs):
    for tf, df in dfs.items():
        df['rsi'] = talib.RSI(df['Close'], timeperiod=RSI_PERIOD)
        slowk, slowd = talib.STOCH(df['High'], df['Low'], df['Close'],
                                   fastk_period=STOCH_K,
                                   slowk_period=STOCH_D,
                                   slowk_matype=0,
                                   slowd_period=STOCH_SLOW,
                                   slowd_matype=0)
        df['slowk'] = slowk
        df['slowd'] = slowd
        df['atr'] = talib.ATR(df['High'], df['Low'], df['Close'], timeperiod=ATR_PERIOD)
    return dfs

# Hitung relative volume dari timeframe 5m
def relative_volume(df_5m):
    if len(df_5m) < 78:
        return 0
    avg_vol = df_5m['Volume'].iloc[:-1].mean()
    cur_vol = df_5m['Volume'].iloc[-1]
    return cur_vol / avg_vol if avg_vol > 0 else 0

# Scan satu ticker dengan multi timeframe dan trailing stop logic
def scan_ticker(ticker):
    dfs = get_multi_tf_data(ticker)
    if dfs is None:
        return

    dfs = calculate_multi_tf_indicators(dfs)
    df_5m = dfs['5m']
    df_15m = dfs['15m']

    last_5m = df_5m.iloc[-1]
    prev_5m = df_5m.iloc[-2]

    last_15m = df_15m.iloc[-1]
    prev_15m = df_15m.iloc[-2]

    # Hitung gap % close pada 5m timeframe
    gap_pct = (last_5m['Close'] - prev_5m['Close']) / prev_5m['Close']
    if gap_pct < MIN_GAP or gap_pct > MAX_GAP:
        return  # Lewati jika gap di luar range 2-5%

    # RVOL filter pada 5m
    rvol = relative_volume(df_5m)
    if rvol < RVOL_THRESHOLD:
        return

    # Multi timeframe confirmation:
    # Buy signal jika Stoch + RSI crossing oversold (20) di kedua timeframes
    buy_signal_5m = (prev_5m['rsi'] < 20 and last_5m['rsi'] > 20) and \
                    (prev_5m['slowk'] < 20 and last_5m['slowk'] > 20)
    buy_signal_15m = (prev_15m['rsi'] < 20 and last_15m['rsi'] > 20) and \
                     (prev_15m['slowk'] < 20 and last_15m['slowk'] > 20)

    # Sell signal jika crossing overbought 80 di kedua timeframes
    sell_signal_5m = (prev_5m['rsi'] > 80 and last_5m['rsi'] < 80) and \
                     (prev_5m['slowk'] > 80 and last_5m['slowk'] < 80)
    sell_signal_15m = (prev_15m['rsi'] > 80 and last_15m['rsi'] < 80) and \
                      (prev_15m['slowk'] > 80 and last_15m['slowk'] < 80)

    if buy_signal_5m and buy_signal_15m:
        atr = last_5m['atr']
        swing_low_5 = df_5m['Low'].iloc[-5:].min()
        stop_loss = max(swing_low_5 - 0.5 * atr, 0)
        entry = last_5m['Close'] * 0.995  # limit entry di bawah harga

        # Trailing stop initial
        trailing_stop = entry - TRAILING_STOP_ATR_MULTIPLIER * atr

        msg = (f"🟢 *BUY Signal*\nTicker: {ticker}\nEntry Limit: {entry:.2f}\n"
               f"Stop Loss: {stop_loss:.2f}\nTrailing Stop Start: {trailing_stop:.2f}\n"
               f"RSI 5m: {last_5m['rsi']:.2f}, 15m: {last_15m['rsi']:.2f}\n"
               f"RVOL: {rvol:.2f}\nGap: {gap_pct:.2%}")
        send_telegram_message(msg)
        save_setup(ticker, 'BUY', entry, stop_loss, trailing_stop)

    elif sell_signal_5m and sell_signal_15m:
        atr = last_5m['atr']
        swing_high_5 = df_5m['High'].iloc[-5:].max()
        stop_loss = min(swing_high_5 + 0.5 * atr, last_5m['Close'] * 1.05)
        entry = last_5m['Close'] * 1.005  # limit entry di atas harga

        trailing_stop = entry + TRAILING_STOP_ATR_MULTIPLIER * atr

        msg = (f"🔴 *SELL Signal*\nTicker: {ticker}\nEntry Limit: {entry:.2f}\n"
               f"Stop Loss: {stop_loss:.2f}\nTrailing Stop Start: {trailing_stop:.2f}\n"
               f"RSI 5m: {last_5m['rsi']:.2f}, 15m: {last_15m['rsi']:.2f}\n"
               f"RVOL: {rvol:.2f}\nGap: {gap_pct:.2%}")
        send_telegram_message(msg)
        save_setup(ticker, 'SELL', entry, stop_loss, trailing_stop)

def save_setup(ticker, direction, entry, stop_loss, trailing_stop):
    setup = {
        "Ticker": ticker,
        "Direction": direction,
        "Entry": entry,
        "StopLoss": stop_loss,
        "TrailingStop": trailing_stop,
        "Timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }
    df = pd.DataFrame([setup])
    if os.path.exists(SETUPS_FILE):
        df.to_csv(SETUPS_FILE, mode='a', header=False, index=False)
    else:
        df.to_csv(SETUPS_FILE, mode='w', header=True, index=False)

def main():
    if not os.path.exists(TICKERS_FILE):
        print(f"{TICKERS_FILE} tidak ditemukan!")
        return

    with open(TICKERS_FILE) as f:
        tickers = [line.strip() for line in f if line.strip()]

    for ticker in tickers:
        try:
            scan_ticker(ticker)
        except Exception as e:
            print(f"Error scanning {ticker}: {e}")

if __name__ == "__main__":
    main()
