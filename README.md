# rsi-comp-relative-bot
# RSI Relative Strategy Bot 📈

A simple Binance Futures trading bot that uses **RSI signals on the relative price** of an altcoin (e.g., XRP) versus BTC to make trading decisions.

## 🔍 Strategy Overview

- Computes the RSI on the ratio of `SYMBOL/BTCUSDT` using 4-hour candles.
- Places **market orders** when RSI crosses:
  - **Above** the lower threshold → **BUY**
  - **Below** the upper threshold → **SELL**
- Each trade includes:
  - Fixed **Stop Loss** (e.g., 3.5%)
  - Fixed **Take Profit** (e.g., 5%)

## ⚙️ Setup

1. Install dependencies:
   ```bash
   pip install binance-futures-connector pandas pandas-ta numpy
