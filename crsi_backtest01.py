import logging
import numpy as np
if not hasattr(np, "NaN"):
    np.NaN = np.nan
import pandas as pd
import pandas_ta as ta
from tqdm import tqdm
from binance.client import Client
from backtesting import Backtest, Strategy

SYMBOL, BTC = "XRPUSDT", "BTCUSDT"
START, END  = "2025-05-11", "2025-06-01"
INTERVAL    = Client.KLINE_INTERVAL_4HOUR

RSI_LEN              = 9
RSI_UPPER, RSI_LOWER = 75, 25

CASH, COMM  = 10_000, 0.0006
SL_PCT, TP_PCT = 3.5, 5.0

API_KEY = ""
API_SECRET = ""

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s ‑ %(levelname)s: %(message)s")

def fetch_klines(client, symbol, start, end, interval):
    gen = client.futures_historical_klines_generator(symbol, interval,
                                                     start, end)
    cols = ['ts', 'Open', 'High', 'Low', 'Close', 'Volume',
            'ct', 'qav', 'nt', 'tbb', 'tbq', 'ig']
    df = pd.DataFrame(list(gen), columns=cols)
    if df.empty:
        raise RuntimeError(f"No data for {symbol}")
    df['ts'] = pd.to_datetime(df['ts'], unit='ms')
    df.set_index('ts', inplace=True)
    df = df[['Open', 'High', 'Low', 'Close', 'Volume']].apply(pd.to_numeric)
    return df

def load_data():
    client = Client(API_KEY, API_SECRET)
    with tqdm(total=2, desc="Download") as bar:
        df_sym = fetch_klines(client, SYMBOL, START, END, INTERVAL); bar.update(1)
        df_btc = fetch_klines(client, BTC,    START, END, INTERVAL); bar.update(1)

    idx = df_sym.index.intersection(df_btc.index)
    df_sym, df_btc = df_sym.loc[idx], df_btc.loc[idx]

    avg_sym = df_sym[['Open','High','Low','Close']].mean(axis=1)
    avg_btc = df_btc[['Open','High','Low','Close']].mean(axis=1)

    df = df_sym.copy()
    df['rel'] = avg_sym / avg_btc
    df['rsi'] = ta.rsi(df['rel'], length=RSI_LEN)
    return df.dropna()

class CompRSI(Strategy):
    # Fixed
    rsi_len     = RSI_LEN
    upper_band  = RSI_UPPER
    lower_band  = RSI_LOWER
    # Tunable
    sl_pct = SL_PCT
    tp_pct = TP_PCT

    def init(self):
        """Mandatory empty init so the class is not abstract."""
        pass

    def next(self):
        if np.isnan(self.data.rsi[-2]) or np.isnan(self.data.rsi[-1]):
            return

        rsi_prev, rsi_now = self.data.rsi[-2], self.data.rsi[-1]
        price = self.data.Close[-1]

        if rsi_prev < self.lower_band <= rsi_now:
            if self.position.is_short:
                self.position.close()
            self.buy(sl=price*(1-self.sl_pct/100),
                     tp=price*(1+self.tp_pct/100))

        elif rsi_prev > self.upper_band >= rsi_now:
            if self.position.is_long:
                self.position.close()
            self.sell(sl=price*(1+self.sl_pct/100),
                      tp=price*(1-self.tp_pct/100))

def main():
    df = load_data()
    print(f"Loaded {len(df):,} candles ({df.index[0]} → {df.index[-1]})")

    bt = Backtest(df, CompRSI, cash=CASH, commission=COMM,
                  exclusive_orders=True)
    stats = bt.run()
    print(stats)

    if stats['# Trades'] == 0:
        print("No trades produced – adjust parameters if necessary.")
        return

    sl_vals = [1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5]
    tp_vals = [1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5]

    best = bt.optimize(sl_pct=sl_vals,
                       tp_pct=tp_vals,
                       maximize='Equity Final [$]')

    opt_sl, opt_tp = best._strategy.sl_pct, best._strategy.tp_pct
    print("\nOptimised parameters")
    print("sl-",opt_sl)
    print("tp-",opt_tp)
    bt_final = Backtest(df, CompRSI, cash=CASH, commission=COMM,
                        exclusive_orders=True)
    final_stats = bt_final.run(sl_pct=opt_sl, tp_pct=opt_tp)
    bt_final.plot(filename=f"CompRSI_{SYMBOL}_optimised.html")
    print("\nFinal stats\n", final_stats)


if __name__ == "__main__":
    main()