import time
import logging
import pandas as pd
import numpy as np
if not hasattr(np, "NaN"):
    np.NaN = np.nan
import pandas_ta as ta
from datetime import datetime, timedelta, timezone
from binance.um_futures import UMFutures
from binance.error import ClientError

# Strategy parameters
SYMBOL = "XRPUSDT"
BTC = "BTCUSDT"
INTERVAL = "4h"
RSI_LEN = 9
RSI_UPPER = 75
RSI_LOWER = 25
STOP_LOSS_PCT = 3.5 
TAKE_PROFIT_PCT = 5.0  
LEVERAGE = 11  
API_KEY = ""
API_SECRET = ""

logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(levelname)s: %(message)s')

client = UMFutures(key=API_KEY, secret=API_SECRET)

try:
    client.change_leverage(symbol=SYMBOL, leverage=LEVERAGE)
    logging.info(f"Leverage for {SYMBOL} set to {LEVERAGE}x.")
except ClientError as error:
    logging.error(f"Failed to set leverage: {error}")

def get_precision(symbol):
    try:
        exchange_info = client.exchange_info()
        for s in exchange_info['symbols']:
            if s['symbol'] == symbol:
                qty_precision = s['quantityPrecision']
                price_precision = s['pricePrecision']
                return qty_precision, price_precision
    except ClientError as error:
        logging.error(f"Error fetching precision: {error}")
    return None, None

def fetch_klines(symbol, interval, lookback_days=5):
    try:
        end_time = datetime.now(timezone.utc)
        start_time = end_time - timedelta(days=lookback_days)
        klines = client.klines(
            symbol=symbol, 
            interval=interval,
            startTime=int(start_time.timestamp() * 1000),
            endTime=int(end_time.timestamp() * 1000),
            limit=500
        )
        
        df = pd.DataFrame(klines, columns=[
            'timestamp', 'Open', 'High', 'Low', 'Close', 'Volume',
            'Close_time', 'Quote_asset_volume', 'Number_of_trades',
            'Taker_buy_base_asset_volume', 'Taker_buy_quote_asset_volume', 'Ignore'
        ])
        
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df.set_index('timestamp', inplace=True)
        numeric_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
        df[numeric_columns] = df[numeric_columns].astype(float)
        
        return df
    except ClientError as error:
        logging.error(f"Error fetching historical data for {symbol}: {error}")
        return pd.DataFrame()

def prepare_data():
    try:
        df_sym = fetch_klines(SYMBOL, INTERVAL)
        df_btc = fetch_klines(BTC, INTERVAL)
        
        if df_sym.empty or df_btc.empty:
            logging.error("Failed to fetch necessary data.")
            return None
        
        # Ensure both dataframes have the same index
        idx = df_sym.index.intersection(df_btc.index)
        df_sym, df_btc = df_sym.loc[idx], df_btc.loc[idx]
        
        # Calculate average price for each asset
        avg_sym = df_sym[['Open', 'High', 'Low', 'Close']].mean(axis=1)
        avg_btc = df_btc[['Open', 'High', 'Low', 'Close']].mean(axis=1)
        
        # Create final dataframe with ratio and RSI
        df = df_sym.copy()
        df['rel'] = avg_sym / avg_btc
        df['rsi'] = ta.rsi(df['rel'], length=RSI_LEN)
        
        return df.dropna()
    except Exception as e:
        logging.error(f"Error in prepare_data: {e}")
        return None

def check_signal(df):
    if len(df) < 2:
        return None
    
    rsi_prev, rsi_now = df.rsi.iloc[-2], df.rsi.iloc[-1]
    price = df.Close.iloc[-1]
    
    if rsi_prev < RSI_LOWER <= rsi_now:
        return {
            'signal': 'BUY',
            'price': price,
            'sl_price': price * (1 - STOP_LOSS_PCT / 100),
            'tp_price': price * (1 + TAKE_PROFIT_PCT / 100)
        }
    elif rsi_prev > RSI_UPPER >= rsi_now:
        return {
            'signal': 'SELL',
            'price': price,
            'sl_price': price * (1 + STOP_LOSS_PCT / 100),
            'tp_price': price * (1 - TAKE_PROFIT_PCT / 100)
        }
    
    return None

def position_open(max_retries=3):
    for attempt in range(max_retries):
        try:
            positions = client.account()['positions']
            for position in positions:
                if position['symbol'] == SYMBOL and float(position['positionAmt']) != 0:
                    return True
            return False
        except ClientError as error:
            backoff = 2 ** attempt
            logging.warning(f"Attempt {attempt+1} to check position failed: {error}. Retrying in {backoff} seconds...")
            time.sleep(backoff)
        except Exception as e:
            backoff = 2 ** attempt
            logging.warning(f"Unexpected error in position_open() (attempt {attempt+1}): {e}. Retrying in {backoff} seconds...")
            time.sleep(backoff)
    
    logging.error(f"All {max_retries} attempts to check position status failed")
    return False 

def cancel_open_orders():
    try:
        client.cancel_open_orders(symbol=SYMBOL)
        logging.info(f"Canceled all open orders for {SYMBOL}")
    except ClientError as error:
        logging.error(f"Error canceling open orders: {error}")

def place_with_retry(order_params, max_retries=3):
    for attempt in range(max_retries):
        try:
            return client.new_order(**order_params)
        except Exception as e:
            backoff = 2 ** attempt
            logging.warning(f"Attempt {attempt+1} failed: {e}. Retrying in {backoff} seconds...")
            time.sleep(backoff)
    logging.error(f"All {max_retries} attempts failed for order: {order_params}")
    return None

def place_trade(signal):
    if position_open():
        logging.info("Position already open. Skipping.")
        return
    
    cancel_open_orders()
    
    qty_precision, price_precision = get_precision(SYMBOL)
    if qty_precision is None or price_precision is None:
        logging.error("Could not retrieve precision information.")
        return
    
    try:
        balance = float(client.account()['totalWalletBalance'])
        current_price = float(client.mark_price(symbol=SYMBOL)['markPrice'])
        
        qty = round((balance * LEVERAGE * 0.95) / current_price, qty_precision)
        
        side = signal['signal']
        opposite_side = 'SELL' if side == 'BUY' else 'BUY'
        
        sl_price = round(signal['sl_price'], price_precision)
        tp_price = round(signal['tp_price'], price_precision)
        
        logging.info(f"Placing {side} order for {qty} {SYMBOL} at market...")
        main_order = place_with_retry({
            'symbol': SYMBOL,
            'side': side,
            'type': 'MARKET',
            'quantity': qty
        })
        
        logging.info(f"Placing stop-loss order at {sl_price}...")
        place_with_retry({
            'symbol': SYMBOL,
            'side': opposite_side,
            'type': 'STOP_MARKET',
            'stopPrice': sl_price,
            'closePosition': True
        })
        
        logging.info(f"Placing take-profit order at {tp_price}...")
        place_with_retry({
            'symbol': SYMBOL,
            'side': opposite_side,
            'type': 'TAKE_PROFIT_MARKET',
            'stopPrice': tp_price,
            'closePosition': True
        })
        
        logging.info(f"Successfully placed {side} order with SL at {sl_price} and TP at {tp_price}")
        while position_open():
            time.sleep(5)
        logging.info("Position closed. Canceling any remaining open orders...")
        cancel_open_orders()
        logging.info("Position closed and no open orders remain.")
        
    except ClientError as error:
        logging.error(f"Trade execution failed: {error}")
    except Exception as e:
        logging.error(f"Unexpected error in place_trade: {e}")

def calculate_next_interval(interval):
    now = datetime.now(timezone.utc)
    
    if interval == '4h':
        target_hour = 4 * ((now.hour // 4) + 1) % 24
        target_date = now.date()
        
        if target_hour < now.hour:
            target_date = now.date() + timedelta(days=1)
            
        next_time = datetime.combine(
            target_date, 
            datetime.min.time(), 
            tzinfo=timezone.utc
        ).replace(hour=target_hour)
        
        if next_time <= now:
            next_time += timedelta(hours=4)
    elif interval == '1h':
        next_time = now.replace(minute=0, second=0, microsecond=0) + timedelta(hours=1)
    elif interval == '15m':
        minutes_offset = now.minute % 15
        next_time = now.replace(minute=now.minute - minutes_offset + 15, second=0, microsecond=0)
        if next_time <= now:
            next_time += timedelta(minutes=15)
    else:
        next_time = now + timedelta(minutes=5)
    
    return (next_time - now).total_seconds()

def run_bot():
    logging.info(f"Starting CompRSI trading bot for {SYMBOL}")
    
    interval_map = {
        '4h': '4h',
        '1h': '1h',
        '15m': '15m'
    }
    binance_interval = interval_map.get(INTERVAL, '4h')
    
    while True:
        try:
            df = prepare_data()
            
            if df is not None and not df.empty:
                signal = check_signal(df)
                
                if signal:
                    logging.info(f"Detected {signal['signal']} signal at {signal['price']}")
                    place_trade(signal)
                else:
                    logging.info("No trade signal detected")
            else:
                logging.warning("Could not prepare data for analysis")
            
            sleep_seconds = calculate_next_interval(binance_interval)
            logging.info(f"Sleeping for {int(sleep_seconds)} seconds until next {INTERVAL} candle...")
            
            time.sleep(sleep_seconds)
            
        except Exception as e:
            logging.error(f"Error in run_bot: {e}")
            time.sleep(60)

if __name__ == "__main__":
    try:
        run_bot()
    except KeyboardInterrupt:
        logging.info("Bot stopped by user")
    except Exception as e:
        logging.error(f"Unexpected error: {e}")