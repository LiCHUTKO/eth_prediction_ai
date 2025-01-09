import ccxt
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, date
import ta
import time
from pathlib import Path

# Configuration
CONFIG = {
    'exchange': 'binance',
    'pairs': ['ETH/USDT', 'ETH/BTC', 'BTC/USDT'],
    'timeframe': '1d',
    'start_date': '2015-07-30',  # Ethereum launch date
    'end_date': date.today().strftime('%Y-%m-%d'),
    'retry_attempts': 3,
    'retry_delay': 5,
    'output_dir': 'data'
}

def setup_output_directory():
    Path(CONFIG['output_dir']).mkdir(exist_ok=True)

def calculate_indicators(df, pair_prefix):
    """Calculate all technical indicators for a given pair"""
    # Base price columns
    close = df[f'{pair_prefix}_close']
    high = df[f'{pair_prefix}_high']
    low = df[f'{pair_prefix}_low']
    volume = df[f'{pair_prefix}_volume']
    
    # RSI
    df[f'{pair_prefix}_rsi'] = ta.momentum.RSIIndicator(close).rsi()
    
    # Moving Averages
    for period in [20, 50, 200]:
        df[f'{pair_prefix}_ma{period}'] = ta.trend.SMAIndicator(close, window=period).sma_indicator()
    
    # Bollinger Bands
    bollinger = ta.volatility.BollingerBands(close)
    df[f'{pair_prefix}_bb_upper'] = bollinger.bollinger_hband()
    df[f'{pair_prefix}_bb_middle'] = bollinger.bollinger_mavg()
    df[f'{pair_prefix}_bb_lower'] = bollinger.bollinger_lband()
    
    # MACD
    macd = ta.trend.MACD(close)
    df[f'{pair_prefix}_macd'] = macd.macd()
    df[f'{pair_prefix}_macd_signal'] = macd.macd_signal()
    df[f'{pair_prefix}_macd_hist'] = macd.macd_diff()
    
    # Stochastic
    stoch = ta.momentum.StochasticOscillator(high, low, close)
    df[f'{pair_prefix}_stoch_k'] = stoch.stoch()
    df[f'{pair_prefix}_stoch_d'] = stoch.stoch_signal()
    
    # ATR
    df[f'{pair_prefix}_atr'] = ta.volatility.AverageTrueRange(high, low, close).average_true_range()
    
    # Fibonacci Retracement
    max_price = close.max()
    min_price = close.min()
    diff = max_price - min_price
    for level in [0, 0.236, 0.382, 0.5, 0.618, 1]:
        df[f'{pair_prefix}_fib_{int(level*1000)}'] = min_price + (diff * level)
    
    # Ichimoku Cloud
    ichimoku = ta.trend.IchimokuIndicator(high, low)
    df[f'{pair_prefix}_ichimoku_conv'] = ichimoku.ichimoku_conversion_line()
    df[f'{pair_prefix}_ichimoku_base'] = ichimoku.ichimoku_base_line()
    
    # Volume (OBV)
    df[f'{pair_prefix}_obv'] = ta.volume.OnBalanceVolumeIndicator(close, volume).on_balance_volume()
    
    # CCI
    df[f'{pair_prefix}_cci'] = ta.trend.CCIIndicator(high, low, close).cci()
    
    return df

def get_historical_data(exchange_id='binance', timeframe='1d'):
    try:
        exchange = getattr(ccxt, exchange_id)()
        pairs = {
            'eth_usdt': 'ETH/USDT',
            'eth_btc': 'ETH/BTC',
            'btc_usdt': 'BTC/USDT'
        }
        
        # Convert dates to timestamps
        since = int(datetime.strptime(CONFIG['start_date'], '%Y-%m-%d').timestamp() * 1000)
        until = int(datetime.strptime(CONFIG['end_date'], '%Y-%m-%d').timestamp() * 1000)
        
        combined_df = pd.DataFrame()
        
        for pair_name, symbol in pairs.items():
            print(f"Fetching {symbol} from {CONFIG['start_date']} to {CONFIG['end_date']}...")
            
            all_ohlcv = []
            current_since = since
            
            while current_since < until:
                ohlcv = exchange.fetch_ohlcv(symbol, timeframe, since=current_since)
                if not ohlcv:
                    break
                    
                all_ohlcv.extend(ohlcv)
                current_since = ohlcv[-1][0] + 1
                
                # Progress update
                current_date = datetime.fromtimestamp(current_since/1000).strftime('%Y-%m-%d')
                print(f"Progress: {current_date}", end='\r')
                
                time.sleep(exchange.rateLimit / 1000)
            
            if all_ohlcv:
                df = pd.DataFrame(all_ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
                df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
                df.set_index('timestamp', inplace=True)
                
                # Filter data up to end_date
                df = df[df.index <= CONFIG['end_date']]
                
                # Rename columns
                df.columns = [f'{pair_name}_{col}' for col in df.columns]
                
                # Calculate indicators
                df = calculate_indicators(df, pair_name)
                
                if combined_df.empty:
                    combined_df = df
                else:
                    combined_df = combined_df.join(df)
            
            print(f"\nCompleted fetching {symbol}")
            
        return combined_df
        
    except Exception as e:
        print(f"Error: {e}")
        return None

def main():
    print("Starting cryptocurrency analysis")
    setup_output_directory()
    
    df = get_historical_data()
    if df is not None:
        filename = Path(CONFIG['output_dir']) / "crypto_data.csv"
        df.to_csv(filename)
        print(f"Data saved to {filename}")
    
    print("Cryptocurrency analysis completed")

if __name__ == "__main__":
    main()