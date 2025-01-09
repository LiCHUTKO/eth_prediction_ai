import pandas as pd
import yfinance as yf

def add_market_data(crypto_df):
    # Define tickers
    tickers = {
        'nasdaq': '^IXIC',
        'sp500': '^GSPC', 
        'dxy': 'DX-Y.NYB',
        'brent': 'BZ=F',
        'wti': 'CL=F',
        'gold': 'GC=F',
        'us10y': '^TNX',
        'us30y': '^TYX'
    }
    
    # Pobierz dane rynkowe i zapisz do tymczasowego pliku
    market_data = {}
    for name, ticker in tickers.items():
        print(f"Pobieram dane dla {name} ({ticker})")
        df = yf.download(ticker, start=crypto_df.index[0], end=crypto_df.index[-1])
        market_data[name] = df

    # Zapisz dane rynkowe do osobnego pliku do weryfikacji
    market_df = pd.concat(market_data, axis=1)
    market_df.to_csv('data/market_data_temp.csv')
    print("Zapisano dane rynkowe do market_data_temp.csv")
    
    # Poczekaj na weryfikację
    input("Sprawdź plik market_data_temp.csv i naciśnij Enter, aby kontynuować...")
    
    # Dodaj kolumny do crypto_df
    for name, df in market_data.items():
        crypto_df[f'{name}_open'] = df['Open']
        crypto_df[f'{name}_high'] = df['High'] 
        crypto_df[f'{name}_low'] = df['Low']
        crypto_df[f'{name}_close'] = df['Close']
        if 'Volume' in df.columns:
            crypto_df[f'{name}_volume'] = df['Volume']
            
    return crypto_df

# Read existing CSV
df = pd.read_csv('data/crypto_data.csv', index_col='timestamp', parse_dates=True)

# Add market data
df_with_market = add_market_data(df)

# Save enhanced dataset
df_with_market.to_csv('crypto_data_enhanced.csv')