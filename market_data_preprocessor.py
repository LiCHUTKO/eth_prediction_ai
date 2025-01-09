import pandas as pd
import yfinance as yf
from pathlib import Path
from datetime import date

# Define data directory
DATA_DIR = Path('data_unprepared')

def download_market_data():
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
    
    # Ensure data directory exists
    DATA_DIR.mkdir(exist_ok=True)
    
    # Download market data
    market_data = {}
    start_date = '2015-07-30'  # Ethereum launch date
    end_date = date.today().strftime('%Y-%m-%d')
    
    for name, ticker in tickers.items():
        print(f"Pobieram dane dla {name} ({ticker})")
        df = yf.download(ticker, start=start_date, end=end_date)
        market_data[name] = df

    # Save market data
    market_df = pd.concat(market_data, axis=1)
    market_df.to_csv(DATA_DIR / 'market_data.csv')
    print("Zapisano dane rynkowe do data_unprepared/market_data.csv")

if __name__ == "__main__":
    download_market_data()