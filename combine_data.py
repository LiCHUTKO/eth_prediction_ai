import pandas as pd

def combine_crypto_market_data():
    # Load both CSV files
    crypto_df = pd.read_csv('crypto_data.csv', index_col='timestamp', parse_dates=True)
    market_df = pd.read_csv('market_data_temp.csv', index_col='Date', parse_dates=True)
    
    # Rename market data columns to match format
    new_columns = {
        ('nasdaq', 'Close'): 'nasdaq_close',
        ('nasdaq', 'High'): 'nasdaq_high', 
        ('nasdaq', 'Low'): 'nasdaq_low',
        ('nasdaq', 'Open'): 'nasdaq_open',
        ('nasdaq', 'Volume'): 'nasdaq_volume',
        
        ('sp500', 'Close'): 'sp500_close',
        ('sp500', 'High'): 'sp500_high',
        ('sp500', 'Low'): 'sp500_low', 
        ('sp500', 'Open'): 'sp500_open',
        ('sp500', 'Volume'): 'sp500_volume',
        
        ('dxy', 'Close'): 'dxy_close',
        ('dxy', 'High'): 'dxy_high',
        ('dxy', 'Low'): 'dxy_low',
        ('dxy', 'Open'): 'dxy_open',
        
        ('brent', 'Close'): 'brent_close', 
        ('brent', 'High'): 'brent_high',
        ('brent', 'Low'): 'brent_low',
        ('brent', 'Open'): 'brent_open',
        ('brent', 'Volume'): 'brent_volume',
        
        ('wti', 'Close'): 'wti_close',
        ('wti', 'High'): 'wti_high',
        ('wti', 'Low'): 'wti_low', 
        ('wti', 'Open'): 'wti_open',
        ('wti', 'Volume'): 'wti_volume',
        
        ('gold', 'Close'): 'gold_close',
        ('gold', 'High'): 'gold_high',
        ('gold', 'Low'): 'gold_low',
        ('gold', 'Open'): 'gold_open',
        ('gold', 'Volume'): 'gold_volume',
        
        ('us10y', 'Close'): 'us10y_close',
        ('us10y', 'High'): 'us10y_high', 
        ('us10y', 'Low'): 'us10y_low',
        ('us10y', 'Open'): 'us10y_open',
        
        ('us30y', 'Close'): 'us30y_close',
        ('us30y', 'High'): 'us30y_high',
        ('us30y', 'Low'): 'us30y_low',
        ('us30y', 'Open'): 'us30y_open'
    }
    
    market_df = market_df.rename(columns=new_columns)
    
    # Combine dataframes
    combined_df = pd.concat([crypto_df, market_df], axis=1)
    
    # Save combined data
    combined_df.to_csv('crypto_data.csv')
    print("Dane zostały połączone i zapisane do crypto_data.csv")

if __name__ == "__main__":
    combine_crypto_market_data()