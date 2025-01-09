import pandas as pd

# Define file paths
market_data_path = 'data_unprepared/market_data.csv'
crypto_data_path = 'data_unprepared/crypto_data.csv'
output_path = 'data/merged_data.csv'

# Read the CSV files, skipping the first two rows and setting the correct header row
market_data = pd.read_csv(market_data_path, skiprows=2)
crypto_data = pd.read_csv(crypto_data_path)

# Rename the 'Ticker' column to 'Date'
market_data.rename(columns={'Ticker': 'Date'}, inplace=True)

# Drop the first row which contains the string "Date"
market_data = market_data[market_data['Date'] != 'Date']

# Ensure the 'Date' column is in datetime format
market_data['Date'] = pd.to_datetime(market_data['Date'])
crypto_data['timestamp'] = pd.to_datetime(crypto_data['timestamp'])

# Merge the dataframes on the 'Date' column
merged_data = pd.merge(market_data, crypto_data, left_on='Date', right_on='timestamp', how='inner')

# Drop the duplicate 'timestamp' column
merged_data.drop(columns=['timestamp'], inplace=True)

# Save the merged dataframe to a new CSV file
merged_data.to_csv(output_path, index=False)

print(f"Merged data saved to {output_path}")