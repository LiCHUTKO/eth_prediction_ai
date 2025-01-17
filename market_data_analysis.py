import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

# Wczytanie danych
data_path = 'data_unprepared/market_data.csv'
data = pd.read_csv(data_path, skiprows=2)

# Ustawienie odpowiednich nazw kolumn
data.columns = ['Date', 'nasdaq_close', 'nasdaq_high', 'nasdaq_low', 'nasdaq_open', 'nasdaq_volume',
                'sp500_close', 'sp500_high', 'sp500_low', 'sp500_open', 'sp500_volume',
                'dxy_close', 'dxy_high', 'dxy_low', 'dxy_open', 'dxy_volume',
                'brent_close', 'brent_high', 'brent_low', 'brent_open', 'brent_volume',
                'wti_close', 'wti_high', 'wti_low', 'wti_open', 'wti_volume',
                'gold_close', 'gold_high', 'gold_low', 'gold_open', 'gold_volume',
                'us10y_close', 'us10y_high', 'us10y_low', 'us10y_open', 'us10y_volume',
                'us30y_close', 'us30y_high', 'us30y_low', 'us30y_open', 'us30y_volume']

# Drop the first row which contains the string "Date"
data = data[data['Date'] != 'Date']

# Konwersja kolumny 'Date' na format daty
data['Date'] = pd.to_datetime(data['Date'])

# Sprawdzenie pierwszych kilku wierszy danych
print("Pierwsze kilka wierszy danych:")
print(data.head())

# Sprawdzenie brakujących wartości
missing_values = data.isnull().sum()
print("\nBrakujące wartości w każdej kolumnie:")
print(missing_values)

# Podstawowe statystyki opisowe
print("\nPodstawowe statystyki opisowe:")
print(data.describe())

# Wizualizacja rozkładu cen zamknięcia NASDAQ
plt.figure(figsize=(10, 6))
sns.histplot(data['nasdaq_close'], bins=50, kde=True)
plt.title('Rozkład cen zamknięcia NASDAQ')
plt.xlabel('Cena zamknięcia NASDAQ')
plt.ylabel('Częstotliwość')
plt.show()

# Wizualizacja trendu cen zamknięcia NASDAQ w czasie
plt.figure(figsize=(15, 7))
plt.plot(data['Date'], data['nasdaq_close'], label='Cena zamknięcia NASDAQ')
plt.title('Trend cen zamknięcia NASDAQ w czasie')
plt.xlabel('Data')
plt.ylabel('Cena zamknięcia NASDAQ')
plt.legend()
plt.show()

# Analiza sezonowości
decomposition = seasonal_decompose(data['nasdaq_close'], model='multiplicative', period=30)
fig = decomposition.plot()
fig.set_size_inches(15, 10)
plt.show()

# Analiza autokorelacji
plt.figure(figsize=(15, 7))
plot_acf(data['nasdaq_close'].dropna(), lags=50)
plt.title('Autokorelacja cen zamknięcia NASDAQ')
plt.show()

plt.figure(figsize=(15, 7))
plot_pacf(data['nasdaq_close'].dropna(), lags=50)
plt.title('Częściowa autokorelacja cen zamknięcia NASDAQ')
plt.show()

# Analiza zmienności
data['nasdaq_diff'] = data['nasdaq_close'].diff()
plt.figure(figsize=(15, 7))
plt.plot(data['Date'], data['nasdaq_diff'], label='Zmienność cen zamknięcia NASDAQ')
plt.title('Zmienność cen zamknięcia NASDAQ w czasie')
plt.xlabel('Data')
plt.ylabel('Zmienność ceny zamknięcia NASDAQ')
plt.legend()
plt.show()

# Korelacja między zmiennymi
correlation_matrix = data.corr()
plt.figure(figsize=(15, 10))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Macierz korelacji')
plt.show()