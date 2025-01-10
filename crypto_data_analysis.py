import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

# Wczytanie danych
data_path = 'data_unprepared/crypto_data.csv'
data = pd.read_csv(data_path, parse_dates=['timestamp'])

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

# Wizualizacja rozkładu cen zamknięcia ETH
plt.figure(figsize=(10, 6))
sns.histplot(data['eth_usdt_close'], bins=50, kde=True)
plt.title('Rozkład cen zamknięcia ETH')
plt.xlabel('Cena zamknięcia ETH (USDT)')
plt.ylabel('Częstotliwość')
plt.show()

# Wizualizacja trendu cen zamknięcia ETH w czasie
plt.figure(figsize=(15, 7))
plt.plot(data['timestamp'], data['eth_usdt_close'], label='Cena zamknięcia ETH')
plt.title('Trend cen zamknięcia ETH w czasie')
plt.xlabel('Data')
plt.ylabel('Cena zamknięcia ETH (USDT)')
plt.legend()
plt.show()

# Analiza sezonowości
decomposition = seasonal_decompose(data['eth_usdt_close'], model='multiplicative', period=30)
fig = decomposition.plot()
fig.set_size_inches(15, 10)
plt.show()

# Analiza autokorelacji
plt.figure(figsize=(15, 7))
plot_acf(data['eth_usdt_close'].dropna(), lags=50)
plt.title('Autokorelacja cen zamknięcia ETH')
plt.show()

plt.figure(figsize=(15, 7))
plot_pacf(data['eth_usdt_close'].dropna(), lags=50)
plt.title('Częściowa autokorelacja cen zamknięcia ETH')
plt.show()

# Analiza zmienności
data['eth_usdt_close_diff'] = data['eth_usdt_close'].diff()
plt.figure(figsize=(15, 7))
plt.plot(data['timestamp'], data['eth_usdt_close_diff'], label='Zmienność cen zamknięcia ETH')
plt.title('Zmienność cen zamknięcia ETH w czasie')
plt.xlabel('Data')
plt.ylabel('Zmienność ceny zamknięcia ETH (USDT)')
plt.legend()
plt.show()

# Korelacja między zmiennymi
correlation_matrix = data.corr()
plt.figure(figsize=(15, 10))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Macierz korelacji')
plt.show()