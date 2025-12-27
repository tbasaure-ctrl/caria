import pandas as pd
import matplotlib.pyplot as plt

# Carga de datos
df = pd.read_csv('signals_history.csv')
df = df.rename(columns={'Unnamed: 0': 'Date'})
df['Date'] = pd.to_datetime(df['Date'])

# Define el umbral para connectivity
umbral = 0.30

# Gráfico de líneas con umbral
plt.figure(figsize=(10, 6))
plt.plot(df['Date'], df['entropy'], label='entropy')
plt.plot(df['Date'], df['connectivity'], label='connectivity')
plt.plot(df['Date'], df['asf'], label='asf')

# Línea horizontal para el umbral
plt.axhline(umbral, linestyle='--', label='umbral connectivity')

plt.xlabel('Date')
plt.ylabel('Value')
plt.title('Señales con umbral de connectivity')
plt.legend()
plt.tight_layout()
plt.show()
