import pandas as pd
import matplotlib.pyplot as plt
import kagglehub

path = kagglehub.dataset_download("risakashiwabara/tokyo-weatherdata")
print("Rruga e dataset-it:", path)

df = pd.read_csv(f"{path}/weather_tokyo_data.csv")

df['Date'] = pd.to_datetime(df['year'].astype(str) + '-' + df['day'], format='%Y-%m/%d')
df.columns = [item.strip() for item in df.columns]
df['temperature'] = df['temperature'].apply(lambda x: float(x.replace('(', '-').replace(')', '')))

df['Month'] = df['Date'].dt.month
monthly_avg_temp = df.groupby('Month')['temperature'].mean()

plt.figure(figsize=(10, 6))
monthly_avg_temp.plot(kind='bar', color='skyblue')
plt.title('Temperatura mesatare mujore në Tokio')
plt.xlabel('Muaji')
plt.ylabel('Temperatura mesatare (°C)')
plt.xticks(ticks=range(12), labels=["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"], rotation=45)
plt.tight_layout()
plt.show()

plt.figure(figsize=(12, 6))
plt.plot(df['Date'], df['temperature'], color='orange', linewidth=1)
plt.title('Trendet e Temperaturës në Tokio')
plt.xlabel('Data')
plt.ylabel('Temperatura (°C)')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

avg_temperature = df['temperature'].mean()
print(f"\nTemperatura mesatare për të gjithë datasetin: {avg_temperature:.2f}°C")

hottest_day = df.loc[df['temperature'].idxmax()]
coldest_day = df.loc[df['temperature'].idxmin()]

print("\nDita më e nxehtë:")
print(hottest_day)
print("\nDita më e ftohtë:")
print(coldest_day)

seasons = {
    'Pranvera': [3, 4, 5],
    'Vera': [6, 7, 8],
    'Vjeshta': [9, 10, 11],
    'Dimri': [12, 1, 2]
}

seasonal_avg_temp = {}
for season, months in seasons.items():
    seasonal_data = df[df['Month'].isin(months)]
    seasonal_avg_temp[season] = seasonal_data['temperature'].mean()

print("\nTemperaturat mesatare për çdo sezon:")
for season, temp in seasonal_avg_temp.items():
    print(f"{season}: {temp:.2f}°C")