import pandas as pd
import matplotlib.pyplot as plt

# Load the saved CSV
df = pd.read_csv('monaco_2023_laps.csv')

# Keep only needed columns
df = df[['Driver', 'Team', 'LapTime']]

# Convert LapTime to seconds
df['LapTime'] = pd.to_timedelta(df['LapTime']).dt.total_seconds()

# Remove NaNs and outliers (pit laps, safety car, etc.)
df = df.dropna(subset=['LapTime'])
df = df[df['LapTime'] > 40]  # Ignore weirdly low lap times

# Calculate average lap time per team
team_avg = df.groupby('Team')['LapTime'].mean().sort_values()

# Plot
plt.figure(figsize=(10,6))
team_avg.plot(kind='bar', color='skyblue')
plt.ylabel('Average Lap Time (seconds)')
plt.title('Average Lap Time per Team - Monaco 2023')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()

# Save chart
plt.savefig('team_avg_lap_times.png')
plt.show()
