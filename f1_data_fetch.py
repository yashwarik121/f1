import fastf1
import pandas as pd

# Enable cache so it doesn't re-download each time
fastf1.Cache.enable_cache('f1_cache')

print("Step 1: Loading session...")
session = fastf1.get_session(2023, 'Monaco', 'R')  # Year, Grand Prix, Race
print("Step 2: Session object created, now loading data (this may take 1–5 mins)...")

session.load()
print("Step 3: Session loaded successfully!")

# Get lap data
laps = session.laps
print(f"Step 4: Total laps fetched: {len(laps)}")

# Save to CSV
df = pd.DataFrame(laps)
df.to_csv('monaco_2023_laps.csv', index=False)
print("Step 5: Lap data saved to monaco_2023_laps.csv")
