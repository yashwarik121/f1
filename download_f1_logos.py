import os
import requests

# Folder to store logos
os.makedirs("assets", exist_ok=True)

# Mapping of F1 teams to their logo URLs
team_logos = {
    "Red Bull Racing": "https://upload.wikimedia.org/wikipedia/en/thumb/7/77/Red_Bull_Racing_logo.svg/320px-Red_Bull_Racing_logo.svg.png",
    "Ferrari": "https://upload.wikimedia.org/wikipedia/en/thumb/d/d8/Scuderia_Ferrari_Logo.svg/320px-Scuderia_Ferrari_Logo.svg.png",
    "Mercedes": "https://upload.wikimedia.org/wikipedia/en/thumb/3/34/Mercedes_AMG_Petronas_F1_Logo.svg/320px-Mercedes_AMG_Petronas_F1_Logo.svg.png",
    "Aston Martin": "https://upload.wikimedia.org/wikipedia/en/thumb/f/f3/Aston_Martin_Aramco_Cognizant_F1_Team_Logo.svg/320px-Aston_Martin_Aramco_Cognizant_F1_Team_Logo.svg.png",
    "Alpine": "https://upload.wikimedia.org/wikipedia/en/thumb/d/d2/Alpine_F1_Team_Logo.svg/320px-Alpine_F1_Team_Logo.svg.png",
    "McLaren": "https://upload.wikimedia.org/wikipedia/en/thumb/f/f7/McLaren_F1_Logo.svg/320px-McLaren_F1_Logo.svg.png",
    "AlphaTauri": "https://upload.wikimedia.org/wikipedia/en/thumb/2/27/Scuderia_AlphaTauri_logo.svg/320px-Scuderia_AlphaTauri_logo.svg.png",
    "Alfa Romeo": "https://upload.wikimedia.org/wikipedia/en/thumb/4/44/Alfa_Romeo_F1_Team_Logo.svg/320px-Alfa_Romeo_F1_Team_Logo.svg.png",
    "Haas": "https://upload.wikimedia.org/wikipedia/en/thumb/3/3e/Haas_F1_Team_logo.svg/320px-Haas_F1_Team_logo.svg.png",
    "Williams": "https://upload.wikimedia.org/wikipedia/en/thumb/7/77/Williams_Grand_Prix_Engineering_Logo.svg/320px-Williams_Grand_Prix_Engineering_Logo.svg.png"
}

# Download each logo
for team, url in team_logos.items():
    filename = f"assets/{team}.png"
    print(f"Downloading {team} logo...")
    response = requests.get(url)
    if response.status_code == 200:
        with open(filename, "wb") as f:
            f.write(response.content)
    else:
        print(f"Failed to download {team} logo.")

print("✅ All logos downloaded into 'assets' folder!")
