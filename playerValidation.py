import requests
from datetime import datetime

username = "russellelliott"

# Define headers with a User-Agent; neccesary for the api.chess.com/pub endpoint
headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
}

# --- Get account creation date and compute age in days ---
profile_url = f"https://api.chess.com/pub/player/{username}"
profile_response = requests.get(profile_url, headers=headers)  # Add headers
print(profile_response)
profile_data = profile_response.json()

joined_timestamp = profile_data["joined"]
joined_date = datetime.fromtimestamp(joined_timestamp)
now = datetime.now()
account_age_days = (now - joined_date).days

# --- Get number of blitz games played ---
blitz_url = f"https://www.chess.com/callback/live/stats/{username}/chart?type=blitz"
blitz_response = requests.get(blitz_url, headers=headers)  # Add headers
blitz_data = blitz_response.json()

num_blitz_games = len(blitz_data)

# --- Output results ---
print(f"Chess.com username: {username}")
print(f"Account age: {account_age_days} days")
print(f"Number of blitz games played: {num_blitz_games}")
