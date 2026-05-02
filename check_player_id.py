from nba_api.stats.static import players

# RUN THIS TO CHECK FOR PLAYER_ID
all_players = players.get_players()
get_id = [p for p in all_players if p["full_name"] == "Stephen Curry"] # Add name here
print(get_id)