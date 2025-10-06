import nfl_data_py as nfl
import nflreadpy as nflr
import pandas as pd
import numpy as np

CURRENT_SEASON = 2025


def import_data(start_year=2014, end_year=2024):
    if end_year > CURRENT_SEASON:
        end_year = CURRENT_SEASON

    # list of years to import data from
    years = range(start_year, end_year + 1)

    # import weekly team stats
    try:
        team_data = nflr.load_team_stats(years).to_pandas()
        print(f"Team stats for {start_year}-{end_year} loaded successfully.")
    except Exception as e:
        print(f"Error loading team stats: {e}")
        # if error, try reduced year range
        years = range(start_year, end_year)
        team_data = nflr.load_team_stats(years).to_pandas()
        print(f"Team stats for {start_year}-{end_year-1} loaded successfully.")

    # import team data
    try:
        team_desc = nfl.import_team_desc()
        team_dict = {row['team_abbr']: row['team_name'] for index, row in team_desc.iterrows()}
        print("Team descriptions loaded successfully.")
    except Exception as e:
        print(f"Error loading team descriptions: {e}") 

    # import schedules data
    try:
        schedule_data = nfl.import_schedules(years)
        print(f"Schedules for {start_year}-{end_year} loaded successfully.")
    except Exception as e:
        print(f"Error loading schedules: {e}")
        # if error, try reduced year range
        years = range(start_year, end_year)
        schedule_data = nfl.import_schedules(years)
        print(f"Schedules for {start_year}-{end_year-1} loaded successfully.")

        
    return team_data, team_dict, schedule_data

def create_team_features(weekly_data, season, week):
    
    season_data = weekly_data[(weekly_data['season'] == season) & (weekly_data['week'] < week)]

    team_features = {}

    for team in season_data['team'].unique():
        continue


def main():
    team_data, team_dict, schedule_data = import_data(2014, 2024)
    
    # Display the first few rows of each DataFrame
    print(team_data.columns)
    #print(schedule_data.head())
    #print(team_dict)

if __name__ == "__main__":
    main()
