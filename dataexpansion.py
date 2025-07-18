import pandas as pd
import numpy as np
from datetime import datetime
import urllib.request
import os


def download_premier_league_data():
    """Download the latest Premier League data from football-data.co.uk"""
    urls = {
        'E0_2324.csv': 'https://www.football-data.co.uk/mmz4281/2324/E0.csv',  # 2023/24
        'E0_2425.csv': 'https://www.football-data.co.uk/mmz4281/2425/E0.csv'  # 2024/25
    }

    for filename, url in urls.items():
        try:
            print(f"Downloading {filename}...")
            urllib.request.urlretrieve(url, filename)
            print(f"Downloaded {filename} successfully")
        except Exception as e:
            print(f"Error downloading {filename}: {e}")

    return list(urls.keys())


def convert_to_standard_format(input_file, output_file, season_end_year, source_format="football-data"):
    """
    Convert Premier League CSV data to standard format:
    Season_End_Year,Wk,Date,Home,HomeGoals,AwayGoals,Away,FTR

    Parameters:
    input_file: Path to input CSV file
    output_file: Path to output CSV file
    season_end_year: Year the season ends (e.g., 2024 for 2023-24 season)
    source_format: "football-data", "fixturedownload", "kaggle", or "auto"
    """

    # Read the CSV file
    df = pd.read_csv(input_file)

    # Print original columns to help identify format
    print("Original columns:", df.columns.tolist())

    # Initialize output dataframe
    output_df = pd.DataFrame()

    # Auto-detect format or use specified format
    if source_format == "auto":
        if 'HomeTeam' in df.columns and 'AwayTeam' in df.columns:
            source_format = "football-data"
        elif 'Home Team' in df.columns and 'Away Team' in df.columns:
            source_format = "fixturedownload"
        elif 'home_team' in df.columns and 'away_team' in df.columns:
            source_format = "kaggle"
        else:
            print("Could not auto-detect format. Please specify source_format parameter.")
            return

    print(f"Using format: {source_format}")

    # Convert based on source format
    if source_format == "football-data":
        # football-data.co.uk format
        output_df['Season_End_Year'] = season_end_year
        output_df['Date'] = pd.to_datetime(df['Date'], format='%d/%m/%Y').dt.strftime('%Y-%m-%d')
        output_df['Home'] = df['HomeTeam']
        output_df['HomeGoals'] = df['FTHG']  # Full Time Home Goals
        output_df['AwayGoals'] = df['FTAG']  # Full Time Away Goals
        output_df['Away'] = df['AwayTeam']
        output_df['FTR'] = df['FTR']  # Full Time Result

        # Calculate week number
        output_df['Date_dt'] = pd.to_datetime(output_df['Date'])
        season_start = pd.to_datetime(f'{season_end_year - 1}-08-01')
        output_df['Wk'] = ((output_df['Date_dt'] - season_start).dt.days // 7) + 1

    elif source_format == "fixturedownload":
        # fixturedownload.com format
        output_df['Season_End_Year'] = season_end_year
        output_df['Date'] = pd.to_datetime(df['Date']).dt.strftime('%Y-%m-%d')
        output_df['Home'] = df['Home Team']
        output_df['Away'] = df['Away Team']

        # These might need to be extracted from Result column if separate goal columns don't exist
        if 'Home Goals' in df.columns and 'Away Goals' in df.columns:
            output_df['HomeGoals'] = df['Home Goals']
            output_df['AwayGoals'] = df['Away Goals']
        elif 'Result' in df.columns:
            # Parse result like "2-1" to extract goals
            result_split = df['Result'].str.split('-', expand=True)
            output_df['HomeGoals'] = pd.to_numeric(result_split[0], errors='coerce')
            output_df['AwayGoals'] = pd.to_numeric(result_split[1], errors='coerce')

        # Calculate FTR
        output_df['FTR'] = np.where(output_df['HomeGoals'] > output_df['AwayGoals'], 'H',
                                    np.where(output_df['HomeGoals'] < output_df['AwayGoals'], 'A', 'D'))

        # Calculate week number
        output_df['Date_dt'] = pd.to_datetime(output_df['Date'])
        season_start = pd.to_datetime(f'{season_end_year - 1}-08-01')
        output_df['Wk'] = ((output_df['Date_dt'] - season_start).dt.days // 7) + 1

    elif source_format == "kaggle":
        # Kaggle format (varies by dataset)
        output_df['Season_End_Year'] = season_end_year
        output_df['Date'] = pd.to_datetime(df['date']).dt.strftime('%Y-%m-%d')
        output_df['Home'] = df['home_team']
        output_df['Away'] = df['away_team']
        output_df['HomeGoals'] = df['home_goals']
        output_df['AwayGoals'] = df['away_goals']

        # Calculate FTR
        output_df['FTR'] = np.where(output_df['HomeGoals'] > output_df['AwayGoals'], 'H',
                                    np.where(output_df['HomeGoals'] < output_df['AwayGoals'], 'A', 'D'))

        # Calculate week number
        output_df['Date_dt'] = pd.to_datetime(output_df['Date'])
        season_start = pd.to_datetime(f'{season_end_year - 1}-08-01')
        output_df['Wk'] = ((output_df['Date_dt'] - season_start).dt.days // 7) + 1

    # Clean up and sort
    output_df = output_df.drop('Date_dt', axis=1, errors='ignore')
    output_df = output_df.sort_values('Date').reset_index(drop=True)

    # Select final columns in correct order
    final_columns = ['Season_End_Year', 'Wk', 'Date', 'Home', 'HomeGoals', 'AwayGoals', 'Away', 'FTR']
    output_df = output_df[final_columns]

    # Remove rows with missing essential data
    output_df = output_df.dropna(subset=['HomeGoals', 'AwayGoals'])

    # Save to file
    output_df.to_csv(output_file, index=False)
    print(f"Converted {len(output_df)} matches to {output_file}")

    return output_df


# Example usage:
if __name__ == "__main__":
    # Download the data files
    print("Downloading Premier League data...")
    downloaded_files = download_premier_league_data()

    # Convert 2023/24 season
    if 'E0_2324.csv' in downloaded_files:
        print("\nConverting 2023/24 season...")
        convert_to_standard_format('E0_2324.csv', 'premier_league_2023_24.csv', 2024)

    # Convert 2024/25 season
    if 'E0_2425.csv' in downloaded_files:
        print("\nConverting 2024/25 season...")
        convert_to_standard_format('E0_2425.csv', 'premier_league_2024_25.csv', 2025)

    print("\nConversion complete!")
    print("Files created:")
    print("- premier_league_2023_24.csv")
    print("- premier_league_2024_25.csv")
    print("\nThese files are now in the same format as your 1993-2023 data.")

    # Optional: Combine all files if you want one master file
    combine_choice = input("\nWould you like to combine both seasons into one file? (y/n): ")
    if combine_choice.lower() == 'y':
        df_2324 = pd.read_csv('premier_league_2023_24.csv')
        df_2425 = pd.read_csv('premier_league_2024_25.csv')
        combined_df = pd.concat([df_2324, df_2425], ignore_index=True)
        combined_df.to_csv('premier_league_2023_25_combined.csv', index=False)
        print("Combined file created: premier_league_2023_25_combined.csv")