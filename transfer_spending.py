import pandas as pd

# Load the dataset
url = "https://raw.githubusercontent.com/ewenme/transfers/master/data/premier-league.csv"
df = pd.read_csv(url)

# Filter to keep only rows with valid fee and season
df = df[df['fee_cleaned'].notna() & df['season'].notna()]

# Convert 'fee_cleaned' to numeric
df['fee_cleaned'] = pd.to_numeric(df['fee_cleaned'], errors='coerce')

# Positive spend for 'in' transfers, negative for 'out'
df['spend'] = df.apply(lambda row: row['fee_cleaned'] if row['transfer_movement'] == 'in' else -row['fee_cleaned'], axis=1)

# Group by club and season
net_spend = df.groupby(['season', 'club_name'])['spend'].sum().reset_index()

# Rename and sort
net_spend.columns = ['Season', 'Club', 'Net Spend (€m)']
net_spend = net_spend.sort_values(by=['Season', 'Net Spend (€m)'], ascending=[True, False])

# Save to CSV
net_spend.to_csv("premier_league_net_spend_by_season.csv", index=False)

print("✅ Net spend saved to 'premier_league_net_spend_by_season.csv'")
