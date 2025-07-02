# import pandas as pd
# import re
# import numpy as np
# import matplotlib.pyplot as plt

# def extract_year(planned_date):
#     if pd.isna(planned_date):
#         return None
#     year_match = re.search(r'(20\d{2})', str(planned_date))
#     if year_match:
#         return int(year_match.group(1))
#     return None

# # Load the data
# df = pd.read_csv('data.csv')

# # Extract years from both columns
# df['Op_Year'] = df['First Operational Date'].apply(extract_year)
# df['Planned_Year'] = df['First Operational Date Note'].apply(extract_year)

# # Initialize Combined_Year column
# df['Combined_Year'] = None

# # For existing clusters, use the First Operational Date
# df.loc[~df['Op_Year'].isna(), 'Combined_Year'] = df['Op_Year']

# # For planned clusters, use the Planned Year from First Operational Date Note if Op_Year is not available
# df.loc[df['Op_Year'].isna() & ~df['Planned_Year'].isna(), 'Combined_Year'] = df['Planned_Year']

# # Count the number of clusters by year
# years = range(2022, 2027)  # Years from 2022 to 2026
# clusters_by_year = {}

# print("Number of clusters by year:")
# print("--------------------------")

# for year in years:
#     # Count clusters for this year
#     year_count = df[df['Combined_Year'] == year].shape[0] 
#     clusters_by_year[year] = year_count
    
#     # Format the year label (add "planned" for 2026 as per user preference)
#     year_label = f"{year} (planned)" if year == 2026 else str(year)
    
#     # Print the count for this year
#     print(f"{year_label}: {year_count} clusters")

# # Calculate total
# total_clusters = sum(clusters_by_year.values())
# print("--------------------------")
# print(f"Total: {total_clusters} clusters")


# # Are these clusters in the same locations?

import pandas as pd
import re

def extract_year(planned_date):
    if pd.isna(planned_date):
        return None
    year_match = re.search(r'(20\d{2})', str(planned_date))
    if year_match:
        return int(year_match.group(1))
    return None

# Load the data
df = pd.read_csv('data.csv')

# Extract years from both columns
df['Op_Year'] = df['First Operational Date'].apply(extract_year)
df['Planned_Year'] = df['First Operational Date Note'].apply(extract_year)

# Filter for clusters with > 100K H100 equivalents
large_clusters = df[df['H100 equivalents'] > 100000].copy()

# Count operational vs planned
operational = large_clusters[~large_clusters['Op_Year'].isna()]
planned = large_clusters[large_clusters['Op_Year'].isna() & ~large_clusters['Planned_Year'].isna()]
unknown = large_clusters[(large_clusters['Op_Year'].isna()) & (large_clusters['Planned_Year'].isna())]

print(f'Large clusters (>100K H100 equivalents):')
print(f'Operational: {len(operational)}')
print(f'Planned: {len(planned)}')
print(f'Unknown status: {len(unknown)}')
print(f'Total: {len(large_clusters)}')

# Breakdown by year including 2027
print('\nBreakdown by year:')
for year in range(2022, 2028):  # Extended to include 2027
    year_operational = operational[operational['Op_Year'] == year]
    year_planned = planned[planned['Planned_Year'] == year]
    
    year_label = f'{year} (planned)' if year >= 2026 else str(year)
    
    if len(year_operational) > 0:
        print(f'{year_label} - Operational: {len(year_operational)}')
    if len(year_planned) > 0:
        print(f'{year_label} - Planned: {len(year_planned)}')
    elif year == 2027:
        print(f'2027 (planned) - Planned: 0')

print('\n2027 planned clusters (if any):')
planned_2027 = planned[planned['Planned_Year'] == 2027]
if len(planned_2027) > 0:
    print(planned_2027[['Name', 'First Operational Date Note', 'H100 equivalents']].sort_values(by='H100 equivalents', ascending=False))
else:
    print('No clusters >100K H100 equivalents planned for 2027 in the dataset.')

