import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter

# Helper function to extract year from date strings
def extract_year(date_str):
    if pd.isna(date_str):
        return None
    
    # Try to extract a year (4 digits) from the string
    import re
    year_match = re.search(r'20\d{2}', str(date_str))
    if year_match:
        return int(year_match.group(0))
    return None

# Load the data
df = pd.read_csv('data.csv')

# Extract years from both columns
df['Op_Year'] = df['First Operational Date'].apply(extract_year)
df['Planned_Year'] = df['First Operational Date Note'].apply(extract_year)

# Create a Year column that combines operational and planned years
# This approach follows the predicted_compute_distribution_in_2027_new.py methodology

# First assign operational years for 2022-2024
df.loc[df['Op_Year'].isin([2022, 2023, 2024]), 'Year'] = df['Op_Year']

# For 2025, include both operational and planned clusters for that year
df.loc[(df['Op_Year'] == 2025) | (df['Planned_Year'] == 2025), 'Year'] = 2025

# For 2026-2027, use planned years
df.loc[df['Planned_Year'].isin([2026, 2027]), 'Year'] = df['Planned_Year']

# Add a Status column to distinguish between operational and planned clusters
df['Status'] = 'Unknown'
df.loc[~df['Op_Year'].isna(), 'Status'] = 'Operational'
df.loc[df['Op_Year'].isna() & ~df['Planned_Year'].isna(), 'Status'] = 'Planned'

# For 2025, we need to be more specific about operational vs planned
df.loc[(df['Year'] == 2025) & ~df['Op_Year'].isna(), 'Status'] = 'Operational'
df.loc[(df['Year'] == 2025) & df['Op_Year'].isna() & ~df['Planned_Year'].isna(), 'Status'] = 'Planned'

# Define the H100 equivalent range we're interested in
min_h100 = 10000  # 10K
max_h100 = 100000  # 100K

# Filter clusters by H100 equivalent size
large_clusters = df[(df['H100 equivalents'] >= min_h100) & 
                    (df['H100 equivalents'] < max_h100) &
                    df['Year'].isin(range(2022, 2028))]

# Count clusters by year and status
years = range(2022, 2028)  # Years from 2022 to 2027
clusters_by_year = {}
operational_by_year = {}
planned_by_year = {}

print(f"\nClusters with {min_h100:,} to {max_h100:,} H100 equivalents by year:")
print("-" * 60)

total_count = 0
total_operational = 0
total_planned = 0

for year in years:
    # Count clusters for this year
    year_clusters = large_clusters[large_clusters['Year'] == year]
    year_count = year_clusters.shape[0]
    clusters_by_year[year] = year_count
    total_count += year_count
    
    # Count operational vs planned
    operational_clusters = year_clusters[year_clusters['Status'] == 'Operational']
    planned_clusters = year_clusters[year_clusters['Status'] == 'Planned']
    
    operational_count = operational_clusters.shape[0]
    planned_count = planned_clusters.shape[0]
    
    operational_by_year[year] = operational_count
    planned_by_year[year] = planned_count
    
    total_operational += operational_count
    total_planned += planned_count
    
    # Format the year label (add "planned" for 2026 and 2027)
    year_label = f"{year} (planned)" if year >= 2026 else str(year)
    
    # Print the count and breakdown for this year
    print(f"{year_label}: {year_count} clusters total ({operational_count} operational, {planned_count} planned)")
    
    if year_count > 0:
        # Print operational clusters
        if operational_count > 0:
            print("  Operational Clusters:")
            for _, cluster in operational_clusters.iterrows():
                h100_equiv = cluster['H100 equivalents']
                print(f"  - {cluster['Name']} ({h100_equiv:,.0f} H100 equivalents)")
        
        # Print planned clusters
        if planned_count > 0:
            print("  Planned Clusters:")
            for _, cluster in planned_clusters.iterrows():
                h100_equiv = cluster['H100 equivalents']
                print(f"  - {cluster['Name']} ({h100_equiv:,.0f} H100 equivalents)")
    print()

print("-" * 60)
print(f"Total clusters between {min_h100:,} and {max_h100:,} H100 equivalents: {total_count}")
print(f"  - Operational: {total_operational}")
print(f"  - Planned: {total_planned}")

# Create a stacked bar chart visualization
plt.figure(figsize=(10, 6))
years_list = list(years)

# Get counts for operational and planned clusters
operational_counts = [operational_by_year.get(year, 0) for year in years_list]
planned_counts = [planned_by_year.get(year, 0) for year in years_list]

# Create year labels with (planned) for 2026 and 2027
year_labels = [f"{year} (planned)" if year >= 2026 else str(year) for year in years_list]

# Create a stacked bar chart with cool colors (viridis palette)
bars_operational = plt.bar(year_labels, operational_counts, label='Operational', 
                          color=plt.cm.viridis(0.2))
bars_planned = plt.bar(year_labels, planned_counts, bottom=operational_counts, 
                      label='Planned', color=plt.cm.viridis(0.7))

# Add value labels on top of each stacked bar
for i, year in enumerate(year_labels):
    operational = operational_counts[i]
    planned = planned_counts[i]
    total = operational + planned
    
    # Only add label if there are clusters
    if total > 0:
        # Add total on top
        plt.text(i, total + 0.1, f'{int(total)}', ha='center', va='bottom')
        
        # Add breakdown inside bars if there's enough space
        if operational > 0 and operational > 1:
            plt.text(i, operational/2, f'{int(operational)}', ha='center', va='center', color='white')
        if planned > 0 and planned > 1:
            plt.text(i, operational + planned/2, f'{int(planned)}', ha='center', va='center', color='white')

plt.title(f"Clusters with {min_h100:,} to {max_h100:,} H100 Equivalents by Year")
plt.ylabel("Number of Clusters")
plt.xlabel("Year")
plt.legend(loc='upper left')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()

# Save the figure
plt.savefig('large_clusters_by_year.png', dpi=300)
print("Chart saved as 'large_clusters_by_year.png'")

# Show the plot
plt.show()
