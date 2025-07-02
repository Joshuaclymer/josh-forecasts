import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re

# Function to extract year from date strings (copied from josh_scrap.py)
def extract_year(planned_date):
    if pd.isna(planned_date):
        return np.nan
    match = re.search(r'(\d{4})', str(planned_date))
    if match:
        return int(match.group(1))
    return np.nan

# Load the data
df = pd.read_csv('data.csv')
print(f"Total rows in data.csv: {len(df)}")

# Print status distribution
status_counts = df['Status'].value_counts()
print("\nStatus distribution:")
print(status_counts)

# Extract years from both columns
df['Op_Year'] = df['First Operational Date'].apply(extract_year)
df['Planned_Year'] = df['First Operational Date Note'].apply(extract_year)

# Print year distribution before filtering
print("\nYear distribution before filtering:")
op_year_counts = df['Op_Year'].value_counts().sort_index()
planned_year_counts = df['Planned_Year'].value_counts().sort_index()
print("Operational years:")
print(op_year_counts)
print("\nPlanned years:")
print(planned_year_counts)

# Map Status to operational/planned for better categorization
def determine_status(row):
    if row['Status'] == 'Existing':
        return 'Operational'
    elif row['Status'] == 'Planned':
        return 'Planned'
    elif row['Status'] == 'Decommissioned':
        return 'Decommissioned'
    return 'Unknown'

df['Cluster_Status'] = df.apply(determine_status, axis=1)

# Filter for years up to 2028 only
# For operational clusters, use Op_Year
# For planned clusters, use Planned_Year
operational_df = df[(df['Cluster_Status'] == 'Operational') & 
                    (~df['Op_Year'].isna()) & 
                    (df['Op_Year'] <= 2028)]

planned_df = df[(df['Cluster_Status'] == 'Planned') & 
               (~df['Planned_Year'].isna()) & 
               (df['Planned_Year'] <= 2028)]

# Combine for total filtered dataset
df_filtered = pd.concat([operational_df, planned_df])

print(f"\nAfter year filtering (≤ 2028): {len(df_filtered)} clusters")

# Define size ranges
size_ranges = [
    (1000, 10000),       # 1K - 10K
    (10000, 100000),     # 10K - 100K
    (100000, 1000000),   # 100K - 1M
    (1000000, 10000000)  # 1M - 10M
]

size_labels = ['1K - 10K', '10K - 100K', '100K - 1M', '1M - 10M']

# Create a function to categorize clusters by size
def categorize_size(h100_equiv):
    if pd.isna(h100_equiv):
        return None
    
    for i, (lower, upper) in enumerate(size_ranges):
        if lower <= h100_equiv < upper:
            return size_labels[i]
    return None

# Print H100 equivalents distribution
print("\nH100 equivalents distribution:")
h100_bins = [0, 1000, 10000, 100000, 1000000, 10000000, float('inf')]
h100_labels = ['<1K', '1K-10K', '10K-100K', '100K-1M', '1M-10M', '>10M']
h100_counts = pd.cut(df['H100 equivalents'].dropna(), bins=h100_bins, labels=h100_labels).value_counts().sort_index()
print(h100_counts)

# Add size category to dataframe
df_filtered['Size_Category'] = df_filtered['H100 equivalents'].apply(categorize_size)

# Filter out rows that don't fall into any of our size categories (H100 < 1000 or H100 >= 10M or NaN)
size_filtered_df = df_filtered[~df_filtered['Size_Category'].isna()]
print(f"\nAfter size filtering (1K ≤ H100 < 10M): {len(size_filtered_df)} clusters")

# Separate operational and planned clusters
operational = size_filtered_df[size_filtered_df['Cluster_Status'] == 'Operational'].copy()
planned = size_filtered_df[size_filtered_df['Cluster_Status'] == 'Planned'].copy()

# Count clusters by size category
operational_counts = operational.groupby('Size_Category').size()
planned_counts = planned.groupby('Size_Category').size()

# Print results
print("\nNumber of Operational Clusters by Size Range:")
for label in size_labels:
    count = operational_counts.get(label, 0)
    print(f"{label}: {count}")

print("\nNumber of Planned Clusters by Size Range:")
for label in size_labels:
    count = planned_counts.get(label, 0)
    print(f"{label}: {count}")

# Create a stacked bar chart
fig, ax = plt.subplots(figsize=(8, 4))  # Using preferred compact figure size

# Ensure all size categories are present in both counts
all_counts = pd.DataFrame({
    'Operational': [operational_counts.get(label, 0) for label in size_labels],
    'Planned': [planned_counts.get(label, 0) for label in size_labels]
}, index=size_labels)

# Create the stacked bar chart with cool color palette
all_counts.plot(kind='bar', stacked=True, ax=ax, width=0.6, 
                color=['#4C72B0', '#55A868'])  # Cool color palette

# Customize the plot
ax.set_title('New Compute Clusters by Size', fontsize=12)  # Simple, clean title
ax.set_xlabel('Cluster Size (H100 equivalents)', fontsize=10)
ax.set_ylabel('Number of Clusters', fontsize=10)
ax.legend(title='Status', loc='upper right')  # Legend on right side

# Format y-axis with 'K' for thousands
from matplotlib.ticker import FuncFormatter
def format_func(x, pos):
    if x >= 1000:
        return f'{x/1000:.0f}K'
    else:
        return f'{x:.0f}'
        
ax.yaxis.set_major_formatter(FuncFormatter(format_func))  # Y-axis formatted with "K"

plt.tight_layout()
plt.savefig('cluster_size_distribution.png', dpi=300)
print("\nChart saved as 'cluster_size_distribution.png'")

# Print total counts
print(f"\nTotal operational clusters: {operational_counts.sum()}")
print(f"Total planned clusters: {planned_counts.sum()}")
print(f"Total clusters: {operational_counts.sum() + planned_counts.sum()}")

# Print missing data analysis
print("\nMissing data analysis:")
print(f"Rows missing H100 equivalents: {df['H100 equivalents'].isna().sum()}")
print(f"Rows missing both Op_Year and Planned_Year: {((df['Op_Year'].isna()) & (df['Planned_Year'].isna())).sum()}")
print(f"Rows with H100 equivalents < 1000: {len(df[df['H100 equivalents'] < 1000].dropna())}")
print(f"Rows with H100 equivalents ≥ 10M: {len(df[df['H100 equivalents'] >= 10000000].dropna())}")
