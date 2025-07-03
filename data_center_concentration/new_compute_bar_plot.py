import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import re

# Read the data
df = pd.read_csv('data.csv')

# Filter out rows with missing H100 equivalents
df = df.dropna(subset=['H100 equivalents'])

# Extract year from planned dates where possible
def extract_year(planned_date):
    if pd.isna(planned_date):
        return None
    # Look for patterns like "Planned 2028" or "Planned Q4 2026"
    year_match = re.search(r'(20\d{2})', str(planned_date))
    if year_match:
        return int(year_match.group(1))
    return None

# Apply the extraction function to the 'First Operational Date Note' column
df['Planned_Year'] = df['First Operational Date Note'].apply(extract_year)

# Extract year from First Operational Date for existing clusters
df['Op_Year'] = pd.to_datetime(df['First Operational Date'], errors='coerce').dt.year

# Create a combined year column that uses either Planned_Year or Op_Year
df['Combined_Year'] = df['Op_Year']
# For planned clusters with a year in the note, use that year
df.loc[(df['Status'] == 'Planned') & (~df['Planned_Year'].isna()), 'Combined_Year'] = df['Planned_Year']

# Define the years for which we want to create histograms
years = list(range(2022, 2027))  # From 2023 to 2030

# Define size categories for the clusters (in H100 equivalents)
size_categories = [
    (0, 1000),       # < 1K H100e
    (1000, 10000),    # 1K-10K H100e
    (10000, 100000),  # 10K-100K H100e
    (100000, 1000000),# 100K-1M H100e
    (1000000, float('inf'))  # > 1M H100e
]

size_labels = [
    '< 1K H100e',
    '1K-10K H100e',
    '10K-100K H100e',
    '100K-1M H100e',
    '> 1M H100e'
]

# Define colors for each size category - modern cool color palette
colors = ['#4363d8', '#3cb44b', '#42d4f4', '#469990', '#000075']

# Create a figure
plt.figure(figsize=(8, 4))

# Initialize arrays to store the data
data_by_year = []
compute_by_year = []
cluster_counts_by_year = []

# Process data for each year
for year in years:
    # Filter for clusters operational in this year
    year_df = df[
        (~df['Combined_Year'].isna()) & (df['Combined_Year'] == year) & 
        (~df['H100 equivalents'].isna())
    ]
    
    # Skip if no data for this year
    if len(year_df) == 0:
        data_by_year.append([0] * len(size_categories))
        compute_by_year.append([0] * len(size_categories))
        cluster_counts_by_year.append(0)
        continue
    
    # Calculate total compute for this year
    total_compute = year_df['H100 equivalents'].sum()
    compute_by_year.append(total_compute)
    
    # Count total clusters for this year
    cluster_counts_by_year.append(len(year_df))
    
    # Initialize data for this year
    year_data = [0] * len(size_categories)
    
    # Count clusters in each size category
    for i, (min_size, max_size) in enumerate(size_categories):
        # Filter clusters in this size category
        category_df = year_df[
            (year_df['H100 equivalents'] >= min_size) & 
            (year_df['H100 equivalents'] < max_size)
        ]
        
        # Calculate total compute in this category
        category_compute = category_df['H100 equivalents'].sum()
        year_data[i] = category_compute
    
    data_by_year.append(year_data)

# Convert to numpy array for easier manipulation
data_array = np.array(data_by_year)

# Create the stacked bar chart
bottom = np.zeros(len(years))

bar_width = 0.7
bar_positions = np.arange(len(years))

# Plot each size category as a layer in the stacked bars
for i in range(len(size_categories)):
    plt.bar(bar_positions, data_array[:, i], bottom=bottom, width=bar_width, 
            label=size_labels[i], color=colors[i])
    bottom += data_array[:, i]

# Add year labels to the x-axis
plt.xticks(bar_positions, [str(year) if year != 2026 else "2026 (planned)" for year in years ])

# y ticks (K)

# Add labels and title
plt.xlabel('Year operational', fontsize=14)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.ylabel('Compute (H100 equivalents)', fontsize=14)
# plt.yscale("log")
plt.title('New Compute Clusters by Size', fontsize=16)

# Add a legend with reversed order
plt.legend(title='Cluster Size Categories', fontsize=12, title_fontsize=12, reverse=True)

# Add cluster count and total compute annotations above each bar
# for i, (count, total) in enumerate(zip(cluster_counts_by_year, compute_by_year)):
#     if count > 0:  # Only add annotations for years with data
        # plt.text(i, bottom[i] + 100000, f'{count} clusters', 
        #         ha='center', va='bottom', fontsize=10)
        # plt.text(i, bottom[i] + 500000, f'Total: {total:,.0f} H100e', 
        #         ha='center', va='bottom', fontsize=10)

# Format y-axis with 'K' to signify thousands
def thousands_formatter(x, pos):
    if x >= 1e6:
        return f'{x/1e6:.0f}M'
    elif x >= 1e3:
        return f'{x/1e3:.0f}K'
    else:
        return f'{x:.0f}'

plt.gca().get_yaxis().set_major_formatter(plt.matplotlib.ticker.FuncFormatter(thousands_formatter))

# Add grid lines for better readability
plt.grid(axis='y', linestyle='--', alpha=0.7)

# Adjust layout
plt.tight_layout()

# Save the figure
plt.savefig('new_compute_distribution_by_year.png', dpi=300, bbox_inches='tight')

# Show the plot
plt.show()

# Print some statistics
print("\nStatistics by Year:")
for i, year in enumerate(years):
    if cluster_counts_by_year[i] > 0:
        print(f"\nYear {year}:")
        # print(f"  Total clusters: {cluster_counts_by_year[i]}")
        # print(f"  Total compute: {compute_by_year[i]:,.0f} H100 equivalents")
        
        # Print distribution by size category
        for j, (min_size, max_size) in enumerate(size_categories):
            category_compute = data_array[i, j]
            if category_compute > 0:
                percentage = (category_compute / compute_by_year[i]) * 100
                print(f"  {size_labels[j]}: {percentage:.1f}% of compute")
    else:
        print(f"\nYear {year}: No clusters found")
