from predicted_compute_distribution_in_2027 import raw_data
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re

# Get the raw data for forecasted clusters in 2026 and 2027
forecasted_df_2026 = raw_data[2026]
forecasted_df_2027 = raw_data[2027]

# Load the data
df = pd.read_csv('data.csv')

# Extract year from planned operational date
def extract_year(planned_date):
    if pd.isna(planned_date):
        return None
    year_match = re.search(r'(20\d{2})', str(planned_date))
    if year_match:
        return int(year_match.group(1))
    return None

# Extract year from First Operational Date Note for planned projects
df['Planned_Year'] = df['First Operational Date Note'].apply(extract_year)

# Filter for planned projects in 2026 and 2027
planned_2026_df = df[
    (df['Status'] == 'Planned') & 
    (df['Planned_Year'] == 2026) & 
    (~df['H100 equivalents'].isna())
]

planned_2027_df = df[
    (df['Status'] == 'Planned') & 
    (df['Planned_Year'] == 2027) & 
    (~df['H100 equivalents'].isna())
]

# Create bins for cluster sizes
bins = [1000, 10000, 100000, 1000000, 10000000, float('inf')]
bin_labels = ['10K', '100K', '1M', '10M']

# Function to calculate distribution of compute across bins
def calculate_distribution(dataframe):
    # Initialize distribution array
    dist = np.zeros(len(bins) - 1)
    
    # Calculate distribution
    for _, row in dataframe.iterrows():
        size = row['H100 equivalents']
        for i in range(len(bins) - 1):
            if bins[i] <= size < bins[i + 1]:
                dist[i] += size
                break
    
    # Normalize to get proportions
    if dist.sum() > 0:
        dist = dist / dist.sum()
    
    return dist

# Calculate distributions for 2026 and 2027
forecasted_distribution_2026 = calculate_distribution(forecasted_df_2026)
planned_distribution_2026 = calculate_distribution(planned_2026_df)
forecasted_distribution_2027 = calculate_distribution(forecasted_df_2027)
planned_distribution_2027 = calculate_distribution(planned_2027_df)

# Create figure with 2x2 grid of subplots
fig, axs = plt.subplots(2, 2, figsize=(8, 8))
# fig.suptitle('Comparison of Compute Cluster Size Distributions', fontsize=14)

# Y positions for the bars (centered between ticks)
y_positions = np.arange(len(bins) - 1) + 0.5

# Plot 2026 distributions (top row)
axs[0, 0].barh(y_positions, forecasted_distribution_2026, height=0.8, color='#4682B4')
axs[0, 0].set_title('2026 Forecasted Distribution', fontsize=12)
# axs[0, 0].set_xlabel('Proportion of Compute')

axs[0, 1].barh(y_positions, planned_distribution_2026, height=0.8, color='#5F9EA0')
axs[0, 1].set_title('2026 Planned Projects', fontsize=12)
# axs[0, 1].set_xlabel('Proportion of Compute')

# Plot 2027 distributions (bottom row)
axs[1, 0].barh(y_positions, forecasted_distribution_2027, height=0.8, color='#4682B4')
axs[1, 0].set_title('2027 Forecasted Distribution', fontsize=12)
# axs[1, 0].set_xlabel('Proportion of Compute')

axs[1, 1].barh(y_positions, planned_distribution_2027, height=0.8, color='#5F9EA0')
axs[1, 1].set_title('2027 Planned Projects', fontsize=12)
# axs[1, 1].set_xlabel('Proportion of Compute')

# Set y-ticks and labels for all subplots
for row in axs:
    for ax in row:
        ax.set_yticks(range(len(bin_labels)))
        ax.set_yticklabels(bin_labels)
        ax.set_ylabel('Cluster Size (H100 equivalents)', fontsize=12)
        ax.grid(axis='x', linestyle='--', alpha=0.7)
        
        # # Add percentage labels to the bars
        # for i, v in enumerate(ax.containers[0]):
        #     width = v.get_width()
        #     if width > 0.01:  # Only show label if proportion is greater than 1%
        #         ax.text(width + 0.01, v.get_y() + v.get_height()/2, 
        #                f'{width:.1%}', va='center', fontsize=8)

# Adjust layout
plt.tight_layout()
plt.subplots_adjust(top=0.85)

# Show the plot
plt.show()