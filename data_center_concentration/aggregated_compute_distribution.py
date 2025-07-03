# Obtain the predicted distribution of compute in 2027 across clusters of various sizes.

# First predict the *amount* of new compute per year knowing that, as of Dec 2024, there is roughly 10M H100 equivalents, and the total amount of compute is increasing by roughly 2.25x per year.

# Then predict how that new compute will be distributed for each year up to 2027.
#1. For years up to 2025, you can just look at how compute is distributed by cluster sizes according to the data in data.csv. And you can assume those distributions are representative.
#2. Then you need to forecast distributions for 2026 and 2027, which you can do via the same methods as in "trends_in_new_compute.py"
#3. Basically, the idea is to use the fact that the mean cluster size of new clusters is increasing by roughly 3.9x per year.
#4. Then you can assume that the distribution of compute across cluster sizes is otherwise the same as it was in 2025, but exponentially shifted by 3.9x.

# After you've done all this, you'll get distributions for *new compute* annually.
# You can now just add these up to predict how the total amount of compute in 2027 will be distributed across cluster sizes (including clusters built in previous years)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
import scipy.stats as stats

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

# Extract year from First Operational Date
df['Year'] = df['First Operational Date'].apply(extract_year)

# Constants based on the problem statement
TOTAL_COMPUTE_2024 = 10_000_000  # 10M H100 equivalents as of Dec 2024
ANNUAL_COMPUTE_GROWTH = 2.25      # Total compute increases 2.25x per year
ANNUAL_CLUSTER_SIZE_GROWTH = 3.9  # Mean cluster size increases 3.9x per year
ADJUSTMENT_FACTOR = 1.5

# Years to analyze
years = [2022, 2023, 2024, 2025, 2026, 2027]

# Calculate the total new compute for each year
total_compute_by_year = {}
for year in years:
    total_compute_by_year[year] = 10000000 * (2.25 **(year - 2024))

new_compute_by_year = {}
for i, year in enumerate(years):
    if i == 0:
        new_compute_by_year[year] = total_compute_by_year[year]
    else:
        new_compute_by_year[year] = total_compute_by_year[year] - total_compute_by_year[year - 1]

# Create simplified bins as requested
def create_simplified_bins():
    bins = [0, 1, 10_000, 100_000, 1_000_000, 10_000_000, float('inf')]
    return bins

# Create the bins
bin_edges = create_simplified_bins()
size_categories = list(zip(bin_edges[:-1], bin_edges[1:]))

# Create tick labels for the bin edges
tick_labels = ['0', '1K', '10K', '100K', '1M', '10M']

# Estimate mean cluster size for each bin -- just assume it's 3x the lower bound
mean_cluster_sizes = []
for i, (min_size, max_size) in enumerate(size_categories):
    mean_cluster_sizes.append(max_size / 3)

# Estimate a sample of new clusters for each year
new_clusters_by_year = {}

for year in years:
    # For years up to 2025, use actual data
    if year <= 2025:
        year_df = df[
            (~df['Year'].isna()) & 
            (df['Year'] == year) & 
            (~df['H100 equivalents'].isna())
        ]

        new_clusters_by_year[year] = np.array(year_df['H100 equivalents'])

    # Otherwise extrapolate the 2025 distribution exponentially
    else:
        clusters_in_2025 = new_clusters_by_year[2025]
        years_after_2025 = year - 2025
        scaling_factor = ANNUAL_CLUSTER_SIZE_GROWTH ** years_after_2025
        new_clusters_by_year[year] = clusters_in_2025 * scaling_factor

# Now apply adjustments to each sample
def adjustment_function_old(sample):
    max_value = sample.max()
    new_items = []
    # now for every value, calculate how many times to duplicate that value
    for item in sample:
        new_items.append(item)
        num_times_to_copy = int(ADJUSTMENT_FACTOR ** np.log(max_value / item))
        for i in range(num_times_to_copy - 1):
            new_items.append(item)
    return np.array(new_items)

# Continuous version - assigns weights to clusters based on their size
def adjustment_function(sample, adjustment_factor):
    max_value = sample.max()
    adjusted_sample = sample.copy()
    weights = np.ones(len(sample))
    
    # Calculate weights for each cluster based on its size relative to the largest
    for i, item in enumerate(sample):
        # Weight is proportional to how much smaller the cluster is compared to the largest
        # The smaller the cluster, the higher the weight
        weights[i] = adjustment_factor ** np.log(max_value / item)
    
    # Return both the original sample and the weights
    return adjusted_sample, weights

adjusted_clusters_by_year = {}
cluster_weights_by_year = {}

for year in years:
    adjusted_clusters_by_year[year], cluster_weights_by_year[year] = adjustment_function(new_clusters_by_year[year], ADJUSTMENT_FACTOR)

# Now calculate the distributions of new compute
new_compute_distributions = {}
for year in years:
    dist = np.zeros(len(size_categories))
    # Use the adjusted clusters and their weights
    for i, cluster_size in enumerate(adjusted_clusters_by_year[year]):
        weight = cluster_weights_by_year[year][i]
        for j, (min_size, max_size) in enumerate(size_categories):
            if min_size <= cluster_size < max_size:
                # Apply the weight to the cluster size contribution
                dist[j] += cluster_size * weight
                break
    # normalize dist
    if dist.sum() > 0:
        dist = dist / dist.sum()
    new_compute_distributions[year] = dist

# Now obtain the total amount of new compute, and the number of new clusters
total_new_compute_by_year = {}
number_of_new_clusters_by_year = {}
for year in years:
    total_new_compute_by_year[year] = new_compute_distributions[year] * new_compute_by_year[year]
    number_of_new_clusters_by_year[year] = total_new_compute_by_year[year] / mean_cluster_sizes

# # Calculate total compute to verify our calculations are correct
total_compute = 0
for year in years:
    total_compute += total_new_compute_by_year[year].sum()

# print("Total compute:", total_compute)
# assert total_compute == total_compute_by_year[2027]

# Now obtain the normalized version of new_compute_distributions_by_year for cleaner plotting
new_compute_distributions_normalized_by_year = {}
for year in years:
    new_compute_distributions_normalized_by_year[year] = total_new_compute_by_year[year] / total_compute

# Now get the total new clusters in 2027 by size
new_clusters_2027 = np.zeros(len(size_categories))
for year in years:
    new_clusters_2027 += number_of_new_clusters_by_year[year]

# ----------- Plot setup 
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 4))

# Use the Viridis colormap which is commonly used in ML papers
from matplotlib.cm import viridis
colors = [viridis(i/8) for i in range(6)]  # Sample 6 colors from the Viridis colormap

# First subplot - Distribution of total compute stacked by year
# Position the bars between the tick marks by offsetting by 0.5
y_centers = np.arange(len(size_categories)) + 0.5

# --------- First Subplot

# Create stacked horizontal bars for the first subplot
left_positions = np.zeros(len(size_categories))
stacked_bars1 = []
year_labels = []

# Create stacked bars for each year's contribution
for i, year in enumerate(years):

    # Create the bar for this year
    bar = ax1.barh(y_centers, new_compute_distributions_normalized_by_year[year], left=left_positions, 
                  color=colors[i % len(colors)], height=1.0, 
                  label=str(year))
    
    # Update the left positions for the next year's bars
    left_positions += new_compute_distributions_normalized_by_year[year]
    
    # Store the bar object for later reference
    stacked_bars1.append(bar)
    year_labels.append(str(year))

# Create tick positions at bin edges
y_ticks = np.arange(len(bin_edges))

# Set y-ticks at the bin edges (excluding infinity)
ax1.set_yticks(y_ticks[:-1])
ax1.set_yticklabels(tick_labels, fontsize=9)

# compute aggregate distribution
aggregate_distribution = np.zeros(len(size_categories))
for year in years:
    aggregate_distribution += new_compute_distributions_normalized_by_year[year]

# Add percentage labels to the total bars
for i, v in enumerate(aggregate_distribution):
    if v > 0.01:  # Only show labels for values > 1%
        # Format as percentage with 1 decimal place
        percentage = f'{v*100:.1f}%'
        # Position the text at the end of the bar
        ax1.text(v, y_centers[i], f' {percentage}', va='center')

# Add labels for first subplot
ax1.set_xlim(0, 1)
ax1.set_xlabel('Proportion of Total Compute', fontsize=12)
ax1.set_ylabel('Compute Cluster Size (H100e)', fontsize=12)
ax1.set_title('Predicted Distribution of Total Compute in 2027', fontsize=12)
ax1.grid(axis='x', linestyle='--', alpha=0.7)
ax1.legend(title='Year Operational', loc='upper right')

# Second subplot -------------- 

# Create stacked horizontal bars for the second subplot
left_positions = np.zeros(len(size_categories))
stacked_bars2 = []

# Create stacked bars for each year's contribution
for i, year in enumerate(years):
    # Get the number of clusters for this year
    year_clusters = number_of_new_clusters_by_year[year]
    
    # Create the bar for this year
    bar = ax2.barh(y_centers, year_clusters, left=left_positions, 
                  color=colors[i % len(colors)], height=1.0, 
                  label=str(year))
    
    # Update the left positions for the next year's bars
    left_positions += year_clusters
    
    # Store the bar object for later reference
    stacked_bars2.append(bar)

# Set y-ticks for second subplot (same as first subplot)
ax2.set_yticks(y_ticks[:-1])
ax2.set_yticklabels(tick_labels, fontsize=9)

# Set x-axis to log scale
ax2.set_xscale('log')

# Add labels for second subplot
ax2.set_xlabel('Number of Compute Clusters', fontsize=12)
ax2.set_ylabel('')  # No need to repeat y-axis label
ax2.set_title('Predicted Number of Compute Clusters in 2027', fontsize=12)
ax2.grid(axis='x', linestyle='--', alpha=0.7)
ax2.legend(title='Year Operational', loc='upper right')

num_clusters = np.zeros(len(size_categories))
for year in years:
    num_clusters += number_of_new_clusters_by_year[year]

# Add value labels to the total bars in the second subplot
print(new_clusters_2027)

# Get the current x-axis limits before adding text
x_min, x_max = ax2.get_xlim()

# Add labels at a fixed position instead of at the actual value
for i, v in enumerate(new_clusters_2027):
    if v > 0:  # Only add labels for non-zero values
        v = round(v)
        if v >= 1000:
            # For thousands, use K suffix with 2 significant figures
            v_thousands = v/1000
            if v_thousands < 10:
                formatted_v = f"{v_thousands:.1f}K"  # e.g., 1.2K, 9.8K
            else:
                formatted_v = f"{round(v_thousands)}K"  # e.g., 12K, 987K
        else:
            # Use regular integer formatting for smaller numbers
            formatted_v = f"{v}"
            
        ax2.text(v, y_centers[i], f" {formatted_v}", va='center', ha='left')

# Adjust layout to ensure labels don't get cut off
fig.tight_layout()

# Add padding inside the subfigures
for ax in [ax2]:
    # Get the current x-axis limits
    x_min, x_max = ax.get_xlim()
    # Add 20% padding to the right side
    ax.set_xlim(x_min, x_max * 3)

# Save the figure with lower DPI to avoid size issues
plt.savefig('predicted_compute_distribution_2027.png', dpi=150, bbox_inches='tight')

# Show the plot
plt.show()

