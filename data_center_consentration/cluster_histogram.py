import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
import math
import datetime
import scipy.stats as stats

# Read the data
df = pd.read_csv('data.csv')

# Filter out rows with missing H100 equivalents
df = df.dropna(subset=['H100 equivalents'])

# Filter for clusters operational after 2024
# First convert the 'First Operational Date' to datetime where possible
def parse_date(date_str):
    if pd.isna(date_str):
        return pd.NaT
    try:
        # Try to parse YYYY-MM-DD format
        return pd.to_datetime(date_str)
    except:
        # If it fails, check if it's just a year
        try:
            if str(date_str).isdigit() and len(str(date_str)) == 4:
                return pd.to_datetime(f"{int(date_str)}-01-01")
            return pd.NaT
        except:
            return pd.NaT

# Apply the parsing function
df['Operational_Date'] = df['First Operational Date'].apply(parse_date)

# Extract year from planned dates where possible
def extract_year(planned_date):
    if pd.isna(planned_date):
        return None
    # Look for patterns like "Planned 2028" or "Planned Q4 2026"
    import re
    year_match = re.search(r'(20\d{2})', str(planned_date))
    if year_match:
        return int(year_match.group(1))
    return None

# Apply the extraction function to the 'First Operational Date Note' column
df['Planned_Year'] = df['First Operational Date Note'].apply(extract_year)

# Set the year cutoff for the first plot (can be changed as needed)
first_plot_year_cutoff = 2030  # Change this value to adjust the cutoff year

# Filter for planned clusters up to the cutoff year
planned_clusters = df[~df['H100 equivalents'].isna()].copy()

# Parse the 'First Operational Date' column for existing clusters
planned_clusters['parsed_date'] = pd.to_datetime(planned_clusters['First Operational Date'], errors='coerce')
planned_clusters['year'] = planned_clusters['parsed_date'].dt.year

# Extract planned years for planned clusters
planned_clusters.loc[planned_clusters['Status'] == 'Planned', 'year'] = \
    planned_clusters.loc[planned_clusters['Status'] == 'Planned', 'First Operational Date'].apply(extract_year)

# Filter to include only planned clusters up to the cutoff year
planned_clusters = planned_clusters[
    (~planned_clusters['year'].isna()) & 
    (planned_clusters['year'] <= first_plot_year_cutoff)
].copy()


print(planned_clusters["H100 equivalents"].sum())

# Create more granular bins with natural breakpoints
bins = [0, 1000, 5000, 10000, 25000, 50000, 100000, 250000, 500000, 1000000, 5000000, 10000000]
bin_labels = ['0-1K', '1K-5K', '5K-10K', '10K-25K', '25K-50K', '50K-100K', '100K-250K', '250K-500K', '500K-1M', '1M-5M', '5M-10M']

# Count clusters in each bin
hist, bin_edges = np.histogram(planned_clusters['H100 equivalents'], bins=bins)

# Create a figure with subplots for five charts
fig, (ax1, ax2, ax3, ax4, ax5) = plt.subplots(1, 5, figsize=(36, 8))

# First subplot - Distribution of clusters by size
ax1.bar(range(len(hist)), hist, align='center', alpha=0.7, color='skyblue', edgecolor='navy')
ax1.set_xticks(range(len(hist)))
ax1.set_xticklabels(bin_labels, rotation=45)
ax1.set_xlabel('Cluster Size (H100 Equivalents)', fontsize=14)
ax1.set_ylabel('Number of Clusters', fontsize=14)
ax1.set_title('Distribution of Clusters by Size', fontsize=16)
ax1.grid(axis='y', linestyle='--', alpha=0.7)

# Add count labels on top of each bar
for i, count in enumerate(hist):
    if count > 0:  # Only add labels for non-zero bars
        ax1.text(i, count + 0.1, str(count), ha='center', fontsize=12, fontweight='bold')

# Second subplot - Proportion of compute accounted for vs number of clusters
# Sort clusters by size (H100 equivalents) in descending order
sorted_clusters = planned_clusters.sort_values('H100 equivalents', ascending=False).reset_index(drop=True)

# Calculate cumulative sum of compute power
total_compute = sorted_clusters['H100 equivalents'].sum()
sorted_clusters['cumulative_proportion'] = sorted_clusters['H100 equivalents'].cumsum() / total_compute * 100

# Create x-axis points (number of clusters)
num_clusters = list(range(1, len(sorted_clusters) + 1))

# Plot the cumulative proportion curve
ax2.plot(num_clusters, sorted_clusters['cumulative_proportion'], 'r-', linewidth=3)
ax2.set_xlabel('Number of Clusters (Largest to Smallest)', fontsize=14)
ax2.set_ylabel('Cumulative % of Total Compute', fontsize=14)
ax2.set_title('Proportion of Compute by Number of Clusters', fontsize=16)
ax2.grid(True, linestyle='--', alpha=0.7)

# Add horizontal lines at key percentages
for pct in [50, 80, 90, 95]:
    ax2.axhline(y=pct, color='gray', linestyle='--', alpha=0.7)
    # Find the number of clusters needed to reach this percentage
    clusters_needed = sorted_clusters[sorted_clusters['cumulative_proportion'] >= pct].index.min() + 1
    ax2.text(len(sorted_clusters) * 0.02, pct + 1, f'{pct}% of compute: {clusters_needed} clusters', 
             fontsize=10, verticalalignment='bottom')

# Third subplot - Cumulative proportion of compute from clusters >= size
# Create a sorted list of cluster sizes (use a more granular range for smoother curve)
min_size = planned_clusters['H100 equivalents'].min()
max_size = planned_clusters['H100 equivalents'].max()

# Create logarithmically spaced points for evaluation
size_thresholds = np.logspace(np.log10(min_size*0.9), np.log10(max_size*1.1), 100)

# Calculate cumulative proportion for each size threshold
cumulative_proportions = []

for size in size_thresholds:
    # Calculate total compute from clusters >= this size
    compute_above_threshold = planned_clusters[planned_clusters['H100 equivalents'] >= size]['H100 equivalents'].sum()
    proportion = (compute_above_threshold / total_compute) * 100
    cumulative_proportions.append(proportion)

# Plot the curve
ax3.plot(size_thresholds, cumulative_proportions, 'g-', linewidth=3)
ax3.set_xscale('log')  # Log scale for cluster size
ax3.set_xlabel('Cluster Size Threshold (H100 Equivalents)', fontsize=14)
ax3.set_ylabel('% of Compute from Clusters ≥ Size', fontsize=14)
ax3.set_title('Cumulative Compute by Cluster Size Threshold', fontsize=16)
ax3.grid(True, linestyle='--', alpha=0.7)

# Add horizontal lines at key percentages
for pct in [50, 80, 90, 95]:
    ax3.axhline(y=pct, color='gray', linestyle='--', alpha=0.7)
    
    # Find the size threshold for this percentage (interpolate if needed)
    try:
        # Find the indices where the percentage crosses our target
        idx_above = next((i for i, p in enumerate(cumulative_proportions) if p >= pct), None)
        idx_below = next((i for i, p in enumerate(cumulative_proportions) if p < pct), None)
        
        if idx_above is not None and idx_below is not None:
            # Interpolate to get a more accurate threshold
            p_above = cumulative_proportions[idx_above]
            p_below = cumulative_proportions[idx_below]
            s_above = size_thresholds[idx_above]
            s_below = size_thresholds[idx_below]
            
            # Linear interpolation in log space
            weight = (pct - p_above) / (p_below - p_above) if p_below != p_above else 0
            size_at_pct = s_above * (s_below/s_above) ** weight
            
            ax3.text(min_size, pct + 1, f'{pct}% from clusters ≥ {size_at_pct:.0f}', 
                    fontsize=10, verticalalignment='bottom')
    except Exception:
        pass
        
# Add vertical lines at key size thresholds
for size in [1000, 10000, 100000, 1000000]:
    if min_size <= size <= max_size:
        ax3.axvline(x=size, color='gray', linestyle=':', alpha=0.5)
        # Find the percentage at this size threshold
        idx = np.abs(size_thresholds - size).argmin()
        pct_at_size = cumulative_proportions[idx]
        if pct_at_size > 5:  # Only label if significant
            ax3.text(size, 5, f'{size:,}', ha='center', fontsize=9, rotation=90)

# Configuration for fourth subplot
# Set the cutoff year for the fourth plot (can be changed as needed)
fourth_plot_year_cutoff = 2025  # Set to a high value like 2030 for all clusters, or specific year like 2025, 2026, etc.

# Fourth subplot - Check for Pareto distribution (log-log plot) with configurable year cutoff
# Extract year information for all clusters
all_clusters = df[~df['H100 equivalents'].isna()].copy()

# Parse dates for existing clusters
all_clusters['parsed_date'] = pd.to_datetime(all_clusters['First Operational Date'], errors='coerce')
all_clusters['year'] = all_clusters['parsed_date'].dt.year

# Extract planned years for planned clusters
def extract_year(text):
    if pd.isna(text):
        return None
    match = re.search(r'(20\d\d)', str(text))
    if match:
        return int(match.group(1))
    return None

all_clusters.loc[all_clusters['Status'] == 'Planned', 'year'] = \
    all_clusters.loc[all_clusters['Status'] == 'Planned', 'First Operational Date'].apply(extract_year)

# Filter based on the cutoff year if specified
if fourth_plot_year_cutoff < 2030:  # Only filter if a reasonable cutoff is set
    all_clusters = all_clusters[~all_clusters['year'].isna() & (all_clusters['year'] <= fourth_plot_year_cutoff)]

# Get all cluster sizes
cluster_sizes = all_clusters['H100 equivalents'].values

# Filter out clusters smaller than 10^3 H100 equivalents
filtered_sizes = cluster_sizes[cluster_sizes >= 1000]
filtered_sizes_sorted = np.sort(filtered_sizes)[::-1]  # Sort in descending order

# Calculate empirical CCDF (1 - CDF) for filtered data
ccdf = np.arange(1, len(filtered_sizes_sorted) + 1) / len(filtered_sizes_sorted)

# Plot on log-log scale
ax4.loglog(filtered_sizes_sorted, ccdf, 'bo', markersize=4, alpha=0.7)

# Fit a power law (Pareto) to the data
# For a Pareto distribution, we expect log(CCDF) = -α * log(x) + constant
# where α is the shape parameter

# Use linear regression on log-transformed data to estimate α
log_sizes = np.log(filtered_sizes_sorted)
log_ccdf = np.log(ccdf)

# Remove any potential infinities or NaNs
valid_indices = np.isfinite(log_sizes) & np.isfinite(log_ccdf)
log_sizes_clean = log_sizes[valid_indices]
log_ccdf_clean = log_ccdf[valid_indices]

# Linear regression
slope, intercept, r_value, p_value, std_err = stats.linregress(log_sizes_clean, log_ccdf_clean)

# The negative of the slope is the shape parameter α
alpha = -slope

# Plot the fitted line
x_fit = np.logspace(np.log10(1000), np.log10(max(filtered_sizes_sorted)), 100)
y_fit = np.exp(intercept) * x_fit**slope
ax4.loglog(x_fit, y_fit, 'r-', linewidth=2, label=f'Pareto fit (α={-slope:.2f})')

# Add labels and title
ax4.set_xlabel('Cluster Size (H100 Equivalents)', fontsize=14)
ax4.set_ylabel('Complementary CDF (1-CDF)', fontsize=14)
if fourth_plot_year_cutoff < 2030:
    ax4.set_title(f'Pareto Check - Clusters ≤ {fourth_plot_year_cutoff}\n(Existing + Planned) ≥ 10³ H100 Equivalents', fontsize=16)
else:
    ax4.set_title('Pareto Check - ALL Clusters\n(Existing + Planned) ≥ 10³ H100 Equivalents', fontsize=16)
ax4.grid(True, which="both", linestyle='--', alpha=0.7)
ax4.legend()

# Add R-squared value to show goodness of fit
ax4.text(0.05, 0.05, f'R² = {r_value**2:.4f}', transform=ax4.transAxes, 
         fontsize=12, bbox=dict(facecolor='white', alpha=0.8))

# Fifth subplot - Pareto distribution check for clusters from 2023 or earlier
# Filter for clusters from 2023 or earlier
clusters_2023_or_earlier = df[(df['Status'] != 'Planned') & (~df['H100 equivalents'].isna())].copy()
clusters_2023_or_earlier['parsed_date'] = pd.to_datetime(clusters_2023_or_earlier['First Operational Date'], errors='coerce')
clusters_2023_or_earlier['year'] = clusters_2023_or_earlier['parsed_date'].dt.year
clusters_2023_or_earlier = clusters_2023_or_earlier[~clusters_2023_or_earlier['year'].isna() & (clusters_2023_or_earlier['year'] <= 2023)]

# Filter out clusters smaller than 10^3 H100 equivalents
filtered_2023_sizes = clusters_2023_or_earlier['H100 equivalents'].values[clusters_2023_or_earlier['H100 equivalents'].values >= 1000]
filtered_2023_sizes_sorted = np.sort(filtered_2023_sizes)[::-1]  # Sort in descending order

# Calculate empirical CCDF (1 - CDF) for filtered 2023 data
ccdf_2023 = np.arange(1, len(filtered_2023_sizes_sorted) + 1) / len(filtered_2023_sizes_sorted)

# Plot on log-log scale
ax5.loglog(filtered_2023_sizes_sorted, ccdf_2023, 'go', markersize=4, alpha=0.7)

# Use linear regression on log-transformed data to estimate α
log_sizes_2023 = np.log(filtered_2023_sizes_sorted)
log_ccdf_2023 = np.log(ccdf_2023)

# Remove any potential infinities or NaNs
valid_indices_2023 = np.isfinite(log_sizes_2023) & np.isfinite(log_ccdf_2023)
log_sizes_2023_clean = log_sizes_2023[valid_indices_2023]
log_ccdf_2023_clean = log_ccdf_2023[valid_indices_2023]

# Linear regression
slope_2023, intercept_2023, r_value_2023, p_value_2023, std_err_2023 = stats.linregress(log_sizes_2023_clean, log_ccdf_2023_clean)

# The negative of the slope is the shape parameter α
alpha_2023 = -slope_2023

# Plot the fitted line
x_fit_2023 = np.logspace(np.log10(1000), np.log10(max(filtered_2023_sizes_sorted)), 100)
y_fit_2023 = np.exp(intercept_2023) * x_fit_2023**slope_2023
ax5.loglog(x_fit_2023, y_fit_2023, 'r-', linewidth=2, label=f'Pareto fit (α={-slope_2023:.2f})')

# Add labels and title
ax5.set_xlabel('Cluster Size (H100 Equivalents)', fontsize=14)
ax5.set_ylabel('Complementary CDF (1-CDF)', fontsize=14)
ax5.set_title('Pareto Check - Clusters ≤ 2023\n≥ 10³ H100 Equivalents', fontsize=16)
ax5.grid(True, which="both", linestyle='--', alpha=0.7)
ax5.legend()

# Add R-squared value to show goodness of fit
ax5.text(0.05, 0.05, f'R² = {r_value_2023**2:.4f}', transform=ax5.transAxes, 
         fontsize=12, bbox=dict(facecolor='white', alpha=0.8))

# Enhance visual appearance
fig.suptitle(f'Supercomputer Cluster Analysis - Planned Clusters Up To {first_plot_year_cutoff}', fontsize=18)
plt.tight_layout(rect=[0, 0, 1, 0.95])  # Adjust for the suptitle

# Add a summary of the data
total_clusters = len(planned_clusters)
max_size = planned_clusters['H100 equivalents'].max()
min_size = planned_clusters['H100 equivalents'].min()
mean_size = planned_clusters['H100 equivalents'].mean()

# plt.figtext(0.5, 0.01, 
#             f"Total clusters: {total_clusters}\n"
#             f"Average size: {mean_size:.2f} H100 equivalents\n"
#             f"Median size: {all_clusters_for_plot['H100 equivalents'].median():.2f} H100 equivalents\n"
#             f"Largest cluster: {max_size:.2f} H100 equivalents\n"
#             f"Smallest cluster: {min_size:.2f} H100 equivalents",
#             ha="center", fontsize=12, bbox={"facecolor":"lightgray", "alpha":0.5, "pad":5})

# Save the figures
plt.savefig('cluster_analysis.png', dpi=300, bbox_inches='tight')

# Show the plot
plt.show()

# Print distribution of clusters by size
print(f"Distribution of planned clusters by size (H100 equivalents) up to {first_plot_year_cutoff}:")
for i, count in enumerate(hist):
    print(f"{bin_labels[i]}: {count} clusters")
print(f"\nTotal planned clusters analyzed: {len(planned_clusters)}")

# Print statistics about compute concentration
print("\nCompute concentration statistics:")
for pct in [50, 80, 90, 95, 99]:
    clusters_needed = sorted_clusters[sorted_clusters['cumulative_proportion'] >= pct].index.min() + 1
    print(f"{pct}% of compute is in the top {clusters_needed} clusters ({clusters_needed/total_clusters*100:.1f}% of all clusters)")

# Calculate Gini coefficient for compute distribution
def gini(x):
    # Mean absolute difference
    mad = np.abs(np.subtract.outer(x, x)).mean()
    # Relative mean absolute difference
    rmad = mad / np.mean(x)
    # Gini coefficient
    return 0.5 * rmad

compute_gini = gini(planned_clusters['H100 equivalents'].values)
print(f"\nGini coefficient for compute distribution: {compute_gini:.4f}")
print(f"(0 = perfect equality, 1 = perfect inequality)")

# Pareto distribution analysis
print("\nPareto distribution analysis (Up to 2025):")
print(f"Estimated shape parameter (α): {alpha:.4f}")
print(f"R-squared of log-log fit: {r_value**2:.4f}")

print("\nPareto distribution analysis (Up to 2023):")
print(f"Estimated shape parameter (α): {alpha_2023:.4f}")
print(f"R-squared of log-log fit: {r_value_2023**2:.4f}")

