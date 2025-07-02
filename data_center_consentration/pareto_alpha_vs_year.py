import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats

# Read the data
df = pd.read_csv('data.csv')

# Filter out rows with missing H100 equivalents and only include non-planned clusters
df = df[(~df['H100 equivalents'].isna()) & (df['Status'] != 'Planned')].copy()

# Parse the 'First Operational Date' column
df['parsed_date'] = pd.to_datetime(df['First Operational Date'], errors='coerce')
df['year'] = df['parsed_date'].dt.year

# Function to calculate alpha for a given year cutoff
def calculate_alpha(year_cutoff):
    # Filter for clusters up to the cutoff year
    clusters_up_to_year = df[~df['year'].isna() & (df['year'] <= year_cutoff)].copy()
    
    # Filter out clusters smaller than 10^3 H100 equivalents
    filtered_sizes = clusters_up_to_year['H100 equivalents'].values[clusters_up_to_year['H100 equivalents'].values >= 1000]
    
    # If we don't have enough data points, return None
    if len(filtered_sizes) < 5:
        return None, None, 0
    
    filtered_sizes_sorted = np.sort(filtered_sizes)[::-1]  # Sort in descending order
    
    # Calculate empirical CCDF (1 - CDF)
    ccdf = np.arange(1, len(filtered_sizes_sorted) + 1) / len(filtered_sizes_sorted)
    
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
    
    # Return alpha, R-squared, and the number of data points
    return alpha, r_value**2, len(filtered_sizes)

# Calculate alpha for a range of year cutoffs from 2022 to 2025
year_cutoffs = range(2022, 2026)
alphas = []
r_squareds = []
data_points = []

print("Year Cutoff | Alpha | R² | Data Points")
print("-" * 40)

for year in year_cutoffs:
    alpha, r_squared, n_points = calculate_alpha(year)
    if alpha is not None:
        alphas.append(alpha)
        r_squareds.append(r_squared)
        data_points.append(n_points)
        print(f"{year} | {alpha:.4f} | {r_squared:.4f} | {n_points}")
    else:
        alphas.append(np.nan)
        r_squareds.append(np.nan)
        data_points.append(0)
        print(f"{year} | N/A | N/A | {n_points}")

# Create a figure with two subplots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

# Plot alpha vs year cutoff
ax1.plot(year_cutoffs, alphas, 'bo-', linewidth=2, markersize=8)
ax1.set_xlabel('Year Cutoff', fontsize=14)
ax1.set_ylabel('Pareto Shape Parameter (α)', fontsize=14)
ax1.set_title('Pareto α vs Year Cutoff\n(Clusters ≥ 10³ H100 Equivalents)', fontsize=16)
ax1.grid(True, linestyle='--', alpha=0.7)

# Add data point labels
for i, (year, alpha) in enumerate(zip(year_cutoffs, alphas)):
    if not np.isnan(alpha):
        ax1.text(year, alpha + 0.01, f'α = {alpha:.3f}', ha='center', va='bottom', fontsize=10)

# Plot R-squared vs year cutoff
ax2.plot(year_cutoffs, r_squareds, 'ro-', linewidth=2, markersize=8)
ax2.set_xlabel('Year Cutoff', fontsize=14)
ax2.set_ylabel('R² of Power Law Fit', fontsize=14)
ax2.set_title('Goodness of Fit vs Year Cutoff\n(Clusters ≥ 10³ H100 Equivalents)', fontsize=16)
ax2.grid(True, linestyle='--', alpha=0.7)

# Add data point labels
for i, (year, r2) in enumerate(zip(year_cutoffs, r_squareds)):
    if not np.isnan(r2):
        ax2.text(year, r2 + 0.01, f'R² = {r2:.3f}', ha='center', va='bottom', fontsize=10)

# Add data point count annotation
for i, (year, n) in enumerate(zip(year_cutoffs, data_points)):
    if n > 0:
        ax1.text(year, 0.5, f'n = {n}', ha='center', va='bottom', fontsize=9, color='gray')

# Enhance visual appearance
fig.suptitle('Pareto Distribution Analysis Over Time', fontsize=18)
plt.tight_layout(rect=[0, 0, 1, 0.95])  # Adjust for the suptitle

# Save the figure
plt.savefig('pareto_alpha_vs_year.png', dpi=300, bbox_inches='tight')

# Show the plot
plt.show()

# Additional analysis: Calculate alpha for each individual year
print("\nAlpha for individual years:")
print("Year | Alpha | R² | Data Points")
print("-" * 40)

yearly_alphas = {}
for year in range(2022, 2026):
    # Filter for clusters from this specific year
    clusters_this_year = df[~df['year'].isna() & (df['year'] == year)].copy()
    
    # Filter out clusters smaller than 10^3 H100 equivalents
    filtered_sizes = clusters_this_year['H100 equivalents'].values[clusters_this_year['H100 equivalents'].values >= 1000]
    
    # If we don't have enough data points, skip
    if len(filtered_sizes) < 5:
        print(f"{year} | N/A | N/A | {len(filtered_sizes)} (insufficient data)")
        continue
    
    filtered_sizes_sorted = np.sort(filtered_sizes)[::-1]  # Sort in descending order
    
    # Calculate empirical CCDF (1 - CDF)
    ccdf = np.arange(1, len(filtered_sizes_sorted) + 1) / len(filtered_sizes_sorted)
    
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
    
    yearly_alphas[year] = alpha
    print(f"{year} | {alpha:.4f} | {r_value**2:.4f} | {len(filtered_sizes)}")
