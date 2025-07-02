import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv('data.csv')

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


# Extract year from First Operational Date for existing clusters
df['Op_Year'] = pd.to_datetime(df['First Operational Date'], errors='coerce').dt.year

# Create a combined year column that uses either Planned_Year or Op_Year
df['Combined_Year'] = df['Op_Year']
# For planned clusters with a year in the note, use that year
df.loc[(df['Status'] == 'Planned') & (~df['Planned_Year'].isna()), 'Combined_Year'] = df['Planned_Year']

# Define the years for which we want to create histograms
years = [2023, 2024, 2025, 2026, 2027]

# Create a figure with 5 subplots side by side, 2 rows (histograms on top, CCDF on bottom)
# Reduce height to fit better on screen
fig, axes = plt.subplots(2, 5, figsize=(25, 8), sharex=False)

# Define bins for the histograms
bins = [0, 1000, 5000, 10000, 25000, 50000, 100000, 250000, 500000, 1000000, 5000000, 10000000]
bin_labels = ['0-1K', '1K-5K', '5K-10K', '10K-25K', '25K-50K', '50K-100K', '100K-250K', '250K-500K', '500K-1M', '1M-5M', '5M-10M']

# Create a histogram and CCDF plot for each year
for i, year in enumerate(years):
    # Filter for clusters operational exactly in this year
    year_df = df[
        (~df['Combined_Year'].isna()) & (df['Combined_Year'] == year) & 
        (~df['H100 equivalents'].isna())
    ]
    
    # Skip if no data for this year
    if len(year_df) == 0:
        axes[0, i].text(0.5, 0.5, f'No clusters in {year}', 
                     horizontalalignment='center', verticalalignment='center',
                     transform=axes[0, i].transAxes, fontsize=12)
        axes[1, i].text(0.5, 0.5, 'No data available', 
                     horizontalalignment='center', verticalalignment='center',
                     transform=axes[1, i].transAxes, fontsize=12)
        continue
    
    # Count clusters in each bin for histogram
    hist, bin_edges = np.histogram(year_df['H100 equivalents'], bins=bins)
    
    # Plot the histogram (top row)
    axes[0, i].bar(range(len(hist)), hist, align='center', alpha=0.7, color='skyblue', edgecolor='navy')
    axes[0, i].set_xticks(range(len(hist)))
    axes[0, i].set_xticklabels(bin_labels, rotation=45, fontsize=8)
    
    # Only add y-label to the first subplot in each row
    if i == 0:
        axes[0, i].set_ylabel('Number of Clusters', fontsize=12)
    
    # Add title with year and total compute
    total_compute = year_df['H100 equivalents'].sum()
    axes[0, i].set_title(f'New in {year}\nTotal: {total_compute:,.0f} H100 eq.\n{len(year_df)} clusters', fontsize=12)
    
    # Add grid to histogram
    axes[0, i].grid(axis='y', linestyle='--', alpha=0.7)
    
    # Create CCDF plot (bottom row)
    # Create logarithmically spaced points for evaluation
    min_size = year_df['H100 equivalents'].min()
    max_size = year_df['H100 equivalents'].max()
    
    # Create logarithmically spaced size thresholds
    size_thresholds = np.logspace(np.log10(min_size*0.9) if min_size > 0 else 0, 
                                 np.log10(max_size*1.1), 100)
    
    # Calculate cumulative proportion of compute from clusters >= each size threshold
    cumulative_proportions = []
    
    for size in size_thresholds:
        # Calculate total compute from clusters >= this size
        compute_above_threshold = year_df[year_df['H100 equivalents'] >= size]['H100 equivalents'].sum()
        proportion = (compute_above_threshold / total_compute) * 100
        cumulative_proportions.append(proportion)
    
    # Plot CCDF with log scale on x-axis only
    axes[1, i].semilogx(size_thresholds, cumulative_proportions, 'b-', linewidth=2)
    axes[1, i].set_xlabel('Cluster Size Threshold (H100 Equivalents)', fontsize=10)
    
    # Add horizontal lines at key percentages
    for pct in [50, 80, 90, 95]:
        axes[1, i].axhline(y=pct, color='gray', linestyle='--', alpha=0.7)
        
        # Find the size threshold for this percentage (interpolate if needed)
        try:
            threshold_idx = next(idx for idx, val in enumerate(cumulative_proportions) if val <= pct)
            threshold = size_thresholds[threshold_idx]
            axes[1, i].text(threshold*1.1, pct+1, f'{pct}%: ≥{threshold:.0f}', 
                         fontsize=8, verticalalignment='bottom')
        except (StopIteration, IndexError):
            pass
    
    # Only add y-label to the first subplot in each row
    if i == 0:
        axes[1, i].set_ylabel('% of Compute from Clusters ≥ Size', fontsize=12)
    
    # Set y-axis limits for CCDF
    axes[1, i].set_ylim([0, 105])
    
    # Add grid to CCDF
    axes[1, i].grid(True, which='both', linestyle='--', alpha=0.7)
    
    # Print statistics for this year
    if len(year_df) > 0:
        print(f"\nYear {year}:")
        print(f"  Total clusters: {len(year_df)}")
        print(f"  Total compute: {total_compute:,.0f} H100 equivalents")
        
        # Print distribution by bin
        for j, count in enumerate(hist):
            if count > 0:  # Only print non-zero bins
                print(f"  {bin_labels[j]}: {count} clusters")
                
        # Print key thresholds for CCDF
        print("  Compute concentration:")
        for pct in [50, 80, 90, 95]:
            try:
                threshold_idx = next(idx for idx, val in enumerate(cumulative_proportions) if val <= pct)
                threshold = size_thresholds[threshold_idx]
                print(f"    {pct}% of compute from clusters ≥ {threshold:.0f} H100 equivalents")
            except (StopIteration, IndexError):
                pass
    else:
        print(f"\nYear {year}: No clusters found")

# Add an overall title
fig.suptitle('Distribution of New Clusters by Size for Each Year', fontsize=16)

# Adjust layout
plt.tight_layout(rect=[0, 0, 1, 0.95])  # Make room for the suptitle
plt.subplots_adjust(hspace=0.3)  # Add space between the rows

# Show the plot
plt.savefig('new_cluster_histograms_by_year.png', dpi=300, bbox_inches='tight')
plt.show()
