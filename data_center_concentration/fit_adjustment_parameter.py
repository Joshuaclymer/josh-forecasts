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

df['Op_Year'] = df['First Operational Date'].apply(extract_year)
df['Planned_Year'] = df['First Operational Date Note'].apply(extract_year)

# Initialize Combined_Year column
df['Year'] = None

# For years 2022-2024, use the First Operational Date
df.loc[df['Op_Year'].isin([2022, 2023, 2024]), 'Year'] = df['Op_Year']

# For 2025, include both operational and planned clusters
df.loc[(df['Op_Year'] == 2025) | (df['Planned_Year'] == 2025), 'Year'] = 2025

# For years 2026-2027, use the Planned Year from First Operational Date Note
df.loc[df['Planned_Year'].isin([2026, 2027]), 'Year'] = df['Planned_Year']

# Constants based on the problem statement
TOTAL_COMPUTE_2024 = 10_000_000  # 10M H100 equivalents as of Dec 2024
ANNUAL_COMPUTE_GROWTH = 2.25      # Total compute increases 2.25x per year
ANNUAL_CLUSTER_SIZE_GROWTH = 3.9  # Mean cluster size increases 3.9x per year
ADJUSTMENT_FACTOR = 1 

# Years to analyze
years = [2023, 2024, 2025, 2026, 2027]

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
    year_df = df[
        (~df['Year'].isna()) & 
        (df['Year'] == year) & 
        (~df['H100 equivalents'].isna())
    ]

    new_clusters_by_year[year] = np.array(year_df['H100 equivalents'])

# Now apply adjustments to each sample
# Discrete version - duplicates clusters based on their size
def adjustment_function_discrete(sample):
    max_value = sample.max()
    new_items = []
    # For every value, calculate how many times to duplicate that value
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


def predict_total_compute_from_factor(original_sample, adjustment_factor):
    adjusted_sample, weights = adjustment_function(original_sample, adjustment_factor)
    total = 0
    for i, item in enumerate(adjusted_sample):
        total += item * weights[i]
    return total

# Plot the predicted total compute for a range of adjustment factors vs the actual total new compute
import matplotlib.pyplot as plt
from matplotlib.cm import viridis

# Define a range of adjustment factors to test
adjustment_factors = np.linspace(1.0, 3.5, 25)  # Test factors from 1.0 to 3.5

# Create a figure with a compact size (11x4) to accommodate legends on the right
plt.figure(figsize=(11, 4))

# Use the Viridis colormap which is a cool color palette
colors = [viridis(i/len(years)) for i in range(len(years))]

# Plot a curve for each year
for i, year in enumerate(years):
    # Skip years with no data
    if len(new_clusters_by_year[year]) == 0:
        continue
        
    # Calculate total compute for each adjustment factor
    totals = []
    for factor in adjustment_factors:
        total = predict_total_compute_from_factor(new_clusters_by_year[year], factor)
        totals.append(total)
    
    # Plot the curve without points
    plt.plot(adjustment_factors, totals, '-', color=colors[i], label=f'{year}')
    
    # Find intersection point where predicted compute matches actual compute
    if year in new_compute_by_year:
        actual_total = new_compute_by_year[year]
        # Find where the predicted line crosses the actual value
        for j in range(len(adjustment_factors) - 1):
            if (totals[j] <= actual_total <= totals[j+1]) or (totals[j] >= actual_total >= totals[j+1]):
                # Linear interpolation to find the exact intersection point
                x1, x2 = adjustment_factors[j], adjustment_factors[j+1]
                y1, y2 = totals[j], totals[j+1]
                if y1 != y2:  # Avoid division by zero
                    intersection_x = x1 + (x2 - x1) * (actual_total - y1) / (y2 - y1)
                    # Highlight the intersection point
                    plt.plot(intersection_x, actual_total, 'o', markersize=8, 
                            color=colors[i], markeredgecolor='black', markeredgewidth=1.5)

# Add horizontal lines for actual total compute for each year
for i, year in enumerate(years):
    # Skip years with no data
    if len(new_clusters_by_year[year]) == 0:
        continue
        
    # Use the actual total new compute for this year from our calculations
    actual_total = new_compute_by_year[year]
    
    # Add a horizontal line for this year's actual total
    plt.axhline(y=actual_total, linestyle='--', color=colors[i], alpha=0.7)

# Create a custom legend
from matplotlib.lines import Line2D

# Create custom legend elements for years
year_legend_elements = []
for i, year in enumerate(years):
    if year in [2022, 2023, 2024, 2025, 2026, 2027]:
        year_text = f"{year}" if year < 2026 else "2026 (planned)" if year == 2026 else "2027 (planned)"
        year_legend_elements.append(Line2D([0], [0], color=colors[i], lw=2, label=year_text))

# Create custom legend elements for line styles
style_legend_elements = [
    Line2D([0], [0], color='gray', lw=2, label='Forecasted by\nunder-reporting'),
    Line2D([0], [0], color='gray', linestyle='--', lw=2, label='Forecasted by\nannual growth'),
    Line2D([0], [0], marker='o', color='w', markerfacecolor='gray', markersize=8, 
           markeredgecolor='black', markeredgewidth=1.5, label='Intersection point')
]

# Format the plot
plt.xlabel('Under-reporting coefficient', fontsize=12)
plt.ylabel('Predicted Total\nNew Compute in Year (H100e)', fontsize=12)
plt.title('Which under-reporting coefficients align with annual compute growth forecasts?', fontsize=12)
plt.grid(linestyle='--', alpha=0.7)

# Add the legends to the right side of the plot with consistent width and more space
first_legend = plt.legend(handles=year_legend_elements, title='Year', loc='center left', 
                         bbox_to_anchor=(1, 0.7), borderaxespad=1.0, frameon=True, 
                         handletextpad=0.5, columnspacing=1.0)

# Set the width of the first legend box
first_legend_box = first_legend.get_bbox_to_anchor()
first_width = first_legend_box.width

# Add the second legend with the same width
plt.gca().add_artist(first_legend)
second_legend = plt.legend(handles=style_legend_elements, title='Legend', loc='center left', 
                          bbox_to_anchor=(1, 0.25), borderaxespad=1.0, frameon=True, 
                          handletextpad=0.5, columnspacing=1.0)

# Ensure both legends have the same width and enough right padding
plt.tight_layout(rect=[0.1, 0, 0.9, 1])  # Adjust the main plot to leave 30% of width for legends

# Set x-axis limits
plt.xlim(1.0, 3.5)

# Set y-axis to log scale
plt.yscale('log')

# Format y-axis with K for thousands and M for millions
from matplotlib.ticker import FuncFormatter
def value_formatter(x, pos):
    if x >= 1000000:
        return f'{x/1000000:.1f}M'
    elif x >= 1000:
        return f'{x/1000:.0f}K'
    else:
        return f'{x:.0f}'
plt.gca().yaxis.set_major_formatter(FuncFormatter(value_formatter))

# No need for another tight_layout call as we already set it above
# plt.tight_layout()

# Save and show the plot
plt.savefig('adjustment_factor_impact.png', dpi=150, bbox_inches='tight')
plt.show()
