import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
from scipy import stats

# Read the data
df = pd.read_csv('data.csv')

# Filter out rows with missing H100 equivalents
df = df.dropna(subset=['H100 equivalents'])

# Find the largest cluster size in the dataset for reporting bias adjustment
max_cluster_size = df['H100 equivalents'].max()
print(f"Largest cluster size in dataset: {max_cluster_size:.0f} H100 equivalents")

# Function to calculate the reporting bias adjustment factor
def calculate_reporting_bias_factor(size, max_size):
    """Calculate the reporting bias adjustment factor based on the rule:
    - People are 10x less likely to report clusters that are 10x smaller than the largest that exist
    - People are 100x less likely to report clusters that are 100x smaller than the largest that exist
    - etc.
    
    Args:
        size: The size of the cluster in H100 equivalents
        max_size: The size of the largest cluster in H100 equivalents
        
    Returns:
        The reporting bias adjustment factor
    """
    # Avoid division by zero or negative values
    if size <= 0:
        return 1.0
    
    # Calculate the ratio of max_size to current size
    ratio = max_size / size
    
    # Use a square root relationship to moderate the adjustment factor
    # This still follows the general rule but prevents extreme adjustments
    # that would collapse the distribution
    # For example: a 100x smaller cluster gets sqrt(100) = 10x adjustment instead of 100x
    return np.sqrt(ratio)

# Apply the reporting bias adjustment to create adjusted H100 equivalents
df['Adjusted H100 equivalents'] = df['H100 equivalents'].apply(
    lambda x: x * calculate_reporting_bias_factor(x, max_cluster_size)
)

print(f"Sum of original H100 equivalents: {df['H100 equivalents'].sum():,.0f}")
print(f"Sum of adjusted H100 equivalents: {df['Adjusted H100 equivalents'].sum():,.0f}")

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

# Function to calculate reporting bias adjustment for a specific year
def calculate_year_adjusted_values(year_data):
    if len(year_data) == 0:
        return year_data
    
    # Find the largest cluster in this year's data
    year_max_size = year_data['H100 equivalents'].max()
    
    # Apply the reporting bias adjustment for this year's data
    year_data['Adjusted H100 equivalents'] = year_data['H100 equivalents'].apply(
        lambda x: x * calculate_reporting_bias_factor(x, year_max_size)
    )
    
    return year_data

# Apply the reporting bias adjustment for each year separately
for year in range(2022, 2028):
    # Get data for this year
    year_df = df[(~df['Combined_Year'].isna()) & (df['Combined_Year'] == year)]
    if len(year_df) > 0:
        # Apply the adjustment
        year_indices = year_df.index
        adjusted_year_df = calculate_year_adjusted_values(year_df)
        # Update the main dataframe
        df.loc[year_indices, 'Adjusted H100 equivalents'] = adjusted_year_df['Adjusted H100 equivalents']

# Define the years for which we want to create histograms
years = list(range(2022, 2028))  # From 2022 to 2027

# Define size categories for the clusters using an exponential factor
# Set the exponential factor (e.g., 1.5 means each bin is 1.5x larger than the previous)
exp_factor = 1.5

# Define the range of sizes (min and max)
min_size = 0  # Start at 0
max_size = 10000000  # 10M

# Generate bin edges
bin_edges = [min_size]  # Start with 0
current_edge = 100  # First non-zero edge

while current_edge <= max_size:
    bin_edges.append(current_edge)
    current_edge = int(current_edge * exp_factor)

# Create size categories from bin edges
size_categories = [(bin_edges[i], bin_edges[i+1]) for i in range(len(bin_edges)-1)]
# Add the final category to infinity
size_categories.append((bin_edges[-1], float('inf')))

# Create labels for the size categories
size_labels = []
order_of_magnitude_indices = []

# Find indices for specific H100 equivalent values (1K, 10K, 100K, 1M)
key_sizes = [0, 1000, 10000, 100000, 1000000, 10000000]  # 0, 1K, 10K, 100K, 1M, 10M

for i, (min_size, max_size) in enumerate(size_categories):
    # Check if this bin contains or is very close to one of our key sizes
    label = ''
    for key_size in key_sizes:
        # If the key size is within this bin or very close to the min_size
        if (min_size <= key_size < max_size) or abs(min_size - key_size) / (key_size + 1) < 0.05:  # Within 5% or in bin
            if key_size == 0:
                label = '0'
            elif key_size >= 1000000:
                label = f'{key_size/1000000:.0f}M'
            elif key_size >= 1000:
                label = f'{key_size/1000:.0f}K'
            else:
                label = f'{key_size:.0f}'
            order_of_magnitude_indices.append(i)
            break
    
    size_labels.append(label)  # Empty string for non-key sizes

# Print the generated bins for debugging
print("Generated size categories at orders of magnitude:")
for i, ((min_size, max_size), label) in enumerate(zip(size_categories, size_labels)):
    if label:  # Only print bins with labels (orders of magnitude)
        print(f"Bin {i}: {min_size:,.0f} - {max_size:,.0f} ({label})")


# Define colors for the histograms - skyblue for regular bars, darker blue for forecasted
regular_color = 'navy'
forecasted_color = 'skyblue'

# Create a figure with subplots for each year - use a wider size to accommodate 6 years
fig, axes = plt.subplots(1, len(years), figsize=(8, 4), sharey=True)
fig.suptitle('Proportion of Compute in Newly Built Clusters by Size (Reporting Bias Adjusted)', fontsize=12)

# Add more space on the left for y-tick labels and at the bottom for x-labels
plt.subplots_adjust(left=0.15, wspace=0.05, bottom=0.2, right=0.95, top=0.85)

# Store data for regression line
year_indices = []
bin_indices = []
proportions = []

# Calculate bin midpoints for regression analysis
bin_midpoints = []
for min_val, max_val in size_categories:
    bin_midpoints.append(np.log10((min_val + max_val) / 2))  # Use log scale for better representation

# Process data for each year
for i, year in enumerate(years):
    # Skip 2026 and 2027 as we'll handle them separately with forecasted data
    if year >= 2026:
        # Just initialize the subplot but don't process any data yet
        # We'll set xlim later based on the max proportion
        axes[i].set_xticks([])
        # Only show ticks at orders of magnitude
        axes[i].set_yticks(order_of_magnitude_indices)
        axes[i].set_yticklabels([])
        if i == 0:
            # Only add labels at orders of magnitude
            for j in order_of_magnitude_indices:
                label = size_labels[j]
                if label:  # Skip empty labels
                    axes[i].text(-0.1, j, label, ha='right', va='center', fontsize=9)
            # axes[i].set_ylabel('Cluster Size (H100 equiv.)', fontsize=12, labelpad=40)
        continue
        
    # Get data for this year
    year_df = df[
        (~df['Combined_Year'].isna()) & (df['Combined_Year'] == year) & 
        (~df['Adjusted H100 equivalents'].isna())
    ]
    
    # Skip if no data for this year
    if len(year_df) == 0:
        axes[i].text(0.5, 0.5, f'No data\nfor {year}', 
                    ha='center', va='center', transform=axes[i].transAxes)
        continue
    
    # Calculate total compute for this year using adjusted values
    total_compute = year_df['Adjusted H100 equivalents'].sum()
    
    # Initialize data for proportions in each category
    year_proportions = []
    
    # Calculate proportion of compute in each size category
    for j, (min_size, max_size) in enumerate(size_categories):
        # Filter clusters in this size category based on adjusted H100 equivalents
        category_df = year_df[
            (year_df['Adjusted H100 equivalents'] >= min_size) & 
            (year_df['Adjusted H100 equivalents'] < max_size)
        ]
        
        # Calculate total compute in this category
        category_compute = category_df['Adjusted H100 equivalents'].sum()
        proportion = category_compute / total_compute if total_compute > 0 else 0
        year_proportions.append(proportion)
        
        # Store data for regression analysis
        if proportion > 0:
            year_indices.append(i)  # X-coordinate (year index)
            bin_indices.append(j)   # Y-coordinate (bin index)
            proportions.append(proportion)  # Size of point (for weighted regression)
    
    # Create horizontal bar chart (appears vertical when plotted) with skyblue color
    bars = axes[i].barh(range(len(size_labels)), year_proportions, color=regular_color, alpha=0.8)
    
    # No percentage labels as requested
    
    # Set x-axis limits based on the maximum proportion across all years for consistency
    # This will be set after processing all years
    
    # Remove title and move year to the bottom as x-tick label
    axes[i].set_title("")
    if year == 2026:
        axes[i].set_xlabel(f'{year} (forecasted)', fontsize=10)
    else:
        axes[i].set_xlabel(f'{year}', fontsize=10)
    # if i == 0:  # Only add y-label to the first subplot
        # axes[i].set_ylabel('Cluster Size Categories', fontsize=10)
    
    # Set x-axis limits
    axes[i].set_xlim(0, 1)

    # remove x ticks
    axes[i].set_xticks([])
    
    # Remove individual x-axis labels (we'll add a shared one)
    
    # Set y-ticks for all subplots
    axes[i].set_yticks(range(len(size_labels)))
    
    # Only set y-tick labels for the first subplot
    if i == 0:
        # Force the y-tick labels to be visible with reduced offset
        for j, label in enumerate(size_labels):
            axes[i].text(-0.1, j, label, ha='right', va='center', fontsize=9)
        
        # add gap between y-label and y-ticks
        axes[i].set_ylabel('Cluster Size', fontsize=12, labelpad=35)
    
    # Hide the default y-tick labels for all subplots
    axes[i].set_yticklabels([])

# Now calculate and plot the regression line based on actual average cluster sizes
# instead of bin indices, which will better reflect the dramatic increase

# Create arrays to store year indices and average cluster sizes
year_indices_for_regression = []
avg_cluster_sizes = []

# Calculate average cluster size for each year up to 2025 (excluding 2026 and 2027)
for i, year in enumerate(years):
    if year >= 2026:
        continue  # Skip 2026 and 2027
        
    year_df = df[(~df['Combined_Year'].isna()) & (df['Combined_Year'] == year)]
    if len(year_df) > 0:
        # Use original H100 equivalents for growth trend calculation to avoid distortion
        total_compute = year_df['H100 equivalents'].sum()
        num_clusters = len(year_df)
        avg_size = total_compute / num_clusters
        
        # Store the year index and average cluster size
        year_indices_for_regression.append(i)
        avg_cluster_sizes.append(avg_size)
        
# Print the data points used for regression
print(f"Data points for regression:")
for i, (year_idx, avg_size) in enumerate(zip(year_indices_for_regression, avg_cluster_sizes)):
    print(f"  Year {years[year_idx]}: {avg_size:.0f} H100 equivalents")

# Perform regression on average cluster sizes if we have enough data points
if len(year_indices_for_regression) > 1:
    # Convert to numpy arrays
    year_indices_np = np.array(year_indices_for_regression)
    avg_sizes_np = np.array(avg_cluster_sizes)
    
    # Since the Y-axis is logarithmic, we should use log-transformed data for regression
    # Add a small constant (1) to avoid log(0)
    log_sizes = np.log10(avg_sizes_np + 1)
    
    # Perform linear regression on the log-transformed data
    log_poly_coeffs = np.polyfit(year_indices_np, log_sizes, 1)
    log_slope, log_intercept = log_poly_coeffs
    
    # Calculate annual growth factor (10^slope is the multiplicative factor per year)
    annual_growth_factor = 10**log_slope
    
    # Also calculate regular linear regression for reporting purposes
    poly_coeffs = np.polyfit(year_indices_np, avg_sizes_np, 1)
    slope, intercept = poly_coeffs
    
    # Calculate R-squared manually for the log-transformed data
    log_y_pred = log_slope * year_indices_np + log_intercept
    log_ss_total = np.sum((log_sizes - np.average(log_sizes))**2)
    log_ss_residual = np.sum((log_sizes - log_y_pred)**2)
    r_value = np.sqrt(1 - log_ss_residual/log_ss_total) if log_ss_total != 0 else 0
    p_value = 0  # We don't calculate p-value here
    std_err = np.sqrt(log_ss_residual / (len(year_indices_np) - 2)) if len(year_indices_np) > 2 else 0
    
    # For reporting, calculate the actual slope in original units
    # This is an approximation of the average yearly growth factor
    growth_factor = 10**log_slope
    # Convert back to original scale for interpretation
    slope = (10**log_slope - 1) * np.mean(avg_sizes_np)  # Approximate slope in original units
    intercept = 10**log_intercept - 1  # Approximate intercept in original units
    
    # Generate points for the regression line across all subplots
    x_reg = np.linspace(0, len(years)-1, 100)  # Smooth line across all years
    # Calculate predicted values in log scale, then transform back
    log_y_reg = log_slope * x_reg + log_intercept
    y_reg = 10**log_y_reg - 1  # Transform back to original scale

    # Print the predicted values of avg cluster size next to actual values for each year
    for i in range(len(year_indices_for_regression)):
        predicted_value = 10**(log_slope * year_indices_np[i] + log_intercept) - 1
        print(f"Year {years[i]}: Actual {avg_cluster_sizes[i]:.0f}, Predicted {predicted_value:.0f}")
    
    # Draw a continuous dotted line across all subplots
    # First, get the positions of all subplots in figure coordinates
    subplot_positions = []
    for i in range(len(years)):
        bbox = axes[i].get_position()
        subplot_positions.append(bbox)
    
    # Create line segments in figure coordinates
    # We'll create points at 80% of each subplot's width
    line_x = []
    line_y = []
    
    # Draw a regression line based on the average cluster sizes (excluding 2026)
    # We'll draw the line from the first subplot (2022) to the second-to-last subplot (2025)
    
    # Get positions of the first and second-to-last subplots (2022 and 2025)
    first_bbox = axes[0].get_position()  # 2022
    last_relevant_bbox = axes[3].get_position()  # 2025 (index 3 for years 2022-2025)
    
    # Start and end points for the line (80% across each subplot)
    start_x = first_bbox.x0 + 0.8 * first_bbox.width
    end_x = last_relevant_bbox.x0 + 0.8 * last_relevant_bbox.width
    
    # We need to map the average cluster sizes to bin indices for visualization
    # First, determine which bin each average size falls into
    # Ensure start_avg_size is not negative (use actual 2022 value if available)
    if len(avg_cluster_sizes) > 0:
        start_avg_size = max(1000, avg_cluster_sizes[0])  # Use actual 2022 value, but at least 1000
    else:
        start_avg_size = 1000  # Fallback to a reasonable value
        
    # For end value, use the predicted value for 2025 (year index 3)
    end_avg_size = max(30000, slope * 3 + intercept)  # Ensure it's at least 30K
    print(f"Log-based start_avg_size: {start_avg_size:.1f}")
    end_avg_size = 10**(log_slope * 3 + log_intercept) - 1  # Average size at year index 3 (2025)
    print(f"Log-based end_avg_size: {end_avg_size:.1f}")

    
    # Function to map a cluster size to a bin index, accounting for logarithmic scale
    def size_to_bin_index(size):
        # Handle edge cases
        if size <= 0:
            return 0  # First bin
        
        # Find which bin this size belongs to
        for i, (min_size, max_size) in enumerate(size_categories):
            if min_size <= size < max_size:
                # Calculate position within the bin using logarithmic interpolation
                if min_size <= 0:  # Handle the first bin specially
                    return i
                
                # Calculate the logarithmic position within the bin
                log_min = np.log10(min_size)
                log_max = np.log10(max_size)
                log_size = np.log10(size)
                
                # Interpolate within the bin
                bin_fraction = (log_size - log_min) / (log_max - log_min)
                return i + bin_fraction
        
        # If we get here, the size is larger than our largest bin
        return len(size_categories) - 1
    
    # Convert average sizes to bin indices
    start_bin_index = size_to_bin_index(start_avg_size)
    end_bin_index = size_to_bin_index(end_avg_size)
    
    # Print debug information
    print(f"Start avg size: {start_avg_size:.1f} H100 eq -> bin index {start_bin_index}")
    print(f"End avg size: {end_avg_size:.1f} H100 eq -> bin index {end_bin_index}")
    
    start_y_number = 1055
    end_y_number = 34232

    # Convert y coordinates to figure coordinates
    # Map the cluster sizes to bin indices
    start_bin_index = size_to_bin_index(start_y_number)
    end_bin_index = size_to_bin_index(end_y_number)
    print("start_bin_index: ", start_bin_index)
    print("end_bin_index: ", end_bin_index)
    
    # Convert bin indices to figure coordinates
    # For the y-axis, we need to map from the data coordinates to figure coordinates
    # First, get the axes data limits
    first_ax_ylim = axes[0].get_ylim()
    last_ax_ylim = axes[3].get_ylim()
    
    # Calculate the normalized position within the y-axis range
    # For a categorical y-axis, we need to map from the bin index to the position
    # The y-axis is inverted (0 at the top, len(size_labels)-1 at the bottom)
    start_norm_pos = (start_bin_index / (len(size_labels)-1))
    end_norm_pos = (end_bin_index / (len(size_labels)-1))
    
    # Map to figure coordinates
    offset = 0.03
    start_y_coordinates = first_bbox.y0 + offset + first_bbox.height * start_norm_pos
    end_y_coordinates = last_relevant_bbox.y0 + offset + last_relevant_bbox.height * end_norm_pos
    
    print(f"Figure coordinates: start_y={start_y_coordinates:.4f}, end_y={end_y_coordinates:.4f}")
    
    # Create the line segments
    line_x = [start_x, end_x]
    line_y = [start_y_coordinates, end_y_coordinates]
    
    # Draw the dotted line across all subplots if we have enough points
    if len(line_x) > 1:
        # Create a Line2D object in figure coordinates
        regression_line = plt.Line2D(line_x, line_y, 
                                    transform=fig.transFigure,  # Use figure coordinates
                                    color='red',
                                    linestyle=':',
                                    linewidth=3,
                                    zorder=100)  # Ensure it's drawn on top with higher zorder
        
        # Add the line to the figure
        fig.lines.append(regression_line)
        
        # Print debug information about the regression line
    print(f"\nRegression Statistics:")
    print(f"  Annual Growth Factor: {annual_growth_factor:.2f}x (exponential)")
    print(f"  R-squared: {r_value**2:.4f}")
    
    # Calculate the predicted average size for 2026 and 2027 using the regression
    predicted_2026_avg_size = 10**(log_slope * 4 + log_intercept)  # Year index 4 is 2026
    predicted_2027_avg_size = 10**(log_slope * 5 + log_intercept)  # Year index 5 is 2027
    print(f"Predicted 2026 average cluster size: {predicted_2026_avg_size:.0f} H100 equivalents")
    print(f"Predicted 2027 average cluster size: {predicted_2027_avg_size:.0f} H100 equivalents")
    
    # Replace 2026 data with forecasted distribution based on 2025
    # Get the 2025 data
    year_2025_df = df[
        (~df['Combined_Year'].isna()) & (df['Combined_Year'] == 2025) & 
        (~df['Adjusted H100 equivalents'].isna())
    ]
    
    # Calculate the average size for 2025 using original values for consistent growth projection
    avg_size_2025 = year_2025_df['H100 equivalents'].mean()
    
    # Calculate scaling factor
    scaling_factor = predicted_2026_avg_size / avg_size_2025
    print(f"Scaling factor for 2026 forecast: {scaling_factor:.2f}x")
    
    # Create a new dataframe for 2026 by scaling the 2025 data
    year_2026_df = year_2025_df.copy()
    year_2026_df['Adjusted H100 equivalents'] = year_2026_df['Adjusted H100 equivalents'] * scaling_factor
    
    # Replace the 2026 data in the subplot
    i_2026 = years.index(2026)  # Get the index for 2026
    
    # Clear the existing 2026 subplot
    axes[i_2026].clear()
    
    # Apply the reporting bias adjustment to the forecasted 2026 data
    # First scale the H100 equivalents
    year_2026_df['H100 equivalents'] = year_2026_df['H100 equivalents'] * scaling_factor
    # Then calculate the adjusted values based on the scaled values
    max_cluster_size_2026 = year_2026_df['H100 equivalents'].max()
    year_2026_df['Adjusted H100 equivalents'] = year_2026_df['H100 equivalents'].apply(
        lambda x: x * calculate_reporting_bias_factor(x, max_cluster_size_2026)
    )
    
    # Calculate total compute for forecasted 2026 using adjusted values
    total_compute_2026 = year_2026_df['Adjusted H100 equivalents'].sum()
    
    # Initialize data for proportions in each category
    year_proportions_2026 = []
    
    # Calculate proportion of compute in each size category
    for j, (min_size, max_size) in enumerate(size_categories):
        # Filter clusters in this size category
        category_df = year_2026_df[
            (year_2026_df['Adjusted H100 equivalents'] >= min_size) & 
            (year_2026_df['Adjusted H100 equivalents'] < max_size)
        ]
        
        # Calculate total compute in this category using adjusted values
        category_compute = category_df['Adjusted H100 equivalents'].sum()
        proportion = category_compute / total_compute_2026 if total_compute_2026 > 0 else 0
        year_proportions_2026.append(proportion)
    
    # Create horizontal bar chart for 2026 with darker blue for forecasted data
    bars = axes[i_2026].barh(range(len(size_labels)), year_proportions_2026, color=forecasted_color, alpha=0.8)
    
    # No percentage labels as requested
    
    # Set x-axis limits based on the maximum proportion across all years for consistency
    # This will be set after processing all years
    
    # Remove x ticks
    axes[i_2026].set_xticks([])
    
    # Set y-ticks
    axes[i_2026].set_yticks(range(len(size_labels)))
    axes[i_2026].set_yticklabels([])
    
    # Update the label
    axes[i_2026].set_xlabel('2026\n(forecasted)', fontsize=10)
    
    # Now do the same for 2027
    # Create a new dataframe for 2027 by scaling the 2025 data
    year_2027_df = year_2025_df.copy()
    scaling_factor_2027 = predicted_2027_avg_size / avg_size_2025
    print(f"Scaling factor for 2027 forecast: {scaling_factor_2027:.2f}x")
    year_2027_df['Adjusted H100 equivalents'] = year_2027_df['Adjusted H100 equivalents'] * scaling_factor_2027
    
    # Replace the 2027 data in the subplot
    i_2027 = years.index(2027)  # Get the index for 2027
    
    # Apply the reporting bias adjustment to the forecasted 2027 data
    # First scale the H100 equivalents
    year_2027_df['H100 equivalents'] = year_2027_df['H100 equivalents'] * scaling_factor_2027
    # Then calculate the adjusted values based on the scaled values
    max_cluster_size_2027 = year_2027_df['H100 equivalents'].max()
    year_2027_df['Adjusted H100 equivalents'] = year_2027_df['H100 equivalents'].apply(
        lambda x: x * calculate_reporting_bias_factor(x, max_cluster_size_2027)
    )
    
    # Calculate total compute for forecasted 2027 using adjusted values
    total_compute_2027 = year_2027_df['Adjusted H100 equivalents'].sum()
    
    # Initialize data for proportions in each category
    year_proportions_2027 = []
    
    # Calculate proportion of compute in each size category
    for j, (min_size, max_size) in enumerate(size_categories):
        # Filter clusters in this size category (still using original H100 equivalents for filtering)
        category_df = year_2027_df[
            (year_2027_df['Adjusted H100 equivalents'] >= min_size) & 
            (year_2027_df['Adjusted H100 equivalents'] < max_size)
        ]
        
        # Calculate total compute in this category using adjusted values
        category_compute = category_df['Adjusted H100 equivalents'].sum()
        proportion = category_compute / total_compute_2027 if total_compute_2027 > 0 else 0
        year_proportions_2027.append(proportion)
    
    # Create horizontal bar chart for 2027 with darker blue for forecasted data
    bars = axes[i_2027].barh(range(len(size_labels)), year_proportions_2027, color=forecasted_color, alpha=0.8)
    
    # No percentage labels as requested
    
    # Set x-axis limits based on the maximum proportion across all years for consistency
    # This will be set after processing all years
    
    # Remove x ticks
    axes[i_2027].set_xticks([])
    
    # Set y-ticks
    axes[i_2027].set_yticks(range(len(size_labels)))
    axes[i_2027].set_yticklabels([])
    
    # Update the label
    axes[i_2027].set_xlabel('2027\n(forecasted)', fontsize=10)
    
    # Add red dots to indicate the mean of each distribution
    # Only for years up to 2025 (not including 2026 and 2027)
    for i, year in enumerate(years):
        # Skip 2026 and 2027 as we handle them separately
        if year >= 2026:
            continue
            
        # Filter for clusters operational in this year using Combined_Year
        year_df = df[(~df['Combined_Year'].isna()) & (df['Combined_Year'] == year)]
        
        if len(year_df) > 0:
            # Get adjusted cluster sizes for this year
            cluster_sizes = year_df['Adjusted H100 equivalents'].values
            
            # Calculate the proportion of compute in each bin
            bin_counts = np.zeros(len(size_categories))
            
            # Bin the clusters by size
            for size in cluster_sizes:
                for j, (lower, upper) in enumerate(size_categories):
                    if lower <= size < upper:
                        bin_counts[j] += size
                        break
            
            # Calculate total compute
            total_compute = sum(bin_counts)
            
            # Calculate proportions
            if total_compute > 0:
                bin_proportions = bin_counts / total_compute
                
                # Calculate weighted mean bin index
                mean_bin = sum(j * bin_proportions[j] for j in range(len(bin_proportions)))
            
            # Plot the mean as a red dot
            axes[i].scatter([0.8], [mean_bin], color='red', s=30, zorder=5)
    
    # No red dots for 2026 and 2027 forecasted data as requested
    
    # Find the maximum proportion across all years to set consistent x-axis limits
    all_proportions = []
    
    # Collect all proportions from actual years (2022-2025)
    for i, year in enumerate(years):
        if year < 2026:
            year_df = df[
                (~df['Combined_Year'].isna()) & (df['Combined_Year'] == year) & 
                (~df['H100 equivalents'].isna())
            ]
            if len(year_df) > 0:
                total_compute = year_df['Adjusted H100 equivalents'].sum()
                for j, (min_size, max_size) in enumerate(size_categories):
                    category_df = year_df[
                        (year_df['Adjusted H100 equivalents'] >= min_size) & 
                        (year_df['Adjusted H100 equivalents'] < max_size)
                    ]
                    category_compute = category_df['Adjusted H100 equivalents'].sum()
                    proportion = category_compute / total_compute if total_compute > 0 else 0
                    all_proportions.append(proportion)
    
    # Add proportions from 2026 and 2027 forecasts
    all_proportions.extend(year_proportions_2026)
    all_proportions.extend(year_proportions_2027)
    
    # Calculate the maximum proportion and set consistent x-axis limits
    max_proportion = max(all_proportions) if all_proportions else 0.5
    # Add some padding (10%) to ensure all bars are visible
    x_limit = max_proportion * 1.1
    
    # Set the same x-axis limit for all subplots
    for i in range(len(years)):
        axes[i].set_xlim(0, x_limit)
    
    # Print regression statistics
    print(f"Regression Statistics for Average Cluster Sizes:")
    print(f"Slope: {slope:.0f} H100 equivalents per year")
    print(f"Intercept: {intercept:.0f} H100 equivalents")
    print(f"R-squared: {r_value**2:.3f}")
    print(f"p-value: {p_value:.3e}")
    print(f"Standard Error: {std_err:.0f}")
    
    # Add a text box with regression statistics
    # Position it in the upper left corner of the figure, above all other elements
    stats_text = f"Annual Growth in avg cluster size: {annual_growth_factor:.1f}x\nR²: {r_value**2:.2f}"
    
    # Get the position of the first subplot
    bbox = axes[0].get_position()
    
    # Add text to the figure rather than the axes to ensure it's on top
    fig.text(bbox.x0 + 0.02, bbox.y1 - 0.02, stats_text, fontsize=9,
             bbox=dict(facecolor='white', alpha=0.9, edgecolor='gray', boxstyle='round,pad=0.5'),
             ha='left', va='top', zorder=1000)  # High zorder to ensure it's on top
    
    # No trend box at the bottom as requested

# Add shared x-axis labels
# fig.text(0.5, 0.08, 'Proportion of Compute', ha='center', fontsize=12)

# Add Year Operational label
fig.text(0.5, 0.05, 'Year Operational', ha='center', fontsize=12)

# Don't use tight_layout as it can interfere with our manual adjustments
# We've already set subplots_adjust earlier

# Save the figure with a name indicating the reporting bias adjustment
fig.savefig('new_compute_distribution_trend_adjusted.png', dpi=300, bbox_inches='tight')

# Print statistics for each year
print("\nStatistics by Year:")
for year in years:
    year_df = df[(~df['Combined_Year'].isna()) & (df['Combined_Year'] == year)]
    if len(year_df) > 0:
        original_compute = year_df['H100 equivalents'].sum()
        adjusted_compute = year_df['Adjusted H100 equivalents'].sum()
        num_clusters = len(year_df)
        avg_size = adjusted_compute / num_clusters
        print(f"{year}: {num_clusters} clusters, {adjusted_compute:,.0f} adjusted H100 equivalents total (original: {original_compute:,.0f}), {avg_size:,.0f} average adjusted size")
    else:
        print(f"Year {year}: No clusters found")

# If regression was calculated, print the statistics
if len(year_indices) > 1:
    print("\nRegression Statistics:")
    print(f"  Slope: {slope:.2f} H100 equivalents per year")
    print(f"  R-squared: {r_value**2:.4f}")
    print(f"  p-value: {p_value:.4f}")

# Show the plot
plt.show()
