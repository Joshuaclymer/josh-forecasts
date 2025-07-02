
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.patches as patches

# Data for the bar plot
categories = [
    "Training\ncompute before\nslowdown",
    'Data centers\nin uncooperative\nnations',
    'Data centers in\ncooperative nations\nnot under verification',
    'Secret\ndata centers in\ncooperative nations',
]

# Values in format: [(low, medium, high), ...]
# None can be used for low or high to indicate no error bar in that direction
values = [(None, 10000000, None), (1000, 300_000, 1_000_000), (10_000, 100_000, 1_000_000), (1000, 20_000, 100_000)]

# All values are used for bars (no special initial compute value)

# Variables to control the position of the arrow and text labels
# Values between -0.5 and len(categories)-0.5 for horizontal positioning
ARROW_X_POSITION = 1.5  # Adjusted position after removing a bar
RIGHT_LABEL_X_POSITION = len(categories) -0.2# Controls where the dotted line labels appear on the right

# Create a figure and axis
fig, ax = plt.subplots(figsize=(13, 6))

# Set consistent font size for all text elements
FONT_SIZE = 16
plt.rcParams.update({'font.size': FONT_SIZE})

# Define colors for the bars - darker grey for first bar, standard grey for others
bar_colors = ['#888888', '#D1D1D1', '#D1D1D1', '#D1D1D1']

# Process the values to extract medium values and error margins
valid_categories = []
medium_values = []
lower_errors = []
upper_errors = []

# Process all categories and their corresponding values
# Use all values for the bars (no special initial compute)
for i in range(len(categories)):
    # Make sure we have enough values
    if i < len(values):
        value_data = values[i]  # Get the corresponding value tuple
        
        valid_categories.append(categories[i])
        low, medium, high = value_data
        medium_values.append(medium)
        
        # Handle None values for error bars
        if low is None:
            lower_errors.append(0)  # No lower error bar
        else:
            lower_errors.append(medium - low)
            
        if high is None:
            upper_errors.append(0)  # No upper error bar
        else:
            upper_errors.append(high - medium)

# Create the bar plot with asymmetric error bars
bars = ax.bar(valid_categories, medium_values, color=bar_colors[:len(valid_categories)], 
            edgecolor='none', width=0.6, 
            yerr=[lower_errors, upper_errors], capsize=5, 
            error_kw={'ecolor': 'black', 'elinewidth': 1.5, 'capthick': 1.5})

# Calculate the decrease factor between the first bar and the second bar (if applicable)
if len(medium_values) >= 2:
    decrease_factor = medium_values[0] / medium_values[1] if medium_values[1] > 0 else 1
else:
    decrease_factor = 1  # Default if not enough values

# Add annotations for each bar - above the bar, right-aligned
for i, bar in enumerate(bars):
    value = medium_values[i]
    
    # Keep H100e only for the first bar
    if i == 0 and value >= 1000000:
        value_text = f'{value/1000000:.0f}M H100e'
    elif value >= 1000000:
        value_text = f'{value/1000000:.0f}M'
    elif value >= 1000:
        value_text = f'{value/1000:.0f}K'
    else:
        value_text = f'{value}'
    
    # Position labels above the bar
    label_y = bar.get_height() * 1.05  # Just above the bar
    
    # Center text for the first bar, right of center for others
    if i == 0:
        label_x = bar.get_x() + bar.get_width()/2  # Center of the bar
        text_align = 'center'
    else:
        label_x = bar.get_x() + bar.get_width()/2 + 0.05  # Right of center of the bar
        text_align = 'left'
    
    # Add white background to ensure readability
    ax.text(label_x, label_y, value_text, 
            ha=text_align, va='bottom', fontsize=FONT_SIZE,
            bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', pad=1))

# Add a horizontal line at the first bar (if applicable)
if len(medium_values) > 0:
    first_bar_value = medium_values[0]
    ax.axhline(y=first_bar_value, color='gray', linestyle='--', alpha=0.7, xmin=0, xmax=0.95)

# Add a text label for the first bar line at the right
# ax.text(RIGHT_LABEL_X_POSITION, first_bar_value*1.1, 'Compute for ASI development before slowdown: 10M H100e', 
#         fontsize=FONT_SIZE, ha='right', va='bottom', 
#         bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', pad=0))

# Add a horizontal line at the second bar (if applicable)
if len(medium_values) > 1:
    second_bar_value = medium_values[1]
    ax.axhline(y=second_bar_value, color='gray', linestyle='--', alpha=0.7, 
               xmin=0, xmax=0.95)

# Define max_bar for use in arrow positioning (if applicable)
if len(medium_values) > 0:
    max_bar = max(medium_values)
else:
    max_bar = 0  # Default if no values

# Add text label for the second dotted line at the right
# ax.text(RIGHT_LABEL_X_POSITION, second_bar_value*1.1, 'Compute for ASI development after slowdown: < 300K H100e', 
#         fontsize=FONT_SIZE, ha='right', va='bottom', 
#         bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', pad=0))


# Add a vertical arrow showing the compute decrease (if applicable)
# Only add the arrow if we have at least 2 bars
if len(medium_values) >= 2:
    # Position the arrow centered above the second bar
    arrow_x = 3.0  # Center above the second bar (index 1)
    arrow_y_start = medium_values[0]  # Start at the top line (first bar value)
    arrow_y_end = medium_values[1] + (medium_values[1] * 0.05)  # End at the bottom line (second bar value) with some padding

# Add the arrow annotation - pointing downward (if applicable)
if len(medium_values) >= 2:
    ax.annotate('',  # Empty text for the arrow itself
                xy=(arrow_x, arrow_y_end),  # Arrow tip points to bottom line
                xytext=(arrow_x, arrow_y_start),  # Arrow starts from top line
                arrowprops=dict(facecolor='black', shrink=0.05, width=1.5, headwidth=8, headlength=8),
                fontsize=12)

# Add the text label to the right of the arrow (if applicable)
if len(medium_values) >= 2:
    # For log scale, we need to calculate the geometric mean for proper vertical centering
    middle_y = np.sqrt(arrow_y_start * arrow_y_end)  # Geometric mean for log scale
    
    # Calculate the decrease factor between the first and second bar
    decrease_factor = medium_values[0] / medium_values[1] if medium_values[1] > 0 else 0
    
    # Add the text label to the right of the arrow
    ax.text(arrow_x + 0.15, middle_y, f'{decrease_factor:.0f}x decrease', 
            fontsize=FONT_SIZE, ha='left', va='center', bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', pad=0))

# Extend x-axis limits to make room for the arrow
ax.set_xlim(-0.5, len(valid_categories))

# Set log scale for y-axis to better visualize the large differences
ax.set_yscale('log')

# This will be set later

# Set labels and title
ax.set_ylabel('H100 equivalents of compute', fontsize=FONT_SIZE)

# Increase tick label font size
ax.tick_params(axis='x', which='major', labelsize=FONT_SIZE)
ax.tick_params(axis='y', which='major', labelsize=12)

# Remove x-axis ticks but keep the labels
ax.tick_params(axis='x', which='both', length=0)

# Ensure the arrow is drawn on top
plt.draw()
ax.figure.canvas.draw()

# Add significant padding to the top of the plot for log scale
# We need a much larger multiplier since we're in log scale
if len(medium_values) > 0:
    max_value = max(medium_values)
    for i, value in enumerate(medium_values):
        if i < len(upper_errors):
            max_value = max(max_value, value + upper_errors[i])
    ax.set_ylim(bottom=None, top=max_value*5)  # Add much more space at the top for log scale





# Adjust layout and display the plot
plt.tight_layout()
plt.savefig('compute_comparison.png', dpi=300, bbox_inches='tight')
plt.show()