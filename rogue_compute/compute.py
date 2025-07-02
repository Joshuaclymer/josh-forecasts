import matplotlib.pyplot as plt
import numpy as np
import matplotlib.patches as patches

# Data for the bar plot
categories = [
    'Data centers\nin uncooperative\nnations',
    'Secret\ndata centers in\ncooperative nations',
    'Visible\ndata centers\nrunning\nhidden workloads',
    'Crowdsourced\ncompute',
]

# Initial compute value (before the slowdown) and values for the bars
initial_compute = 10000000  # 10M H100 equivalents
values = [300000, 100000, 100000, 10000]  # Values for the remaining categories

# Variables to control the position of the arrow and text labels
# Values between -0.5 and len(categories)-0.5 for horizontal positioning
ARROW_X_POSITION = 2  # Default: positioned at the first bar
RIGHT_LABEL_X_POSITION = len(categories) -0.2# Controls where the dotted line labels appear on the right

# Create a figure and axis
fig, ax = plt.subplots(figsize=(12, 6))

# Set consistent font size for all text elements
FONT_SIZE = 16
plt.rcParams.update({'font.size': FONT_SIZE})

# Define colors for the bars - using a standard blue color for all bars
bar_colors = ['skyblue', 'skyblue', 'skyblue', 'skyblue']

# Create the bar plot with vertical bars
values_modified = values.copy()
# values_modified[1] = 250000
bars = ax.bar(categories, values_modified, color=bar_colors, edgecolor='none', width=0.6)

# Calculate the decrease factor between the initial compute and the largest bar
decrease_factor = initial_compute / values[0]

# Add annotations for each bar - just the values without H100e
for i, bar in enumerate(bars):
    value = values[i]
    if value >= 1000000:
        value_text = f'{value/1000000:.0f}M'
    elif value >= 1000:
        value_text = f'{value/1000:.0f}K H100e'
    else:
        value_text = f'{value}'
    
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() * 1.05, 
            value_text, ha='center', va='bottom', fontsize=FONT_SIZE)

# Add a horizontal line at the initial compute level
ax.axhline(y=initial_compute, color='gray', linestyle='--', alpha=0.7, xmin=0, xmax=0.95)

# Add a text label for the initial compute level at the right of the line
ax.text(RIGHT_LABEL_X_POSITION, initial_compute*1.1, 'Compute for ASI development before slowdown: 10M H100e', 
        fontsize=FONT_SIZE, ha='right', va='bottom', 
        bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', pad=0))

# Add a horizontal line at the largest bar
max_bar = max(values)
max_bar_index = values.index(max_bar)
ax.axhline(y=max_bar, color='gray', linestyle='--', alpha=0.7, 
           xmin=0, xmax=0.95)

# Add text label for the lower dotted line at the right
ax.text(RIGHT_LABEL_X_POSITION, max_bar*1.1, 'Compute for ASI development after slowdown: < 300K H100e', 
        fontsize=FONT_SIZE, ha='right', va='bottom', 
        bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', pad=0))


# Add a vertical arrow showing the compute decrease
# Position the arrow according to the control variable
arrow_x = ARROW_X_POSITION  # Use the control variable
arrow_y_start = initial_compute  # Start at the initial compute level
arrow_y_end = max_bar * 1.7  # End higher above the largest bar

# Add the arrow annotation - place the arrow vertically
ax.annotate('',  # Empty text for the arrow itself
            xy=(arrow_x, arrow_y_end), 
            xytext=(arrow_x, arrow_y_start),
            arrowprops=dict(facecolor='black', shrink=0.05, width=1.5, headwidth=8, headlength=8),
            fontsize=12)

# Add the text label to the right of the arrow
# For log scale, we need to calculate the geometric mean for proper vertical centering
middle_y = np.sqrt(arrow_y_start * arrow_y_end)  # Geometric mean for log scale
ax.text(arrow_x + 0.05, middle_y, f'30x decrease', 
        fontsize=FONT_SIZE, ha='left', va='center', bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', pad=0))

# Extend x-axis limits to make room for the arrow
ax.set_xlim(-0.5, len(categories))

# Set log scale for y-axis to better visualize the large differences
ax.set_yscale('log')

# This will be set later

# Set labels and title
ax.set_ylabel('80th percentile bound\non H100 equivalents\nfor ASI development', fontsize=FONT_SIZE)

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
ax.set_ylim(bottom=None, top=initial_compute*5)  # Add much more space at the top for log scale





# Adjust layout and display the plot
plt.tight_layout()
plt.savefig('compute_comparison.png', dpi=300, bbox_inches='tight')
plt.show()