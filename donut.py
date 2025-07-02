import matplotlib.pyplot as plt
import numpy as np

# Set consistent font size for all text elements
FONT_SIZE = 16
plt.rcParams.update({'font.size': FONT_SIZE})

# Create figure and axis
fig, ax = plt.subplots(figsize=(12, 6))

# Data for the donut chart
sizes = [75, 25]  # 75 H100e for Tier 1, 25 H100e for Tier 2
labels = ['Required Tier 1 share', 'All Tier 2 countries limit']
colors = ['#cceabb', '#f7a072']

# Plot the donut chart
wedges, texts = ax.pie(
    sizes, 
    labels=None,
    colors=colors, 
    wedgeprops=dict(width=0.4, edgecolor='none', linewidth=1),  # No edge color as per user preference
    startangle=90, 
    counterclock=True
)

# Draw the center circle for donut shape
centre_circle = plt.Circle((0, 0), 0.6, fc='white')
fig.gca().add_artist(centre_circle)

# Add annotations with lines for key segments
# For Tier 1 share (green segment) - left side
angle = 180  # Points to the left side of the chart
x = np.cos(np.radians(angle)) * 0.8
y = np.sin(np.radians(angle)) * 0.8
ax.annotate('', xy=(x, y), xytext=(x-0.5, y),
            arrowprops=dict(arrowstyle='-', color='black'))

# Add a dot at the arrow start
ax.plot(x, y, 'o', color='black', markersize=5)

# Add the text label with H100e count prominently displayed
# Position text to the right of the arrow as per user preference
ax.text(x-0.6, y, '75 H100e', 
        ha='right', va='center', fontsize=FONT_SIZE, fontweight='bold',
        bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', pad=0.3))
ax.text(x-0.6, y-0.15, 'Required Tier 1 share', 
        ha='right', va='center', fontsize=FONT_SIZE-2, color='gray')

# For Tier 2 share (orange segment) - right side
angle = 0  # Points to the right side of the chart
x = np.cos(np.radians(angle)) * 0.8
y = np.sin(np.radians(angle)) * 0.8
ax.annotate('', xy=(x, y), xytext=(x+0.5, y),
            arrowprops=dict(arrowstyle='-', color='black'))

# Add a dot at the arrow start
ax.plot(x, y, 'o', color='black', markersize=5)

# Add the text label with H100e count prominently displayed
# Position text to the right of the arrow as per user preference
ax.text(x+0.6, y, '25 H100e', 
        ha='left', va='center', fontsize=FONT_SIZE, fontweight='bold',
        bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', pad=0.3))
ax.text(x+0.6, y-0.15, 'All Tier 2 countries limit', 
        ha='left', va='center', fontsize=FONT_SIZE-2, color='gray')

# Equal aspect ratio ensures that pie is drawn as a circle
ax.axis('equal')

# Remove axes
ax.set_axis_off()

# Add padding to prevent labels from being cut off
plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1)

# Save the figure
plt.savefig('compute_distribution_donut.png', dpi=300, bbox_inches='tight')

# Display the plot
plt.show()