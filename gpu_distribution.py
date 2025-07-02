import matplotlib.pyplot as plt
import numpy as np

# Set consistent font size for all text elements
FONT_SIZE = 16
plt.rcParams.update({'font.size': FONT_SIZE})

# Data for country groups pie chart
country_groups = [
    'The PRC and the US', 
    'Key US partners (Australia, Denmark, United Kingdom, etc)',
    'US adversaries (Russia, Iran, North Korea, etc)',
    'Other nations', 
]
group_values = [58, 20, 0.3, 10]  # In millions of H100 equivalents

# Data for countries with most ownership
countries = [
    'US', 
    'PRC',
    'UAE', 
    'UK', 
    'France', 
    'India',
]

# current distrib
country_values = [50, 8, 5, 2, 2, 1]  # In millions of H100 equivalents

# Create a figure with two subplots side by side
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

# Function to format labels with values
def autopct_format(values):
    def my_format(pct):
        total = sum(values)
        val = int(round(pct*total/100.0))
        return f'{pct:.1f}%\n({val}M)'
    return my_format

# First pie chart: Country Groups
colors1 = ['#2A623D', '#4682B4', '#D2B48C', '#7F2B2B']
wedges1, texts1, autotexts1 = ax1.pie(
    group_values, 
    labels=None,
    autopct=autopct_format(group_values),
    startangle=90,
    colors=colors1,
    wedgeprops={'edgecolor': 'white', 'linewidth': 1}
)

# Second pie chart: Countries with most ownership
colors2 = ['#2A623D', '#E6B800', '#4682B4', '#9370DB', '#D2B48C']
wedges2, texts2, autotexts2 = ax2.pie(
    country_values, 
    labels=None,
    autopct=autopct_format(country_values),
    startangle=90,
    colors=colors2,
    wedgeprops={'edgecolor': 'white', 'linewidth': 1}
)

# Set properties for all text elements
for autotext in autotexts1 + autotexts2:
    autotext.set_fontsize(FONT_SIZE - 2)
    autotext.set_bbox(dict(facecolor='white', alpha=0.7, edgecolor='none', pad=0.3))

# Add legends outside the pie charts
ax1.legend(
    wedges1, 
    country_groups,
    loc="center left",
    bbox_to_anchor=(0, 0.5),
    fontsize=FONT_SIZE - 2
)
ax2.legend(
    wedges2, 
    countries,
    loc="center left",
    bbox_to_anchor=(0, 0.5),
    fontsize=FONT_SIZE - 2
)

# Add titles
ax1.set_title('Compute Distribution by Country Groups', fontsize=FONT_SIZE)
ax2.set_title('Countries with Most GPU Ownership', fontsize=FONT_SIZE)

# Equal aspect ratio ensures that pie is drawn as a circle
ax1.set_aspect('equal')
ax2.set_aspect('equal')

plt.tight_layout()
plt.savefig('gpu_distribution_charts.png', dpi=300, bbox_inches='tight')
plt.show()
