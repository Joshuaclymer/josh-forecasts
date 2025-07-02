import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# Set consistent font size and bar width variables
FONT_SIZE = 16
BAR_WIDTH = 0.7
plt.rcParams.update({'font.size': FONT_SIZE})

# ---- AI INVESTMENT DATA ----
# AI investment data by country (in billions of dollars)
ai_countries = [
    "USA", "China", "UK", "Canada", "Israel", 
    "Germany", "India", "France", "South Korea", "Singapore", 
    "Sweden", "Japan", "Australia", "Switzerland", "UAE"
]

ai_investment_amounts = [
    470.92, 119.32, 28.17, 15.31, 14.96, 
    13.27, 11.29, 11.10, 8.96, 7.27, 
    7.27, 5.89, 3.99, 3.90, 3.67
]

# Calculate percentage of total AI investment
total_ai_investment = sum(ai_investment_amounts)
ai_percentages = [(amount / total_ai_investment) * 100 for amount in ai_investment_amounts]

# Sort the AI investment percentages in descending order
ai_percentages.sort(reverse=True)

# ---- GOOGLE DATA CENTERS DATA ----
# List of countries for each Google data center
google_dc_countries = [
    "USA", "USA", "Chile", "Taiwan", "USA", "USA", "USA", "USA", "India", "Qatar", "Ireland", "Netherlands",
    "Germany", "Denmark", "Belgium", "Finland", "USA", "Hong Kong", "Japan", "Indonesia", "Japan", "USA",
    "USA", "Singapore", "UK", "USA", "Spain", "Australia", "Netherlands", "USA", "Italy", "USA", "Canada",
    "India", "USA", "Japan", "Brazil", "USA", "France", "USA", "Chile", "USA", "USA", "South Korea", "Australia",
    "Israel", "USA", "Canada", "Italy", "Brazil", "Poland", "Singapore", "USA", "Switzerland", "Austria",
    "Germany", "Saudi Arabia", "Greece", "USA", "Kuwait", "Malaysia", "New Zealand", "Norway", "Mexico",
    "South Africa", "Sweden", "Taiwan", "Thailand", "Taiwan", "USA", "UK", "Uruguay"
]

# Count how many data centers are in each country
google_dc_counts = pd.Series(google_dc_countries).value_counts()

# Convert counts to percentages
total_dc = google_dc_counts.sum()
google_dc_percentages = [(count / total_dc) * 100 for count in google_dc_counts]

# Sort the Google data center percentages in descending order
google_dc_percentages.sort(reverse=True)

# ---- AGGREGATE THE DISTRIBUTIONS ----
# Determine the maximum number of bars needed
max_bars = max(len(ai_percentages), len(google_dc_percentages))

# Pad the shorter list with zeros to match the length of the longer list
while len(ai_percentages) < max_bars:
    ai_percentages.append(0)
    
while len(google_dc_percentages) < max_bars:
    google_dc_percentages.append(0)

# Calculate the average percentage for each position
average_percentages = [(ai + google) / 2 for ai, google in zip(ai_percentages, google_dc_percentages)]

# Take the top 15 values for visualization
top_15_avg = average_percentages[:15]

# ---- CREATE THE VISUALIZATION ----
# Create a figure with a single plot
fig, ax = plt.subplots(figsize=(12, 6))

# Plot the Average (Final Forecast)
bars = ax.bar(range(len(top_15_avg)), top_15_avg, 
        color='skyblue', edgecolor='none', width=BAR_WIDTH)

# Add value labels on top of each bar
for i, bar in enumerate(bars):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height + 0.5,
            f'{top_15_avg[i]:.1f}%',
            ha='center', va='bottom', fontsize=FONT_SIZE-2,
            bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', pad=1))

# Set x-axis labels as position numbers
ax.set_xticks(range(len(top_15_avg)))
ax.set_xticklabels([f'#{i+1}' for i in range(len(top_15_avg))], fontsize=FONT_SIZE)

# Set labels and title
ax.set_ylabel("Percentage of global share (%)", fontsize=FONT_SIZE)
ax.set_title("Aggregate Forecast (Average of AI Investment and Google Data Centers Distributions)", fontsize=FONT_SIZE)

# Format y-axis with percentage signs
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.0f}%'))

# Adjust layout for better display
plt.tight_layout()

# Save the plot
plt.savefig('aggregate_forecast_plot.png', dpi=300, bbox_inches='tight')
plt.show()