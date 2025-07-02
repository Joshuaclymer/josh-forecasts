import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# Set consistent font size and bar width variables
FONT_SIZE = 16
BAR_WIDTH = 0.7  # Variable to control bar width
plt.rcParams.update({'font.size': FONT_SIZE})

# Create a figure with two subplots side by side
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(22, 6))

# ---- AI INVESTMENT PLOT (LEFT) ----

# AI investment data by country (in billions of dollars)
countries = [
    "USA", "China", "UK", "Canada", "Israel", 
    "Germany", "India", "France", "South Korea", "Singapore", 
    "Sweden", "Japan", "Australia", "Switzerland", "UAE"
]

investment_amounts = [
    470.92, 119.32, 28.17, 15.31, 14.96, 
    13.27, 11.29, 11.10, 8.96, 7.27, 
    7.27, 5.89, 3.99, 3.90, 3.67
]

# Create a DataFrame for easier manipulation
investment_data = pd.DataFrame({
    'Country': countries,
    'Investment': investment_amounts
})

# Calculate percentage of total investment
total_investment = investment_data['Investment'].sum()
investment_data['Percentage'] = (investment_data['Investment'] / total_investment) * 100

# Plot AI investment data on the left subplot
bars1 = ax1.bar(range(len(investment_data)), investment_data['Percentage'], 
               color='skyblue', edgecolor='none', width=BAR_WIDTH)

# Format x-axis labels for the left subplot
ax1.set_xticks(range(len(investment_data)))
ax1.set_xticklabels(investment_data['Country'], rotation=90, fontsize=FONT_SIZE-2)
ax1.tick_params(axis='y', which='major', labelsize=10)

# Add percentage signs to Y-axis tick labels
from matplotlib.ticker import FuncFormatter
ax1.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f'{y:.0f}%'))

# Add labels and title for the left subplot
ax1.set_ylabel("Percentage of global share (%)", fontsize=FONT_SIZE)
ax1.set_title("Global Distribution of AI Investment in 2024", fontsize=FONT_SIZE)

# ---- GOOGLE DATACENTERS PLOT (RIGHT) ----

# List of countries for each Google data center
countries = [
    "USA", "USA", "Chile", "Taiwan", "USA", "USA", "USA", "USA", "India", "Qatar", "Ireland", "Netherlands",
    "Germany", "Denmark", "Belgium", "Finland", "USA", "Hong Kong", "Japan", "Indonesia", "Japan", "USA",
    "USA", "Singapore", "UK", "USA", "Spain", "Australia", "Netherlands", "USA", "Italy", "USA", "Canada",
    "India", "USA", "Japan", "Brazil", "USA", "France", "USA", "Chile", "USA", "USA", "South Korea", "Australia",
    "Israel", "USA", "Canada", "Italy", "Brazil", "Poland", "Singapore", "USA", "Switzerland", "Austria",
    "Germany", "Saudi Arabia", "Greece", "USA", "Kuwait", "Malaysia", "New Zealand", "Norway", "Mexico",
    "South Africa", "Sweden", "Taiwan", "Thailand", "Taiwan", "USA", "UK", "Uruguay"
]

# Count how many data centers are in each country
country_counts = pd.Series(countries).value_counts()

# Convert counts to percentages
total = country_counts.sum()
country_percentages = (country_counts / total) * 100

# Sort by percentage descending and take only the top 15 countries
country_percentages = country_percentages.sort_values(ascending=False).head(15)

# Plot Google datacenters data on the right subplot (top 15 countries only)
bars2 = ax2.bar(range(len(country_percentages)), country_percentages.values, 
               color='skyblue', edgecolor='none', width=BAR_WIDTH)

# Format x-axis labels for the right subplot
ax2.set_xticks(range(len(country_percentages)))
ax2.set_xticklabels(country_percentages.index, rotation=90, fontsize=FONT_SIZE-2)
ax2.tick_params(axis='y', which='major', labelsize=10)

# Add percentage signs to Y-axis tick labels
ax2.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f'{y:.0f}%'))

# Add labels and title for the right subplot
# ax2.set_ylabel("Percentage of Google\nData Centers Located\nIn Nation (%)", fontsize=13)
ax2.set_title("Global Distribution of (non-AI) Google Data Centers", fontsize=FONT_SIZE)

# Adjust layout for better display
plt.tight_layout()
# Add bottom padding to accommodate the larger x-axis labels and reduce space between plots
plt.subplots_adjust(bottom=0.35, wspace=0.07)

# Save the combined plot
plt.savefig('combined_distribution_plots.png', dpi=300, bbox_inches='tight')
plt.show()
