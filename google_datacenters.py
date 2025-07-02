import matplotlib.pyplot as plt
import pandas as pd

# List of countries for each Google data center (as provided)
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

# Sort by percentage descending
country_percentages = country_percentages.sort_values(ascending=False)

# Set consistent font size for all text elements
FONT_SIZE = 16
plt.rcParams.update({'font.size': FONT_SIZE})

# Plotting
plt.figure(figsize=(12, 5))  # Less tall, wider aspect ratio
plt.bar(country_percentages.index, country_percentages.values, color='skyblue', edgecolor='none')  # No outlines around bars
plt.xticks(rotation=90, fontsize=FONT_SIZE-2)  # Larger x-axis labels
plt.yticks(fontsize=10)  # Consistent y-axis label size
plt.ylabel("Percentage of Google\nData Centers Located\nIn Nation (%)", fontsize=13)
plt.title("Global distribution of Google Datacenters", fontsize=FONT_SIZE *1.3)

# Add more bottom padding to accommodate the larger x-axis labels
plt.subplots_adjust(bottom=0.40)

# Save and show the plot
plt.savefig('google_datacenters_distribution.png', dpi=300, bbox_inches='tight')
plt.show()