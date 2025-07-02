import matplotlib.pyplot as plt
import pandas as pd

# AI investment data by country (in billions of dollars)
countries = [
    "United States", "China", "United Kingdom", "Canada", "Israel", 
    "Germany", "India", "France", "South Korea", "Singapore", 
    "Sweden", "Japan", "Australia", "Switzerland", "United Arab Emirates"
]

investment_amounts = [
    470.92, 119.32, 28.17, 15.31, 14.96, 
    13.27, 11.29, 11.10, 8.96, 7.27, 
    7.27, 5.89, 3.99, 3.90, 3.67
]

# Create a DataFrame for easier manipulation
data = pd.DataFrame({
    'Country': countries,
    'Investment': investment_amounts
})

# Calculate percentage of total investment
total_investment = data['Investment'].sum()
data['Percentage'] = (data['Investment'] / total_investment) * 100

# Set consistent font size for all text elements
FONT_SIZE = 16
plt.rcParams.update({'font.size': FONT_SIZE})

# Plotting
plt.figure(figsize=(8, 6))  # Match the Google data centers plot dimensions
plt.bar(data['Country'], data['Percentage'], color='skyblue', edgecolor='none', width=0.7)  # No outlines around bars, narrower width

# Format x-axis labels
plt.xticks(rotation=90, fontsize=FONT_SIZE-2)  # Larger x-axis labels
plt.yticks(fontsize=10)  # Consistent y-axis label size

# Add labels and title
plt.ylabel("Percentage of Global AI\nInvestment (%)", fontsize=13)
plt.title("Global Distribution of AI Investment in 2024", fontsize=FONT_SIZE * 1.3)

# Add value labels on top of each bar
# for i, p in enumerate(data['Percentage']):
#     plt.text(i, p + 0.5, f"${data['Investment'][i]:.1f}B", 
#              ha='center', fontsize=FONT_SIZE-4)

# Adjust layout for better display
plt.tight_layout()
# Add bottom padding to accommodate the larger x-axis labels
plt.subplots_adjust(bottom=0.40)
plt.show()

# Save the plot
# """