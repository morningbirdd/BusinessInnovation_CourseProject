import pandas as pd
import matplotlib.pyplot as plt

# Read the data file containing prediction results
test_data = pd.read_csv('test_predictions_xgboost.csv')

# Define trip distance ranges and labels
distance_ranges = [0, 5, 15,  25, float('inf')]
labels = ['<5', '5-15', '15-25', '25+']

# Group samples based on trip distance ranges
test_data['distance_group'] = pd.cut(test_data['trip_distance'], bins=distance_ranges, labels=labels)

# Calculate the number of samples in each trip distance range
group_counts = test_data['distance_group'].value_counts()

# Define colors for the pie chart
colors = ['#ff9999', '#66b3ff', '#99ff99', '#ffcc99', '#c2c2f0']

# Create the pie chart
fig, ax = plt.subplots(figsize=(10, 10))
ax.pie(group_counts, labels=group_counts.index, autopct='%1.1f%%', colors=colors, textprops={'fontsize': 14})
ax.set_title('Trip Distance Distribution', fontsize=18)
ax.axis('equal')  # Ensure the pie chart is circular

# Add a legend
plt.legend(labels=labels, loc='upper right', bbox_to_anchor=(1.2, 0.9), fontsize=14)

# Display the plot
plt.tight_layout()
plt.show()
