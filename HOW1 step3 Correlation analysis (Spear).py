import pandas as pd
import numpy as np
import os
from scipy.stats import spearmanr
import matplotlib.pyplot as plt
import seaborn as sns

# Read the data
train_processed = pd.read_csv('train_processed.csv')

# Select the feature columns to analyze
features = ['pickup_longitude', 'pickup_latitude', 'dropoff_longitude', 'dropoff_latitude', 'passenger_count', 'trip_distance', 
            'pickup_year','pickup_month','pickup_day','pickup_hour','pickup_minute','pickup_second','fare_amount']
print("Selected feature columns:", features)

# Extract the data of selected feature columns
df_summary = train_processed[features]
print("Extracted feature data:")
print(df_summary.head())

# Calculate Spearman's rank correlation coefficient
def spearman_correlation(df_summary, target_col):
    # Remove the target column
    features_without_target = [col for col in df_summary.columns if col != target_col]
    corr_data = []
    for feature in features_without_target:
        corr, _ = spearmanr(df_summary[feature], df_summary[target_col])
        corr_data.append((feature, corr))
    
    corr_data.sort(key=lambda x: abs(x[1]), reverse=True)
    
    print(f"{target_col}'s Spearman rank correlation coefficient:")
    for feature, corr in corr_data:
        print(f"{feature}: {corr}")
    
    print("\nSorted correlation coefficients:")
    for feature, corr in corr_data:
        print(f"{feature}: {corr}")
    
    # Create a DataFrame for visualization
    corr_df = pd.DataFrame(corr_data, columns=['Feature', 'Correlation'])
    
    # Create a bar plot
    plt.figure(figsize=(10, 6))
    sns.barplot(x='Correlation', y='Feature', data=corr_df)
    plt.title(f"Spearman Rank Correlation with {target_col}")
    plt.xlabel('Correlation Coefficient')
    plt.ylabel('Feature')
    plt.tight_layout()
    plt.savefig(f"{result_dir}/spearman_correlation_plot.png")
    plt.show()
    
    return corr_data

# Heat map function
def heat_map(df_summary):
    df_heat = df_summary.apply(lambda x: x.astype(float))
    heat_data = df_heat.corr()
    print(heat_data)
    plt.figure(figsize=(12, 10))
    sns.set(font_scale=0.8)  
    sns.heatmap(heat_data, annot=True, fmt=".2f", cmap="coolwarm", cbar_kws={"shrink": 0.8})
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(f"{result_dir}/heatmap.png", dpi=300, bbox_inches='tight')
    plt.show()


# Mapping table
column_mapping = {
    'pickup_longitude': 'Pickup Longitude',
    'pickup_latitude': 'Pickup Latitude',
    'dropoff_longitude': 'Dropoff Longitude',
    'dropoff_latitude': 'Dropoff Latitude',
    'passenger_count': 'Passenger Count',
    'trip_distance': 'Trip Distance',
    'fare_amount': 'Fare Amount'
}
print(column_mapping)

# Directory to save the results
result_dir = 'results'


# Run the correlation analysis
print("Starting correlation analysis...")
corr_results = spearman_correlation(df_summary, 'fare_amount')

# Save the correlation results to a file
with open(f"{result_dir}/spearman_correlation_results.txt", 'w') as file:
    file.write("Spearman's Rank Correlation Coefficients:\n")
    for feature, corr in corr_results:
        file.write(f"{feature}: {corr}\n")

print("Correlation analysis completed. Results saved to file.")

# Generate heat map
print("Generating heat map...")
heat_map(df_summary)
print("Heat map generated and saved.")
