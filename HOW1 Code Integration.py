import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from scipy.stats import spearmanr
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.tree import DecisionTreeRegressor, export_graphviz
from graphviz import Source
from xgboost import XGBRegressor
import shap
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

# Step 1: Data Preprocessing
# Read training and validation set data
train_data = pd.read_csv('train.csv')
test_data = pd.read_csv('test.csv', usecols=['key', 'pickup_datetime', 'pickup_longitude', 'pickup_latitude',
                                              'dropoff_longitude', 'dropoff_latitude', 'passenger_count'])

# Check for missing values and remove rows containing missing values
print("Training set missing value checking:")
print(train_data.isnull().sum())
print("Validation set missing value checking:")
print(test_data.isnull().sum())
train_data = train_data.dropna()
test_data = test_data.dropna()

# Remove rows containing 0 values
columns_to_check = ['pickup_longitude', 'pickup_latitude', 'dropoff_longitude', 'dropoff_latitude', 'passenger_count']
train_data = train_data[(train_data[columns_to_check] != 0).all(axis=1)]
test_data = test_data[(test_data[columns_to_check] != 0).all(axis=1)]

# Check for incorrectly formatted rows and remove them
def is_valid_row(row):
    try:
        float(row['pickup_longitude'])
        float(row['pickup_latitude'])
        float(row['dropoff_longitude'])
        float(row['dropoff_latitude'])
        return True
    except ValueError:
        return False

train_data = train_data[train_data.apply(is_valid_row, axis=1)]
test_data = test_data[test_data.apply(is_valid_row, axis=1)]

# Remove rows with longitude or latitude outside of New York City limits
min_longitude = -74.25
max_longitude = -73.70
min_latitude = 40.50
max_latitude = 40.92

train_data = train_data[
    (train_data['pickup_longitude'] >= min_longitude) & (train_data['pickup_longitude'] <= max_longitude) &
    (train_data['dropoff_longitude'] >= min_longitude) & (train_data['dropoff_longitude'] <= max_longitude) &
    (train_data['pickup_latitude'] >= min_latitude) & (train_data['pickup_latitude'] <= max_latitude) &
    (train_data['dropoff_latitude'] >= min_latitude) & (train_data['dropoff_latitude'] <= max_latitude)
]

test_data = test_data[
    (test_data['pickup_longitude'] >= min_longitude) & (test_data['pickup_longitude'] <= max_longitude) &
    (test_data['dropoff_longitude'] >= min_longitude) & (test_data['dropoff_longitude'] <= max_longitude) &
    (test_data['pickup_latitude'] >= min_latitude) & (test_data['pickup_latitude'] <= max_latitude) &
    (test_data['dropoff_latitude'] >= min_latitude) & (test_data['dropoff_latitude'] <= max_latitude)
]

# Detect and remove outliers using boxplot
def remove_outliers_using_boxplot(data, column):
    q1 = data[column].quantile(0.25)
    q3 = data[column].quantile(0.75)
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    outliers = data[(data[column] < lower_bound) | (data[column] > upper_bound)]
    data = data[(data[column] >= lower_bound) & (data[column] <= upper_bound)]
    return data, outliers

train_data, fare_amount_outliers = remove_outliers_using_boxplot(train_data, 'fare_amount')
train_data, passenger_count_outliers = remove_outliers_using_boxplot(train_data, 'passenger_count')
test_data, _ = remove_outliers_using_boxplot(test_data, 'passenger_count')

# Extract features
train_data['pickup_datetime'] = pd.to_datetime(train_data['pickup_datetime'])
train_data['pickup_year'] = train_data['pickup_datetime'].dt.year
train_data['pickup_month'] = train_data['pickup_datetime'].dt.month
train_data['pickup_day'] = train_data['pickup_datetime'].dt.day
train_data['pickup_hour'] = train_data['pickup_datetime'].dt.hour
train_data['pickup_minute'] = train_data['pickup_datetime'].dt.minute
train_data['pickup_second'] = train_data['pickup_datetime'].dt.second

test_data['pickup_datetime'] = pd.to_datetime(test_data['pickup_datetime'])
test_data['pickup_year'] = test_data['pickup_datetime'].dt.year
test_data['pickup_month'] = test_data['pickup_datetime'].dt.month
test_data['pickup_day'] = test_data['pickup_datetime'].dt.day
test_data['pickup_hour'] = test_data['pickup_datetime'].dt.hour
test_data['pickup_minute'] = test_data['pickup_datetime'].dt.minute
test_data['pickup_second'] = test_data['pickup_datetime'].dt.second

# Calculate trip distance
def haversine_distance(lon1, lat1, lon2, lat2):
    R = 6371  # Earth radius in kilometers
    dlon = np.radians(lon2 - lon1)
    dlat = np.radians(lat2 - lat1)
    a = np.sin(dlat/2)**2 + np.cos(np.radians(lat1)) * np.cos(np.radians(lat2)) * np.sin(dlon/2)**2
    c = 2 * np.arcsin(np.sqrt(a))
    return R * c

train_data['trip_distance'] = haversine_distance(train_data['pickup_longitude'], train_data['pickup_latitude'],
                                                 train_data['dropoff_longitude'], train_data['dropoff_latitude'])
test_data['trip_distance'] = haversine_distance(test_data['pickup_longitude'], test_data['pickup_latitude'],
                                                test_data['dropoff_longitude'], test_data['dropoff_latitude'])

# Save preprocessed training and validation set data
train_processed = train_data.drop(['key', 'pickup_datetime'], axis=1)
test_processed = test_data.drop(['key', 'pickup_datetime'], axis=1)
train_processed.to_csv('train_processed.csv', index=False)
test_processed.to_csv('test_processed.csv', index=False)

# Step 2: Feature Engineering
# Read pre-processed data
train_data = pd.read_csv('train_processed.csv')
print("Data loaded. Shape:", train_data.shape)

# Calculate travel distance
train_data['trip_distance'] = haversine_distance(train_data['pickup_longitude'], train_data['pickup_latitude'],
                                                 train_data['dropoff_longitude'], train_data['dropoff_latitude'])

# Split the data into training and testing sets
X = train_data.drop(['fare_amount'], axis=1)
y = train_data['fare_amount']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 3: Correlation Analysis (Spearman)
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
    plt.savefig(f"results/spearman_correlation_plot.png")
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
    plt.savefig(f"results/heatmap.png", dpi=300, bbox_inches='tight')
    plt.show()

# Directory to save the results
result_dir = 'results'

# Check if the directory exists, if not, create it
if not os.path.exists(result_dir):
    os.makedirs(result_dir)

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

# Step 4: Multiple Linear Regression Analysis (MLRA)
# Selection of characteristics and target variables
features = ['pickup_longitude', 'pickup_latitude', 'dropoff_longitude', 
            'dropoff_latitude', 'passenger_count', 'trip_distance',
            'pickup_year','pickup_month','pickup_day','pickup_hour',
            'pickup_minute','pickup_second']
target = 'fare_amount'

# Divide the training set and test set
X = train_processed[features]
y = train_processed[target]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Creating multiple linear regression models
model = LinearRegression()

# Training model
model.fit(X_train, y_train)

# Output regression coefficients and intercepts
coefficients = pd.DataFrame({'Feature': features, 'Coefficient': model.coef_})
print('Coefficients:')
print(coefficients)
print('Intercept:', model.intercept_)

# Scatterplot of actual vs. predicted values
y_pred = model.predict(X_test)
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred)
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.title('Actual vs. Predicted Values')
plt.tight_layout()
plt.savefig('result/actual_vs_predicted.png')
plt.show()

# Printing actual and predicted values
actual_vs_predicted = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
print('Actual vs. Predicted Values:')
print(actual_vs_predicted.head(10))

# Residual plot
residuals = y_test - y_pred
plt.figure(figsize=(8, 6))
plt.scatter(y_pred, residuals)
plt.xlabel('Predicted Values')
plt.ylabel('Residuals')
plt.title('Residual Plot')
plt.axhline(y=0, color='r', linestyle='--')
plt.tight_layout()
plt.savefig('result/residual_plot.png')
plt.show()

# Print Residuals
residual_df = pd.DataFrame({'Predicted': y_pred, 'Residual': residuals})
print('Residuals:')
print(residual_df.head(10))

# Calculation of assessment indicators
mse = mean_squared_error(y_test, y_pred)
rmse = mean_squared_error(y_test, y_pred, squared=False)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Squared Error (MSE): {mse:.4f}")
print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")
print(f"Mean Absolute Error (MAE): {mae:.4f}")
print(f"R-squared (R2): {r2:.4f}")

# Step 5: Decision Tree Regression
# Load the preprocessed data
train_processed = pd.read_csv('train_processed.csv')
print("Data loaded. Shape:", train_processed.shape)

# Select features and target variable
features = ['pickup_longitude', 'pickup_latitude', 'dropoff_longitude',
            'dropoff_latitude', 'passenger_count', 'trip_distance',
            'pickup_year','pickup_month','pickup_day','pickup_hour',
            'pickup_minute','pickup_second']
target = 'fare_amount'
print("Features:", features)
print("Target:", target)

# Split the data into training and testing sets
X = train_processed[features]
y = train_processed[target]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print("Train set shape:", X_train.shape, y_train.shape)
print("Test set shape:", X_test.shape, y_test.shape)

# Create a decision tree regression model with the best parameters
best_params = {'max_depth': 9, 'min_samples_leaf': 4, 'min_samples_split': 10, 'random_state': 42}
dt_model = DecisionTreeRegressor(**best_params)
print("Decision Tree model created with best parameters.")

# Train the model
dt_model.fit(X_train, y_train)
print("Model trained.")

# Visualize the decision tree
dot_data = export_graphviz(dt_model, out_file=None, 
                           feature_names=features,  
                           filled=True, rounded=True,  
                           special_characters=True)
graph = Source(dot_data)
graph.render("decision_tree", format="png")
print("Decision tree visualization saved as 'decision_tree.png'.")

# Make predictions on the test set
y_pred = dt_model.predict(X_test)
print("Predictions made.")

# Calculate evaluation metrics
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Squared Error (MSE): {mse:.4f}")
print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")
print(f"Mean Absolute Error (MAE): {mae:.4f}")
print(f"R-squared (R2): {r2:.4f}")

# Feature importances
feature_importances = pd.DataFrame({'Feature': features, 'Importance': dt_model.feature_importances_})
print("Feature Importances:")
print(feature_importances)

# Plot feature importances
plt.figure(figsize=(10, 6))
plt.bar(features, dt_model.feature_importances_)
plt.xlabel('Features')
plt.ylabel('Importance')
plt.title('Feature Importances')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('result/decision_tree_feature_importances.png')
plt.show()

# Scatter plot of actual vs. predicted values
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred)
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.title('Actual vs. Predicted Values')
plt.tight_layout()
plt.savefig('result/decision_tree_actual_vs_predicted.png')
plt.show()

# Residual plot
residuals = y_test - y_pred
plt.figure(figsize=(8, 6))
plt.scatter(y_pred, residuals)
plt.xlabel('Predicted Values')
plt.ylabel('Residuals')
plt.title('Residual Plot')
plt.axhline(y=0, color='r', linestyle='--')
plt.tight_layout()
plt.savefig('result/decision_tree_residual_plot.png')
plt.show()


# Step 6: XGBoost Regression
# Read the preprocessed data
train_processed = pd.read_csv('train_processed.csv')
print("Data loaded. Shape:", train_processed.shape)

# Selection of characteristics and target variables
features = ['pickup_longitude', 'pickup_latitude', 'dropoff_longitude', 'dropoff_latitude', 'passenger_count', 'trip_distance','pickup_year','pickup_month','pickup_day','pickup_hour','pickup_minute','pickup_second']
target = 'fare_amount'
print("Features:", features)
print("Target:", target)

# Delineate training and test sets
X = train_processed[features]
y = train_processed[target]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print("Train set shape:", X_train.shape, y_train.shape)
print("Test set shape:", X_test.shape, y_test.shape)

# Setting the parameter distribution for random searches
param_dist = {
    'n_estimators': [50, 100, 200],
    'learning_rate': [0.01, 0.1, 0.2],
    'max_depth': [3, 5, 7],
    'subsample': [0.8, 1.0],
    'colsample_bytree': [0.8, 1.0],
    'random_state': [42]
}

# Create an XGBoost model
xgb_model = XGBRegressor()

# Conduct random searches and cross-validation
random_search = RandomizedSearchCV(estimator=xgb_model, param_distributions=param_dist, n_iter=20, cv=3, n_jobs=1, verbose=2)
random_search.fit(X_train, y_train)

# Output optimal parameters
print("Best parameters found:")
print(random_search.best_params_)

# Create XGBoost models using optimal parameters
best_xgb_model = random_search.best_estimator_
print("Best XGBoost model created.")

# Predictions on test sets
y_pred = best_xgb_model.predict(X_test)
print("Predictions made.")

# Calculate assessment indicators
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Squared Error (MSE): {mse:.4f}")
print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")
print(f"Mean Absolute Error (MAE): {mae:.4f}")
print(f"R-squared (R2): {r2:.4f}")

# Scatter plot of actual vs. predicted values
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred)
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.title('Actual vs. Predicted Values')
plt.tight_layout()
plt.savefig('result/xgb_actual_vs_predicted.png')
plt.show()

# Residual plots
residuals = y_test - y_pred
plt.figure(figsize=(8, 6))
plt.scatter(y_pred, residuals)
plt.xlabel('Predicted Values')
plt.ylabel('Residuals')
plt.title('Residual Plot')
plt.axhline(y=0, color='r', linestyle='--')
plt.tight_layout()
plt.savefig('result/xgb_residual_plot.png')
plt.show()

# Characterization importance
feature_importances = pd.DataFrame({'Feature': features, 'Importance': best_xgb_model.feature_importances_})
print("Feature Importances:")
print(feature_importances)

# Mapping the significance of features
plt.figure(figsize=(10, 6))
plt.bar(features, best_xgb_model.feature_importances_)
plt.xlabel('Features')
plt.ylabel('Importance')
plt.title('Feature Importances')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('result/xgb_feature_importances.png')
plt.show()

# Step 7: XGBoost Prediction
# Read the preprocessed training data
train_processed = pd.read_csv('train_processed.csv')
print("Training data loaded. Shape:", train_processed.shape)

# Select features and target variable
features = ['pickup_longitude', 'pickup_latitude', 'dropoff_longitude', 'dropoff_latitude', 'passenger_count', 'trip_distance', 'pickup_year', 'pickup_month', 'pickup_day', 'pickup_hour', 'pickup_minute', 'pickup_second']
target = 'fare_amount'
print("Features:", features)
print("Target:", target)

# Split features and target variable
X_train = train_processed[features]
y_train = train_processed[target]

# Create XGBoost model with the best parameters
best_params = {'subsample': 1.0, 'random_state': 42, 'n_estimators': 200, 'max_depth': 7, 'learning_rate': 0.2, 'colsample_bytree': 0.8}
xgb_model = XGBRegressor(**best_params)
print("XGBoost model created with best parameters.")

# Train the model
xgb_model.fit(X_train, y_train)
print("Model trained.")

# Read the validation set data
test_data = pd.read_csv('test_processed.csv')
print("Validation data loaded. Shape:", test_data.shape)

# Select features for the validation set
X_test = test_data[features]

# Make predictions on the validation set
y_pred = xgb_model.predict(X_test)
print("Predictions made.")

# Add the predicted results to the validation set dataframe
test_data['fare_amount'] = y_pred

# Save the predicted results to a file
output_file = 'test_predictions_xgboost.csv'
test_data.to_csv(output_file, index=False)
print(f"Predictions saved to {output_file}.")

# Step 8: User Group Segmentation
# Read the data file containing prediction results
test_data = pd.read_csv('test_predictions_xgboost.csv')

# Define trip distance ranges and labels
distance_ranges = [0, 5, 15, 25, float('inf')]
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

# Step 9: SHAP Analysis
# Train the XGBoost model
xgb_model = XGBRegressor(**best_params)
xgb_model.fit(X_train, y_train)

# Create the SHAP explainer
explainer = shap.TreeExplainer(xgb_model)
shap_values = explainer.shap_values(X_test)

# Calculate global feature importance
global_importance = pd.DataFrame(list(zip(X_test.columns, np.abs(shap_values).mean(0))), 
                                 columns=['Feature', 'Importance'])
global_importance = global_importance.sort_values('Importance', ascending=False)
print("Global Feature Importance:")
print(global_importance)

# SHAP value visualization
shap.summary_plot(shap_values, X_test, plot_type="bar")

# Summary plot with all sample points
shap.summary_plot(shap_values, X_test)

# Dependence plot for 'trip_distance' and 'pickup_hour' and 'passenger_count' interaction
shap.dependence_plot('trip_distance', shap_values, X_test, display_features=X_test, interaction_index='pickup_hour')
