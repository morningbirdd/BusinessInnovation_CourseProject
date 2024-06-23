import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor, export_graphviz
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
from graphviz import Source

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
