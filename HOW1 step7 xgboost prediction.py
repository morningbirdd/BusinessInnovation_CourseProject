import pandas as pd
import numpy as np
from xgboost import XGBRegressor

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

# Create XGBoost model with best parameters
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
