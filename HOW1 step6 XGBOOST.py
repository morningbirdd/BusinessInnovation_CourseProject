import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

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