import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt

# Read pre-processed data
train_processed = pd.read_csv('train_processed.csv')

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

# training model
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
plt.savefig('actual_vs_predicted.png')
plt.show()

# Printing actual and predicted values
actual_vs_predicted = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
print('Actual vs. Predicted Values:')
print(actual_vs_predicted.head(10))

# residual plot
residuals = y_test - y_pred
plt.figure(figsize=(8, 6))
plt.scatter(y_pred, residuals)
plt.xlabel('Predicted Values')
plt.ylabel('Residuals')
plt.title('Residual Plot')
plt.axhline(y=0, color='r', linestyle='--')
plt.tight_layout()
plt.savefig('residual_plot.png')
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
