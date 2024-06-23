import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# Read training and validation set data
train_data = pd.read_csv('train.csv')
test_data = pd.read_csv('test.csv', usecols=['key', 'pickup_datetime', 'pickup_longitude', 'pickup_latitude',
                                              'dropoff_longitude', 'dropoff_latitude', 'passenger_count'])

# Check for missing values
print("Training set missing value checking:")
print(train_data.isnull().sum())
print("Validation set missing value checking:")
print(test_data.isnull().sum())

# Remove rows containing missing values
train_data = train_data.dropna()
test_data = test_data.dropna()

# Remove rows containing 0 values
columns_to_check = ['pickup_longitude', 'pickup_latitude', 'dropoff_longitude', 'dropoff_latitude', 'passenger_count']
train_data = train_data[(train_data[columns_to_check] != 0).all(axis=1)]
test_data = test_data[(test_data[columns_to_check] != 0).all(axis=1)]

# Check for incorrectly formatted rows
def is_valid_row(row):
    try:
        float(row['pickup_longitude'])
        float(row['pickup_latitude'])
        float(row['dropoff_longitude'])
        float(row['dropoff_latitude'])
        return True
    except ValueError:
        return False

# Remove incorrectly formatted rows
train_data = train_data[train_data.apply(is_valid_row, axis=1)]
test_data = test_data[test_data.apply(is_valid_row, axis=1)]

# Define the longitude and latitude range of New York City
min_longitude = -74.25
max_longitude = -73.70
min_latitude = 40.50
max_latitude = 40.92

# Remove rows with longitude or latitude outside of New York City limits
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

# Perform outlier removal for 'fare_amount' and 'passenger_count'
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
