import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from scipy.stats import spearmanr

# Read pre-processed data
train_data = pd.read_csv('train_processed.csv')
print("Data loaded. Shape:", train_data.shape)

#Calculate travel distance
def haversine_distance(lon1, lat1, lon2, lat2):
    R = 6371  # Radius of the Earth in kilometers
    dlon = np.radians(lon2 - lon1)
    dlat = np.radians(lat2 - lat1)
    a = np.sin(dlat/2)**2 + np.cos(np.radians(lat1)) * np.cos(np.radians(lat2)) * np.sin(dlon/2)**2
    c = 2 * np.arcsin(np.sqrt(a))
    return R * c

train_data['trip_distance'] = haversine_distance(train_data['pickup_longitude'], train_data['pickup_latitude'],
                                                 train_data['dropoff_longitude'], train_data['dropoff_latitude'])

# 划分训练集和测试集
X = train_data.drop(['key', 'pickup_datetime', 'fare_amount'], axis=1)
y = train_data['fare_amount']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# # 保存预处理后的数据
# train_processed = pd.concat([X_train, y_train], axis=1)
# test_processed = pd.concat([X_test, y_test], axis=1)
# train_processed.to_csv('train_processed.csv', index=False)
# test_processed.to_csv('test_processed.csv', index=False)

train_processed = pd.read_csv('train_processed.csv')
# 选择要分析的特征列
features = ['pickup_longitude', 'pickup_latitude', 'dropoff_longitude', 'dropoff_latitude', 
            'passenger_count', 'trip_distance', 'pickup_year', 'pickup_month', 'pickup_day', 
            'pickup_hour', 'pickup_minute', 'pickup_second', 'fare_amount']
print("选择的特征列:", features)

# 提取所选特征列的数据
df_summary = train_processed[features]
print("提取的特征数据:")
print(df_summary.head())

# 计算斯皮尔曼等级相关系数
def spearman_correlation(df_summary, target_col):
    print(f"开始计算与 {target_col} 的斯皮尔曼等级相关系数...")
    features_without_target = [col for col in df_summary.columns if col != target_col]
    corr_data = []
    
    for feature in features_without_target:
        corr, _ = spearmanr(df_summary[feature], df_summary[target_col])
        corr_data.append((feature, corr))
    
    corr_data.sort(key=lambda x: abs(x[1]), reverse=True)
    
    print(f"与 {target_col} 的斯皮尔曼等级相关系数:")
    for feature, corr in corr_data:
        print(f"{feature}: {corr}")
    
    print("\n排序后的相关性系数:")
    for feature, corr in corr_data:
        print(f"{feature}: {corr}")
    
    return corr_data

# 映射表
column_mapping = {
    'pickup_longitude': 'Pickup Longitude',
    'pickup_latitude': 'Pickup Latitude',
    'dropoff_longitude': 'Dropoff Longitude',
    'dropoff_latitude': 'Dropoff Latitude',
    'passenger_count': 'Passenger Count',
    'trip_distance': 'Trip Distance',
    'pickup_year': 'Pickup Year',
    'pickup_month': 'Pickup Month',
    'pickup_day': 'Pickup Day',
    'pickup_hour': 'Pickup Hour',
    'pickup_minute': 'Pickup Minute',
    'pickup_second': 'Pickup Second',
    'fare_amount': 'Fare Amount'
}
print(column_mapping)

# 结果保存目录
result_dir = 'results'

# 检查目录是否存在,如果不存在则创建
if not os.path.exists(result_dir):
    os.makedirs(result_dir)

# 运行相关性分析
print("开始运行相关性分析...")
corr_results = spearman_correlation(df_summary, 'fare_amount')

# 将相关性结果保存到文件
with open(f"{result_dir}/spearman_correlation_results.txt", 'w') as file:
    file.write("Spearman's Rank Correlation Coefficients:\n")
    for feature, corr in corr_results:
        file.write(f"{feature}: {corr}\n")

print("相关性分析完成。结果已保存到文件。")
