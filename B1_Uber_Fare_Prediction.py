import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# 1. Load the Dataset
df = pd.read_csv('uber.csv')
print(df.head())
print(df.shape)
df.info()
print(df.describe())

# 2. Pre-processing
# Drop columns that carry no predictive information
df = df.drop(['Unnamed: 0', 'key'], axis=1)
print(df.isnull().sum())

# Fill the single missing drop-off coordinate with the column mean
df['dropoff_latitude'] = df['dropoff_latitude'].fillna(df['dropoff_latitude'].mean())
df['dropoff_longitude'] = df['dropoff_longitude'].fillna(df['dropoff_longitude'].mean())
print(df.isnull().sum())

# Convert pickup_datetime to datetime and extract useful time features
df['pickup_datetime'] = pd.to_datetime(df['pickup_datetime'], errors='coerce')
df = df.assign(hour=df.pickup_datetime.dt.hour,
               day=df.pickup_datetime.dt.day,
               month=df.pickup_datetime.dt.month,
               year=df.pickup_datetime.dt.year,
               dayofweek=df.pickup_datetime.dt.dayofweek)
df = df.drop('pickup_datetime', axis=1)
print(df.head())

# Haversine distance (km) between pickup and drop-off
def haversine(lon1, lat1, lon2, lat2):
    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])
    dlon, dlat = lon2 - lon1, lat2 - lat1
    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    return 2 * 6371 * np.arcsin(np.sqrt(a))

df['distance_km'] = haversine(df['pickup_longitude'], df['pickup_latitude'],
                              df['dropoff_longitude'], df['dropoff_latitude'])
print(df[['distance_km']].describe())

# 3. Identify Outliers
cols = ['fare_amount', 'distance_km', 'passenger_count',
        'pickup_longitude', 'pickup_latitude', 'dropoff_longitude', 'dropoff_latitude']
fig, axes = plt.subplots(2, 4, figsize=(16, 7))
for ax, c in zip(axes.flatten(), cols):
    sns.boxplot(y=df[c], ax=ax)
    ax.set_title(c)
axes.flatten()[-1].axis('off')
plt.suptitle('Box plots before outlier removal')
plt.tight_layout(); plt.show()

# Remove impossible values first (domain rules)
df = df[(df.fare_amount > 0) &
        (df.passenger_count > 0) & (df.passenger_count <= 6) &
        (df.pickup_latitude.between(-90, 90)) & (df.dropoff_latitude.between(-90, 90)) &
        (df.pickup_longitude.between(-180, 180)) & (df.dropoff_longitude.between(-180, 180)) &
        (df.distance_km > 0)].copy()

# Then treat remaining outliers with the IQR method (cap at the whiskers)
def remove_outlier_iqr(df, col):
    Q1, Q3 = df[col].quantile(0.25), df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower, upper = Q1 - 1.5 * IQR, Q3 + 1.5 * IQR
    df[col] = np.clip(df[col], lower, upper)
    return df

for c in ['fare_amount', 'distance_km', 'pickup_longitude', 'pickup_latitude',
          'dropoff_longitude', 'dropoff_latitude']:
    df = remove_outlier_iqr(df, c)

print("Shape after cleaning:", df.shape)

fig, axes = plt.subplots(2, 4, figsize=(16, 7))
for ax, c in zip(axes.flatten(), cols):
    sns.boxplot(y=df[c], ax=ax)
    ax.set_title(c)
axes.flatten()[-1].axis('off')
plt.suptitle('Box plots after outlier treatment')
plt.tight_layout(); plt.show()

# 4. Check the Correlation
corr = df.corr()
print(corr)

plt.figure(figsize=(12, 8))
sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm')
plt.title('Correlation Matrix')
plt.show()

print(corr['fare_amount'].sort_values(ascending=False))

# 5. Train-Test Split
from sklearn.model_selection import train_test_split

X = df.drop('fare_amount', axis=1)
y = df['fare_amount']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print("Train:", X_train.shape, " Test:", X_test.shape)

# 6. Linear Regression
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

lr = LinearRegression()
lr.fit(X_train, y_train)
y_pred_lr = lr.predict(X_test)

r2_lr   = r2_score(y_test, y_pred_lr)
rmse_lr = np.sqrt(mean_squared_error(y_test, y_pred_lr))
mae_lr  = mean_absolute_error(y_test, y_pred_lr)
print(f"Linear Regression -> R2: {r2_lr:.4f} | RMSE: {rmse_lr:.4f} | MAE: {mae_lr:.4f}")

# 7. Random Forest Regression
from sklearn.ensemble import RandomForestRegressor

rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)

r2_rf   = r2_score(y_test, y_pred_rf)
rmse_rf = np.sqrt(mean_squared_error(y_test, y_pred_rf))
mae_rf  = mean_absolute_error(y_test, y_pred_rf)
print(f"Random Forest     -> R2: {r2_rf:.4f} | RMSE: {rmse_rf:.4f} | MAE: {mae_rf:.4f}")

# 8. Model Comparison
results = pd.DataFrame({
    'Model': ['Linear Regression', 'Random Forest'],
    'R2 Score': [r2_lr, r2_rf],
    'RMSE': [rmse_lr, rmse_rf],
    'MAE': [mae_lr, mae_rf]
})
print(results)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
for ax, pred, name in zip(axes, [y_pred_lr, y_pred_rf], ['Linear Regression', 'Random Forest']):
    ax.scatter(y_test[:2000], pred[:2000], alpha=0.3, s=8)
    ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
    ax.set_xlabel('Actual fare'); ax.set_ylabel('Predicted fare'); ax.set_title(name)
plt.tight_layout(); plt.show()
