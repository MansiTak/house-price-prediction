import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import root_mean_squared_error, mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor
# from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np
import joblib

# Step 1: Load the dataset
data = pd.read_csv('train.csv')

# Display first few rows and dataset information
print("First 5 rows of the dataset:")
print(data.head())
print("\nDataset Info:")
print(data.info())

# Step 2: Handle missing values
# Check for missing values
missing_values = data.isnull().sum()
print("\nMissing values in each column:")
print(missing_values[missing_values > 0])
print(data.select_dtypes(include='object').columns.tolist())
for col in data.select_dtypes(include='object').columns:
    # print(f"{col}: {data[col].nunique()} unique values")
    print(col, data[col].unique())
# Common mapping for ordinal columns
quality_map = {'Ex': 5, 'Gd': 4, 'TA': 3, 'Fa': 2, 'Po': 1, None: 0}

# Apply to ordinal columns
ordinal_cols = ['ExterQual', 'ExterCond', 'BsmtQual', 'BsmtCond', 'HeatingQC',
                'KitchenQual', 'FireplaceQu', 'GarageQual', 'GarageCond', 'PoolQC']

for col in ordinal_cols:
    data[col] = data[col].map(quality_map)

# Custom map for Fence
fence_map = {'GdPrv': 4, 'MnPrv': 3, 'GdWo': 2, 'MnWw': 1, None: 0}
data['Fence'] = data['Fence'].map(fence_map)

# Label Encoding for binary columns
binary_cols = ['CentralAir', 'Alley']
for col in binary_cols:
    data[col] = data[col].map({'Y': 1, 'N': 0, None: 0})  # If there's None, consider as 0

# One-Hot Encoding for nominal features
nominal_cols = ['MSZoning', 'Street', 'LotShape', 'LandContour', 'Utilities', 
                'LotConfig', 'LandSlope', 'Neighborhood', 'Condition1', 'Condition2', 
                'BldgType', 'HouseStyle', 'RoofStyle', 'RoofMatl', 'Exterior1st', 
                'Exterior2nd', 'MasVnrType', 'Foundation', 'BsmtExposure', 'BsmtFinType1', 
                'BsmtFinType2', 'Heating', 'Electrical', 'Functional', 'GarageType', 
                'GarageFinish', 'PavedDrive', 'MiscFeature', 'SaleType', 'SaleCondition']

data = pd.get_dummies(data, columns=nominal_cols, drop_first=True)



# Assuming 'data' is your preprocessed DataFrame

# 1. Handle missing values (fill NaN with the most frequent value for categorical and median for numerical)
data.fillna(data.mode().iloc[0], inplace=True)  # Fill categorical columns
data.fillna(data.median(), inplace=True)  # Fill numerical columns (if needed)

# 2. Define Features (X) and Target (y)
X = data.drop('SalePrice', axis=1)  # Features
y = data['SalePrice']  # Target variable

# 3. Drop constant columns (if any)
X = X.loc[:, X.nunique() > 1]  # Only keep columns with more than 1 unique value

# 4. Split the data into Training and Testing sets (80% training, 20% testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 5. Scale the features using StandardScaler
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Check the type and shape of the scaled data
print(f"Type of X_train_scaled: {type(X_train_scaled)}")
print(f"Shape of X_train_scaled: {X_train_scaled.shape}")
print(f"Type of X_test_scaled: {type(X_test_scaled)}")
print(f"Shape of X_test_scaled: {X_test_scaled.shape}")

# Initialize the model (Linear Regression)
model = LinearRegression()

# Train the model on the scaled training data
model.fit(X_train_scaled, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test_scaled)

# Calculate evaluation metrics
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse1 = root_mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)  # Calculate RMSE manually
r2 = r2_score(y_test, y_pred)

# Print out the evaluation metrics
print(f"Mean Absolute Error (MAE): {mae}")
print(f"Mean Squared Error (MSE): {mse}")
print(f"Root Mean Squared Error (RMSE): {rmse,rmse1}")
print(f"R-squared (R²): {r2}")

# Save the model to a file for later use
joblib.dump(model, "house_price_model.pkl")


# Initialize the model
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)

# Train the model
rf_model.fit(X_train, y_train)

# Make predictions
y_pred_rf = rf_model.predict(X_test)

# Evaluate the model's performance
mae_rf = mean_absolute_error(y_test, y_pred_rf)
mse_rf = mean_squared_error(y_test, y_pred_rf)
rmse_rf = root_mean_squared_error(y_test, y_pred_rf)

# rmse_rf = np.sqrt(mse)  # Calculate RMSE manually
# rmse_rf = np.mean(np.sqrt(np.mean((y_test - y_pred_rf) ** 2, axis=0)))
# rmse_rf = np.sqrt(np.mean((y_test - y_pred_rf) ** 2))
# rmse_rf = mean_squared_error(y_test, y_pred_rf, multioutput='raw_values')
# rmse_rf = mean_squared_error(y_test, y_pred_rf, squared=False)

# Print the results
print(f"22Random Forest MAE: {mae_rf}")
print(f"22Random Forest MSE: {mse_rf}")
print(f"22Random Forest RMSE: {rmse_rf}")

from sklearn.ensemble import GradientBoostingRegressor

# Train Gradient Boosting model
gb_model = GradientBoostingRegressor(n_estimators=100, learning_rate=0.05, random_state=42)
gb_model.fit(X_train, y_train)

# Make predictions
y_pred_gb = gb_model.predict(X_test)

# Calculate error metrics
mae_gb = mean_absolute_error(y_test, y_pred_gb)
mse_gb = mean_squared_error(y_test, y_pred_gb)
rmse_gb = root_mean_squared_error(y_test, y_pred_gb)

print(f"Gradient Boosting MAE: {mae_gb}")
print(f"Gradient Boosting MSE: {mse_gb}")
print(f"Gradient Boosting RMSE: {rmse_gb}")


import matplotlib.pyplot as plt
import pandas as pd

# Ensure y_test is a Series with a reset index
if isinstance(y_test, pd.Series):
    y_test = y_test.reset_index(drop=True)

# 1. Sample Predictions
print("Sample Predictions:")
for actual, predicted in zip(y_test[:10], y_pred[:10]):
    print(f"Actual: {actual}, Predicted: {predicted}")

# 2. Actual vs Predicted Plot
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred, alpha=0.5)
plt.xlabel("Actual Prices")
plt.ylabel("Predicted Prices")
plt.title("Actual vs Predicted House Prices")
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color='red')
plt.grid(True)
plt.tight_layout()
plt.savefig("actual_vs_predicted.png")  # ✅ save figure
plt.show()

# 3. Residuals Plot
residuals = y_test - y_pred
plt.figure(figsize=(8, 6))
plt.scatter(y_pred, residuals, alpha=0.5)
plt.axhline(0, color='red', linestyle='--')
plt.xlabel("Predicted Prices")
plt.ylabel("Residuals")
plt.title("Residuals vs Predicted Prices")
plt.grid(True)
plt.tight_layout()
plt.savefig("residuals_plot.png")  # ✅ save figure
plt.show()
