import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, root_mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline

# Step 1: Load and inspect data
data = pd.read_csv('train.csv')

print("First 5 rows of the dataset:")
print(data.head())
print("\nDataset Info:")
print(data.info())

# Step 2: Handle missing values & encode
missing_values = data.isnull().sum()
print("\nMissing values in each column:")
print(missing_values[missing_values > 0])

# Quality mapping
quality_map = {'Ex': 5, 'Gd': 4, 'TA': 3, 'Fa': 2, 'Po': 1}
ordinal_cols = ['ExterQual', 'ExterCond', 'BsmtQual', 'BsmtCond', 'HeatingQC',
                'KitchenQual', 'FireplaceQu', 'GarageQual', 'GarageCond', 'PoolQC']
for col in ordinal_cols:
    data[col] = data[col].map(quality_map).fillna(0)

# Fence mapping
fence_map = {'GdPrv': 4, 'MnPrv': 3, 'GdWo': 2, 'MnWw': 1}
data['Fence'] = data['Fence'].map(fence_map).fillna(0)

# Binary columns
binary_cols = ['CentralAir', 'Alley']
for col in binary_cols:
    data[col] = data[col].map({'Y': 1, 'N': 0}).fillna(0)

# One-hot encode nominal columns
nominal_cols = ['MSZoning', 'Street', 'LotShape', 'LandContour', 'Utilities', 
                'LotConfig', 'LandSlope', 'Neighborhood', 'Condition1', 'Condition2', 
                'BldgType', 'HouseStyle', 'RoofStyle', 'RoofMatl', 'Exterior1st', 
                'Exterior2nd', 'MasVnrType', 'Foundation', 'BsmtExposure', 'BsmtFinType1', 
                'BsmtFinType2', 'Heating', 'Electrical', 'Functional', 'GarageType', 
                'GarageFinish', 'PavedDrive', 'MiscFeature', 'SaleType', 'SaleCondition']

data = pd.get_dummies(data, columns=nominal_cols, drop_first=True)

# Impute any remaining missing values
data.fillna(data.mode().iloc[0], inplace=True)
data.fillna(data.median(numeric_only=True), inplace=True)

# Step 3: Feature-target split
X = data.drop('SalePrice', axis=1)
y = data['SalePrice']

# Drop constant columns
X = X.loc[:, X.nunique() > 1]

# Save template input
X.iloc[[0]].to_csv("input_template.csv", index=False)
 
# Step 4: Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 5: Create & train pipeline
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('model', RandomForestRegressor(n_estimators=100, random_state=42))
])

pipeline.fit(X_train, y_train)
 
# Step 6: Evaluate model
y_pred_rf = pipeline.predict(X_test)

mae_rf = mean_absolute_error(y_test, y_pred_rf)
mse_rf = mean_squared_error(y_test, y_pred_rf)
rmse_rf = root_mean_squared_error(y_test, y_pred_rf)

print(f"\n✅ Random Forest MAE: {mae_rf}")
print(f"✅ Random Forest MSE: {mse_rf}")
print(f"✅ Random Forest RMSE: {rmse_rf}")
 
# Step 7: Save pipeline and metadata
joblib.dump(X_train.columns.tolist(), 'model_columns.pkl')
joblib.dump(pipeline, 'model_pipeline.pkl')
print("\n✅ Model pipeline and column names saved successfully.")


import matplotlib.pyplot as plt

# Ensure y_test is a Series with a reset index
if isinstance(y_test, pd.Series):
    y_test = y_test.reset_index(drop=True)

# Ensure y_pred is a NumPy array or Series with matching index
y_pred = pd.Series(y_pred_rf).reset_index(drop=True)

# 1. Sample Predictions
print("\n🔍 Sample Predictions:")
for actual, predicted in zip(y_test[:10], y_pred[:10]):
    print(f"Actual: {actual}, Predicted: {round(predicted, 2)}")

# 2. Actual vs Predicted Plot
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred, alpha=0.5, color='blue')
plt.xlabel("Actual Prices")
plt.ylabel("Predicted Prices")
plt.title("Actual vs Predicted House Prices")
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color='red', linewidth=2)
plt.grid(True)
plt.tight_layout()
plt.savefig("actual_vs_predicted.png")
plt.show()

# 3. Residuals Plot
residuals = y_test - y_pred
plt.figure(figsize=(8, 6))
plt.scatter(y_pred, residuals, alpha=0.5, color='purple')
plt.axhline(0, color='red', linestyle='--', linewidth=2)
plt.xlabel("Predicted Prices")
plt.ylabel("Residuals")
plt.title("Residuals vs Predicted Prices")
plt.grid(True)
plt.tight_layout()
plt.savefig("residuals_plot.png")
plt.close()


pd.Series(y_test).to_csv("y_test.csv", index=False)
pd.Series(y_pred_rf).to_csv("y_pred.csv", index=False)

