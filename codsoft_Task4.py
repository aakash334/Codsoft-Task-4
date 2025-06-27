import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression # A common linear regression model
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler

# --- Step 1: Load or Simulate the Dataset ---
# Since a specific dataset for Sales Prediction was not provided,
# we will simulate a dataset based on common features mentioned (advertising, target audience, platform).
# In a real scenario, you would replace this with pd.read_csv('your_sales_data.csv').

print("--- Simulating Sales Data for Demonstration ---")
np.random.seed(42) # For reproducibility

# Number of data points
num_samples = 1000

# Simulate features
advertising_expenditure = np.random.rand(num_samples) * 1000 + 100 # Spending between 100 and 1100
platform = np.random.choice(['Online', 'TV', 'Radio'], num_samples)
target_audience_segment = np.random.choice(['Youth', 'Adult', 'Senior'], num_samples)
economic_index = np.random.rand(num_samples) * 10 + 90 # Index between 90 and 100

# Simulate sales based on a linear relationship with some noise
# Sales = (Ad Exp * 0.5) + (Platform Effect) + (Audience Effect) + (Economic Index * 2) + Noise
sales = (advertising_expenditure * 0.5 +
         np.where(platform == 'Online', 50, np.where(platform == 'TV', 30, 10)) + # Platform impact
         np.where(target_audience_segment == 'Youth', 20, np.where(target_audience_segment == 'Adult', 10, 5)) + # Audience impact
         (economic_index * 2) +
         np.random.randn(num_samples) * 50) # Add random noise

df = pd.DataFrame({
    'advertising_expenditure': advertising_expenditure,
    'platform': platform,
    'target_audience_segment': target_audience_segment,
    'economic_index': economic_index,
    'sales': sales
})

print(f"Simulated dataset with {num_samples} samples created successfully!")
print("Please replace this section with loading your actual dataset if available:")
print("e.g., df = pd.read_csv('your_sales_data.csv')")

# --- Step 2: Exploratory Data Analysis (EDA) ---
print("\n--- Exploratory Data Analysis ---")
print("Data Head:\n", df.head())
print("\nData Info:\n")
df.info()
print("\nDescriptive Statistics:\n", df.describe(include='all'))
print("\nMissing values before handling:\n", df.isnull().sum()) # Should be 0 for simulated data

# Visualize the distribution of the target variable 'sales'
plt.figure(figsize=(10, 6))
sns.histplot(df['sales'], kde=True, bins=30)
plt.title('Distribution of Sales')
plt.xlabel('Sales')
plt.ylabel('Count')
plt.show()

# Visualize relationship between 'advertising_expenditure' and 'sales'
plt.figure(figsize=(10, 6))
sns.scatterplot(x='advertising_expenditure', y='sales', data=df)
plt.title('Sales vs. Advertising Expenditure')
plt.xlabel('Advertising Expenditure')
plt.ylabel('Sales')
plt.show()

# Visualize sales by 'platform'
plt.figure(figsize=(8, 5))
sns.boxplot(x='platform', y='sales', data=df)
plt.title('Sales by Platform')
plt.xlabel('Platform')
plt.ylabel('Sales')
plt.show()

# --- Step 3: Data Preprocessing ---
# For simulated data, missing value handling is not strictly necessary,
# but it's kept for consistency with the ML pipeline.

# Handle missing values (if any in real data)
# For simplicity, numerical missing values will be filled with median,
# and categorical with 'Unknown' (though not expected in this simulated data).
numerical_cols = ['advertising_expenditure', 'economic_index']
categorical_cols = ['platform', 'target_audience_segment']

for col in numerical_cols:
    if df[col].isnull().any():
        df[col].fillna(df[col].median(), inplace=True)
        print(f"Filled missing values in '{col}' with median.")

for col in categorical_cols:
    if df[col].isnull().any():
        df[col].fillna('Unknown', inplace=True)
        print(f"Filled missing values in '{col}' with 'Unknown'.")

print("\nMissing values after handling (should be 0 for simulated data):\n", df.isnull().sum())

# One-Hot Encode categorical features: 'platform', 'target_audience_segment'
# Regression models require numerical input.
df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

print("\nDataFrame head after preprocessing:\n", df.head())
print("\nDataFrame info after preprocessing:\n")
df.info()

# --- Step 4: Define Features (X) and Target (y) ---
# X contains the independent variables (features) used for prediction.
# y contains the dependent variable (the 'sales' we want to predict).
X = df.drop('sales', axis=1) # All columns except 'sales'
y = df['sales']             # The 'sales' column

print("\nFeatures (X) sample after preprocessing:\n", X.head())
print("\nTarget (y) sample after preprocessing:\n", y.head())

# --- Step 5: Split the Data into Training and Testing Sets ---
# 80% for training, 20% for testing. `random_state` ensures reproducibility.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"\nTraining set size: {X_train.shape[0]} samples")
print(f"Testing set size: {X_test.shape[0]} samples")

# --- Step 6: Feature Scaling ---
# Scaling numerical features is crucial for linear models like Linear Regression
# and can also benefit tree-based models like Random Forest.
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Convert scaled arrays back to DataFrame for easier inspection (optional).
X_train_scaled = pd.DataFrame(X_train_scaled, columns=X.columns, index=X_train.index)
X_test_scaled = pd.DataFrame(X_test_scaled, columns=X.columns, index=X_test.index)

print("\nFeatures (X) sample after scaling:\n", X_train_scaled.head())

# --- Step 7: Choose and Train Machine Learning Models & Fine-tune hyperparameters ---
# We will train and tune two common regression models: Random Forest Regressor and Linear Regression.

# Model 1: Random Forest Regressor
print("\n--- Training Random Forest Regressor ---")
rf_regressor = RandomForestRegressor(random_state=42)

# Define hyperparameters for GridSearchCV for Random Forest Regressor.
param_grid_rf = {
    'n_estimators': [50, 100, 200], # Number of trees
    'max_depth': [10, 20, None], # Maximum depth
    'min_samples_leaf': [1, 5] # Minimum samples per leaf
}

# `scoring='neg_mean_squared_error'` is used because GridSearchCV minimizes a score,
# so we use the negative of MSE to maximize (minimize negative MSE).
grid_search_rf = GridSearchCV(estimator=rf_regressor, param_grid=param_grid_rf,
                              cv=3, n_jobs=-1, verbose=1, scoring='neg_mean_squared_error')
grid_search_rf.fit(X_train_scaled, y_train)

best_rf_regressor = grid_search_rf.best_estimator_
print(f"Best Random Forest Regressor Parameters: {grid_search_rf.best_params_}")
print(f"Best Random Forest Regressor Training MSE (CV): {-grid_search_rf.best_score_:.4f}")

# Make predictions and evaluate Random Forest Regressor.
y_pred_rf = best_rf_regressor.predict(X_test_scaled)
print("\n--- Random Forest Regressor Model Evaluation ---")
mse_rf = mean_squared_error(y_test, y_pred_rf)
rmse_rf = np.sqrt(mse_rf)
mae_rf = mean_absolute_error(y_test, y_pred_rf)
r2_rf = r2_score(y_test, y_pred_rf)

print(f"Random Forest Test Mean Squared Error (MSE): {mse_rf:.4f}")
print(f"Random Forest Test Root Mean Squared Error (RMSE): {rmse_rf:.4f}")
print(f"Random Forest Test Mean Absolute Error (MAE): {mae_rf:.4f}")
print(f"Random Forest Test R-squared (R2): {r2_rf:.4f}")


# Model 2: Linear Regression (no specific hyperparameters to tune for basic LinearRegression)
print("\n--- Training Linear Regression Model ---")
lr_model = LinearRegression() # Simple Linear Regression

# For Linear Regression, there are no hyperparameters like alpha to tune in basic form,
# so GridSearchCV is not strictly necessary for this specific model, but it's kept
# for consistency if you want to add more complex linear models (e.g., Ridge, Lasso).
# Here, we'll just train it directly.
lr_model.fit(X_train_scaled, y_train)

# Make predictions and evaluate Linear Regression.
y_pred_lr = lr_model.predict(X_test_scaled)
print("\n--- Linear Regression Model Evaluation ---")
mse_lr = mean_squared_error(y_test, y_pred_lr)
rmse_lr = np.sqrt(mse_lr)
mae_lr = mean_absolute_error(y_test, y_pred_lr)
r2_lr = r2_score(y_test, y_pred_lr)

print(f"Linear Regression Test Mean Squared Error (MSE): {mse_lr:.4f}")
print(f"Linear Regression Test Root Mean Squared Error (RMSE): {rmse_lr:.4f}")
print(f"Linear Regression Test Mean Absolute Error (MAE): {mae_lr:.4f}")
print(f"Linear Regression Test R-squared (R2): {r2_lr:.4f}")


# --- Step 8: Compare Evaluation Metrics of Various Regression Algorithms ---
print("\n--- Model Comparison Summary ---")
print(f"Random Forest Regressor Test R-squared (R2): {r2_rf:.4f}")
print(f"Linear Regression Test R-squared (R2): {r2_lr:.4f}")

print("\nLower MSE/RMSE/MAE and higher R2 (closer to 1) indicate better model performance for regression.")
print("The model with the highest R2 score and lowest error metrics is generally preferred.")
