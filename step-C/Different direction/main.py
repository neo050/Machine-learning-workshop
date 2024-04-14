import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import StandardScaler
from ucimlrepo import fetch_ucirepo

# Fetch the dataset
wine_quality = fetch_ucirepo(id=186)

# Data (as pandas DataFrames)
X = wine_quality.data.features.copy()  # Features, ensuring we work on a copy to avoid SettingWithCopyWarning
y = wine_quality.data.targets  # Target variable

# Feature Engineering: Create a new feature sulfur_dioxide_ratio
X['sulfur_dioxide_ratio'] = X['free_sulfur_dioxide'] / X['total_sulfur_dioxide']

# Normalize the feature matrix
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split the dataset into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Convert y_train to a NumPy array and flatten it
y_train_array = y_train.values.ravel()

# Before using SMOTE, let's print the class distribution to understand the imbalance
unique, counts = np.unique(y_train_array, return_counts=True)
print("Class distribution before SMOTE:\n", dict(zip(unique, counts)))

# Proceed with the SMOTE oversampling
# Adjust the k_neighbors parameter if necessary
# Note: Ensure you have enough samples in each class before setting the k_neighbors for SMOTE
k_neighbors_smote = min([2, *counts]) - 1
k_neighbors_smote = max(1, k_neighbors_smote)  # Ensure k_neighbors is at least 1
print("Using k_neighbors for SMOTE:", k_neighbors_smote)

smote = SMOTE(random_state=42, k_neighbors=k_neighbors_smote)
X_train_sm, y_train_sm = smote.fit_resample(X_train, y_train_array)

# After applying SMOTE, check the class distribution again
unique_after, counts_after = np.unique(y_train_sm, return_counts=True)
print("Class distribution after SMOTE:\n", dict(zip(unique_after, counts_after)))


# Model initialization and training
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train_sm, y_train_sm)  # Train on the oversampled data

# Predictions and evaluation
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f"MSE: {mse}")
print(f"R^2 Score: {r2}")

# Hyperparameter tuning with GridSearchCV
param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5]
}
grid_search = GridSearchCV(estimator=RandomForestRegressor(random_state=42),
                           param_grid=param_grid,
                           cv=5,
                           scoring='neg_mean_squared_error',
                           n_jobs=-1)
grid_search.fit(X_train_sm, y_train_sm)
print("Best parameters:", grid_search.best_params_)
print("Best MSE score:", -grid_search.best_score_)

# Cross-validation to evaluate the model
cv_scores = cross_val_score(model, X_train_sm, y_train_sm, cv=5, scoring='neg_mean_squared_error')
print(f"CV MSE: {-np.mean(cv_scores)}")

# Feature importances
features = np.append(wine_quality.data.features.columns, 'sulfur_dioxide_ratio')  # Ensure this matches the columns in X before scaling
importances = model.feature_importances_
indices = np.argsort(importances)[::-1]
plt.figure(figsize=(12, 8))
plt.title('Feature Importance')
plt.bar(range(X_train_sm.shape[1]), importances[indices], align='center')  # Correct the range to match the number of features after SMOTE
plt.xticks(range(X_train_sm.shape[1]), features[indices], rotation=90)  # Ensure features array matches the processed data
plt.tight_layout()
plt.show()

