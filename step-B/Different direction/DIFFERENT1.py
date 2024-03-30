import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from scipy.stats import f_oneway, chi2_contingency
from sklearn.linear_model import LogisticRegression

# Load the dataset
df = pd.read_csv("../../prepared_drug_overdose_dataset.csv")  # Adjust path as necessary

# Data Cleaning and Preparation
df.dropna(inplace=True)  # Handling missing values
df_encoded = pd.get_dummies(df, columns=['STUB_NAME', 'STUB_LABEL'])  # One-hot encoding for categorical variables

# Normalize 'ESTIMATE' column
scaler = StandardScaler()
df_encoded['ESTIMATE'] = scaler.fit_transform(df_encoded[['ESTIMATE']])

# Descriptive Statistics
print("Descriptive Statistics:")
print(df_encoded.describe())

# Visualization
## Time Series Analysis for Overdose Death Rates Over Time by Demographic Groups
for column in df_encoded.columns:
    if 'Sex_' in column or 'Race_' in column or 'AGE_' in column:
        plt.figure(figsize=(12, 6))
        sns.lineplot(data=df, x='YEAR', y='ESTIMATE', estimator='mean', errorbar=None, hue=df[column])
        plt.title(f'Average Drug Overdose Death Rates Over Time by {column}')
        plt.xlabel('Year')
        plt.ylabel('Normalized Death Rate')
        plt.show()

# Correlation and Causation Analysis
# Correlation Matrix for Encoded Dataset
correlation_matrix = df_encoded.corr()
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, cmap='coolwarm', vmax=0.3, vmin=-0.3)
plt.title('Correlation Matrix for Encoded Dataset')
plt.show()

# Advanced Hypothesis Testing - Chi-Square Test for Independence between Sex and Overdose Rates
chi2, p, dof, ex = chi2_contingency(pd.crosstab(df['STUB_LABEL'].str.contains('Male'), df['ESTIMATE']))
print(f"Chi-Square Test between gender and overdose rates: chi2={chi2}, p={p}")

# Predictive Modeling - Logistic Regression Example for Binary Outcome
X = df_encoded.drop(['ESTIMATE', 'YEAR'], axis=1)  # Feature selection
y = df_encoded['ESTIMATE']  # Correcting the target variable for regression
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Regression Model
regressor = RandomForestRegressor(n_estimators=100, random_state=42)
regressor.fit(X_train, y_train)  # Correct use of y_train
y_pred = regressor.predict(X_test)
print(f"Regression Model MSE: {mean_squared_error(y_test, y_pred)}")

# Logistic Regression for a binary outcome demonstration
y_binary = pd.qcut(df_encoded['ESTIMATE'], 2, labels=False)
X_train, X_test, y_binary_train, y_binary_test = train_test_split(X, y_binary, test_size=0.2, random_state=42)

logistic_model = LogisticRegression(max_iter=1000, random_state=42)
logistic_model.fit(X_train, y_binary_train)
y_binary_pred = logistic_model.predict(X_test)
print(f"Logistic Regression Model Accuracy: {logistic_model.score(X_test, y_binary_test)}")

# Feature Importance Analysis from RandomForestRegressor
importances = regressor.feature_importances_
features = X.columns
importance_df = pd.DataFrame({'Feature': features, 'Importance': importances}).sort_values(by='Importance', ascending=False)
plt.figure(figsize=(10, 8))
sns.barplot(x='Importance', y='Feature', data=importance_df.head(20))
plt.title('Top 20 Feature Importances from RandomForest')
plt.show()

# Cluster Analysis for Group Identification
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)
kmeans = KMeans(n_clusters=3, random_state=42)
clusters = kmeans.fit_predict(X_pca)

plt.figure(figsize=(8, 6))
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=clusters, cmap='viridis')
plt.title('PCA-based Clustering with KMeans')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.show()
