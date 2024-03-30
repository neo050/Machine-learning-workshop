import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from scipy.stats import chi2_contingency, f_oneway
import statsmodels.api as sm
from statsmodels.formula.api import ols

# Load the dataset
df = pd.read_csv("../prepared_drug_overdose_dataset.csv")  # Adjust the path as needed

# Data Cleaning and Preparation
df.dropna(subset=['ESTIMATE'], inplace=True)

# One-hot encode categorical variables 'STUB_NAME' and 'STUB_LABEL'
encoder = OneHotEncoder()
encoded_data = encoder.fit_transform(df[['STUB_NAME', 'STUB_LABEL']]).toarray()
encoded_feature_names = encoder.get_feature_names_out(['STUB_NAME', 'STUB_LABEL'])
df_encoded = pd.DataFrame(encoded_data, columns=encoded_feature_names)

# Normalize 'ESTIMATE'
scaler = StandardScaler()
df['ESTIMATE_norm'] = scaler.fit_transform(df[['ESTIMATE']])

# Merge the one-hot encoded features with the original dataframe
df_final = pd.concat([df.reset_index(drop=True), df_encoded.reset_index(drop=True)], axis=1)

# Descriptive Statistics and Data Exploration
print(df_final.describe())

# Visualization: Time Series Analysis and Comparative Plots
sns.lineplot(data=df, x='YEAR', y='ESTIMATE', hue='STUB_LABEL')
plt.title('Drug Overdose Death Rates Over Time by Demographic Group')
plt.show()

# Comparative plots for 'STUB_LABEL'
sns.boxplot(data=df, x='ESTIMATE', y='STUB_LABEL')
plt.title('Overdose Death Rates by Demographic Group')
plt.show()


# Ensure df_final contains only numeric data before calculating the correlation matrix
df_final_numeric = df_final.select_dtypes(include=[np.number])
# Now you can safely compute the correlation matrix on numeric data
correlation_matrix = df_final_numeric.corr()
# Continue with plotting the heatmap for the correlation matrix
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, cmap='coolwarm')
plt.title('Correlation Matrix of Features')
plt.show()


"""""
# Correlation Analysis
correlation_matrix = df_final.corr()
sns.heatmap(correlation_matrix, cmap='coolwarm')
plt.title('Correlation Matrix of Features')
plt.show()
"""

# Hypothesis Testing: ANOVA for 'STUB_LABEL' categories
for category in df['STUB_LABEL'].unique():
    df_group = df[df['STUB_LABEL'] == category]['ESTIMATE']
    print(f"ANOVA for {category}: {f_oneway(df_group, df['ESTIMATE'])[1]}")

# Predictive Modeling: Random Forest Regressor
X = df_final.drop(['ESTIMATE', 'ESTIMATE_norm', 'STUB_NAME', 'STUB_LABEL', 'YEAR'], axis=1)
y = df_final['ESTIMATE_norm']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

regressor = RandomForestRegressor(n_estimators=100, random_state=42)
regressor.fit(X_train, y_train)
y_pred = regressor.predict(X_test)
print(f"Model MSE: {mean_squared_error(y_test, y_pred)}")
print(f"Model R^2: {r2_score(y_test, y_pred)}")

# Feature Importance from Random Forest
feature_importance = regressor.feature_importances_
plt.barh(np.arange(len(feature_importance)), feature_importance)
plt.yticks(np.arange(len(X.columns)), X.columns)
plt.title('Feature Importances in Random Forest Model')
plt.show()

# PCA and Clustering Analysis
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)
kmeans = KMeans(n_clusters=3, random_state=42)
clusters = kmeans.fit_predict(X_pca)
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=clusters, cmap='viridis')
plt.title('Cluster Analysis with PCA and KMeans')
plt.show()
