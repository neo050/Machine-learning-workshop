import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import mean_squared_error, accuracy_score
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from scipy.stats import chi2_contingency, f_oneway
from statsmodels.stats.multicomp import pairwise_tukeyhsd
import statsmodels.api as sm

# Load and prepare dataset
df = pd.read_csv("../../prepared_drug_overdose_dataset.csv")

# Data preprocessing
encoder = OneHotEncoder()
encoded = encoder.fit_transform(df[['STUB_NAME', 'STUB_LABEL']])
encoded_features = encoder.get_feature_names_out(['STUB_NAME', 'STUB_LABEL'])
df_encoded = pd.concat([df, pd.DataFrame(encoded.toarray(), columns=encoded_features)], axis=1).drop(['STUB_NAME', 'STUB_LABEL'], axis=1)

# Normalize 'ESTIMATE'
scaler = StandardScaler()
df_encoded['ESTIMATE_norm'] = scaler.fit_transform(df_encoded[['ESTIMATE']])

# Descriptive statistics
print("Descriptive Statistics:")
print(df_encoded.describe())

# Visualization: Time Series Analysis
plt.figure(figsize=(12, 6))
sns.lineplot(x='YEAR', y='ESTIMATE_norm', data=df_encoded, hue='Sex_Male')  # Example for 'Sex_Male'
plt.title('Overdose Death Rates Over Time by Gender')
plt.xlabel('Year')
plt.ylabel('Normalized Death Rate')
plt.legend(title='Gender')
plt.show()

# Correlation Analysis
corr_matrix = df_encoded.corr()
plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=False, cmap='coolwarm')
plt.title('Correlation Matrix')
plt.show()

# Hypothesis Testing: ANOVA and Chi-square
# ANOVA Example for Age Groups
f_val, p_val = f_oneway(df_encoded[df_encoded['AGE_15-24 years'] == 1]['ESTIMATE_norm'],
                        df_encoded[df_encoded['AGE_25-34 years'] == 1]['ESTIMATE_norm'])
print(f"ANOVA F-value: {f_val}, P-value: {p_val}")

# Chi-square Test for Independence between Sex and Overdose Rates
chi2, p, dof, expected = chi2_contingency(pd.crosstab(df['Sex_Male'], df['ESTIMATE']))
print(f"Chi-Square Test: chi2={chi2}, p={p}")

# Predictive Modeling: RandomForestRegressor for Feature Importance
X = df_encoded.drop(['ESTIMATE', 'ESTIMATE_norm'], axis=1)
y = df_encoded['ESTIMATE_norm']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

print(f"MSE: {mean_squared_error(y_test, y_pred)}")

# Feature Importance
importances = model.feature_importances_
indices = np.argsort(importances)[::-1]
plt.figure(figsize=(10, 8))
plt.title('Feature Importance')
plt.bar(range(X_train.shape[1]), importances[indices], align='center')
plt.xticks(range(X_train.shape[1]), [X_train.columns[i] for i in indices], rotation=90)
plt.tight_layout()
plt.show()

# Clustering: KMeans after PCA
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)
kmeans = KMeans(n_clusters=3, random_state=42)
clusters = kmeans.fit_predict(X_pca)

plt.figure(figsize=(8, 6))
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=clusters, cmap='viridis')
plt.title('Cluster Analysis with KMeans (PCA-Reduced Data)')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.show()
