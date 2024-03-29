import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer
from feature_engine.encoding import RareLabelEncoder


def extract_gender_ethnicity(label):
    if "Male:" in label:
        gender = "Male"
    elif "Female:" in label:
        gender = "Female"
    else:
        gender = "Unknown"

    if ':' in label:
        ethnicity = label.split(":")[-1].strip()
    else:
        ethnicity = "All"
    return gender, ethnicity


# Load the dataset
df = pd.read_csv("../prepared_drug_overdose_dataset.csv")
# Apply the function to each row and create new columns for Gender and Ethnicity
df[['Gender', 'Ethnicity']] = df.apply(lambda row: pd.Series(extract_gender_ethnicity(row['STUB_LABEL'])), axis=1)

# Handle rare labels in 'Ethnicity'
rare_encoder = RareLabelEncoder(tol=0.05, n_categories=1, replace_with='Rare')
df['Ethnicity'] = rare_encoder.fit_transform(df[['Ethnicity']])

# Impute missing ESTIMATE values before encoding
df['ESTIMATE'] = pd.to_numeric(df['ESTIMATE'], errors='coerce')
imputer = SimpleImputer(strategy='median')
df['ESTIMATE'] = imputer.fit_transform(df[['ESTIMATE']].values)

# Encode 'Gender' and the transformed 'Ethnicity' with pd.get_dummies
df_encoded = pd.get_dummies(df, columns=['Gender', 'Ethnicity'])

# Prepare X and y for feature selection, ensuring y is numeric and has no missing values
X = df_encoded.drop(['ESTIMATE', 'STUB_LABEL'], axis=1)  # Excluding non-relevant or already encoded columns
y = df_encoded['ESTIMATE']

# Apply SelectKBest for feature selection
selector = SelectKBest(score_func=f_classif, k=10)  # Adjust k as needed
X_new = selector.fit_transform(X, y)
selected_features = X.columns[selector.get_support()]
print("Selected features:", selected_features)

# Standardizing features for PCA
X_scaled = StandardScaler().fit_transform(X_new)

# Performing PCA
pca = PCA(n_components=2)
principalComponents = pca.fit_transform(X_scaled)

# Creating a DataFrame for PCA results visualization
principalDf = pd.DataFrame(data=principalComponents, columns=['Principal Component 1', 'Principal Component 2'])

# Visualizing the PCA results
plt.figure(figsize=(8, 6))
plt.scatter(principalDf['Principal Component 1'], principalDf['Principal Component 2'], s=50)
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.title('PCA of Dataset')
plt.show()
