import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve, precision_recall_curve
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import StandardScaler
from ucimlrepo import fetch_ucirepo

# Fetch the dataset
wine_quality = fetch_ucirepo(id=186)

# Data preparation
X = wine_quality.data.features.copy()  # Features
y = wine_quality.data.targets  # Target variable
y_binary = (y['quality'] >= 5).astype(int)  # Convert to binary classification

# Feature Engineering
X['sulfur_dioxide_ratio'] = X['free_sulfur_dioxide'] / X['total_sulfur_dioxide']
X['alcohol_volatile_acidity_interaction'] = X['alcohol'] * X['volatile_acidity']
X['alcohol_sulfur_dioxide_ratio_interaction'] = X['alcohol'] * X['sulfur_dioxide_ratio']

# Normalize the feature matrix
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split the dataset, ensuring stratified split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_binary, test_size=0.2, random_state=42, stratify=y_binary)

# Handling class imbalance with SMOTE
smote = SMOTE(random_state=42)
X_train_sm, y_train_sm = smote.fit_resample(X_train, y_train)

# Model training and selection
rf_model = RandomForestClassifier(n_estimators=200, max_depth=10, min_samples_split=2, min_samples_leaf=1, random_state=42)
rf_model.fit(X_train_sm, y_train_sm)

svm_model = SVC(kernel='linear', probability=True, random_state=42)
svm_model.fit(X_train_sm, y_train_sm)

# 10-fold cross-validation
cv = StratifiedKFold(n_splits=10, random_state=42, shuffle=True)
rf_cv_scores = cross_val_score(rf_model, X_train_sm, y_train_sm, cv=cv, scoring='roc_auc')
svm_cv_scores = cross_val_score(svm_model, X_train_sm, y_train_sm, cv=cv, scoring='roc_auc')

best_model = rf_model if np.mean(rf_cv_scores) > np.mean(svm_cv_scores) else svm_model

# Model evaluation
y_pred = best_model.predict(X_test)
print(f"Classification Report:\n{classification_report(y_test, y_pred)}")

# Confusion Matrix as a heatmap
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap='Blues', xticklabels=['Predicted 0', 'Predicted 1'], yticklabels=['Actual 0', 'Actual 1'])
plt.title('Confusion Matrix')
plt.ylabel('Actual label')
plt.xlabel('Predicted label')
plt.show()

# ROC Curve
roc_auc = roc_auc_score(y_test, best_model.predict_proba(X_test)[:, 1])
fpr, tpr, _ = roc_curve(y_test, best_model.predict_proba(X_test)[:, 1])
plt.figure()
plt.plot(fpr, tpr, label=f'Best Model ROC Curve (area = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc="lower right")
plt.show()

# Precision-Recall Curve
plt.figure(figsize=(8, 6))
precision, recall, _ = precision_recall_curve(y_test, best_model.predict_proba(X_test)[:, 1])
plt.plot(recall, precision, color='blue')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
plt.show()

# Histogram of Predicted Probabilities
plt.figure(figsize=(8, 6))
probabilities = best_model.predict_proba(X_test)[:, 1]
plt.hist(probabilities, bins=10, color='skyblue', edgecolor='black')
plt.title('Histogram of Predicted Probabilities')
plt.xlabel('Predicted Probability of Positive Class')
plt.ylabel('Frequency')
plt.show()

# Feature importances, if RandomForest is the best model
if best_model == rf_model:
    features = np.append(X.columns, ['sulfur_dioxide_ratio', 'alcohol_volatile_acidity_interaction', 'alcohol_sulfur_dioxide_ratio_interaction'])
    importances = best_model.feature_importances_
    indices = np.argsort(importances)[::-1]
    plt.figure(figsize=(12, 8))
    plt.title('Feature Importances')
    plt.bar(range(X.shape[1]), importances[indices], align='center')
    plt.xticks(range(X.shape[1]), features[indices], rotation=90)
    plt.tight_layout()
    plt.show()
