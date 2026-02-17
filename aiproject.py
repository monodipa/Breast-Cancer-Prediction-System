# ==========================================================
# Breast Cancer Prediction System (Load from CSV)
# ==========================================================

# --- Import Libraries ---
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score, roc_curve
)

sns.set(style="whitegrid", palette="muted")

# ==========================================================
# STEP 1: Load Dataset from CSV
# ==========================================================
# <-- CHANGE THIS PATH to your CSV file -->
dataset_path = "C://Users//KIIT0001//Downloads//newdata.csv"

df = pd.read_csv(dataset_path)
print("Dataset loaded successfully!")
print("Shape:", df.shape)
print("\nColumns:", list(df.columns))
print("\nFirst 5 rows:\n", df.head())

# ==========================================================
# STEP 2: Prepare target column
# ==========================================================
# Example: If your CSV has 'diagnosis' column with 'M'/'B'
if 'diagnosis' in df.columns:
    df['target'] = df['diagnosis'].map({'M': 0, 'B': 1})
    df['diagnosis'] = df['target'].map({0: 'Malignant', 1: 'Benign'})
elif 'target' in df.columns:
    df['diagnosis'] = df['target'].map({0: 'Malignant', 1: 'Benign'})
else:
    raise ValueError("Dataset must have a 'diagnosis' or 'target' column")

print("\nTarget distribution:\n", df['diagnosis'].value_counts())

# ==========================================================
# STEP 3: Exploratory Data Analysis (EDA)
# ==========================================================

# Countplot for diagnosis
plt.figure(figsize=(5,4))
sns.countplot(x='diagnosis', data=df)
plt.title("Diagnosis Count (Benign vs Malignant)")
plt.show()

# Pie chart of diagnosis
plt.figure(figsize=(5,5))
plt.pie(df['diagnosis'].value_counts(),
        labels=['Benign', 'Malignant'],
        autopct='%1.1f%%', colors=['#6ab04c','#eb4d4b'])
plt.title("Diagnosis Distribution")
plt.show()

# Correlation heatmap
plt.figure(figsize=(12,10))
sns.heatmap(df.corr(numeric_only=True), cmap='coolwarm', center=0)
plt.title("Feature Correlation Heatmap")
plt.show()

# Top 10 features correlated with target
corr = df.corr(numeric_only=True)['target'].abs().sort_values(ascending=False)[1:11]
plt.figure(figsize=(8,5))
sns.barplot(x=corr.values, y=corr.index, palette='viridis')
plt.title("Top 10 Features Most Correlated with Diagnosis")
plt.xlabel("Correlation Strength")
plt.ylabel("Feature")
plt.show()

# Distribution plots for key features
key_features = corr.index[:4]  # top 4 features
for feature in key_features:
    plt.figure(figsize=(6,4))
    sns.histplot(df, x=feature, hue='diagnosis', kde=True, element='step')
    plt.title(f"Distribution of {feature}")
    plt.show()

# Boxplots
for feature in key_features:
    plt.figure(figsize=(6,4))
    sns.boxplot(x='diagnosis', y=feature, data=df, palette='coolwarm')
    plt.title(f"{feature} by Diagnosis")
    plt.show()

# Violin plot for one feature
plt.figure(figsize=(8,5))
sns.violinplot(x='diagnosis', y=key_features[0], data=df, palette='Set2')
plt.title(f"Violin Plot: {key_features[0]} vs Diagnosis")
plt.show()

# Pairplot of top features
sns.pairplot(df[list(key_features)+['diagnosis']], hue='diagnosis', palette='husl')
plt.show()

# ==========================================================
# STEP 4: Data Preprocessing
# ==========================================================
X = df.drop(['diagnosis', 'target'], axis=1)
y = df['target']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)

# ==========================================================
# STEP 5: Train Multiple Models
# ==========================================================
models = {
    "Logistic Regression": LogisticRegression(max_iter=5000),
    "Random Forest": RandomForestClassifier(random_state=42),
    "SVM (RBF Kernel)": SVC(probability=True, random_state=42)
}

results = {}

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    
    results[name] = {
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred),
        "Recall": recall_score(y_test, y_pred),
        "F1 Score": f1_score(y_test, y_pred),
        "ROC AUC": roc_auc_score(y_test, y_proba),
        "Confusion Matrix": confusion_matrix(y_test, y_pred)
    }

# ==========================================================
# STEP 6: Model Performance Comparison
# ==========================================================
result_df = pd.DataFrame(results).T
print("\n Model Performance Comparison:\n")
print(result_df[['Accuracy', 'Precision', 'Recall', 'F1 Score', 'ROC AUC']])

# Comparison bar chart
result_df[['Accuracy', 'F1 Score', 'ROC AUC']].plot(kind='bar', figsize=(10,5))
plt.title("Model Performance Comparison")
plt.ylabel("Score")
plt.ylim(0.8, 1.0)
plt.xticks(rotation=45)
plt.show()

# ==========================================================
# STEP 7: Confusion Matrix Visualization
# ==========================================================
for name, metrics in results.items():
    cm = metrics["Confusion Matrix"]
    plt.figure(figsize=(4,4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.title(f"Confusion Matrix - {name}")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.show()

# ==========================================================
# STEP 8: ROC Curves
# ==========================================================
plt.figure(figsize=(8,6))
for name, model in models.items():
    y_proba = model.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    auc = roc_auc_score(y_test, y_proba)
    plt.plot(fpr, tpr, label=f"{name} (AUC = {auc:.3f})")

plt.plot([0,1], [0,1], 'k--')
plt.title("ROC Curves for All Models")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend()
plt.show()

# ==========================================================
# STEP 9: Best Model Selection
# ==========================================================
best_model_name = result_df['ROC AUC'].idxmax()
print(f"\n Best Performing Model: {best_model_name}")
print("\nDetailed Classification Report:\n")
y_pred_best = models[best_model_name].predict(X_test)
print(classification_report(y_test, y_pred_best))
