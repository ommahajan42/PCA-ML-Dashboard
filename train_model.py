import os
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


# ==========================================
# 1. LOAD DATASET
# ==========================================

DATASET_PATH = "dataset/Iris_Dataset.csv"

df = pd.read_csv(DATASET_PATH)

print("Dataset loaded successfully!")
print("Dataset Shape:", df.shape)


# ==========================================
# 2. FEATURES AND TARGET
# ==========================================

features = [
    "sepal length (cm)",
    "sepal width (cm)",
    "petal length (cm)",
    "petal width (cm)"
]

X = df[features]
y = df["target"]


# ==========================================
# 3. TRAIN / TEST SPLIT
# ==========================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("Training Samples:", len(X_train))
print("Testing Samples:", len(X_test))


# ==========================================
# 4. STANDARDIZATION
# ==========================================

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# ==========================================
# 5. PCA
# ==========================================

pca = PCA(n_components=2)

X_train_pca = pca.fit_transform(X_train_scaled)
X_test_pca = pca.transform(X_test_scaled)


# ==========================================
# 6. LOGISTIC REGRESSION
# ==========================================

classifier = LogisticRegression(max_iter=200)

classifier.fit(X_train_pca, y_train)


# ==========================================
# 7. PREDICTION
# ==========================================

y_pred = classifier.predict(X_test_pca)

accuracy = accuracy_score(y_test, y_pred)


# ==========================================
# 8. RESULTS
# ==========================================

variance = pca.explained_variance_ratio_.sum() * 100

print("\n======================================")
print("       PCA ML PROJECT RESULTS")
print("======================================")

print("Original Features :", X.shape[1])
print("Reduced Features  :", X_train_pca.shape[1])
print(f"Variance Retained : {variance:.2f}%")
print(f"Model Accuracy    : {accuracy * 100:.2f}%")

print("======================================")


# ==========================================
# 9. CREATE MODEL DIRECTORY
# ==========================================

os.makedirs("model", exist_ok=True)


# ==========================================
# 10. SAVE MODELS
# ==========================================

joblib.dump(scaler, "model/scaler.pkl")
joblib.dump(pca, "model/pca.pkl")
joblib.dump(classifier, "model/classifier.pkl")

print("\nModels saved successfully!")
print("✓ scaler.pkl")
print("✓ pca.pkl")
print("✓ classifier.pkl")