from flask import Flask, render_template, request
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix

app = Flask(__name__)

# ==========================================
# LOAD TRAINED MODELS
# ==========================================

scaler = joblib.load("model/scaler.pkl")
pca = joblib.load("model/pca.pkl")
classifier = joblib.load("model/classifier.pkl")

# ==========================================
# LOAD DATASET
# ==========================================

df = pd.read_csv("dataset/Iris_Dataset.csv")

feature_names = [
    "sepal length (cm)",
    "sepal width (cm)",
    "petal length (cm)",
    "petal width (cm)"
]

species_names = [
    "Setosa",
    "Versicolor",
    "Virginica"
]

# ==========================================
# PCA DATA FOR VISUALIZATION
# ==========================================

X = df[feature_names]
y = df["target"]

# Transform complete dataset using already-trained scaler and PCA
X_scaled = scaler.transform(X)
X_pca = pca.transform(X_scaled)

# PCA points for chart
pca_data = {
    "pc1": X_pca[:, 0].round(3).tolist(),
    "pc2": X_pca[:, 1].round(3).tolist(),
    "species": df["species"].str.lower().tolist()
}

# ==========================================
# TEST SET EVALUATION
# ==========================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# IMPORTANT:
# Use the already-trained scaler and PCA
X_test_scaled = scaler.transform(X_test)
X_test_pca = pca.transform(X_test_scaled)

# Prediction on test set
y_test_pred = classifier.predict(X_test_pca)

# Accuracy
model_accuracy = accuracy_score(
    y_test,
    y_test_pred
) * 100

# Confusion Matrix
cm = confusion_matrix(
    y_test,
    y_test_pred
)

confusion_data = cm.tolist()

# ==========================================
# VARIANCE RETAINED
# ==========================================

variance_retained = (
    pca.explained_variance_ratio_.sum() * 100
)


# ==========================================
# DASHBOARD ROUTE
# ==========================================

@app.route("/", methods=["GET", "POST"])
def dashboard():

    prediction = None
    pc1 = None
    pc2 = None

    if request.method == "POST":

        try:

            # Get input values
            sepal_length = float(
                request.form["sepal_length"]
            )

            sepal_width = float(
                request.form["sepal_width"]
            )

            petal_length = float(
                request.form["petal_length"]
            )

            petal_width = float(
                request.form["petal_width"]
            )

            # Create input DataFrame
            input_data = pd.DataFrame(
                [[
                    sepal_length,
                    sepal_width,
                    petal_length,
                    petal_width
                ]],
                columns=feature_names
            )

            # ==================================
            # STANDARDIZATION
            # ==================================

            scaled_data = scaler.transform(
                input_data
            )

            # ==================================
            # PCA TRANSFORMATION
            # ==================================

            pca_input = pca.transform(
                scaled_data
            )

            # PC values
            pc1 = round(
                float(pca_input[0][0]),
                3
            )

            pc2 = round(
                float(pca_input[0][1]),
                3
            )

            # ==================================
            # PREDICTION
            # ==================================

            prediction_class = classifier.predict(
                pca_input
            )[0]

            prediction = species_names[
                prediction_class
            ]

        except Exception as e:

            prediction = "Error: " + str(e)

    # ==========================================
    # SEND DATA TO HTML
    # ==========================================

    return render_template(

        "dashboard.html",

        prediction=prediction,

        pc1=pc1,

        pc2=pc2,

        variance=round(
            variance_retained,
            2
        ),

        model_accuracy=round(
            model_accuracy,
            2
        ),

        pca_data=pca_data,

        confusion_data=confusion_data
    )


# ==========================================
# RUN FLASK
# ==========================================

if __name__ == "__main__":
    app.run(debug=True)