"""
Model training
Heart Disease Risk Prediction (UCI Cleveland dataset, 303 patients)

Models:
  1. Logistic Regression -> does the patient have heart disease? (yes/no)
  2. Random Forest       -> same question, to compare with model 1
  3. Linear Regression   -> predict the maximum heart rate (thalach)

Steps:
  1. Get the cleaned, split and scaled data from data_preprocessing.py
  2. Train the three models on the training patients
  3. Predict for the test patients
  4. Save the predictions to MongoDB (predictions collection)
  5. Save the trained models to disk for the Streamlit app

Run:  python train_models.py
Full evaluation (metrics and graphs) is done in evaluate_models.py.
"""

import os
from datetime import datetime

import joblib
import pandas as pd
from pymongo import MongoClient
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression

from data_preprocessing import prepare_data


def title(text):
    print("\n" + "=" * 60)
    print(text)
    print("=" * 60)


# ---------------------------------------------------------------
# 1. Data from data_preprocessing.py
# ---------------------------------------------------------------
title("1. Load prepared data (from MongoDB via data_preprocessing.py)")
data = prepare_data()
print("Training patients:", len(data["X_cls_train"]), "| Test patients:", len(data["X_cls_test"]))


# ---------------------------------------------------------------
# 2. Train the models
# ---------------------------------------------------------------
title("2. Train the models")

# Model 1: Logistic Regression
# Gives each input a weight, adds them up and turns the total into a
# probability between 0 and 1. Above 0.5 means "Disease".
log_reg = LogisticRegression(max_iter=1000)
log_reg.fit(data["X_cls_train"], data["y_cls_train"])
print("Logistic Regression trained")

# Model 2: Random Forest
# Builds 100 decision trees on random parts of the data; each tree votes
# and the majority wins.
rand_forest = RandomForestClassifier(n_estimators=100, random_state=42)
rand_forest.fit(data["X_cls_train"], data["y_cls_train"])
print("Random Forest trained (100 trees)")

# Model 3: Linear Regression
# Fits a straight-line formula: heart rate = b0 + b1*age + b2*sex + ...
lin_reg = LinearRegression()
lin_reg.fit(data["X_reg_train"], data["y_reg_train"])
print("Linear Regression trained")


# ---------------------------------------------------------------
# 3. Predict for the test patients
# ---------------------------------------------------------------
title("3. Predictions on the 61 test patients")
lr_pred = log_reg.predict(data["X_cls_test"])
lr_prob = log_reg.predict_proba(data["X_cls_test"])[:, 1]   # chance of disease
rf_pred = rand_forest.predict(data["X_cls_test"])
rf_prob = rand_forest.predict_proba(data["X_cls_test"])[:, 1]
hr_pred = lin_reg.predict(data["X_reg_test"])

results = pd.DataFrame({
    "patient_id": data["test_patient_ids"].values,
    "actual_target": data["y_cls_test"].values,
    "logistic_pred": lr_pred,
    "logistic_prob": lr_prob.round(3),
    "forest_pred": rf_pred,
    "forest_prob": rf_prob.round(3),
    "actual_thalach": data["y_reg_test"].values,
    "predicted_thalach": hr_pred.round(1),
})
print(results.head(10).to_string(index=False))

# Quick check only (full metrics in evaluate_models.py)
print("\nQuick check, share of test patients predicted correctly:")
print("  Logistic Regression:", round((results.logistic_pred == results.actual_target).mean() * 100, 1), "%")
print("  Random Forest:      ", round((results.forest_pred == results.actual_target).mean() * 100, 1), "%")
print("  Linear Regression average error:",
      round((results.actual_thalach - results.predicted_thalach).abs().mean(), 1), "beats per minute")


# ---------------------------------------------------------------
# 4. Save predictions to MongoDB
# ---------------------------------------------------------------
title("4. Save predictions to MongoDB")
client = MongoClient("mongodb://localhost:27017/")
pred_col = client["heart_disease_db"]["predictions"]
pred_col.drop()

labels = {0: "No Disease", 1: "Disease"}
docs = []
for row in results.to_dict("records"):
    docs.append({
        "patient_id": row["patient_id"],
        "actual": labels[int(row["actual_target"])],
        "logistic_regression": {"prediction": labels[int(row["logistic_pred"])],
                                "probability": float(row["logistic_prob"])},
        "random_forest": {"prediction": labels[int(row["forest_pred"])],
                          "probability": float(row["forest_prob"])},
        "max_heart_rate": {"actual": int(row["actual_thalach"]),
                           "predicted": float(row["predicted_thalach"])},
        "created_at": datetime.now(),
    })
pred_col.insert_many(docs)
pred_col.create_index("patient_id", unique=True)
print("Saved", pred_col.count_documents({}), "prediction documents to heart_disease_db.predictions")
print("\nExample document:")
print(pred_col.find_one({}, {"_id": 0}))

# A query on the saved predictions: patients the two models disagree on
disagree = pred_col.count_documents(
    {"$expr": {"$ne": ["$logistic_regression.prediction", "$random_forest.prediction"]}})
print("\nTest patients where the two models disagree:", disagree)


# ---------------------------------------------------------------
# 5. Save the trained models for the Streamlit app
# ---------------------------------------------------------------
title("5. Save trained models")
os.makedirs("models", exist_ok=True)
joblib.dump({
    "logistic_regression": log_reg,
    "random_forest": rand_forest,
    "linear_regression": lin_reg,
    "cls_scaler": data["cls_scaler"],
    "reg_scaler": data["reg_scaler"],
    "cls_columns": list(data["X_cls_train"].columns),
    "reg_columns": list(data["X_reg_train"].columns),
}, "models/heart_models.pkl")
print("Saved models/heart_models.pkl")

print("\nDone.")
