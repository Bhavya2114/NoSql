"""
Phase 5: Evaluation and graphs
Heart Disease Risk Prediction (UCI Cleveland dataset, 303 patients)

Steps:
  1. Read the test predictions from MongoDB (saved in Phase 4)
  2. Classification metrics: Accuracy, Precision, Recall, F1
  3. Regression metrics: R² and MAE
  4. Save all metrics to MongoDB (model_metrics collection)
  5. Draw the graphs into the graphs/ folder

Run:  python phase5_evaluation.py   (after phase4_models.py)
"""

import os

import joblib
import matplotlib
matplotlib.use("Agg")  # save graphs to files without opening windows
import matplotlib.pyplot as plt
import pandas as pd
from pymongo import MongoClient
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             mean_absolute_error, precision_score, r2_score,
                             recall_score)


def title(text):
    print("\n" + "=" * 60)
    print(text)
    print("=" * 60)


# ---------------------------------------------------------------
# 1. Read predictions from MongoDB
# ---------------------------------------------------------------
title("1. Read predictions from MongoDB")
client = MongoClient("mongodb://localhost:27017/")
db = client["heart_disease_db"]
docs = list(db["predictions"].find({}, {"_id": 0}))
print("Loaded", len(docs), "test patients from heart_disease_db.predictions")

# 1 = Disease, 0 = No Disease
to_num = {"No Disease": 0, "Disease": 1}
actual = [to_num[d["actual"]] for d in docs]
predicted = {
    "Logistic Regression": [to_num[d["logistic_regression"]["prediction"]] for d in docs],
    "Random Forest": [to_num[d["random_forest"]["prediction"]] for d in docs],
}
hr_actual = [d["max_heart_rate"]["actual"] for d in docs]
hr_predicted = [d["max_heart_rate"]["predicted"] for d in docs]


# ---------------------------------------------------------------
# 2. Classification metrics
# ---------------------------------------------------------------
title("2. Classification metrics (disease yes/no)")
print("Accuracy  = share of all patients predicted correctly")
print("Precision = of patients we said have disease, how many really do")
print("Recall    = of patients who really have disease, how many we caught")
print("F1        = balance of precision and recall\n")

cls_metrics = {}
for name, pred in predicted.items():
    cls_metrics[name] = {
        "accuracy": round(accuracy_score(actual, pred), 3),
        "precision": round(precision_score(actual, pred), 3),
        "recall": round(recall_score(actual, pred), 3),
        "f1": round(f1_score(actual, pred), 3),
    }
print(pd.DataFrame(cls_metrics).T.to_string())

for name, pred in predicted.items():
    tn, fp, fn, tp = confusion_matrix(actual, pred).ravel()
    print(f"\n{name}: {tp} sick caught, {tn} healthy correct, "
          f"{fp} false alarms, {fn} sick patients missed")


# ---------------------------------------------------------------
# 3. Regression metrics
# ---------------------------------------------------------------
title("3. Regression metrics (max heart rate)")
r2 = round(r2_score(hr_actual, hr_predicted), 3)
mae = round(mean_absolute_error(hr_actual, hr_predicted), 2)
print("R²  =", r2, " (1.0 = perfect, 0 = no better than guessing the average)")
print("MAE =", mae, "beats per minute (average size of the error)")


# ---------------------------------------------------------------
# 4. Save metrics to MongoDB
# ---------------------------------------------------------------
title("4. Save metrics to MongoDB")
metrics_col = db["model_metrics"]
metrics_col.drop()
metrics_col.insert_many(
    [{"model": name, "task": "Heart disease (yes/no)", **m} for name, m in cls_metrics.items()]
    + [{"model": "Linear Regression", "task": "Max heart rate", "r2": r2, "mae": mae}])
best = metrics_col.find_one({"accuracy": {"$exists": True}}, sort=[("accuracy", -1)])
print("Saved", metrics_col.count_documents({}), "documents to heart_disease_db.model_metrics")
print("Best classification model (query sorted by accuracy):", best["model"])


# ---------------------------------------------------------------
# 5. Graphs
# ---------------------------------------------------------------
title("5. Graphs")
os.makedirs("graphs", exist_ok=True)
saved = []

# a) Confusion matrices for both models
fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
for ax, (name, pred) in zip(axes, predicted.items()):
    cm = confusion_matrix(actual, pred)
    ax.imshow(cm, cmap="Blues")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, cm[i, j], ha="center", va="center", fontsize=18,
                    color="white" if cm[i, j] > cm.max() / 2 else "black")
    ax.set_xticks([0, 1], ["No Disease", "Disease"])
    ax.set_yticks([0, 1], ["No Disease", "Disease"])
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(name)
fig.suptitle("Confusion matrices (61 test patients)")
fig.tight_layout()
fig.savefig("graphs/confusion_matrices.png", dpi=120)
saved.append("confusion_matrices.png")

# b) Model comparison bar chart
fig, ax = plt.subplots(figsize=(8, 4.5))
names = ["accuracy", "precision", "recall", "f1"]
x = range(len(names))
for k, (name, m) in enumerate(cls_metrics.items()):
    vals = [m[n] for n in names]
    bars = ax.bar([i + k * 0.38 for i in x], vals, width=0.38, label=name)
    ax.bar_label(bars, fmt="%.2f", fontsize=9)
ax.set_xticks([i + 0.19 for i in x], ["Accuracy", "Precision", "Recall", "F1"])
ax.set_ylim(0, 1.2)
ax.set_title("Logistic Regression vs Random Forest")
ax.legend(loc="upper center", ncol=2)
fig.tight_layout()
fig.savefig("graphs/model_comparison.png", dpi=120)
saved.append("model_comparison.png")

# c) Actual vs predicted max heart rate
fig, ax = plt.subplots(figsize=(6, 6))
ax.scatter(hr_actual, hr_predicted, alpha=0.7)
low, high = min(hr_actual + hr_predicted) - 5, max(hr_actual + hr_predicted) + 5
ax.plot([low, high], [low, high], "r--", label="Perfect prediction")
ax.set_xlabel("Actual max heart rate")
ax.set_ylabel("Predicted max heart rate")
ax.set_title(f"Linear Regression: R² = {r2}, MAE = {mae}")
ax.legend()
fig.tight_layout()
fig.savefig("graphs/heart_rate_actual_vs_predicted.png", dpi=120)
saved.append("heart_rate_actual_vs_predicted.png")

# d) Which inputs matter most (Random Forest feature importance)
models = joblib.load("models/heart_models.pkl")
readable = {"age": "Age", "sex": "Sex (male)", "trestbps": "Resting BP", "chol": "Cholesterol",
            "fbs": "High fasting sugar", "thalach": "Max heart rate", "exang": "Exercise chest pain",
            "oldpeak": "ST depression", "ca": "Blocked vessels", "cp_2": "Chest pain: atypical",
            "cp_3": "Chest pain: non-anginal", "cp_4": "Chest pain: none", "restecg_1": "ECG: ST-T abnormal",
            "restecg_2": "ECG: LV enlarged", "slope_2": "Slope: flat", "slope_3": "Slope: down",
            "thal_6": "Thal: fixed defect", "thal_7": "Thal: reversible defect"}
importance = pd.Series(models["random_forest"].feature_importances_,
                       index=[readable[c] for c in models["cls_columns"]]).sort_values()
fig, ax = plt.subplots(figsize=(8, 6))
importance.plot.barh(ax=ax)
ax.set_title("Random Forest: which inputs matter most")
ax.set_xlabel("Importance")
fig.tight_layout()
fig.savefig("graphs/feature_importance.png", dpi=120)
saved.append("feature_importance.png")
print("Top 5 most important inputs:", list(importance.sort_values(ascending=False).index[:5]))

# e) Disease rate by age group and gender (from the patients collection)
patients = pd.DataFrame(list(db["patients"].find({}, {"_id": 0, "age": 1, "sex": 1, "num": 1})))
patients["has_disease"] = patients["num"] > 0
patients["age_group"] = pd.cut(patients["age"], [0, 40, 50, 60, 70, 120],
                               labels=["<40", "40-49", "50-59", "60-69", "70+"], right=False)
patients["gender"] = patients["sex"].map({1: "Male", 0: "Female"})
rate = patients.groupby(["age_group", "gender"], observed=True)["has_disease"].mean().unstack() * 100
fig, ax = plt.subplots(figsize=(8, 4.5))
rate.plot.bar(ax=ax, rot=0)
ax.set_ylabel("% with heart disease")
ax.set_xlabel("Age group")
ax.set_title("Heart disease rate by age and gender")
fig.tight_layout()
fig.savefig("graphs/disease_by_age_gender.png", dpi=120)
saved.append("disease_by_age_gender.png")

print("Saved graphs in graphs/:")
for s in saved:
    print("  ", s)

print("\nPhase 5 complete.")
