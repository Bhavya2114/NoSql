"""
Streamlit web app
Heart Disease Risk Prediction (UCI Cleveland dataset, 303 patients)

Pages:
  1. Predict   - enter a patient's details, get predictions from all three
                 models, and save the result to MongoDB (app_predictions)
  2. Dashboard - model metrics, dataset numbers from MongoDB, graphs,
                 and the history of predictions made in the app

Run:  streamlit run app.py   (after train_models.py and evaluate_models.py)
"""

import os
from datetime import datetime

import joblib
import pandas as pd
import streamlit as st
from pymongo import MongoClient

st.set_page_config(page_title="Heart Disease Risk", page_icon="❤️", layout="wide")


# ---------------------------------------------------------------
# Load models and connect to MongoDB (cached so it runs only once)
# ---------------------------------------------------------------
@st.cache_resource
def load_models():
    return joblib.load("models/heart_models.pkl")


@st.cache_resource
def get_db():
    return MongoClient("mongodb://localhost:27017/")["heart_disease_db"]


models = load_models()
db = get_db()


def make_input_row(values, columns):
    """Turn the form values into the same columns the models were trained on.
    The category answers become 0/1 columns, the same one-hot encoding as data_preprocessing.py."""
    row = {col: 0 for col in columns}
    for name in ["age", "sex", "trestbps", "chol", "fbs", "thalach", "exang", "oldpeak", "ca"]:
        if name in row:
            row[name] = values[name]
    for name in ["cp", "restecg", "slope", "thal"]:
        col = f"{name}_{values[name]}"   # e.g. cp_4
        if col in row:                   # the first category has no column (drop_first)
            row[col] = 1
    return pd.DataFrame([row], columns=columns)


# ---------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------
st.sidebar.title("❤️ Heart Disease Risk")
page = st.sidebar.radio("Page", ["Predict", "Dashboard"])
st.sidebar.caption("Data: UCI Heart Disease (Cleveland), 303 patients, stored in MongoDB")


# ---------------------------------------------------------------
# Page 1: Predict
# ---------------------------------------------------------------
if page == "Predict":
    st.title("Heart disease risk prediction")
    st.write("Enter the patient's details and press **Predict**.")

    with st.form("patient_form"):
        c1, c2, c3 = st.columns(3)
        with c1:
            name = st.text_input("Patient name (optional)")
            age = st.number_input("Age", 20, 100, 55)
            sex = st.selectbox("Sex", ["Male", "Female"])
            cp = st.selectbox("Chest pain type", ["Typical angina", "Atypical angina",
                                                  "Non-anginal pain", "No symptoms"])
            exang = st.selectbox("Chest pain during exercise?", ["No", "Yes"])
        with c2:
            trestbps = st.number_input("Resting blood pressure (mm Hg)", 80, 220, 130)
            chol = st.number_input("Cholesterol (mg/dl)", 100, 600, 240)
            fbs = st.selectbox("Fasting blood sugar above 120 mg/dl?", ["No", "Yes"])
            restecg = st.selectbox("Resting ECG result", ["Normal", "ST-T wave abnormality",
                                                          "Left ventricle enlarged"])
        with c3:
            thalach = st.number_input("Max heart rate in exercise test", 60, 220, 150)
            oldpeak = st.number_input("ST depression (oldpeak)", 0.0, 7.0, 1.0, step=0.1)
            slope = st.selectbox("ST slope at peak exercise", ["Upsloping", "Flat", "Downsloping"])
            ca = st.selectbox("Major vessels seen in X-ray (0-3)", [0, 1, 2, 3])
            thal = st.selectbox("Thal test result", ["Normal", "Fixed defect", "Reversible defect"])
        submitted = st.form_submit_button("Predict", type="primary")

    if submitted:
        # Convert the answers back to the codes used in the dataset
        values = {
            "age": age, "sex": 1 if sex == "Male" else 0,
            "cp": ["Typical angina", "Atypical angina", "Non-anginal pain", "No symptoms"].index(cp) + 1,
            "trestbps": trestbps, "chol": chol, "fbs": 1 if fbs == "Yes" else 0,
            "restecg": ["Normal", "ST-T wave abnormality", "Left ventricle enlarged"].index(restecg),
            "thalach": thalach, "exang": 1 if exang == "Yes" else 0, "oldpeak": oldpeak,
            "slope": ["Upsloping", "Flat", "Downsloping"].index(slope) + 1, "ca": ca,
            "thal": {"Normal": 3, "Fixed defect": 6, "Reversible defect": 7}[thal],
        }

        # Same columns and same scaling as in training
        X_cls = pd.DataFrame(models["cls_scaler"].transform(make_input_row(values, models["cls_columns"])),
                             columns=models["cls_columns"])
        X_reg = pd.DataFrame(models["reg_scaler"].transform(make_input_row(values, models["reg_columns"])),
                             columns=models["reg_columns"])

        lr_prob = float(models["logistic_regression"].predict_proba(X_cls)[0, 1])
        rf_prob = float(models["random_forest"].predict_proba(X_cls)[0, 1])
        expected_hr = float(models["linear_regression"].predict(X_reg)[0])

        label = lambda p: "Disease" if p >= 0.5 else "No Disease"

        st.subheader("Result")
        if rf_prob >= 0.5:
            st.error(f"⚠️ High risk of heart disease ({rf_prob:.0%} chance, Random Forest)")
        else:
            st.success(f"✅ Low risk of heart disease ({rf_prob:.0%} chance, Random Forest)")

        m1, m2, m3 = st.columns(3)
        m1.metric("Random Forest (best model)", label(rf_prob), f"{rf_prob:.0%} chance", delta_color="off")
        m2.metric("Logistic Regression", label(lr_prob), f"{lr_prob:.0%} chance", delta_color="off")
        m3.metric("Expected max heart rate (Linear Regression)", f"{expected_hr:.0f} bpm",
                  f"{thalach - expected_hr:+.0f} vs entered {thalach}", delta_color="off")

        # Save this prediction to MongoDB
        db["app_predictions"].insert_one({
            "patient_name": name or "Unknown",
            "inputs": values,
            "random_forest": {"prediction": label(rf_prob), "probability": round(rf_prob, 3)},
            "logistic_regression": {"prediction": label(lr_prob), "probability": round(lr_prob, 3)},
            "expected_max_heart_rate": round(expected_hr, 1),
            "created_at": datetime.now(),
        })
        st.caption("Saved to MongoDB: heart_disease_db.app_predictions")


# ---------------------------------------------------------------
# Page 2: Dashboard
# ---------------------------------------------------------------
else:
    st.title("Dashboard")

    # Dataset numbers straight from MongoDB
    total = db["patients"].count_documents({})
    sick = db["patients"].count_documents({"num": {"$gt": 0}})
    made = db["app_predictions"].count_documents({})
    d1, d2, d3, d4 = st.columns(4)
    d1.metric("Patients in MongoDB", total)
    d2.metric("With heart disease", sick)
    d3.metric("Without heart disease", total - sick)
    d4.metric("Predictions made in app", made)

    # Model results saved by evaluate_models.py
    st.subheader("Model results (61 test patients)")
    metrics = list(db["model_metrics"].find({}, {"_id": 0}))
    cls = pd.DataFrame([m for m in metrics if "accuracy" in m]).set_index("model")
    cls = cls[["accuracy", "precision", "recall", "f1"]]
    st.dataframe((cls * 100).round(1).astype(str) + " %", width="stretch")
    reg = next(m for m in metrics if m["model"] == "Linear Regression")
    st.write(f"**Linear Regression (max heart rate):** R² = {reg['r2']}, MAE = {reg['mae']} bpm")

    # Aggregation: disease rate by chest pain type
    st.subheader("Disease rate by chest pain type (MongoDB aggregation)")
    pain = {1: "Typical angina", 2: "Atypical angina", 3: "Non-anginal pain", 4: "No symptoms"}
    rows = db["patients"].aggregate([
        {"$group": {"_id": "$cp", "total": {"$sum": 1},
                    "sick": {"$sum": {"$cond": [{"$gt": ["$num", 0]}, 1, 0]}}}},
        {"$sort": {"_id": 1}},
    ])
    rate = pd.DataFrame([{"Chest pain type": pain[r["_id"]],
                          "% with disease": round(r["sick"] / r["total"] * 100, 1)} for r in rows])
    st.bar_chart(rate.set_index("Chest pain type"))

    # Graphs drawn by evaluate_models.py
    st.subheader("Graphs")
    graphs = ["model_comparison.png", "confusion_matrices.png", "feature_importance.png",
              "heart_rate_actual_vs_predicted.png", "disease_by_age_gender.png"]
    g1, g2 = st.columns(2)
    for i, g in enumerate(graphs):
        path = os.path.join("graphs", g)
        if os.path.exists(path):
            (g1 if i % 2 == 0 else g2).image(path, width="stretch")

    # History of app predictions
    st.subheader("Recent predictions made in the app")
    history = list(db["app_predictions"].find().sort("created_at", -1).limit(10))
    if history:
        st.dataframe(pd.DataFrame([{
            "Time": h["created_at"].strftime("%d %b %H:%M"),
            "Patient": h["patient_name"],
            "Age": h["inputs"]["age"],
            "Random Forest": h["random_forest"]["prediction"],
            "Chance": f"{h['random_forest']['probability']:.0%}",
            "Logistic Regression": h["logistic_regression"]["prediction"],
        } for h in history]), width="stretch", hide_index=True)
    else:
        st.info("No predictions yet. Use the Predict page to make one.")
