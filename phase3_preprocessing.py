"""
Phase 3: Data preprocessing
Heart Disease Risk Prediction (UCI Cleveland dataset, 303 patients)

Steps:
  1. Fetch the patients from MongoDB into a pandas DataFrame
  2. Check and fill missing values
  3. Create the yes/no target column (heart disease or not)
  4. Encode the category columns (one-hot encoding)
  5. Split into train (80%) and test (20%) sets
  6. Scale the numbers (StandardScaler)
  7. Save the cleaned data back to MongoDB (patients_clean)

Run:  python phase3_preprocessing.py
Phase 4 imports prepare_data() from this file.
"""

import pandas as pd
from pymongo import MongoClient
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Columns whose numbers are codes for a category (not real amounts)
CATEGORY_COLUMNS = ["cp", "restecg", "slope", "thal"]


def load_from_mongodb():
    """Step 1: read every patient document from MongoDB into pandas."""
    client = MongoClient("mongodb://localhost:27017/")
    docs = list(client["heart_disease_db"]["patients"].find({}, {"_id": 0}))
    return pd.DataFrame(docs)


def clean(df):
    """Steps 2 to 4: fill missing values, create target, encode categories."""
    df = df.copy()

    # Step 2: fill missing values with the most common value (mode)
    for col in ["ca", "thal"]:
        df[col] = df[col].fillna(df[col].mode()[0]).astype(int)

    # Step 3: target = 1 if any heart disease (num 1 to 4), else 0
    df["target"] = (df["num"] > 0).astype(int)

    # Step 4: one-hot encode the category columns
    # e.g. cp (1,2,3,4) becomes cp_2, cp_3, cp_4 with 0/1 values
    df[CATEGORY_COLUMNS] = df[CATEGORY_COLUMNS].astype(int)
    df = pd.get_dummies(df, columns=CATEGORY_COLUMNS, drop_first=True, dtype=int)
    return df


def prepare_data():
    """Run all steps and return everything the models in Phase 4 need."""
    raw = load_from_mongodb()
    df = clean(raw)

    # Columns not used as inputs: the ID, the original diagnosis and its labels
    drop = ["patient_id", "num", "diagnosis", "target"]

    # Model 1 and 2 (classification): predict target from all health values
    X_cls = df.drop(columns=drop)
    y_cls = df["target"]

    # Model 3 (regression): predict max heart rate (thalach) from the others
    X_reg = df.drop(columns=drop + ["thalach"])
    y_reg = df["thalach"]

    # Step 5: one 80/20 split of patients, used by all models.
    # stratify keeps the same disease/no-disease ratio in train and test.
    train_idx, test_idx = train_test_split(
        df.index, test_size=0.2, random_state=42, stratify=y_cls)

    # Step 6: scale numbers to the same range (mean 0, std 1).
    # The scaler learns only from the training data, so the test set stays unseen.
    cls_scaler = StandardScaler().fit(X_cls.loc[train_idx])
    reg_scaler = StandardScaler().fit(X_reg.loc[train_idx])

    def scaled(scaler, X, idx):
        return pd.DataFrame(scaler.transform(X.loc[idx]), columns=X.columns, index=idx)

    return {
        "raw": raw, "clean": df,
        "X_cls_train": scaled(cls_scaler, X_cls, train_idx),
        "X_cls_test": scaled(cls_scaler, X_cls, test_idx),
        "y_cls_train": y_cls.loc[train_idx], "y_cls_test": y_cls.loc[test_idx],
        "X_reg_train": scaled(reg_scaler, X_reg, train_idx),
        "X_reg_test": scaled(reg_scaler, X_reg, test_idx),
        "y_reg_train": y_reg.loc[train_idx], "y_reg_test": y_reg.loc[test_idx],
        "cls_scaler": cls_scaler, "reg_scaler": reg_scaler,
        "test_patient_ids": df.loc[test_idx, "patient_id"],
    }


def title(text):
    print("\n" + "=" * 60)
    print(text)
    print("=" * 60)


if __name__ == "__main__":
    data = prepare_data()
    raw, df = data["raw"], data["clean"]

    title("1. Data fetched from MongoDB")
    print("Shape (rows, columns):", raw.shape)
    print(raw.head())

    title("2. Missing values")
    print("Before cleaning:")
    print(raw.isna().sum()[raw.isna().sum() > 0])
    print("Filled ca with", raw["ca"].mode()[0], "and thal with", raw["thal"].mode()[0],
          "(the most common value)")
    print("After cleaning, missing values left:", df.isna().sum().sum())

    title("3. Target column")
    print("target = 1 if num > 0 (has heart disease), else 0")
    print(df["target"].value_counts().rename({0: "No Disease (0)", 1: "Disease (1)"}))

    title("4. One-hot encoding of category columns")
    print("Encoded:", CATEGORY_COLUMNS)
    print("New columns:", [c for c in df.columns if c.split("_")[0] in CATEGORY_COLUMNS])

    title("5. Train / test split (80% / 20%)")
    print("Train patients:", len(data["X_cls_train"]), "| Test patients:", len(data["X_cls_test"]))
    print("Disease share in train:", round(data["y_cls_train"].mean() * 100, 1), "%",
          "| in test:", round(data["y_cls_test"].mean() * 100, 1), "%")
    print("Classification inputs:", data["X_cls_train"].shape[1], "columns")
    print("Regression inputs (no thalach):", data["X_reg_train"].shape[1], "columns")

    title("6. Scaling (StandardScaler)")
    print("Training data after scaling, first 5 columns (mean is about 0, std about 1):")
    print(data["X_cls_train"].iloc[:, :5].describe().loc[["mean", "std"]].round(2))

    title("7. Save cleaned data to MongoDB")
    client = MongoClient("mongodb://localhost:27017/")
    clean_col = client["heart_disease_db"]["patients_clean"]
    clean_col.drop()
    clean_col.insert_many(df.to_dict("records"))
    print("Saved", clean_col.count_documents({}), "cleaned patients to heart_disease_db.patients_clean")

    print("\nPhase 3 complete.")
