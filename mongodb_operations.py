"""
MongoDB / PyMongo operations
Heart Disease Risk Prediction (UCI Cleveland dataset, 303 patients)

What this file shows:
  1. Connect to local MongoDB
  2. CREATE  - insert all patients (insert_many) and one new patient (insert_one)
  3. READ    - find with filters, projection, sorting, counting
  4. UPDATE  - update_one and update_many
  5. DELETE  - delete_one
  6. INDEX   - create indexes and prove MongoDB uses them
  7. AGGREGATION - group-by style reports on the patients

Run:  python mongodb_operations.py
"""

import pandas as pd
from pymongo import MongoClient, ASCENDING, DESCENDING


def title(text):
    print("\n" + "=" * 60)
    print(text)
    print("=" * 60)


# ---------------------------------------------------------------
# 1. CONNECT
# ---------------------------------------------------------------
title("1. Connect to MongoDB")
client = MongoClient("mongodb://localhost:27017/")
db = client["heart_disease_db"]
patients = db["patients"]
print("Connected. MongoDB version:", client.server_info()["version"])
print("Database:", db.name, "| Collection:", patients.name)


# ---------------------------------------------------------------
# 2. CREATE - load the CSV and insert every patient
# ---------------------------------------------------------------
title("2. CREATE - insert all patients from the CSV")
df = pd.read_csv("data/heart_raw.csv")

# Every column except oldpeak holds whole numbers, so store them as integers
for col in df.columns:
    if col != "oldpeak":
        df[col] = df[col].astype("Int64")

# Give every patient a simple ID: P001, P002, ...
df.insert(0, "patient_id", [f"P{i:03d}" for i in range(1, len(df) + 1)])

# Turn each row into a dictionary (one MongoDB document).
# Missing values (NaN) become None, which MongoDB stores as null.
records = df.astype(object).where(df.notna(), None).to_dict("records")

patients.drop()  # start fresh every time the script runs
result = patients.insert_many(records)
print("Inserted", len(result.inserted_ids), "patients")

print("\nOne document looks like this:")
print(patients.find_one({}, {"_id": 0}))

# insert_one: add a new (made-up) patient to show single insert
new_patient = {
    "patient_id": "P999", "age": 45, "sex": 1, "cp": 2, "trestbps": 130,
    "chol": 240, "fbs": 0, "restecg": 0, "thalach": 160, "exang": 0,
    "oldpeak": 0.5, "slope": 1, "ca": 0, "thal": 3, "num": 0,
}
patients.insert_one(new_patient)
print("\nInserted test patient P999. Total now:", patients.count_documents({}))


# ---------------------------------------------------------------
# 3. READ - queries
# ---------------------------------------------------------------
title("3. READ - queries")

# a) Patients older than 60 with cholesterol above 300
query = {"age": {"$gt": 60}, "chol": {"$gt": 300}}
print("a) Age > 60 AND cholesterol > 300:", patients.count_documents(query), "patients")
for p in patients.find(query, {"_id": 0, "patient_id": 1, "age": 1, "chol": 1}).limit(5):
    print("  ", p)

# b) Top 5 highest cholesterol (projection + sort + limit)
print("\nb) Top 5 highest cholesterol:")
for p in patients.find({}, {"_id": 0, "patient_id": 1, "age": 1, "chol": 1}) \
                 .sort("chol", DESCENDING).limit(5):
    print("  ", p)

# c) Patients with heart disease (num 1 to 4)
print("\nc) Patients with heart disease (num > 0):",
      patients.count_documents({"num": {"$gt": 0}}))

# d) Women with chest pain from exercise ($and of two conditions)
print("d) Women who get chest pain during exercise:",
      patients.count_documents({"sex": 0, "exang": 1}))

# e) Rows with missing values (cleaned later in data_preprocessing.py)
missing = {"$or": [{"ca": None}, {"thal": None}]}
print("e) Patients with a missing ca or thal value:", patients.count_documents(missing))
for p in patients.find(missing, {"_id": 0, "patient_id": 1, "ca": 1, "thal": 1}):
    print("  ", p)


# ---------------------------------------------------------------
# 4. UPDATE
# ---------------------------------------------------------------
title("4. UPDATE")

# update_one: change the test patient's cholesterol
patients.update_one({"patient_id": "P999"}, {"$set": {"chol": 210}})
print("update_one -> P999 cholesterol is now",
      patients.find_one({"patient_id": "P999"})["chol"])

# update_many: add a readable 'diagnosis' label to every patient
r1 = patients.update_many({"num": 0}, {"$set": {"diagnosis": "No Disease"}})
r2 = patients.update_many({"num": {"$gt": 0}}, {"$set": {"diagnosis": "Disease"}})
print("update_many -> labelled", r1.modified_count, "as 'No Disease' and",
      r2.modified_count, "as 'Disease'")


# ---------------------------------------------------------------
# 5. DELETE
# ---------------------------------------------------------------
title("5. DELETE")
r = patients.delete_one({"patient_id": "P999"})
print("delete_one -> removed", r.deleted_count, "test patient. Total now:",
      patients.count_documents({}))


# ---------------------------------------------------------------
# 6. INDEXES
# ---------------------------------------------------------------
title("6. INDEXES")
patients.create_index([("patient_id", ASCENDING)], unique=True)
patients.create_index([("age", ASCENDING)])
print("Indexes:", list(patients.index_information().keys()))

# explain() shows whether MongoDB used the index (IXSCAN) or scanned everything (COLLSCAN)
plan = patients.find({"age": {"$gt": 60}}).explain()["queryPlanner"]["winningPlan"]
stage = plan.get("inputStage", plan).get("stage")
print("Query on age uses stage:", stage, "(IXSCAN = index used)")


# ---------------------------------------------------------------
# 7. AGGREGATIONS
# ---------------------------------------------------------------
title("7. AGGREGATIONS")

# a) How many patients have / do not have heart disease
print("a) Patients by diagnosis:")
for row in patients.aggregate([
    {"$group": {"_id": "$diagnosis", "count": {"$sum": 1}}},
    {"$sort": {"count": -1}},
]):
    print("  ", row)

# b) Average health values for each group
print("\nb) Average values by diagnosis:")
for row in patients.aggregate([
    {"$group": {
        "_id": "$diagnosis",
        "avg_age": {"$avg": "$age"},
        "avg_cholesterol": {"$avg": "$chol"},
        "avg_blood_pressure": {"$avg": "$trestbps"},
        "avg_max_heart_rate": {"$avg": "$thalach"},
    }},
    {"$project": {"avg_age": {"$round": ["$avg_age", 1]},
                  "avg_cholesterol": {"$round": ["$avg_cholesterol", 1]},
                  "avg_blood_pressure": {"$round": ["$avg_blood_pressure", 1]},
                  "avg_max_heart_rate": {"$round": ["$avg_max_heart_rate", 1]}}},
]):
    print("  ", row)

# c) Disease rate for men and women
print("\nc) Disease rate by gender:")
for row in patients.aggregate([
    {"$group": {
        "_id": {"$cond": [{"$eq": ["$sex", 1]}, "Male", "Female"]},
        "total": {"$sum": 1},
        "with_disease": {"$sum": {"$cond": [{"$gt": ["$num", 0]}, 1, 0]}},
    }},
    {"$project": {"total": 1, "with_disease": 1,
                  "disease_percent": {"$round": [
                      {"$multiply": [{"$divide": ["$with_disease", "$total"]}, 100]}, 1]}}},
]):
    print("  ", row)

# d) Disease rate by age group ($bucket)
print("\nd) Disease rate by age group:")
for row in patients.aggregate([
    {"$bucket": {
        "groupBy": "$age",
        "boundaries": [0, 40, 50, 60, 70, 120],
        "output": {"total": {"$sum": 1},
                   "with_disease": {"$sum": {"$cond": [{"$gt": ["$num", 0]}, 1, 0]}}},
    }},
]):
    labels = {0: "below 40", 40: "40-49", 50: "50-59", 60: "60-69", 70: "70+"}
    pct = round(row["with_disease"] / row["total"] * 100, 1)
    print(f"   {labels[row['_id']]:>9}: {row['total']:3d} patients, {pct}% with disease")

# e) Disease rate by chest pain type
print("\ne) Disease rate by chest pain type:")
pain_names = {1: "Typical angina", 2: "Atypical angina",
              3: "Non-anginal pain", 4: "No symptoms"}
for row in patients.aggregate([
    {"$group": {"_id": "$cp", "total": {"$sum": 1},
                "with_disease": {"$sum": {"$cond": [{"$gt": ["$num", 0]}, 1, 0]}}}},
    {"$sort": {"_id": 1}},
]):
    pct = round(row["with_disease"] / row["total"] * 100, 1)
    print(f"   {pain_names[int(row['_id'])]:>16}: {row['total']:3d} patients, {pct}% with disease")

print("\nDone. Data is stored in heart_disease_db.patients")
