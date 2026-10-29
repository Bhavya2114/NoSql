# Heart Disease Risk Prediction using MongoDB and Machine Learning

College CA project. The UCI Heart Disease (Cleveland) dataset of 303 patients is stored in MongoDB with PyMongo, and three models are trained on it:

| Model | Predicts | Result on 61 test patients |
|---|---|---|
| Logistic Regression | Heart disease (yes/no) | Accuracy 82.0%, F1 80.7% |
| Random Forest | Heart disease (yes/no) | Accuracy 86.9%, Recall 89.3%, F1 86.2% |
| Linear Regression | Maximum heart rate | R² 0.22, MAE 16.2 bpm |

A Streamlit app lets you enter a patient's details, see the prediction and save it to MongoDB.

## Files

| File | What it does |
|---|---|
| `phase2_mongodb.py` | Loads the data into MongoDB; CRUD, queries, indexes, aggregations |
| `phase3_preprocessing.py` | Fetches data from MongoDB, fills missing values, encodes, splits, scales |
| `phase4_models.py` | Trains the three models and saves predictions to MongoDB |
| `phase5_evaluation.py` | Calculates the metrics, saves them to MongoDB, draws the graphs |
| `app.py` | Streamlit app (Predict page and Dashboard page) |
| `data/` | Dataset and simple meanings of each column |
| `graphs/` | Evaluation graphs |
| `phaseN_output.txt` | Printed output of each script |

## How to run

Needs Python 3 and MongoDB running on `localhost:27017`.

```
pip install -r requirements.txt
python phase2_mongodb.py
python phase3_preprocessing.py
python phase4_models.py
python phase5_evaluation.py
streamlit run app.py
```

## Dataset

UCI Machine Learning Repository, [Heart Disease dataset](https://archive.ics.uci.edu/dataset/45/heart+disease) (`processed.cleveland.data`).
