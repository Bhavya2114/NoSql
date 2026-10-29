# UCI Heart Disease (Cleveland) – Column Meanings

Source: UCI Machine Learning Repository, `processed.cleveland.data`
Size: 303 patients, 14 columns. Missing values: `ca` (4 rows), `thal` (2 rows), shown as `?` in the raw file.

| Column | Simple meaning | Values |
|---|---|---|
| age | Patient's age in years | 29 to 77 |
| sex | Gender | 1 = male, 0 = female |
| cp | Chest pain type | 1 = typical angina, 2 = atypical angina, 3 = non-anginal pain, 4 = no symptoms (asymptomatic) |
| trestbps | Resting blood pressure when admitted (mm Hg) | 94 to 200 |
| chol | Cholesterol level in blood (mg/dl) | 126 to 564 |
| fbs | Is fasting blood sugar above 120 mg/dl? | 1 = yes, 0 = no |
| restecg | Resting ECG (heart electrical test) result | 0 = normal, 1 = ST-T wave abnormality, 2 = left ventricle enlarged |
| thalach | Maximum heart rate reached during exercise test | 71 to 202 (target for Linear Regression) |
| exang | Did exercise cause chest pain? | 1 = yes, 0 = no |
| oldpeak | How much the ECG "ST" line drops during exercise compared to rest (higher = worse) | 0.0 to 6.2 |
| slope | Shape of the ST line at peak exercise | 1 = upsloping (good), 2 = flat, 3 = downsloping (bad) |
| ca | Number of major blood vessels seen in an X-ray dye test | 0 to 3 |
| thal | Thalassemia (blood disorder) test result | 3 = normal, 6 = fixed defect, 7 = reversible defect |
| num | Heart disease diagnosis | 0 = no disease, 1 to 4 = disease (increasing severity) |

## Target for classification
`num` will be turned into a yes/no column `target`: 0 = no disease (164 patients), 1 = disease (139 patients, i.e. num 1 to 4). This gives a fairly balanced dataset for Logistic Regression and Random Forest.
