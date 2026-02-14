# Heart Disease Prediction — ML Assignment

## Problem Statement
Heart disease is one of the leading causes of death worldwide. Early detection can significantly improve patient outcomes and reduce healthcare costs.

The objective of this project is to build and evaluate multiple machine learning classification models to predict whether a patient has heart disease based on clinical and demographic features.

The models are trained on historical patient data and evaluated using multiple performance metrics to determine the most reliable model for prediction.

---

## Dataset Description
The dataset used in this project contains medical and demographic attributes of patients, along with a target variable indicating the presence of heart disease.

### Target Variable
- **HeartDisease**
  - `1` → Presence of heart disease  
  - `0` → No heart disease  

### Feature Categories
The dataset includes health, lifestyle, and demographic attributes that are known risk factors for heart disease:

- **BMI** — Body Mass Index, indicating body fat based on height and weight.
- **Smoking** — Whether the individual has smoked at least 100 cigarettes in their lifetime.
- **AlcoholDrinking** — Indicates heavy alcohol consumption.
- **Stroke** — Whether the individual has ever had a stroke.
- **DiffWalking** — Difficulty walking or climbing stairs, indicating mobility issues.
- **Sex** — Biological sex of the individual.
- **AgeCategory** — Age group of the individual.
- **Race** — Self-identified race of the individual.
- **Diabetic** — Indicates whether the individual has diabetes or pre-diabetes.
- **PhysicalActivity** — Whether the individual engages in regular physical activity.
- **GenHealth** — Self-reported general health status.
- **SleepTime** — Average number of hours of sleep per day.
- **Asthma** — Whether the individual has asthma.
- **KidneyDisease** — Whether the individual has kidney disease.
- **SkinCancer** — Whether the individual has had skin cancer.

### Dataset Usage
- Training set → Model training  
- Test set → Model evaluation  

---

## Models Used
The following classification models were implemented and evaluated:

1. Logistic Regression  
2. Decision Tree Classifier  
3. K-Nearest Neighbors (kNN)  
4. Naive Bayes (Gaussian)  
5. Random Forest (Ensemble)  
6. XGBoost (Ensemble)  

Each model was evaluated using:

- Accuracy  
- AUC Score  
- Precision  
- Recall  
- F1 Score  
- Matthews Correlation Coefficient (MCC)  

---

## Model Performance Comparison

### Evaluation Metrics on Test Dataset

| Model | Accuracy | AUC | Precision | Recall | F1 | MCC |
|-------|----------|-----|-----------|--------|-----|-----|
| Logistic Regression | 0.75 | 0.84 | 0.22 | 0.77 | 0.34 | 0.31 |
| Random Forest       | 0.76 | 0.82 | 0.22 | 0.71 | 0.34 | 0.30 |
| XGBoost             | 0.75 | 0.82 | 0.22 | 0.74 | 0.34 | 0.30 |
| Naive Bayes         | 0.75 | 0.82 | 0.21 | 0.73 | 0.33 | 0.30 |
| Decision Tree       | 0.71 | 0.80 | 0.20 | 0.75 | 0.31 | 0.28 |
| kNN                 | 0.89 | 0.70 | 0.30 | 0.17 | 0.22 | 0.17 |


---

## Model Observations

| ML Model | Observation about model performance |
|----------|------------------------------------|
| **Logistic Regression** | Balanced performance with strong recall and highest AUC. Suitable for detecting heart disease cases. |
| **Decision Tree** | Lower accuracy and MCC, indicating overfitting and instability compared to ensemble methods. |
| **kNN** | Highest accuracy but extremely low recall, meaning it fails to detect many heart disease cases — not suitable for medical use. |
| **Naive Bayes** | Stable performance but slightly lower precision; assumptions of feature independence may limit performance. |
| **Random Forest** | Strong overall performance with good balance across metrics; robust against overfitting. |
| **XGBoost** | Comparable to Random Forest with strong recall and balanced metrics; effective ensemble model. |

---
