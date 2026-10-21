# Changelog - Actividad 5: Diabetes MLOps Project

All notable changes to the data pipeline, training models, and experiment tracking for this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to Semantic Versioning where applicable.

---

## [1.0.0] - 2026-10-21

### Added
- **Repository Architecture:** Created project file directory layout (`datos/datos_ini`, `datos/datos_limp`, `fuentes/`).
- **Data Acquisition Script (`datos_prep.py`):**
  - Ingestion of public Pima Indians Diabetes Dataset into `datos/datos_ini/diabetes_original.csv`.
  - Implemented missing data cleaning replacing biologically impossible zeros (`Glucose`, `BloodPressure`, `SkinThickness`, `Insulin`, `BMI`) with column medians.
  - Automated duplicate row removal and dataset standardization.
  - Exported clean dataset to `datos/datos_limp/diabetes_limpio.csv`.
- **Exploratory Data Analysis:** Added feature correlation heatmap and feature distribution plotting functions.
- **Model Training & Experimentation (`train.py` / `entrena.ipynb`):**
  - Stratified 80/20 train-test split with fixed seed (`random_state=42`) for full reproducibility.
  - Feature scaling pipeline using `StandardScaler`.
  - Grid Search 5-Fold Cross Validation (`GridSearchCV`) for Logistic Regression and Random Forest models.
- **MLOps & MLflow Integration:**
  - Automated experiment tracking under `Proyecto_Diabetes_MLOps`.
  - Logging of hyperparameter configurations ($C$, `n_estimators`, `max_depth`).
  - Standardized metrics tracking (`accuracy`, `precision`, `recall`, `f1_score`, `roc_auc`).
  - Model serialization logging via `mlflow.sklearn.log_model` using `cloudpickle` format.
  - Visual evaluation artifacts (Confusion Matrix, Comparative ROC Curves).
- **Documentation:** Created complete `README.md` guide and technical overview.

---

## [0.1.0] - Initial Setup

### Added
- Repository initialization on GitHub with basic structure.