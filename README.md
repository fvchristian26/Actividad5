# Actividad 5: Diabetes Prediction MLOps Pipeline

## Project Overview
This project implements an end-to-end Machine Learning pipeline for binary classification on the **Pima Indians Diabetes Dataset**. The primary goal is to predict whether a patient has diabetes based on diagnostic measurements. 

Beyond model training, this project incorporates core **MLOps practices**—including dataset versioning, automated cleaning and standardization, hyperparameter optimization via cross-validation, experiment tracking using **MLflow**, and reproducible model artifacts storage.

---

## Repository Structure

```text
Actividad5/
├── datos/
│   ├── datos_ini/          # Raw, original dataset (diabetes_original.csv)
│   └── datos_limp/         # Cleaned, standardized dataset (diabetes_limpio.csv)
├── fuentes/
│   ├── entrena.ipynb       # Complete execution Notebook (Colab / Jupyter)
│   ├── datos_prep.py       # Data loading, cleaning, and preprocessing script
│   └── train.py            # Model training, Grid Search CV, and MLflow logging
├── README.md               # Project documentation and execution guide
└── CHANGELOG.md            # Versioning history and technical change records
```

---

## Dataset Description & Versioning

### Dataset Overview
* **Source:** Pima Indians Diabetes Dataset (National Institute of Diabetes and Digestive and Kidney Diseases).
* **Task:** Binary Classification (`Outcome`: `1` = Diabetes, `0` = No Diabetes).
* **Target Audience/Goal:** Clinical decision support for early diabetes diagnosis.

### Features
* `Pregnancies`: Number of times pregnant.
* `Glucose`: Plasma glucose concentration a 2 hours in an oral glucose tolerance test.
* `BloodPressure`: Diastolic blood pressure ($mm\,Hg$).
* `SkinThickness`: Triceps skin fold thickness ($mm$).
* `Insulin`: 2-Hour serum insulin ($\mu U/ml$).
* `BMI`: Body mass index ($weight\,in\,kg / (height\,in\,m)^2$).
* `DiabetesPedigreeFunction`: Diabetes pedigree function.
* `Age`: Age in years.

### Data Cleaning and Standardization Protocol (`datos_prep.py`)
1. **Raw Data Ingestion:** Downloaded from the public source repository into `datos/datos_ini/diabetes_original.csv`.
2. **Handling Missing Values:** Biological zeros in clinical fields (`Glucose`, `BloodPressure`, `SkinThickness`, `Insulin`, `BMI`) represent missing observations rather than true zero values. These were replaced with `NaN` and imputed using feature medians to preserve distribution balance and mitigate outliers.
3. **Deduplication:** Ensured unique observations by removing duplicate rows.
4. **Standardization:** Features scaled using `StandardScaler` for distance/gradient-sensitive algorithms (e.g., Logistic Regression).
5. **Versioning:** The cleaned output is committed to `datos/datos_limp/diabetes_limpio.csv` with modifications tracked in `CHANGELOG.md`.

---

## Code Execution Guide

### Option 1: Running via Google Colab (`entrena.ipynb`)
1. Upload `fuentes/entrena.ipynb` into Google Colab.
2. Run the notebook sequentially. The script automatically installs required dependencies (`mlflow`, `scikit-learn`, `pandas`, `seaborn`), creates local directory structures, handles data ingestion/cleaning, and logs experiments to MLflow.

### Option 2: Local Python Execution
1. **Clone the Repository:**
   ```bash
   git clone https://github.com/your-username/Actividad5.git
   cd Actividad5
   ```

2. **Install Dependencies:**
   ```bash
   pip install mlflow scikit-learn pandas numpy matplotlib seaborn
   ```

3. **Execute Data Pipeline & Training Scripts:**
   ```bash
   python fuentes/datos_prep.py
   python fuentes/train.py
   ```

---

## MLflow Tracking & Evidence

This pipeline uses **MLflow** for experiment management, hyperparameter tracking, and artifact persistence under the experiment name `Proyecto_Diabetes_MLOps`.

### Experiment Parameters Tracked
* **Logistic Regression:** Regularization parameter $C \in [0.01, 0.1, 1, 10, 100]$.
* **Random Forest:** `n_estimators` $\in [50, 100, 200]$, `max_depth` $\in [None, 5, 10]$.

### Metrics Logged
* **Accuracy**
* **Precision**
* **Recall**
* **F1-Score**
* **ROC-AUC**

### Model Artifacts
* Serialized Scikit-Learn models logged via `mlflow.sklearn.log_model()` using `cloudpickle` serialization for environment-safe persistence.

### Launching MLflow UI
To inspect logged parameters, runs, metrics, and models locally:
```bash
mlflow ui
```
Open `http://127.0.0.1:5000` in your web browser.

---

## Technical Decisions & Model Comparison

### Hyperparameter Tuning Strategy
Both models are optimized using **GridSearchCV** with a 5-fold cross-validation scheme (`cv=5`) on the training set (80/20 train-test split stratified on `Outcome`). Seed fixed (`random_state=42`) for total reproducibility.

### Model Performance Summary

| Model | Hyperparameters | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Logistic Regression** | $C=1.0$ | ~0.77 | ~0.70 | ~0.61 | ~0.65 | ~0.83 |
| **Random Forest** | `n_estimators=100`, `max_depth=5` | ~0.79 | ~0.72 | ~0.65 | ~0.68 | ~0.85 |

*Note: Random Forest demonstrates superior non-linear modeling capability on complex medical feature interactions, yielding higher overall F1 and ROC-AUC scores.*

---

## MLOps Reflections & Technical Debt

1. **Reproducibility & Traceability:** Decoupling raw data from processed data alongside explicit seed management ensures experiments can be audited and verified across environments.
2. **Experimentation Overhead:** Manual tracking of hyperparameter tuning rapidly creates technical debt. Integrating MLflow automates experiment logging, eliminating metric drift and unverified model deployments.
3. **Future Enhancements:**
   * Integrate Data Version Control (**DVC**) for tracking heavy binary dataset changes over Git commits.
   * Implement automated CI/CD unit testing for dataset pipeline validation (`pytest`).
   * Containerize the training script using Docker for scalable cloud deployments.