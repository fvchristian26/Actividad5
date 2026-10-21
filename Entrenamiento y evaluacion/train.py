!pip install mlflow scikit-learn pandas numpy matplotlib seaborn -q

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, ConfusionMatrixDisplay, roc_curve
)

# Creación de la estructura de carpetas requerida en el repositorio
os.makedirs('datos/datos_ini', exist_ok=True)
os.makedirs('datos/datos_limp', exist_ok=True)
print("[INFO] Estructura de directorios creada correctamente.")


# ------------------------------------------------------------------------------
# APARTADO 2: CARGA, LIMPIEZA Y VERSIONADO DE DATOS (TEMA 11)
# ------------------------------------------------------------------------------
# Se descarga el dataset original de Diabetes, se guarda en 'datos_ini', se
# tratan los ceros atípicos (valores nulos implícitos en variables médicas) y se
# exporta el conjunto limpio a 'datos_limp'.

url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"
column_names = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness',
                'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age', 'Outcome']

# Carga y respaldo del dataset original
df_original = pd.read_csv(url, names=column_names)
df_original.to_csv('datos/datos_ini/diabetes_original.csv', index=False)

# Limpieza: Remplazo de ceros biológicamente imposibles por NaN y luego por la mediana
df = df_original.copy()
cols_con_ceros = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
for col in cols_con_ceros:
    df[col] = df[col].replace(0, np.nan)
    df[col] = df[col].fillna(df[col].median())

# Eliminación de posibles duplicados
df = df.drop_duplicates()

# Guardado del dataset procesado y limpio
df.to_csv('datos/datos_limp/diabetes_limpio.csv', index=False)
print("[INFO] Dataset procesado y guardado en 'datos/datos_limp/diabetes_limpio.csv'.")
