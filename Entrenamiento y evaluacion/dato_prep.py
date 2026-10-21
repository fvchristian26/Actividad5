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


# ------------------------------------------------------------------------------
# APARTADO 3: ANÁLISIS EXPLORATORIO DE DATOS (EDA Y VISUALIZACIONES)
# ------------------------------------------------------------------------------
# Visualizaciones relevantes para identificar la distribución de las variables y
# posibles correlaciones antes del modelado.

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Visualización 1: Distribución de la Glucosa
sns.histplot(df['Glucose'], kde=True, ax=axes[0], color='teal')
axes[0].set_title('Distribución de Glucosa (Datos Limpios)')

# Visualización 2: Matriz de Correlación
sns.heatmap(df.corr(), annot=True, fmt=".2f", cmap='Blues', ax=axes[1])
axes[1].set_title('Matriz de Correlación entre Variables')

plt.tight_layout()
plt.show()


# ------------------------------------------------------------------------------
# APARTADO 4: PREPARACIÓN Y DIVISIÓN DEL DATASET (TRAIN / TEST)
# ------------------------------------------------------------------------------
# División estratificada (80% entrenamiento, 20% prueba) con semilla fija (random_state=42)
# para asegurar reproducibilidad[cite: 1]. Aplicación de escalado de características.

X = df.drop('Outcome', axis=1)
y = df['Outcome']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Escalado estándar (requerido para modelos basados en gradientes/distancias como Regresión Logística)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(f"[INFO] Tamaño conjunto entrenamiento: {X_train.shape[0]} muestras")
print(f"[INFO] Tamaño conjunto de prueba: {X_test.shape[0]} muestras")