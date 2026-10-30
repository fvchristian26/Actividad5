# Usar imagen oficial de Python ligera
FROM python:3.9-slim

# Directorio de trabajo dentro del contenedor
WORKDIR /app

# Copiar e instalar las dependencias
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copiar todo el código fuente, modelo y archivos del proyecto
COPY . /app/

# Exponer el puerto predeterminado (por ejemplo, 8000 si usas FastAPI, o 5000 si usas Flask)
EXPOSE 8000

# Comando para ejecutar la aplicación (cambia 'app:app' por tu archivo y variable principal)
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]