import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from tensorflow import keras
from tensorflow.keras import layers

# 1. Cargar el dataset
df = pd.read_csv('dataset_geometria.csv')

# 2. Separar características (X) y etiquetas (y)
X = df.drop('target', axis=1).values
y = df['target'].values

# 3. Preprocesamiento
# Convertir nombres (círculo, etc.) a números (0, 1, 2, 3)
encoder = LabelEncoder()
y_encoded = encoder.fit_transform(y)

# Escalar los datos (Crucial: pone todos los valores en el mismo rango)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Dividir en Entrenamiento (80%) y Prueba (20%)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_encoded, test_size=0.2, random_state=42)

# 4. Definir la Arquitectura de la Red
model = keras.Sequential([
    layers.Dense(16, activation='relu', input_shape=(4,)), # Capa entrada: 4 neuronas
    layers.Dense(8, activation='relu'),                    # Capa oculta
    layers.Dense(4, activation='softmax')                  # Capa salida: una por figura
])

# 5. Compilación
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# 6. Entrenamiento
print("\nIniciando entrenamiento...")
history = model.fit(
    X_train, y_train,
    epochs=100,
    batch_size=16,
    validation_split=0.1,
    verbose=1
)

# 7. Evaluación final
loss, acc = model.evaluate(X_test, y_test, verbose=0)
print(f"\n✔ Precisión final en el set de prueba: {acc*100:.2f}%")

# Guardar el modelo y el escalador para usar después
model.save('modelo_figuras.keras')
import joblib
joblib.dump(scaler, 'scaler.pkl')
joblib.dump(encoder, 'encoder.pkl')