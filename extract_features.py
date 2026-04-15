import cv2
import numpy as np
import os
import pandas as pd

# Configuración
BASE_DIR = "dataset_imagenes"
SHAPES = ["triangulo", "cuadrado", "pentagono", "circulo"]

def extraer_descriptores(ruta_imagen):
    # 1. Cargar y preprocesar
    img = cv2.imread(ruta_imagen)
    if img is None: return None
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Binarizamos para asegurar que los bordes sean claros
    _, thresh = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)
    
    # 2. Encontrar el contorno de la figura
    cnts, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts: return None
    
    # Nos quedamos con el contorno más grande (por si hay ruido)
    c = max(cnts, key=cv2.contourArea)
    
    # --- CÁLCULO DE MÉTRICAS ---
    
    # A. Vértices (Aproximación poligonal)
    peri = cv2.arcLength(c, True)
    approx = cv2.approxPolyDP(c, 0.03 * peri, True)
    vertices = len(approx)
    
    # B. Circularidad (Independiente del tamaño)
    area = cv2.contourArea(c)
    perimetro = cv2.arcLength(c, True)
    circularidad = (4 * np.pi * area) / (perimetro**2) if perimetro > 0 else 0
    
    # C. Relación de Aspecto (Ancho / Alto)
    x, y, w, h = cv2.boundingRect(c)
    aspect_ratio = float(w) / h
    
    # D. Solidez (Área / Área del Casco Convexo)
    hull = cv2.convexHull(c)
    hull_area = cv2.contourArea(hull)
    solidez = float(area) / hull_area if hull_area > 0 else 0
    
    return [vertices, circularidad, aspect_ratio, solidez]

# --- PROCESAMIENTO GENERAL ---
print("Extrayendo características de las imágenes...")
dataset = []

for shape in SHAPES:
    folder_path = os.path.join(BASE_DIR, shape)
    for filename in os.listdir(folder_path):
        features = extraer_descriptores(os.path.join(folder_path, filename))
        if features:
            # Agregamos los datos y la etiqueta (target)
            dataset.append(features + [shape])

# Crear DataFrame y guardar
df = pd.DataFrame(dataset, columns=['vertices', 'circularidad', 'aspect_ratio', 'solidez', 'target'])
df.to_csv('dataset_geometria.csv', index=False)

print(f"\n✔ ¡Listo! Se procesaron {len(df)} imágenes.")
print("Archivo 'dataset_geometria.csv' generado en la raíz.")