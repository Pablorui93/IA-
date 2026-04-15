import os
import random
import numpy as np
from PIL import Image, ImageDraw

# Configuración básica
SHAPES = ["triangulo", "cuadrado", "pentagono", "circulo"]
COUNT = 500  
SIZE = (128, 128)

def create_folders():
    """Crea la estructura de carpetas para el dataset."""
    if not os.path.exists("dataset_imagenes"):
        os.makedirs("dataset_imagenes")
    for shape in SHAPES:
        os.makedirs(f"dataset_imagenes/{shape}", exist_ok=True)

def draw_shape(draw, shape, center, size):
    """Dibuja la figura geométrica en el lienzo."""
    x, y = center
    r = size // 2
    if shape == "triangulo":
        points = [(x, y - r), (x - r, y + r), (x + r, y + r)]
    elif shape == "cuadrado":
        points = [(x - r, y - r), (x + r, y - r), (x + r, y + r), (x - r, y + r)]
    elif shape == "pentagono":
        points = [(x + r * np.cos(np.radians(72*i - 90)), 
                   y + r * np.sin(np.radians(72*i - 90))) for i in range(5)]
    elif shape == "circulo":
        draw.ellipse([x - r, y - r, x + r, y + r], outline="black", width=3)
        return
    
    draw.polygon(points, outline="black", width=3)

# --- EJECUCIÓN ---
create_folders()

for shape in SHAPES:
    print(f"Generando {shape}...")
    for i in range(COUNT):
        # Crear lienzo blanco
        img = Image.new("RGB", SIZE, "white")
        draw = ImageDraw.Draw(img)
        
        # Aleatoriedad para robustez
        random_center = (random.randint(50, 78), random.randint(50, 78))
        random_size = random.randint(35, 55)
        
        draw_shape(draw, shape, random_center, random_size)
        
        # Rotar la imagen aleatoriamente
        img = img.rotate(random.randint(0, 360), fillcolor="white")
        
        # Guardar archivo
        img.save(f"dataset_imagenes/{shape}/{shape}_{i:03d}.png")

print("\n✔ ¡Proceso terminado! Revisá la carpeta 'dataset_imagenes'.")