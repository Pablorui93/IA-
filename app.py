import streamlit as st
import cv2
import numpy as np
import joblib
from tensorflow import keras
from PIL import Image

# Configuración de la página
st.set_page_config(page_title="Clasificador Geométrico RNA", page_icon="📐")

# 1. Cargar el cerebro de la red y herramientas
@st.cache_resource # Esto evita que el modelo se recargue cada vez que tocás un botón
def load_assets():
    model = keras.models.load_model('modelo_figuras.keras')
    scaler = joblib.load('scaler.pkl')
    encoder = joblib.load('encoder.pkl')
    return model, scaler, encoder

model, scaler, encoder = load_assets()

# --- INTERFAZ ---
st.title("📐 Identificador de Figuras Geométricas")
st.subheader("Proyecto de Redes Neuronales Artificiales")
st.write("Subí una imagen (triángulo, cuadrado, pentágono o círculo) y dejá que la RNA haga su magia.")

uploaded_file = st.file_uploader("Elegí una imagen...", type=["png", "jpg", "jpeg"])


if uploaded_file is not None:
    # 1. Abrir imagen
    image = Image.open(uploaded_file)
    st.image(image, caption='Imagen subida', width=300)
    
    # 2. Convertir a OpenCV correctamente
    img_array = np.array(image.convert('RGB'))
    img_cv = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
    
    # 3. Procesamiento de imagen
    gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
    
    # Probamos un umbral más flexible (127 en lugar de 200)
    _, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)
    
    # --- VISUALIZACIÓN DE DEBUG ---
    st.write("### Vista de Debug (OpenCV)")
    col_debug1, col_debug2 = st.columns(2)
    with col_debug1:
        st.image(gray, caption="Escala de Grises", width=200)
    with col_debug2:
        st.image(thresh, caption="Binarizada (Threshold)", width=200)
    # ------------------------------

    # 4. Búsqueda de contornos
    cnts, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if len(cnts) > 10: 
        st.warning("Hay demasiado ruido en la imagen. ¿Seguro que es una figura geométrica?")
    else:
        st.success(f"¡Se detectaron {len(cnts)} contornos!")
        c = max(cnts, key=cv2.contourArea)
        # if cnts:
    #     st.success(f"¡Se detectaron {len(cnts)} contornos!")
    #     c = max(cnts, key=cv2.contourArea)
        
    #     # ... (Acá seguí con el resto de tu código de predicción igual que antes)
    #     # Extraer características, escalar, predecir...
        
    # else:
    #     st.error("❌ OpenCV no detectó ningún contorno. Revisá la imagen binarizada arriba.")
    #     st.info("Tip: Si la imagen 'Binarizada' se ve toda negra, el umbral (Threshold) está fallando.")

