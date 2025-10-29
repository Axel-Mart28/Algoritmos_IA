# --- 4a. ANÁLISIS DE TEXTURAS (GLCM - Adaptado para tu Imagen) ---

# Objetivo: Calcular características de textura GLCM de tu imagen.
# Bibliotecas: scikit-image, opencv-python, matplotlib
#              (Instala con: pip install scikit-image opencv-python matplotlib)

import numpy as np
import cv2 # Para cargar la imagen
from skimage.feature import graycomatrix, graycoprops # Funciones para GLCM
import matplotlib.pyplot as plt # Para mostrar la imagen
import os # Para construir la ruta

print("--- 4a. Análisis de Texturas  ---")

# --- P1: Cargar TU Imagen ---
print("Cargando imagen...")
# Obtener la ruta del directorio donde se encuentra ESTE script .py
script_dir = os.path.dirname(os.path.abspath(__file__))
nombre_archivo_imagen = '000_imagen_ejemplo.jpg'
# -----------------------------------------------------
ruta_completa_imagen = os.path.join(script_dir, nombre_archivo_imagen)
print(f"Buscando imagen en: {ruta_completa_imagen}")

# Cargar la imagen en escala de grises
imagen_gris = cv2.imread(ruta_completa_imagen, cv2.IMREAD_GRAYSCALE)

# Verificar si la imagen se cargó
if imagen_gris is None:
    error_msg = (f"¡ERROR! No se pudo cargar la imagen '{ruta_completa_imagen}'.\n"
                 f"Verifica la ruta y el nombre del archivo.")
    raise FileNotFoundError(error_msg)
else:
    print(f"Imagen '{nombre_archivo_imagen}' cargada. Dimensiones: {imagen_gris.shape}")

# --- P2: Seleccionar un Parche (Opcional, pero recomendado) ---
# GLCM funciona mejor en regiones con textura relativamente uniforme.
# Analizar toda la imagen puede mezclar texturas. Vamos a tomar un parche central.
h, w = imagen_gris.shape
tam_parche = 64 # Tamaño del parche (ej. 64x64 píxeles)
if h > tam_parche and w > tam_parche:
    # Calcular coordenadas para centrar el parche
    y_start = (h - tam_parche) // 2
    x_start = (w - tam_parche) // 2
    parche_textura = imagen_gris[y_start : y_start + tam_parche,
                                 x_start : x_start + tam_parche]
    print(f"Extrayendo un parche central de {tam_parche}x{tam_parche} píxeles.")
else:
    # Si la imagen es más pequeña que el parche, usar la imagen completa
    print("La imagen es pequeña, usando la imagen completa como parche.")
    parche_textura = imagen_gris

# --- P3: Calcular la GLCM ---
# Cuantizar la imagen a menos niveles de gris (mejora GLCM y reduce tamaño)
niveles = 8 # Reducir a 8 niveles (0-7)
parche_cuantizado = (parche_textura / (256 / niveles)).astype(np.uint8)

print(f"\nCalculando GLCM para el parche (cuantizado a {niveles} niveles)...")
# Calcular GLCM para distancia 1 en 4 direcciones
glcm = graycomatrix(parche_cuantizado,
                    distances=[1],
                    angles=[0, np.pi/4, np.pi/2, np.pi*3/4],
                    levels=niveles, # Usar el número de niveles cuantizados
                    symmetric=True,
                    normed=True)

print(f"GLCM calculada (shape: {glcm.shape})") # (niveles, niveles, dist, ang)

# --- P4: Calcular Propiedades de Textura ---
propiedades = ['contrast', 'dissimilarity', 'homogeneity', 'energy', 'correlation', 'ASM']
features_textura = {}

print("\nCalculando propiedades de textura (promedio sobre ángulos):")
for prop in propiedades:
    try:
        valores_prop = graycoprops(glcm, prop=prop)
        features_textura[prop] = np.mean(valores_prop)
        print(f"  - {prop.capitalize()}: {features_textura[prop]:.4f}")
    except Exception as e:
        print(f"  - No se pudo calcular {prop}: {e}") # Manejar posibles errores

# --- P5: Visualizar ---
try:
    plt.figure(figsize=(10, 5))

    plt.subplot(1, 2, 1)
    plt.imshow(imagen_gris, cmap='gray')
    # Dibujar un rectángulo mostrando el parche analizado
    if h > tam_parche and w > tam_parche:
        rect = plt.Rectangle((x_start, y_start), tam_parche, tam_parche,
                              edgecolor='red', facecolor='none', linewidth=2)
        plt.gca().add_patch(rect)
    plt.title("Imagen Original (Gris)")
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.imshow(parche_cuantizado, cmap='gray', interpolation='nearest', vmin=0, vmax=niveles-1)
    plt.title(f"Parche Analizado ({tam_parche}x{tam_parche}, {niveles} niveles)")
    plt.axis('off')
    # 
    plt.tight_layout()
    plt.show()

except Exception as e:
    print(f"\nNo se pudo mostrar la imagen: {e}")

print("\nConclusión Texturas:")
print("Se calcularon características GLCM de un parche de tu imagen.")
print("Estos números (contraste, homogeneidad, etc.) describen la textura")
print("y pueden usarse para clasificar diferentes tipos de superficies.")