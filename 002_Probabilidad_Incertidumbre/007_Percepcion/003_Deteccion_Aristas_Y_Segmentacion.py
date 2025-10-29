# --- 3. DETECCIÓN DE ARISTAS (CANNY) Y SEGMENTACIÓN (OTSU) ---

import numpy as np
import cv2 # Usaremos OpenCV
import matplotlib.pyplot as plt
import os # Para construir la ruta

print("--- 3. Detección de Aristas y Segmentación ---")

# --- P1: Cargar la Imagen (Igual que antes) ---
print("Cargando imagen...")
script_dir = os.path.dirname(os.path.abspath(__file__))
nombre_archivo_imagen = '000_imagen_ejemplo.jpg'
ruta_completa_imagen = os.path.join(script_dir, nombre_archivo_imagen)
imagen_original_color = cv2.imread(ruta_completa_imagen) # Cargar en color esta vez
imagen_gris = cv2.cvtColor(imagen_original_color, cv2.COLOR_BGR2GRAY) # Convertir a escala de grises

if imagen_gris is None:
    error_msg = (f"¡ERROR! No se pudo cargar la imagen '{ruta_completa_imagen}'. Verifica.")
    raise FileNotFoundError(error_msg)
else:
    imagen_gris_float = imagen_gris.astype(np.float32) # Usar float para Sobel si es necesario
    print(f"Imagen '{nombre_archivo_imagen}' cargada. Dimensiones: {imagen_gris.shape}")

# --- P2: Detección de Aristas con Canny ---
# Canny es un algoritmo multi-paso:
# 1. Suavizado Gaussiano (para reducir ruido).
# 2. Cálculo de Gradientes (similar a Sobel).
# 3. Supresión de No-Máximos (adelgaza los bordes).
# 4. Umbralización por Histéresis (conecta bordes débiles a fuertes).
print("\nAplicando Detector de Aristas Canny...")
# Necesita dos umbrales: umbral_bajo y umbral_alto
# Pixeles con gradiente > umbral_alto son aristas seguras.
# Pixeles con gradiente < umbral_bajo se descartan.
# Pixeles entre los dos umbrales se aceptan SOLO si están conectados a aristas seguras.
umbral_bajo = 50
umbral_alto = 150
# Es importante usar la imagen en uint8 para Canny
imagen_canny = cv2.Canny(imagen_gris, umbral_bajo, umbral_alto)
print(f"Detector Canny aplicado (umbrales: {umbral_bajo}, {umbral_alto}).")
# 

# --- P3: Segmentación con Umbralización de Otsu ---
# La umbralización simple requiere elegir un umbral manualmente (como hicimos antes).
# El método de Otsu *encuentra automáticamente* un umbral óptimo
# que separa los píxeles en dos clases (ej. fondo y objeto)
# maximizando la varianza entre las clases.
print("\nAplicando Segmentación por Umbralización de Otsu...")
# cv2.threshold devuelve el umbral encontrado y la imagen umbralizada
# cv2.THRESH_BINARY: Píxeles > umbral van a maxval (255), el resto a 0.
# cv2.THRESH_OTSU: Indica que use el método de Otsu para encontrar el umbral.
umbral_otsu, imagen_otsu = cv2.threshold(
    imagen_gris,      # Imagen de entrada (escala de grises, uint8)
    0,                # Valor de umbral (se ignora si se usa THRESH_OTSU)
    255,              # Valor máximo a asignar
    cv2.THRESH_BINARY + cv2.THRESH_OTSU # Combinar tipo binario con método Otsu
)
print(f"Umbral de Otsu calculado automáticamente: {umbral_otsu}")
print("Umbralización de Otsu aplicada.")
# 

# --- P4: Mostrar Resultados ---
print("\nMostrando resultados...")
try:
    plt.figure(figsize=(15, 5)) # Figura más ancha

    # 1. Imagen Original en Gris
    plt.subplot(1, 3, 1)
    plt.imshow(imagen_gris, cmap='gray')
    plt.title("Original (Gris)")
    plt.axis('off')

    # 2. Aristas Canny
    plt.subplot(1, 3, 2)
    plt.imshow(imagen_canny, cmap='gray')
    plt.title(f"Aristas (Canny {umbral_bajo}-{umbral_alto})")
    plt.axis('off')

    # 3. Segmentación Otsu
    plt.subplot(1, 3, 3)
    plt.imshow(imagen_otsu, cmap='gray')
    plt.title(f"Segmentación (Otsu T={int(umbral_otsu)})")
    plt.axis('off')

    plt.tight_layout()
    plt.show()

except NameError:
     print("\nError al mostrar imágenes. Asegúrate de tener Matplotlib instalado:")
     print("pip install matplotlib")
except Exception as e:
     print(f"\nOcurrió un error al mostrar las imágenes: {e}")


print("\nConclusión:")
print("Canny es un detector de aristas robusto que produce bordes delgados.")
print("La Umbralización de Otsu segmenta automáticamente la imagen en dos clases")
print("(útil para separar objetos del fondo si el histograma es bimodal).")
print("Existen métodos de segmentación mucho más avanzados (k-Means, Watershed, Deep Learning).")