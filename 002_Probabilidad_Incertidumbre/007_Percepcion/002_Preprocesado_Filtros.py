# --- 2. PREPROCESADO: FILTROS (Ejemplo: Desenfoque Promedio - Modificado) ---

# --- 2. PREPROCESADO: FILTROS APLICADOS A TU IMAGEN (Corregido) ---

import numpy as np
import cv2 # Usaremos OpenCV para filtros eficientes
import matplotlib.pyplot as plt # Para mostrar resultados
import os

# --- P1: Cargar tu Imagen ---
print("--- Cargando Imagen ---")
# Obtener la ruta del directorio donde se encuentra ESTE script .py
script_dir = os.path.dirname(os.path.abspath(__file__))
# Nombre del archivo de imagen
nombre_archivo_imagen = '000_imagen_ejemplo.jpg'
# Combinar la ruta del script con el nombre del archivo para obtener la ruta completa
ruta_completa_imagen = os.path.join(script_dir, nombre_archivo_imagen)

print(f"Buscando imagen en: {ruta_completa_imagen}") # Muestra la ruta que intentará usar

# Cargar la imagen en escala de grises usando OpenCV con la ruta completa
imagen_original = cv2.imread(ruta_completa_imagen, cv2.IMREAD_GRAYSCALE)

# Verificar si la imagen se cargó correctamente
if imagen_original is None:
    # Si falla, dar un mensaje claro y salir
    error_msg = (f"¡ERROR! No se pudo cargar la imagen '{ruta_completa_imagen}'.\n"
                 f"Verifica:\n"
                 f"1. Que el nombre del archivo '{nombre_archivo_imagen}' sea correcto.\n"
                 f"2. Que el archivo esté en la MISMA CARPETA ('{os.path.basename(script_dir)}') que el script.\n"
                 f"3. Que el archivo de imagen no esté corrupto.")
    raise FileNotFoundError(error_msg)
else:
    # Si se cargó, convertir a float32 para cálculos
    imagen_original = imagen_original.astype(np.float32)
    print(f"Imagen '{nombre_archivo_imagen}' cargada exitosamente. Dimensiones: {imagen_original.shape}")

# --- IMPORTANTE: YA NO HAY BLOQUE QUE SOBRESCRIBA imagen_original ---

print("\n--- Aplicando Filtros ---")

# --- P2: Aplicar Filtro de Desenfoque Promedio (Blur) ---
# Usa un kernel de promedio (como el que definimos antes).
# (ksize es el tamaño del kernel, ej. (3, 3) para un kernel 3x3)
ksize_blur = (5, 5) # Probemos un kernel 5x5 para un desenfoque más notable
imagen_blur = cv2.blur(imagen_original, ksize_blur)
print(f"Filtro de Promedio {ksize_blur} aplicado.")
# 

# --- P3: Aplicar Filtro Gaussiano (Gaussian Blur) ---
# Similar al promedio, pero usa un kernel Gaussiano (pesos siguen una curva de campana).
# Es mejor para suavizar ruido preservando un poco más los bordes.
ksize_gauss = (5, 5)
sigma_gauss = 1.5 # Desviación estándar del Gaussiano (controla el desenfoque)
imagen_gauss = cv2.GaussianBlur(imagen_original, ksize_gauss, sigma_gauss)
print(f"Filtro Gaussiano {ksize_gauss} (sigma={sigma_gauss}) aplicado.")
# 

# --- P4: Aplicar Filtro Laplaciano (Detección de Aristas Simple) ---
# Detecta bordes calculando la segunda derivada. Resalta cambios bruscos.
print("Aplicando Filtro Laplaciano...")
# --- CORRECCIÓN: Asegurarse de que la entrada sea float64 si el destino es float64 ---
imagen_float64 = imagen_original.astype(np.float64) # Convertir a float64
imagen_laplacian = cv2.Laplacian(imagen_float64, cv2.CV_64F, ksize=3)
# Convertir de nuevo a un formato visible (tomando el valor absoluto y escalando a 0-255)
imagen_laplacian_display = cv2.convertScaleAbs(imagen_laplacian)
print("Filtro Laplaciano (k=3) aplicado.")
# 

# --- P5: Aplicar Filtro de Sobel (Detección de Aristas con Dirección) ---
# Calcula el gradiente en dirección X y Y.
print("Aplicando Filtros de Sobel...")
# --- CORRECCIÓN: Usar la imagen float64 ---
# Gradiente X (detecta bordes verticales)
sobel_x = cv2.Sobel(imagen_float64, cv2.CV_64F, 1, 0, ksize=3) # dx=1, dy=0
sobel_x_display = cv2.convertScaleAbs(sobel_x)

# Gradiente Y (detecta bordes horizontales)
sobel_y = cv2.Sobel(imagen_float64, cv2.CV_64F, 0, 1, ksize=3) # dx=0, dy=1
sobel_y_display = cv2.convertScaleAbs(sobel_y)

# Magnitud del Gradiente (combina Gx y Gy)
# Usar los resultados de Sobel en float64 para mayor precisión antes de convertir
magnitud_sobel = np.sqrt(sobel_x**2 + sobel_y**2)
magnitud_sobel_display = cv2.convertScaleAbs(magnitud_sobel)
print("Filtros de Sobel (Gx, Gy, Magnitud) aplicados.")
# 

# (El resto del código P6 para mostrar los resultados sigue igual)
# ...
# 

# --- P6: Mostrar Resultados con Matplotlib ---
print("\nMostrando resultados...")
try:
    plt.figure(figsize=(15, 10)) # Figura más grande

    # 1. Imagen Original
    plt.subplot(2, 3, 1) # 2 filas, 3 columnas, posición 1
    plt.imshow(imagen_original, cmap='gray')
    plt.title("Original")
    plt.axis('off')

    # 2. Desenfoque Promedio
    plt.subplot(2, 3, 2)
    plt.imshow(imagen_blur, cmap='gray')
    plt.title(f"Promedio {ksize_blur}")
    plt.axis('off')

    # 3. Desenfoque Gaussiano
    plt.subplot(2, 3, 3)
    plt.imshow(imagen_gauss, cmap='gray')
    plt.title(f"Gaussiano {ksize_gauss} $\sigma$={sigma_gauss}") # Usando LaTeX para sigma
    plt.axis('off')

    # 4. Laplaciano
    plt.subplot(2, 3, 4)
    plt.imshow(imagen_laplacian_display, cmap='gray')
    plt.title("Laplaciano (Bordes)")
    plt.axis('off')


    # 6. Magnitud Sobel
    plt.subplot(2, 3, 5) # Cambiado a posición 5
    plt.imshow(magnitud_sobel_display, cmap='gray')
    plt.title("Sobel Magnitud (Bordes)")
    plt.axis('off')

    # Placeholder para el sexto subplot si quitas Gx/Gy
    plt.subplot(2, 3, 6).axis('off')


    plt.tight_layout() # Ajustar espaciado
    plt.show() # Mostrar la ventana

except NameError:
     print("\nError al mostrar imágenes. Asegúrate de tener Matplotlib instalado:")
     print("pip install matplotlib")
except Exception as e:
     print(f"\nOcurrió un error al mostrar las imágenes: {e}")


print("\nConclusión:")
print("Se aplicaron varios filtros comunes usando OpenCV.")
print("- Blur/Gaussiano suavizan la imagen (reducen ruido).")
print("- Laplaciano y Sobel detectan aristas (cambios de intensidad).")
