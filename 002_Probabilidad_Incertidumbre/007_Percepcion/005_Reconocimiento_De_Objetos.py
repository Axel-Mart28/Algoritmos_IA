# ALGORITMO DE RECONOCIMIENTO DE OBJETOS: COINCIDENCIA DE PLANTILLAS (TEMPLATE MATCHING) ---

# Este algoritmo implementa la "Coincidencia de Plantillas", una técnica
# simple de visión por computador para encontrar una imagen pequeña (plantilla)
# dentro de una imagen más grande (fuente).
#
# Definición:
# Desliza la imagen de la plantilla sobre la imagen fuente y calcula
# una métrica de "similitud" o "diferencia" en cada ubicación.
# La ubicación donde la métrica indica la mejor coincidencia se considera
# la posición del objeto (plantilla) encontrado.
#
# Componentes:
# 1. Imagen Fuente (donde buscar).
# 2. Imagen Plantilla (lo que se busca).
# 3. Método de Comparación (ej. `cv2.TM_SQDIFF`).
#
# Aplicaciones:
# - Encontrar objetos con apariencia, escala y rotación fijas.
# - Buscar iconos en interfaces gráficas.
# - Alinear imágenes en procesos industriales simples.
#
# Ventajas:
# - Muy simple y rápido para objetos que no cambian de apariencia.
#
# Desventajas:
# - Muy sensible a cambios de escala, rotación, iluminación, perspectiva y oclusión.


# --- P1: Importar Bibliotecas ---
import cv2        # Importa la biblioteca OpenCV para visión por computador
import numpy as np # Importa NumPy para manejo numérico (aunque no se usa mucho aquí)

# --- P2: Cargar Imágenes ---
imagen = cv2.imread("000_imagen_ejemplo.jpg")
template = cv2.imread("00_template_3.png")

# Verificar si las imágenes se cargaron correctamente
if imagen is None: # Si `imread` no pudo encontrar/abrir la imagen, devuelve None
    raise FileNotFoundError("¡No se pudo cargar '000_imagen_ejemplo.jpg'! Verifica la ruta/nombre.") # Lanza error
if template is None: # Comprobar también la plantilla
    raise FileNotFoundError("¡No se pudo cargar '00_template_3.png'! Verifica la ruta/nombre.") # Lanza error

# --- P3: Redimensionar Imagen Principal (Opcional) ---
# Define las nuevas dimensiones deseadas para la imagen principal
alto = 600 # Nueva altura en píxeles
ancho = 400 # Nueva anchura en píxeles
# Redimensiona la imagen principal a las nuevas dimensiones
# Esto puede ayudar si la imagen es muy grande, pero puede afectar la detección si el objeto se vuelve muy pequeño
imagen = cv2.resize(imagen, (ancho, alto))

# --- P4: Convertir a Escala de Grises ---
# Convierte la imagen principal de BGR (color por defecto en OpenCV) a escala de grises
imagen_gray = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)
# Convierte la imagen plantilla de BGR a escala de grises
template_gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
# (La coincidencia a menudo funciona mejor o es más rápida en escala de grises)

# --- P5: Aplicar Coincidencia de Plantillas ---
# Desliza `template_gray` sobre `imagen_gray` usando el método `TM_SQDIFF`
# `TM_SQDIFF`: Calcula la Suma de las Diferencias al Cuadrado. Menor valor = Mejor coincidencia.
res = cv2.matchTemplate(imagen_gray, template_gray, cv2.TM_SQDIFF)
# `res` es un mapa de resultados (imagen flotante) donde cada píxel (x,y)
# contiene el resultado de la comparación si la esquina superior izquierda
# de la plantilla estuviera en (x,y).

# --- P6: Encontrar la Mejor Coincidencia ---
# Encuentra los valores mínimo y máximo en el mapa `res`, y sus ubicaciones (coordenadas)
min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
# - min_val: El valor mínimo encontrado (la mejor coincidencia para TM_SQDIFF).
# - max_val: El valor máximo encontrado.
# - min_loc: Las coordenadas (x, y) de la esquina superior izquierda donde ocurrió min_val.
# - max_loc: Las coordenadas (x, y) donde ocurrió max_val.

# Imprime los valores encontrados (útil para depuración)
print(f"Valor Mínimo (Mejor Coincidencia): {min_val:.2f}, Ubicación: {min_loc}")
print(f"Valor Máximo: {max_val:.2f}, Ubicación: {max_loc}")

# --- P7: Definir el Rectángulo de la Coincidencia ---
# Como usamos TM_SQDIFF, la mejor ubicación está en min_loc.
# x1, y1 son las coordenadas de la esquina superior izquierda de la coincidencia
x1, y1 = min_loc
# Calcular las coordenadas de la esquina inferior derecha del rectángulo:
# x2 = x1 + ancho_plantilla
# y2 = y1 + alto_plantilla
# `template.shape[1]` es el ancho de la plantilla (número de columnas)
# `template.shape[0]` es el alto de la plantilla (número de filas)
x2 = min_loc[0] + template.shape[1]
y2 = min_loc[1] + template.shape[0]

# --- P8: Dibujar el Rectángulo ---
# Dibuja un rectángulo en la imagen *original* (en color)
cv2.rectangle(
    imagen,       # Imagen donde dibujar
    (x1, y1),     # Coordenadas de la esquina superior izquierda (int)
    (x2, y2),     # Coordenadas de la esquina inferior derecha (int)
    (0, 255, 0),  # Color del rectángulo en BGR (Verde)
    2             # Grosor de la línea del rectángulo
)

# --- P9: Mostrar Resultados ---
# Muestra la imagen original con el rectángulo dibujado en una ventana llamada "Imagen"
cv2.imshow("Imagen", imagen)
# Muestra la imagen de la plantilla en una ventana llamada "Template"
cv2.imshow("Template", template)
# Espera indefinidamente hasta que se presione cualquier tecla
cv2.waitKey(0)
# Cierra todas las ventanas abiertas por OpenCV
cv2.destroyAllWindows()

print("\nConclusión:")
print("Se encontró la ubicación en la imagen principal donde la plantilla")
print("tenía la menor diferencia (mejor coincidencia con TM_SQDIFF).")
print("Se dibujó un rectángulo verde en esa ubicación.")