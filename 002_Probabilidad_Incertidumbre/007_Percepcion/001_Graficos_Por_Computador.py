# ---1. GRÁFICOS POR COMPUTADOR (Ejemplo Simple 2D con Matplotlib) ---

# Este programa SÍ genera una imagen simple usando matplotlib.
# Demuestra el concepto de definir formas y colores
# y "renderizarlos" en un lienzo 2D.

import matplotlib.pyplot as plt # Biblioteca para graficar
import matplotlib.patches as patches # Para dibujar formas (círculos, rectángulos)

print("--- 1. Gráficos por Computador ---")

# --- 1. Crear la "Escena" (Figura y Ejes) ---
fig, ax = plt.subplots() # Crea una figura y un conjunto de ejes (nuestro lienzo)
ax.set_aspect('equal', adjustable='box') # Asegura que los círculos se vean redondos
ax.set_xlim(0, 10) # Define el ancho del lienzo (eje X de 0 a 10)
ax.set_ylim(0, 10) # Define el alto del lienzo (eje Y de 0 a 10)
ax.set_title("Gráficos Simples 2D") # Título de la imagen
ax.grid(True) # Añadir una rejilla

# --- 2. "Modelar" y "Transformar" (Definir Formas y Posiciones) ---

# a. Un círculo rojo en (3, 7) con radio 1.5
circulo = patches.Circle(
    (3, 7),              # Centro (x, y) - Transformación de Traslación
    radius=1.5,          # Tamaño - Transformación de Escala (implícita)
    color='red',         # Propiedad de "Material"
    alpha=0.7            # Transparencia
)

# b. Un rectángulo azul desde (5, 1) hasta (8, 4)
rectangulo = patches.Rectangle(
    (5, 1),              # Esquina inferior izquierda (x, y)
    3,                   # Ancho (8 - 5)
    3,                   # Alto (4 - 1)
    linewidth=1,         # Grosor del borde
    edgecolor='blue',    # Color del borde
    facecolor='lightblue' # Color de relleno ("Material")
)

# c. Un polígono (triángulo) verde
puntos_triangulo = [[1, 1], [4, 1], [2.5, 4]] # Vértices [[x1,y1], [x2,y2], ...]
triangulo = patches.Polygon(
    puntos_triangulo,
    closed=True,         # Cerrar el polígono
    color='green',
    alpha=0.5
)

# --- 3. "Renderizar" (Añadir las formas al lienzo) ---
ax.add_patch(circulo)    # Añadir el círculo a los ejes
ax.add_patch(rectangulo) # Añadir el rectángulo
ax.add_patch(triangulo)  # Añadir el triángulo

# --- 4. Mostrar la Imagen Final ---
print("Generando imagen 2D simple...")
# 
plt.show() # Muestra la ventana con la imagen generada

print("\nConclusión:")
print("Este ejemplo usó matplotlib para 'renderizar' formas 2D básicas.")
print("Define coordenadas (Modelado/Transformación) y colores (Materiales)")
print("para crear una imagen final, ilustrando los principios básicos")
print("de los gráficos por computador de una manera muy simplificada.")