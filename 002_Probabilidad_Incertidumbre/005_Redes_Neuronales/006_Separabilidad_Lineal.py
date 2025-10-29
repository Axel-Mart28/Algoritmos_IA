# ALGORITMO DE SEPARABILIDAD LINEAL (LINEAR SEPARABILITY) 

# Este es un CONCEPTO geométrico clave en Machine Learning,
# especialmente relevante para clasificadores lineales como el
# Perceptrón, ADALINE y SVM con kernel lineal.
#
# Definición:
# Un conjunto de datos con dos clases (ej. 'Clase 0' y 'Clase 1') es
# "linealmente separable" si existe al menos *una línea recta*
# (en 2D), un plano (en 3D), o un hiperplano (en dimensiones > 3)
# que pueda separar *perfectamente* todos los puntos de una clase
# de todos los puntos de la otra clase.
#
# ¿Por qué es importante?
# - Los modelos de *una sola neurona* (Perceptrón, ADALINE)
#   funcionan encontrando precisamente esta línea (o hiperplano)
#   separadora.
# - Por lo tanto, estos modelos *solo pueden* resolver problemas
#   que son linealmente separables.
# - Si los datos *no* son linealmente separables, estos modelos
#   *nunca* convergerán a una solución perfecta (el Perceptrón
#   oscilará, ADALINE encontrará la "mejor" línea posible pero
#   con error).
#
# Ejemplos:
# - Puertas Lógicas AND, OR, NAND, NOR: SON linealmente separables
#   Un Perceptrón puede aprenderlas.
# - Puerta Lógica XOR: NO ES linealmente separable
#   Un Perceptrón simple no puede aprenderla. Se necesita
#   una red multicapa (como MADALINE o un MLP).
#
# ¿Cómo funciona este programa?
# Usaremos `matplotlib` para *visualizar* dos conjuntos de datos:
# 1. Los datos de la puerta AND (linealmente separable).
# 2. Los datos de la puerta XOR (no linealmente separable).
# Esto hará obvia la diferencia geométrica.

import matplotlib.pyplot as plt # Para graficar
import numpy as np # Para los arrays de datos

# --- P1: Datos de Ejemplo ---

# Datos de la puerta AND
X_and = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y_and = np.array([0, 0, 0, 1]) # Clase 0, Clase 0, Clase 0, Clase 1

# Datos de la puerta XOR
X_xor = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y_xor = np.array([0, 1, 1, 0]) # Clase 0, Clase 1, Clase 1, Clase 0

# --- P2: Función de Visualización ---

def visualizar_datos(X, y, titulo):
    """ Grafica los puntos de datos 2D coloreados por su clase """
    plt.figure(figsize=(6, 5)) # Crear figura
    
    # Separar los puntos por clase para colorearlos
    clase_0 = X[y == 0] # Puntos donde y es 0
    clase_1 = X[y == 1] # Puntos donde y es 1
    
    # Graficar los puntos
    plt.scatter(clase_0[:, 0], clase_0[:, 1], color='red', marker='o', label='Clase 0', s=100)
    plt.scatter(clase_1[:, 0], clase_1[:, 1], color='blue', marker='x', label='Clase 1', s=100)
    
    # Añadir títulos y leyenda
    plt.title(titulo) # Título del gráfico
    plt.xlabel("Característica 1 (x1)") # Etiqueta Eje X
    plt.ylabel("Característica 2 (x2)") # Etiqueta Eje Y
    plt.legend() # Mostrar leyenda
    plt.grid(True) # Añadir rejilla
    plt.xlim(-0.5, 1.5) # Ajustar límites eje X
    plt.ylim(-0.5, 1.5) # Ajustar límites eje Y
    plt.axhline(0, color='black', linewidth=0.5) # Eje X
    plt.axvline(0, color='black', linewidth=0.5) # Eje Y
    
    # Mostrar el gráfico
    plt.show() # Mostrar la ventana

# --- P3: Visualizar los Datos ---
print("---Visualización de Separabilidad Lineal ---") # Título

# 1. Visualizar AND
print("\nVisualizando datos de la Puerta AND...")
visualizar_datos(X_and, y_and, "Puerta Lógica AND (Linealmente Separable)")
# 
print("  Observa cómo puedes dibujar UNA línea recta (ej. x1+x2=1.5)")
print("  para separar los puntos azules (Clase 1) de los rojos (Clase 0).")

# 2. Visualizar XOR
print("\nVisualizando datos de la Puerta XOR...")
visualizar_datos(X_xor, y_xor, "Puerta Lógica XOR (NO Linealmente Separable)")
# 
print("  Observa cómo es IMPOSIBLE dibujar UNA sola línea recta")
print("  para separar los puntos azules de los rojos.")
print("  Necesitarías al menos DOS líneas (o una curva).")

print("\nConclusión:")
print("La Separabilidad Lineal es una propiedad de los *datos*.")
print("Los modelos de una sola capa (Perceptrón, ADALINE) solo")
print("pueden resolver problemas si los datos tienen esta propiedad.")
print("Para problemas no separables (como XOR), necesitamos modelos")
print("más complejos, como las Redes Multicapa.")