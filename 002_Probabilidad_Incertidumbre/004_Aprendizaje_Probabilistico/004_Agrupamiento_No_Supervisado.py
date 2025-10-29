# Algoritmo de AGRUPAMIENTO NO SUPERVISADO

# Este es un CONCEPTO, no un algoritmo único. Es el nombre de un
# *tipo* de problema de Machine Learning (Aprendizaje No Supervisado).
#
# Definición:
# Es el proceso de tomar un conjunto de datos (X) que *NO TIENE ETIQUETAS*
# (no hay "respuestas correctas") y encontrar una estructura o
# patrones ocultos en él.
#
# La "estructura" que buscamos es, por lo general, un "agrupamiento" o "cluster".
#
# Objetivo:
# Agrupar los puntos de datos de tal manera que los puntos
# en el *mismo grupo* sean muy *similares* entre sí, y
# muy *diferentes* a los puntos de *otros grupos*.
#
# ¿Cómo funciona este programa?
# 1. Definiremos un conjunto de datos de entrada 'X' (sin etiquetas).
# 2. Mostraremos cuál es el *objetivo* (la "salida deseada"): una lista de "etiquetas de cluster" que el algoritmo debería *inventar* o *descubrir*.
#
# Componentes:
# 1. Los Datos (X): Una lista de vectores (ej. coordenadas [x, y]).
# 2. Una Métrica de Similitud: ¿Cómo medimos si dos puntos
#    son "similares"? (ej. Distancia Euclidiana).
# 3. El número de clusters (k): ¿Cuántos grupos queremos encontrar?
#    (A menudo, 'k' es un parámetro que *nosotros* debemos elegir).
#
# Aplicaciones:
# - Segmentación de Clientes: Agrupar clientes con comportamientos
#   de compra similares (sin saber las categorías de antemano).
# - Agrupación de Noticias: Agrupar artículos que hablan del mismo
#   tema (ej. 'Deportes', 'Política').
# - Compresión de Imágenes: Agrupar 50,000 colores en solo 16 "clusters" de color.
# - Detección de Anomalías: Encontrar puntos que no pertenecen a *ningún* grupo.
#
# Ventajas:
# - No requiere el costoso trabajo humano de "etiquetar" los datos.
# - Puede descubrir patrones y categorías que no esperábamos.
#
# Desventajas:
# - ¿Cómo sabemos si los clusters encontrados son "correctos" o "útiles"?
#   La evaluación es difícil.
# - Es muy sensible a la elección de 'k' (el número de grupos).
#
# Ejemplo de uso:
# - Un algoritmo como "k-Medias" (tema #6) tomará los datos 'X'
#   de este programa y encontrará las etiquetas de cluster.

import math 

# --- P1: Los Datos de Entrada (El Problema) ---

# Estos son nuestros datos. Son "No Supervisados" porque solo
# tenemos las características (ej. [edad, gasto]),
# pero *NO* tenemos etiquetas (ej. "cliente_bueno", "cliente_malo").
#
# Imaginemos 4 clientes (edad, gasto en $)
datos_X = [
    [25, 800],  # Cliente A (joven, gasta mucho)
    [50, 200],  # Cliente B (mayor, gasta poco)
    [30, 900],  # Cliente C (joven, gasta mucho)
    [45, 150]   # Cliente D (mayor, gasta poco)
]

# Definimos cuántos grupos (clusters) queremos encontrar
K = 2 # Queremos encontrar 2 grupos

print(" Agrupamiento No Supervisado (El Problema)") # Título
print(f"Datos de entrada (NO Supervisados) X:") # Imprime los datos
for i, punto in enumerate(datos_X): # Itera sobre los datos
    print(f"  Punto {i} (Cliente {chr(65+i)}): {punto}") # Imprime [25, 800], etc.

print(f"\nObjetivo: Encontrar K={K} grupos (clusters) en estos datos.") # Imprime el objetivo

# --- P2: La Salida Deseada (La Solución) ---

# Un algoritmo de clustering (como k-Medias) debería tomar los
# datos 'X' y producir una lista de "etiquetas de cluster".
#
# El algoritmo "inventa" las etiquetas (ej. Cluster 0, Cluster 1).
#
# (Esto es lo que *esperaríamos* que un algoritmo encontrara)
salida_deseada_Y = [
    0,  # Cliente A -> Pertenece al Grupo 0 (jóvenes, gasto alto)
    1,  # Cliente B -> Pertenece al Grupo 1 (mayores, gasto bajo)
    0,  # Cliente C -> Pertenece al Grupo 0
    1   # Cliente D -> Pertenece al Grupo 1
]

print("\nSalida deseada (lo que un algoritmo debe encontrar):") # Imprime el objetivo
for i, etiqueta in enumerate(salida_deseada_Y): # Itera sobre las etiquetas
    print(f"  Punto {i} -> Cluster {etiqueta}") # Imprime Punto 0 -> Cluster 0

print("\nConclusión:")
print("El Agrupamiento No Supervisado es la *tarea* de tomar")
print("los 'Datos de entrada X' (sin etiquetas) y *producir*")
print("las 'Etiquetas de Cluster Y' (inventadas).")
print("\nLos siguientes algoritmos (k-Medias, EM) son los")
print("métodos que *realizan* esta tarea.")