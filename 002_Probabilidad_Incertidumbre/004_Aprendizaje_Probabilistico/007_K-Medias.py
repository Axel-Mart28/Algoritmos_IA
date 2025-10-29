# ALGORITMO K-MEDIAS (K-MEANS)

# Este es un algoritmo de *Aprendizaje No Supervisado* utilizado para *Agrupamiento* (Clustering).
# Es el algoritmo principal para la tarea definida en el tema #4 (Agrupamiento No Supervisado).
#
# Definición:
# k-Means es un algoritmo *iterativo* que particiona un conjunto de datos
# en 'k' grupos distintos (clusters), donde 'k' es un número que
# *nosotros* especificamos de antemano.
#
#Objetivo:
# Minimizar la "inercia" o "varianza intra-cluster". En otras palabras,
# encontrar 'k' puntos centrales (llamados "centroides") de tal manera
# que la suma de las distancias al cuadrado desde cada punto de datos
# hasta el centroide de *su* cluster sea lo más pequeña posible.
#
# ¿Cómo funciona? (El ciclo Asignar-Actualizar):
# 
# 1. INICIALIZACIÓN:
#    - Elige 'k' puntos de datos al azar para que sean los
#      *centroides iniciales* de los clusters.
#
# 2. BUCLE ITERATIVO (repetir hasta que no haya cambios):
#    a. PASO DE ASIGNACIÓN (Assignment Step):
#       - Para *cada* punto de datos:
#       - Calcula la distancia desde ese punto a *cada uno* de los 'k' centroides.
#       - Asigna el punto al cluster cuyo centroide esté *más cerca*.
#
#    b. PASO DE ACTUALIZACIÓN (Update Step):
#       - Para *cada* cluster (del 1 al k):
#       - Calcula la *nueva posición* del centroide.
#       - El nuevo centroide es el *promedio* (la media) de todos los
#         puntos que fueron asignados a ese cluster en el paso anterior.
#
# 3. TERMINACIÓN:
#    - El algoritmo termina cuando, en un ciclo completo, *ningún* punto
#      de datos cambia de cluster (o cuando se alcanza un número
#      máximo de iteraciones). Los centroides finales definen los clusters.
#
# Componentes:
# 1. Los Datos (X): Puntos sin etiquetas (ej. [x, y]).
# 2. El parámetro 'k': El número *deseado* de clusters.
# 3. Una Métrica de Distancia: (Generalmente Euclidiana).
# 4. Centroides: Los puntos centrales (medias) de cada cluster.
#
# Aplicaciones:
# - Las mismas que el Agrupamiento No Supervisado (tema #4):
#   Segmentación de clientes, compresión de imágenes, etc.
#
# Ventajas:
# - Muy rápido y simple de implementar.
# - Escalable a grandes conjuntos de datos.
# - Fácil de interpretar.
#
# Desventajas:
# - ¡Muy sensible a la *inicialización* aleatoria! Puede converger
#   a una solución "mala" (óptimo local) si los centroides
#   iniciales son malos. (Solución: ejecutarlo varias veces).
# - Necesita que *especifiquemos* 'k' de antemano (¿cómo saber
#   cuántos clusters hay realmente?).
# - Asume que los clusters son "esféricos" y de tamaño similar.
#   Falla con clusters de formas raras o densidades diferentes.
# - Sensible a la escala de los datos (igual que k-NN, necesita normalización).
#
# Ejemplo de uso:
# - Tomaremos los datos de clientes (edad, gasto) del tema #4
#   y le pediremos a k-Means que encuentre k=2 clusters.

import math # Para la distancia euclidiana
import random # Para la inicialización aleatoria
import copy # Para comparar si los centroides han cambiado

# --- P1: Función de Distancia Euclidiana ---
# (La misma que usamos en k-NN)

def distancia_euclidiana(punto1, punto2): # Calcula la distancia entre dos puntos
    """ Calcula la distancia euclidiana entre dos puntos (listas o tuplas) """
    if len(punto1) != len(punto2): # Comprobar dimensiones
        raise ValueError("Los puntos deben tener la misma dimensión") # Error
    suma_cuadrados = sum([(p1 - p2)**2 for p1, p2 in zip(punto1, punto2)]) # Suma de (x1-x2)^2 + (y1-y2)^2 ...
    return math.sqrt(suma_cuadrados) # Devuelve la raíz cuadrada

# --- P2: Algoritmo k-Means ---

class KMeans: # Clase para el algoritmo k-Means
    def __init__(self, k=2, max_iter=100, tol=1e-4): # Constructor
        self.k = k # Número de clusters deseado
        self.max_iter = max_iter # Máximo de iteraciones para evitar bucles infinitos
        self.tol = tol # Tolerancia para la convergencia (si los centroides se mueven menos que esto)
        self.centroides = [] # Lista para almacenar las posiciones de los centroides
        self.etiquetas = [] # Lista para almacenar la asignación de cluster de cada punto

    def fit(self, X): # El método principal que ejecuta el algoritmo
        """ Encuentra los k clusters en los datos X """
        
        num_puntos, num_dims = len(X), len(X[0]) # Obtener tamaño de los datos
        
        # --- 1. INICIALIZACIÓN ALEATORIA ---
        # Elegir k puntos de X al azar como centroides iniciales
        indices_iniciales = random.sample(range(num_puntos), self.k) # Elige k índices únicos
        self.centroides = [X[i] for i in indices_iniciales] # Guarda los puntos correspondientes
        
        print(f"Centroides Iniciales (aleatorios): {self.centroides}") # Mensaje
        
        # --- 2. BUCLE ITERATIVO (Asignar-Actualizar) ---
        for iteracion in range(self.max_iter): # Repetir hasta max_iter
            print(f"  Iteración {iteracion + 1}:") # Mensaje
            
            # --- 2a. PASO DE ASIGNACIÓN ---
            # Crear una lista para guardar las asignaciones de esta iteración
            asignaciones = [0] * num_puntos # [0, 0, 0, ...]
            
            for i, punto in enumerate(X): # Para cada punto de datos
                # Calcular la distancia a cada centroide
                distancias_centroides = [distancia_euclidiana(punto, c) for c in self.centroides]
                # Encontrar el índice del centroide más cercano
                cluster_asignado = distancias_centroides.index(min(distancias_centroides))
                # Guardar la asignación (ej. punto 'i' pertenece al cluster 0)
                asignaciones[i] = cluster_asignado
                
            print(f"    Asignaciones: {asignaciones}") # Mensaje (ver cómo cambian los puntos de cluster)
            
            # --- 2b. PASO DE ACTUALIZACIÓN ---
            # Guardar los centroides *antiguos* para comprobar convergencia
            centroides_antiguos = copy.deepcopy(self.centroides) # Copia profunda
            
            # Calcular los nuevos centroides (la media de los puntos asignados)
            for j in range(self.k): # Para cada cluster (0, 1, ..., k-1)
                # Encontrar todos los puntos asignados a este cluster 'j'
                puntos_en_cluster = [X[i] for i, asign in enumerate(asignaciones) if asign == j]
                
                # Si un cluster quedó vacío (raro, pero posible), no mover su centroide
                if not puntos_en_cluster:
                    continue # Saltar a la siguiente iteración del bucle 'j'
                    
                # Calcular la media (promedio) de esos puntos
                # (Sumar todas las coordenadas x, dividir por #puntos; sumar y, dividir...)
                nuevo_centroide = [] # Lista para las nuevas coordenadas
                for dim in range(num_dims): # Iterar sobre x, y, ...
                    # Suma de la coordenada 'dim' para todos los puntos en el cluster
                    suma_dim = sum(p[dim] for p in puntos_en_cluster)
                    # Calcular la media
                    media_dim = suma_dim / len(puntos_en_cluster)
                    nuevo_centroide.append(media_dim) # Añadir la media a la lista
                    
                # Actualizar el centroide del cluster 'j'
                self.centroides[j] = nuevo_centroide
                
            print(f"    Nuevos Centroides: {self.centroides}") # Mensaje
            
            # --- 3. COMPROBAR CONVERGENCIA ---
            # Calcular cuánto se movieron los centroides
            movimiento_total = 0.0
            for c_nuevo, c_antiguo in zip(self.centroides, centroides_antiguos):
                movimiento_total += distancia_euclidiana(c_nuevo, c_antiguo)
                
            print(f"    Movimiento de centroides: {movimiento_total:.4f}") # Mensaje
            
            # Si el movimiento es menor que la tolerancia, hemos terminado
            if movimiento_total < self.tol:
                print(f"  Algoritmo convergido en {iteracion + 1} iteraciones.") # Mensaje
                break # Salir del bucle 'for iteracion'
                
        # Guardar las asignaciones finales
        self.etiquetas = asignaciones # Guardar la última asignación

# --- P3: Ejecutar k-Means ---
print("--- Algoritmo k-Means Clustering ---") # Título

# 1. Los Datos (los mismos del tema #4)
datos_X = [
    [25, 800],  # Cliente A (joven, gasta mucho) -> Esperado Cluster 0
    [50, 200],  # Cliente B (mayor, gasta poco) -> Esperado Cluster 1
    [30, 900],  # Cliente C (joven, gasta mucho) -> Esperado Cluster 0
    [45, 150]   # Cliente D (mayor, gasta poco) -> Esperado Cluster 1
]
K = 2 # Queremos 2 clusters

print(f"Datos de entrada (X): {datos_X}") # Mensaje
print(f"Número de clusters deseado (k): {K}") # Mensaje

# 2. Crear y Ejecutar el algoritmo k-Means
kmeans = KMeans(k=K, max_iter=10) # Crear instancia (max 10 iteraciones)
kmeans.fit(datos_X) # Ejecutar el algoritmo

# 3. Imprimir los resultados finales
print("\n--- Resultados Finales ---")
print(f"Centroides finales encontrados: {kmeans.centroides}") # Posición de los centros
print(f"Asignación final de clusters (Etiquetas): {kmeans.etiquetas}") # La lista [0, 1, 0, 1] (o [1, 0, 1, 0])

# Imprimir qué puntos pertenecen a qué cluster
for cluster_id in range(K): # Iterar 0, 1
    puntos = [datos_X[i] for i, label in enumerate(kmeans.etiquetas) if label == cluster_id]
    print(f"  Cluster {cluster_id}: {puntos}")

print("\nConclusión:")
print("k-Means encontró iterativamente los centroides y asignó")
print("cada punto al cluster más cercano, separando exitosamente")
print("a los clientes 'jóvenes/gasto alto' de los 'mayores/gasto bajo'.")