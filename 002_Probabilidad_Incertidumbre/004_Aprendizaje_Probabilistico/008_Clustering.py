# -ALGORITMO DE CLUSTERING

# Este es un pseudo-algoritmo que describe los PASOS GENERALES
# del proceso de Clustering (Agrupamiento No Supervisado).
# No es un algoritmo específico como k-Means, sino la idea general.
#
# Definición:
# Es el proceso de tomar datos (X) SIN ETIQUETAS y dividirlos
# en 'k' grupos (clusters) basados en su similitud.
#
# Objetivo:
# Maximizar la similitud *dentro* de cada cluster y
# maximizar la diferencia *entre* clusters.
#
# ¿Cómo funciona (Pasos Conceptuales)?
# 1. ELEGIR 'k': Decidir cuántos clusters (grupos) queremos encontrar.
# 2. INICIALIZAR: Crear una "idea inicial" de dónde están los clusters.
#    (ej. elegir k puntos al azar como centros).
# 3. REPETIR (hasta que no haya cambios):
#    a. ASIGNAR PUNTOS: Para cada punto de datos, decidir a qué
#       cluster pertenece basándose en cuál "representante"
#       (ej. centroide) está más cerca o es más similar.
#    b. ACTUALIZAR REPRESENTANTES: Recalcular la posición o
#       descripción de cada "representante" del cluster basándose
#       en los puntos que le fueron asignados.
#       (ej. el nuevo centroide es la media de los puntos).
# 4. FINALIZAR: Devolver las asignaciones finales (qué punto
#    pertenece a qué cluster) y/o los representantes finales.
#
# Componentes Clave (Abstactos):
# 1. Datos (X): Puntos sin etiquetas.
# 2. Número de Clusters (k).
# 3. Medida de Similitud/Distancia: ¿Cómo se compara un punto
#    con un cluster o con otro punto?
# 4. Representante del Cluster: ¿Cómo se resume un cluster?
#    (ej. un centroide, una distribución de probabilidad).
# 5. Criterio de Convergencia: ¿Cuándo paramos de iterar?

# --- Pseudo-Código (No ejecutable) ---

# def clustering_general(Datos_X, k):
#
#     # 1. Inicializar Representantes (ej. centroides aleatorios)
#     representantes = inicializar_representantes(Datos_X, k)
#     asignaciones_anteriores = None # Para comprobar convergencia
#
#     # 2. Bucle Iterativo
#     repetir_max_veces o hasta_convergencia:
#
#         # --- 3a. Paso de Asignación ---
#         asignaciones_actuales = [] # Lista para [cluster_punto_0, cluster_punto_1, ...]
#         para cada punto P en Datos_X:
#             # Calcular similitud/distancia de P a cada representante R
#             similitudes = [calcular_similitud(P, R) for R in representantes]
#             # Encontrar el representante más similar/cercano
#             mejor_cluster_id = encontrar_mejor_cluster(similitudes)
#             asignaciones_actuales.append(mejor_cluster_id)
#
#         # --- Comprobar Convergencia ---
#         si asignaciones_actuales == asignaciones_anteriores:
#             romper_bucle # ¡Convergencia! No hubo cambios.
#         asignaciones_anteriores = asignaciones_actuales
#
#         # --- 3b. Paso de Actualización ---
#         nuevos_representantes = []
#         para cada cluster_id de 0 a k-1:
#             # Encontrar todos los puntos asignados a este cluster
#             puntos_del_cluster = obtener_puntos(Datos_X, asignaciones_actuales, cluster_id)
#             # Calcular el nuevo representante (ej. la media)
#             nuevo_R = calcular_nuevo_representante(puntos_del_cluster)
#             nuevos_representantes.append(nuevo_R)
#         representantes = nuevos_representantes # Actualizar
#
#     # 4. Devolver Resultado
#     return asignaciones_actuales, representantes

# --- Ejemplo de Ejecución (Conceptual) ---
print("--- Clustering ---") # Título
Datos_X = [[25, 800], [50, 200], [30, 900], [45, 150]] # Datos
k = 2 # Clusters deseados

print(f"Datos: {Datos_X}")
print(f"k: {k}")

print("\nPasos Conceptuales:")
print("1. Elegir k=2 centroides iniciales (ej. [25, 800] y [50, 200]).")
print("2. Iteración 1 - Asignar:")
print("   - Punto [30, 900] está más cerca de [25, 800] -> Cluster 0")
print("   - Punto [45, 150] está más cerca de [50, 200] -> Cluster 1")
print("   -> Asignaciones: [0, 1, 0, 1]")
print("3. Iteración 1 - Actualizar:")
print("   - Nuevo Centroide 0 = media([25, 800], [30, 900]) = [27.5, 850]")
print("   - Nuevo Centroide 1 = media([50, 200], [45, 150]) = [47.5, 175]")
print("4. Iteración 2 - Asignar:")
print("   - (Calcular distancias a los *nuevos* centroides...)")
print("   - (Probablemente las asignaciones [0, 1, 0, 1] no cambien)")
print("5. Iteración 2 - Actualizar:")
print("   - (Los centroides probablemente no cambien mucho)")
print("6. Convergencia: Como las asignaciones no cambiaron, el algoritmo termina.")
print("\nResultado Final:")
print("  - Asignaciones: [0, 1, 0, 1]")
print("  - Centroides: {[27.5, 850], [47.5, 175]}")