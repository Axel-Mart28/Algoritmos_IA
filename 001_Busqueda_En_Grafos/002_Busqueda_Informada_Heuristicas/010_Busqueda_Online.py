# Este algoritmo de busqueda online es una técnica utilizada en inteligencia artificial para navegar en entornos desconocidos.
#Es decir, cuando el grafo es totalmente o parcialmente desconocido.
# La característica principal de este algoritmo es que toma decisiones basadas en la información disponible en el momento, sin tener un conocimiento completo del entorno.
#El agente explora el entorno, actualiza su conocimiento y ajusta su camino en función de la información que va descubriendo.
# Entre sus ventajas se encuentran:
# - Adaptabilidad: Puede adaptarse a cambios en el entorno en tiempo real.
# - Eficiencia en memoria: No requiere almacenar todo el grafo, solo la información relevante.
# - Aplicabilidad: Útil en situaciones donde el entorno es dinámico o desconocido.
# Entre sus desventajas se encuentran:
# - Subóptimo: Puede no encontrar la solución óptima debido a la falta de información completa.
# - Dependencia de la heurística: La calidad de las decisiones depende en gran medida de la heurística utilizada.

import math # Libreria para funciones matemáticas

# Grafo real (el agente no lo conoce completamente desde el inicio)
grafo_real = {
    'A': {'B': 2, 'C': 4},
    'B': {'D': 7, 'E': 3},
    'C': {'F': 5},
    'D': {},
    'E': {'G': 2},
    'F': {},
    'G': {}
}

# Heurística estimada (menor = mejor)
heuristica = {
    'A': 7,
    'B': 6,
    'C': 5,
    'D': 4,
    'E': 2,
    'F': 3,
    'G': 0  # Meta
}


def busqueda_online_LRTA(grafo_real, heuristica, inicio, objetivo, max_iter=20): # Función de búsqueda online
    actual = inicio # Nodo actual comienza en el nodo de inicio
    grafo_conocido = {}  # Lo que el agente descubre
    camino = [actual] # Lista para registrar el camino seguido

    print("=== Búsqueda Online ===\n")
    print(f"Inicio en nodo: {inicio}, objetivo: {objetivo}\n")

    for paso in range(max_iter): # Bucle principal de la búsqueda
        print(f"--- Paso {paso + 1} ---")
        if actual == objetivo: # Si se alcanza el objetivo
            print("Objetivo alcanzado!\n")
            break

        # Descubrir vecinos al llegar a este nodo
        vecinos = grafo_real.get(actual, {}) # Obtener vecinos del grafo real
        grafo_conocido[actual] = vecinos # Actualizar grafo conocido

        print(f"Nodo actual: {actual}")
        print(f"Vecinos descubiertos: {vecinos}")

        # Calcular heurística "actualizada" para vecinos
        costos_estimados = {} # Diccionario para costos estimados
        for v, costo in vecinos.items(): # Para cada vecino
            h = heuristica.get(v, math.inf) # Obtener heurística del vecino
            costos_estimados[v] = costo + h # Costo estimado total

        if not costos_estimados:
            print("Sin vecinos disponibles. Fin de búsqueda.\n")
            break

        # Elegir el mejor vecino según costo estimado
        siguiente = min(costos_estimados, key=costos_estimados.get) # Nodo con menor costo estimado
        mejor_costo = costos_estimados[siguiente] # Mejor costo estimado

        print(f"Mejor opción: {siguiente} (costo estimado: {mejor_costo})\n")

        # Aprendizaje local: actualizar heurística del nodo actual
        heuristica[actual] = mejor_costo # Actualizar heurística localmente
        print(f"Actualizando heurística de {actual} a {heuristica[actual]}\n")

        # Moverse al siguiente nodo
        actual = siguiente # Actualizar nodo actual
        camino.append(actual) # Registrar el nodo en el camino

    print("Camino recorrido:", " → ".join(camino))
    print("Heurística aprendida:", heuristica)
    return camino # Devolver el camino seguido

# Ejemplo de ejecución
camino = busqueda_online_LRTA(grafo_real, heuristica, 'A', 'G')
