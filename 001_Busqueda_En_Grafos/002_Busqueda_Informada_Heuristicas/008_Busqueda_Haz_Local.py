# Este algoritmo es una implementación de la búsqueda local, una técnica heurística utilizada para encontrar soluciones aproximadas a problemas de optimización.
# La búsqueda local comienza con una solución inicial y explora sus vecinos para encontrar una solución mejor.
# La característica principal de este algoritmo es que solo considera soluciones vecinas inmediatas y se mueve hacia la mejor solución encontrada en cada paso.
# La búsqueda de haz local (Local Beam Search) es una mejora de la ascensión de colinas.
# En lugar de trabajar con una sola solución, el algoritmo mantiene k posibles estados simultáneamente (llamados haz o beam).

#Entre sus ventajas se encuentran:
# - Diversidad de soluciones: Al mantener múltiples estados, el algoritmo puede explorar diferentes áreas del espacio de soluciones, lo que reduce la probabilidad de quedarse atrapado en óptimos locales.
# - Flexibilidad: El tamaño del haz (k) puede ajustarse según los recursos disponibles y la complejidad del problema.

# Sin embargo, una de sus desventajas es quepuede quedarse atrapado en óptimos locales, ya que no explora soluciones más lejanas que podrían ser mejores.
# Otra deventaja es que requiere mas memoria y calculos.

#Implementacion de grafo y heuristica
grafo = {
    'A': ['B', 'C', 'D'],
    'B': ['E', 'F'],
    'C': ['G'],
    'D': ['H', 'I'],
    'E': [],
    'F': ['J'],
    'G': [],
    'H': [],
    'I': [],
    'J': []
}

# Heurística (distancia estimada a la meta, menor es mejor)
heuristica = {
    'A': 9,
    'B': 7,
    'C': 6,
    'D': 5,
    'E': 3,
    'F': 2,
    'G': 1,
    'H': 4,
    'I': 3,
    'J': 0  # Meta
}

def busqueda_haz_local(grafo, heuristica, estados_iniciales, objetivo, k=2, max_iter=10): # Función de búsqueda de haz local
    haz = estados_iniciales[:]  # Copiamos los estados iniciales
    print("=== Búsqueda de Haz Local ===\n")
    print(f"Estados iniciales: {haz}\n")

    for iteracion in range(max_iter): # Bucle principal de la búsqueda
        print(f"--- Iteración {iteracion + 1} ---") 
        nuevos_estados = [] #Lista para almacenar nuevos estados

        # Generar todos los vecinos de los estados actuales
        for estado in haz: # Iterar sobre cada estado en el haz
            vecinos = grafo.get(estado, []) # Obtener vecinos del estado actual
            print(f"Vecinos de {estado}: {vecinos}") # Mostrar vecinos
            nuevos_estados.extend(vecinos) # Agregar vecinos a la lista de nuevos estados

        # Si no hay nuevos estados, detener
        if not nuevos_estados:
            print("⚠️ No hay más vecinos, fin de búsqueda.\n")
            break

        # Ordenar los nuevos estados por su heurística
        nuevos_estados = sorted(nuevos_estados, key=lambda n: heuristica[n])

        # Seleccionar los mejores k estados
        haz = nuevos_estados[:k]

        print(f"Estados seleccionados para el siguiente haz: {haz}")
        print(f"Heurísticas: {[heuristica[h] for h in haz]}\n")

        # Verificar si se alcanzó el objetivo
        if objetivo in haz:
            print("Objetivo alcanzado!\n")
            break

    print("Haz final:", haz)
    return haz

# Ejemplo de ejecución
haz_inicial = ['A', 'B']  # Comenzamos con dos estados iniciales
resultado = busqueda_haz_local(grafo, heuristica, haz_inicial, 'J', k=2)
print("Resultado final:", resultado)
