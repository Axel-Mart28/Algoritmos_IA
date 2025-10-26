# Este algoritmo se basa en la idea de "subir la colina" hacia la solución más prometedora.
# La característica principal de este algoritmo es que siempre se mueve hacia el vecino con la mejor heurística, es decir, el que tiene el valor más bajo.
#A diferencia de otros algoritmos informados, la búsqueda de ascensión de colinas no mantiene una lista de nodos abiertos o cerrados, sino que simplemente se mueve hacia el vecino más prometedor en cada paso.
# Sin embargo, este enfoque puede llevar a quedarse atrapado en óptimos locales, ya que no considera otras posibles rutas que podrían llevar a una solución mejor.

import random


grafo = {
    'A': {'B': 6, 'C': 3},
    'B': {'D': 5, 'E': 4},
    'C': {'F': 2, 'G': 1},
    'D': {}, 'E': {}, 'F': {}, 'G': {}
}

# Heurísticas (entre menor, mejor)
heuristica = {
    'A': 7,
    'B': 6,
    'C': 3,
    'D': 5,
    'E': 4,
    'F': 2,
    'G': 0  # Meta
}


def ascension_de_colinas(grafo, heuristica, inicio, objetivo): # Función de búsqueda por ascensión de colinas
    actual = inicio # Nodo actual comienza en el nodo de inicio
    camino = [actual] # Lista para registrar el camino seguido
    
    print("=== Búsqueda por Ascensión de Colinas ===\n") #
    print(f"Inicio en nodo: {actual} (h={heuristica[actual]})\n")

    while True: # Bucle principal de la búsqueda
        vecinos = grafo[actual] # Obtener los vecinos del nodo actual
         # Si no hay vecinos, se detiene la búsqueda
        if not vecinos:
            print("No hay vecinos, se detiene la búsqueda.\n")
            break

        # Encontrar el vecino con la mejor (menor) heurística
        mejor_vecino = min(vecinos, key=lambda n: heuristica[n]) # Nodo con menor h(n)
        mejor_valor = heuristica[mejor_vecino] # Valor heurístico del mejor vecino

        print(f"Vecinos de {actual}: {list(vecinos.keys())}") # Mostrar vecinos
        for v in vecinos: # Mostrar valor heurístico de cada vecino
            print(f"  h({v}) = {heuristica[v]}") # Valor heurístico del vecino
        print(f"→ Mejor vecino: {mejor_vecino} (h={mejor_valor})") # Mejor vecino elegido

        # Si el mejor vecino no mejora, detenerse
        if mejor_valor >= heuristica[actual]:
            print("\nNo hay mejora, alcanzado óptimo local.\n")
            break

        # Moverse al mejor vecino
        actual = mejor_vecino 
        camino.append(actual) # Registrar el nodo en el camino

        print(f"Subiendo hacia: {actual} (h={heuristica[actual]})\n")

        # Si se llega al objetivo
        if actual == objetivo:
            print("Objetivo alcanzado!\n")
            break

    return camino # Devolver el camino seguido

# Ejemplo de ejecución
camino = ascension_de_colinas(grafo, heuristica, 'A', 'G')
print("Camino seguido:", " → ".join(camino))
