#Este es el algoritmo de búsqueda tabú, una técnica metaheurística utilizada para resolver problemas de optimización.
# La Búsqueda Tabú es una extensión de los algoritmos locales (como ascencion de la colina y Temple Simulado), pero introduce una memoria a corto plazo llamada lista tabú, que evita volver a visitar soluciones anteriores.
#Esto permite al algoritmo explorar nuevas zonas del espacio de busqueda
# Entre sus ventajas estan:
# - Evita ciclos: Al mantener una lista de movimientos prohibidos, el algoritmo evita caer en ciclos y quedarse atrapado en óptimos locales.
# - Flexibilidad: Puede adaptarse a una amplia variedad de problemas de optimización.

#Entre sus desventajas estan:
# - Complejidad: La gestión de la lista tabú y la selección de parámetros adecuados pueden aumentar la complejidad del algoritmo.
# - No garantiza la optimalidad: Aunque puede escapar de óptimos locales, no garantiza encontrar la solución óptima global.

#Implementacion de grafo y heuristica
grafo = {
    'A': ['B', 'C', 'D'],
    'B': ['E', 'F'],
    'C': ['G'],
    'D': [],
    'E': [],
    'F': [],
    'G': []
}

# Heurística (distancia estimada a la meta, menor = mejor)
heuristica = {
    'A': 7,
    'B': 6,
    'C': 4,
    'D': 5,
    'E': 3,
    'F': 2,
    'G': 0  # Meta
}

def busqueda_tabu(grafo, heuristica, inicio, objetivo, tamaño_tabu=3, max_iter=20): # Función de búsqueda tabú
    actual = inicio # Nodo actual comienza en el nodo de inicio
    mejor = actual # Mejor nodo encontrado
    lista_tabu = []  # Memoria de nodos prohibidos
    camino = [actual] # Lista para registrar el camino seguido

    print("=== Búsqueda Tabú ===\n")
    print(f"Inicio en nodo: {actual} (h={heuristica[actual]})\n")

    for iteracion in range(max_iter): # Bucle principal de la búsqueda
        vecinos = grafo[actual] # Obtener los vecinos del nodo actual

        if not vecinos: # Si no hay vecinos, detener la búsqueda
            print("⚠️ No hay vecinos disponibles, se detiene la búsqueda.\n")
            break

        # Filtrar vecinos que no estén en la lista tabú
        candidatos = [v for v in vecinos if v not in lista_tabu]

        if not candidatos: # Si todos los vecinos están en la lista tabú, detener la búsqueda
            print("Todos los vecinos están en la lista tabú, se detiene.\n")
            break

        # Escoger el vecino con mejor heurística
        mejor_vecino = min(candidatos, key=lambda n: heuristica[n])

        # Mostrar información del paso
        print(f"Iteración {iteracion+1}")
        print(f"Nodo actual: {actual} (h={heuristica[actual]})")
        print(f"Vecinos disponibles: {vecinos}")
        print(f"Vecinos no tabú: {candidatos}")
        print(f"Mejor vecino elegido: {mejor_vecino} (h={heuristica[mejor_vecino]})\n")

        # Actualizar lista tabú
        lista_tabu.append(actual)
        if len(lista_tabu) > tamaño_tabu:
            lista_tabu.pop(0)

        # Mover al mejor vecino
        actual = mejor_vecino
        camino.append(actual)

        # Si mejora, actualizar el mejor global
        if heuristica[actual] < heuristica[mejor]:
            mejor = actual

        # Si se llega a la meta
        if actual == objetivo:
            print("Objetivo alcanzado!\n")
            break

    print("Lista Tabú final:", lista_tabu)
    print("Camino recorrido:", " → ".join(camino))
    print("Mejor nodo encontrado:", mejor, f"(h={heuristica[mejor]})")

    return camino, mejor # Devolver el camino seguido y el mejor nodo encontrado

# Ejemplo de ejecución
camino, mejor = busqueda_tabu(grafo, heuristica, 'A', 'G')
