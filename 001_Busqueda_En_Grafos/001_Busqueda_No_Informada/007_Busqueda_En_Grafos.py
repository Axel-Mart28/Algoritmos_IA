from collections import deque

# Algoritmo de Búsqueda en Grafos (Genérico)
# La característica principal de este algoritmo es que evita entrar en un bucle infinito de nodos ya visitados
#La diferencia que tiene con los otros algoritmos es que este utiliza un conjunto de nodos explorados para evitar visitar el mismo nodo más de una vez.

# El grafo de siempre
grafo = {
    'A': ['B', 'C', 'D'], 'B': [], 'C': ['E', 'F'],
    'D': [], 'E': [], 'F': ['G'], 'G': []
}

def busqueda_en_grafos_generica(grafo, nodo_inicial, nodo_objetivo):
    
    # 1. La FRONTERA (la estructura que define el tipo de búsqueda)
    #    Usaremos una COLA (deque) para hacer una búsqueda tipo BFS.
    frontera = deque([nodo_inicial])
    
    # 2. El conjunto de EXPLORADOS (LA CLAVE de "Búsqueda en Grafos")
    #    Usamos un 'set' porque es más rápido para buscar.
    explorados = set()
    
    padres = {nodo_inicial: None} # Para reconstruir el camino luego

    while frontera: # Mientras haya nodos por explorar
        
        # a. Sacar un nodo de la frontera
        nodo_actual = frontera.popleft() # .popleft() -> COLA (BFS)
                                         # .pop()     -> PILA (DFS)

        # b. ¿Es el objetivo?
        if nodo_actual == nodo_objetivo:
            print(f"¡Objetivo '{nodo_objetivo}' encontrado!")
            
            # Reconstruir camino desde el inicial hasta el objetivo
            camino = []
            while nodo_actual is not None:
                camino.insert(0, nodo_actual)
                nodo_actual = padres[nodo_actual]
            return camino

        # c. Añadir el nodo a EXPLORADOS
        explorados.add(nodo_actual)

        # d. Encontrar vecinos
        for vecino in grafo[nodo_actual]:
            
            # e. ¡LA REGLA DE BÚSQUEDA EN GRAFOS!
            #    Si el vecino NO está en EXPLORADOS y NO está en la FRONTERA, entonces...
            if vecino not in explorados and vecino not in frontera:
                
                # ...lo añadimos a la frontera para visitarlo después.
                frontera.append(vecino)
                padres[vecino] = nodo_actual # Guardamos su padre

    return "Objetivo no encontrado"

# --- Ejecución ---
print("--- Ejecutando Búsqueda en Grafos (con base BFS) ---")
camino_encontrado = busqueda_en_grafos_generica(grafo, 'A', 'G')
print(f"Camino: {camino_encontrado}")

camino_encontrado_2 = busqueda_en_grafos_generica(grafo, 'A', 'D')
print(f"\nCamino: {camino_encontrado_2}")