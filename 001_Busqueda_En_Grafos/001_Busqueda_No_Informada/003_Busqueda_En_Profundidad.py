# No necesitamos deque, una lista normal de Python funciona como pila (stack)

# Usamos el mismo grafo
grafo = {
    'A': ['B', 'C', 'D'], #A tiene tres vecinos: B, C y D
    'B': [], #B no tiene vecinos
    'C': ['E', 'F'], #C tiene dos vecinos: E y F
    'D': [], #D no tiene vecinos
    'E': [], #E no tiene vecinos
    'F': ['G'], #F tiene un vecino: G
    'G': [] #G no tiene vecinos
}

def busqueda_en_profundidad(grafo, nodo_inicial):
    # En lugar de 'queue' (cola), la llamamos 'stack' (pila)
    stack = [nodo_inicial]  ### <-- CAMBIO 1: De deque() a una lista []

    visitados = []

    while stack: # Mientras la pila no esté vacía
        # Sacamos el último elemento de la pila (en vez de el primero)
        nodo_actual = stack.pop()  ### <-- CAMBIO 2: De popleft() a pop()

        if nodo_actual not in visitados:
            visitados.append(nodo_actual)

            vecinos = grafo[nodo_actual]
            # Agregamos los vecinos a la pila
            for vecino in vecinos:
                if vecino not in visitados:
                    stack.append(vecino)

    return visitados

# Comenzando la búsqueda en profundidad desde el nodo 'A':
result = busqueda_en_profundidad(grafo, 'A')
print("Recorrido en profundidad:", result)