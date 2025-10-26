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

def busqueda_profundidad_limitada(grafo, nodo_inicial, limite):
    # La pila ahora guarda tuplas: (nodo, profundidad)
    # Empezamos en el nodo inicial, en profundidad 0
    stack = [(nodo_inicial, 0)]  # Se le agrega el 0 como profundidad inicial

    visitados = [] # Lista de nodos visitados

    while stack: # Mientras la pila no esté vacía
        # Sacamos tanto el nodo como su profundidad actual
        nodo_actual, profundidad = stack.pop() ### <-- CAMBIO 2

        if nodo_actual not in visitados:
            visitados.append(nodo_actual)

            #Implementacion del limite de profundidad
            # Solo exploramos los vecinos SI AÚN NO HEMOS LLEGADO AL LÍMITE
            if profundidad < limite:  ### <-- CAMBIO 3
                vecinos = grafo[nodo_actual]
                for vecino in vecinos:
                    if vecino not in visitados:
                        # Añadimos el vecino con la profundidad incrementada
                        stack.append((vecino, profundidad + 1)) 

    return visitados

#Probando el algoritmo con diferentes límites:

print("--- Límite = 0 ---")
# Solo debe visitar el nodo inicial
result_0 = busqueda_profundidad_limitada(grafo, 'A', 0)
print(result_0)

print("\n--- Límite = 1 ---")
# Debe visitar 'A' y sus vecinos directos
result_1 = busqueda_profundidad_limitada(grafo, 'A', 1)
print(result_1)

print("\n--- Límite = 2 ---")
# Debe visitar 'A', sus vecinos, y los vecinos de sus vecinos
result_2 = busqueda_profundidad_limitada(grafo, 'A', 2)
print(result_2)

print("\n--- Límite = 3 ---")
# Ahora sí debería alcanzar el grafo completo
result_3 = busqueda_profundidad_limitada(grafo, 'A', 3)
print(result_3)