# Este algoitmo es usado para llegar a la meta de la manera mas rapida posible.
# Se basa en obtener informacion adicional del entorno para tomar mejores decisiones.


# Grafo representado como lista de adyacencia
grafo = {
    'A': ['B', 'C', 'D'],
    'B': ['E'],
    'C': ['F'],
    'D': [],
    'E': ['G'],
    'F': ['G'],
    'G': []
}

# Heurística: estimación de qué tan cerca está cada nodo de la meta 'G'
heuristica = { #Definicion de la funcion heuristica (h(n)), donde n es el nodo
    
# Valor estimado de costo desde el nodo hasta el objetivo 'G'
    'A': 6, 
    'B': 4, 
    'C': 3,
    'D': 5,
    'E': 2,
    'F': 1,
    'G': 0
}

# Función que realiza una búsqueda orientada por heuristica
def busqueda_por_heuristica(grafo, heuristica, inicio, objetivo):
    actual = inicio # Nodo actual comienza en el nodo de inicio
    camino = [actual] # Lista para registrar el camino seguido
    
    print("Inicio de la búsqueda informada (solo heuristica):")
    print(f"Nodo inicial: {inicio}, Nodo objetivo: {objetivo}\n")
    
    while actual != objetivo: # Mientras no se alcance el objetivo
        vecinos = grafo[actual] # Obtener los nodos vecinos
        if not vecinos: # Si no hay vecinos, se detiene la búsqueda
            print(f"No hay más caminos desde {actual}. La búsqueda se detiene.")
            break

        # Mostrar los valores heurísticos de los vecinos
        print(f"Vecinos de {actual}: {vecinos}") # Mostrar vecinos
        for v in vecinos: # Mostrar valor heurístico de cada vecino
            print(f"  h({v}) = {heuristica[v]}") # Valor heurístico del vecino

        # Elegir el vecino con el valor heurístico más bajo
        siguiente = min(vecinos, key=lambda n: heuristica[n]) # Nodo con menor h(n)
        print(f"→ Elegido el nodo {siguiente} con h({siguiente}) = {heuristica[siguiente]}\n") # Nodo siguiente elegido

        actual = siguiente # Moverse al siguiente nodo
        camino.append(actual) # Registrar el nodo en el camino

        if actual == objetivo: # Si se alcanza el objetivo
            print(" Objetivo alcanzado!\n") 
            break

    return camino # Devolver el camino seguido

# Ejemplo de ejecución
camino = busqueda_por_heuristica(grafo, heuristica, 'A', 'G')
print("Camino seguido:", " → ".join(camino))
