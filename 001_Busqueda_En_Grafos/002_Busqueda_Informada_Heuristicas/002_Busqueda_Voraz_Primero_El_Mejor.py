# Este algoritmo es una implementación de la búsqueda voraz, que utiliza una función heurística para guiar la búsqueda hacia el objetivo de manera eficiente.
# La diferencia principal con la búsqueda por heurística simple es que la búsqueda voraz siempre elige el nodo que parece mejor enfunción de la heurística, sin considerar el costo acumulado desde el nodo inicial.
# Elige el nodo con el valor heurístico más bajo en cada paso, con el objetivo de llegar al objetivo rápidamente.

# Grafo representado como lista
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
heuristica = {
    'A': 6,
    'B': 4,
    'C': 3,
    'D': 5,
    'E': 2,
    'F': 1,
    'G': 0
}

from queue import PriorityQueue  # Cola con prioridad para ordenar por heurística

def busqueda_voraz(grafo, heuristica, inicio, objetivo): # Función de búsqueda voraz primero el mejor
    # Cola de prioridad: guarda tuplas (heurística, nodo)
    frontera = PriorityQueue() # Inicializa la frontera, la cual ordena los nodos por su valor heurístico
    frontera.put((heuristica[inicio], inicio)) # Agrega el nodo inicial a la frontera
    
    visitados = set() # Conjunto de nodos visitados
    padres = {inicio: None} # Diccionario para reconstruir el camino, el diccionario hace refencia al nodo padre de cada nodo

    print("=== Búsqueda Voraz Primero el Mejor ===\n")
    
    while not frontera.empty(): # Mientras haya nodos en la frontera
        _, actual = frontera.get() # Obtener el nodo con la heurística más baja
        
        if actual in visitados: # Si ya fue visitado,
            continue # saltar al siguiente
        visitados.add(actual) # Marcar el nodo como visitado
        
        print(f"Explorando nodo: {actual} (h={heuristica[actual]})")
        
        if actual == objetivo: # Si se alcanza el objetivo
            print("\nObjetivo alcanzado!")
            break

        # Agregar vecinos a la frontera según su heurística
        for vecino in grafo[actual]: # Iterar sobre los vecinos del nodo actual
            if vecino not in visitados: # Si el vecino no ha sido visitado
                frontera.put((heuristica[vecino], vecino)) # Agregar a la frontera con su valor heurístico
                padres[vecino] = actual # Registrar el padre del vecino
                print(f"  Se agrega a la frontera: {vecino} (h={heuristica[vecino]})") # Agregar vecino a la frontera

        print()

    # Reconstruir el camino
    camino = []
    nodo = objetivo # Comenzar desde el objetivo
    while nodo is not None: # Mientras no se llegue al nodo inicial
        camino.insert(0, nodo) # Insertar el nodo al inicio del camino
        nodo = padres.get(nodo) # Mover al padre del nodo
    
    return camino # Devolver el camino encontrado

# Ejemplo de ejecución
camino = busqueda_voraz(grafo, heuristica, 'A', 'G')
print("\nCamino encontrado:", " → ".join(camino))


