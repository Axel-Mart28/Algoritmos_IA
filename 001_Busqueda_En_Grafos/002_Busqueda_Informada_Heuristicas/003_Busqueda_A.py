# En este algoritmo se implementa la búsqueda A*, que es una extensión de la búsqueda por costo uniforme y la búsqueda voraz.
# La búsqueda A* utiliza tanto el costo acumulado desde el nodo inicial como una función heurística para estimar el costo total hasta el objetivo.
# En cada paso, elige el nodo con el valor más bajo de f(n) = g(n) + h(n), donde g(n) es el costo desde el inicio hasta el nodo n, y h(n) es la heurística estimada desde n hasta el objetivo.

from queue import PriorityQueue # Esta librería permite manejar una cola de prioridad, de manera que los elementos con menor valor tengan mayor prioridad.


# Grafo con costos por cada arista, los aristas en un grafo representan conexiones entre nodos con un costo
grafo = {
    'A': {'B': 2, 'C': 4, 'D': 6},
    'B': {'E': 3},
    'C': {'F': 2},
    'D': {},
    'E': {'G': 2},
    'F': {'G': 3},
    'G': {}
}

# Heurística: estimación del costo restante hacia 'G'
heuristica = {
    'A': 6,
    'B': 4,
    'C': 3,
    'D': 5,
    'E': 2,
    'F': 1,
    'G': 0
}

def busqueda_a_estrella(grafo, heuristica, inicio, objetivo): # Función de búsqueda A*
    # Cola de prioridad: (f, nodo)
    frontera = PriorityQueue() # Inicializa la frontera
    frontera.put((0, inicio)) # Agrega el nodo inicial con f=0
    
    # Costos desde el inicio hasta cada nodo
    costo_g = {inicio: 0}
    
    # Para reconstruir el camino final
    padres = {inicio: None}

    print("=== Búsqueda A* (A estrella) ===\n")

    while not frontera.empty(): # Mientras haya nodos en la frontera
        f_actual, actual = frontera.get() # Obtener el nodo con el f más bajo
        print(f"Explorando nodo: {actual} (f={f_actual}, g={costo_g[actual]}, h={heuristica[actual]})") # Mostrar nodo actual y sus valores f, g, h

        # Verificar si alcanzamos el objetivo
        if actual == objetivo: # Si se alcanza el objetivo
            print("\nObjetivo alcanzado!")
            break

        for vecino, costo in grafo[actual].items(): # Iterar sobre los vecinos y sus costos
            nuevo_costo_g = costo_g[actual] + costo # Calcular nuevo costo g
            f_vecino = nuevo_costo_g + heuristica[vecino] # Calcular f(n) = g(n) + h(n)

            # Si es la primera vez que visitamos el nodo o encontramos un camino más barato
            if vecino not in costo_g or nuevo_costo_g < costo_g[vecino]: # Si encontramos un camino más barato
                costo_g[vecino] = nuevo_costo_g # Actualizar costo g
                frontera.put((f_vecino, vecino)) # Agregar/actualizar en la frontera
                padres[vecino] = actual # Registrar el padre del vecino
                print(f"  Se agrega/actualiza {vecino}: g={nuevo_costo_g}, h={heuristica[vecino]}, f={f_vecino}") # Mostrar detalles del vecino agregado

        print()

    # Reconstruir camino
    camino = [] # Lista para el camino óptimo
    nodo = objetivo # Comenzar desde el objetivo
    while nodo is not None: # Mientras no lleguemos al inicio
        camino.insert(0, nodo) # Insertar el nodo al inicio del camino
        nodo = padres.get(nodo) # Mover al padre del nodo
    
    return camino, costo_g.get(objetivo, float('inf')) # Devolver el camino y el costo total

# Ejemplo de ejecución
camino, costo_total = busqueda_a_estrella(grafo, heuristica, 'A', 'G') 
print("\nCamino óptimo encontrado:", " → ".join(camino)) # Mostrar el camino óptimo encontrado
print("Costo total del camino:", costo_total) # Mostrar el costo total del camino encontrado

