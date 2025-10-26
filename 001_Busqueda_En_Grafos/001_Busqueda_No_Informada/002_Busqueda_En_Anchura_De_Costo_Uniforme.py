# El objetivo de este algoritmo es encontrar el camino con menor costo total
# Ahora cada vecino debe de tener un costo.

import heapq  # Importamos heapq para usarlo como una cola de prioridad

# El grafo ahora incluye el "costo" para ir de un nodo a otro
# Formato: 'Nodo': [('Vecino1', costo1), ('Vecino2', costo2), ...]
grafo_con_costos = {
    'A': [('B', 5), ('C', 2), ('D', 8)], # A->B cuesta 5, A->C cuesta 2, etc.
    'B': [],
    'C': [('E', 4), ('F', 1)],
    'D': [],
    'E': [],
    'F': [('G', 3)],
    'G': []
}


def busqueda_costo_uniforme(grafo, nodo_inicial):
    # --- Diferencia 1: La cola de prioridad ---
    # Almacena tuplas: (costo_acumulado, nodo)
    # Empezamos con costo 0 para el nodo inicial.
    priority_queue = [(0, nodo_inicial)]
    
    # --- Diferencia 2: Almacén de costos ---
    # Un diccionario para llevar el costo mínimo encontrado
    # hasta ahora para llegar a cada nodo.
    costos = {nodo_inicial: 0}
    
    # --- Diferencia 3: Almacén de recorrido ---
    # Guardaremos el orden en que visitamos los nodos
    # (sacamos de la cola de prioridad)
    recorrido = []
    
    # Usar un set() es más eficiente para buscar
    visitados = set() 

    while priority_queue: # Mientras la cola de prioridad no esté vacía
        
        # --- Diferencia 4: Sacar de la cola ---
        # Sacamos el nodo con el MENOR costo acumulado
        costo_actual, nodo_actual = heapq.heappop(priority_queue)

        # Si ya lo visitamos (por un camino anterior más corto), lo ignoramos
        if nodo_actual in visitados:
            continue
            
        # Lo marcamos como visitado y lo agregamos al recorrido
        visitados.add(nodo_actual)
        recorrido.append(nodo_actual)

        # Exploramos los vecinos
        for vecino, peso_arista in grafo[nodo_actual]:
            if vecino not in visitados:
                # Calculamos el nuevo costo para llegar a ese vecino
                nuevo_costo = costo_actual + peso_arista
                
                # --- Diferencia 5: Lógica de actualización ---
                # Si no lo habíamos alcanzado O encontramos un camino MÁS BARATO
                if vecino not in costos or nuevo_costo < costos[vecino]:
                    # Actualizamos su costo mínimo
                    costos[vecino] = nuevo_costo
                    # Agregamos al vecino a la cola de prioridad con su nuevo costo
                    heapq.heappush(priority_queue, (nuevo_costo, vecino))

    return recorrido, costos

# --- Ejecutando la búsqueda ---
recorrido_ucs, costos_finales = busqueda_costo_uniforme(grafo_con_costos, 'A')

print("Recorrido en Costo Uniforme:", recorrido_ucs)
print("Costos finales desde 'A':", costos_finales)