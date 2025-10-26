from collections import deque

# Algoritmo de Búsqueda Bidireccional
#La caracteristica principal de este algoritmo es que realiza dos búsquedas simultáneas:
# una desde el nodo inicial y otra desde el nodo objetivo, es decir busca de adelante hacia atras y viceversa al mismo tiempo.
# El grafo original de los ejemplos:
grafo = {
    'A': ['B', 'C', 'D'],
    'B': [],
    'C': ['E', 'F'],
    'D': [],
    'E': [],
    'F': ['G'],
    'G': []
}

# --- 1. Función para crear el grafo inverso ---
#Es necesario porque nos permite ir "hacia atrás" desde el nodo objetivo.
#El nodo objetivo se convierte en el punto de partida para la segunda búsqueda.

def crear_grafo_inverso(g):
    """
    Invierte todas las 'flechas' del grafo.
    Si A -> C, en el inverso C -> A.
    """
    g_inverso = {nodo: [] for nodo in g} # Inicializa el nuevo grafo
    for nodo_origen, vecinos in g.items():
        for nodo_destino in vecinos:
            # Agrega la arista invertida
            g_inverso[nodo_destino].append(nodo_origen)
    return g_inverso

# --- 2. Función para reconstruir el camino ---

def reconstruir_camino(padres_inicio, padres_fin, nodo_encuentro):
    """
    Une los dos caminos en el punto de encuentro.
    """
    camino = [nodo_encuentro]
    
    # 1. Reconstruye el camino desde el INICIO hasta el encuentro
    temp = nodo_encuentro
    while padres_inicio[temp] is not None:
        camino.insert(0, padres_inicio[temp]) # Añade al *principio* de la lista
        temp = padres_inicio[temp]
        
    # 2. Reconstruye el camino desde el FIN hasta el encuentro
    temp = nodo_encuentro
    while padres_fin[temp] is not None:
        camino.append(padres_fin[temp]) # Añade al *final* de la lista
        temp = padres_fin[temp]
        
    return camino

# --- 3. Implementacio del algoritmo de Búsqueda Bidireccional ---

def busqueda_bidireccional(grafo, grafo_inverso, nodo_inicial, nodo_objetivo):
    
    # --- Configuración para la búsqueda DESDE EL INICIO ---
    queue_inicio = deque([nodo_inicial])
    # Guardamos {nodo: padre} para reconstruir el camino
    padres_inicio = {nodo_inicial: None} 
    
    # --- Configuración para la búsqueda DESDE EL FIN ---
    queue_fin = deque([nodo_objetivo])
    # Guardamos {nodo: padre} para reconstruir el camino
    padres_fin = {nodo_objetivo: None}

    while queue_inicio and queue_fin: # Mientras ambos exploradores tengan dónde buscar

        # ---PASO 1: Mover el explorador del INICIO ---
        nodo_actual_i = queue_inicio.popleft()
        
        # Comprobación de encuentro
        if nodo_actual_i in padres_fin:
            print(f"¡Encuentro en el nodo {nodo_actual_i} (explorado desde INICIO)!")
            return reconstruir_camino(padres_inicio, padres_fin, nodo_actual_i)
            
        # Explorar vecinos hacia adelante
        for vecino in grafo[nodo_actual_i]:
            if vecino not in padres_inicio:
                padres_inicio[vecino] = nodo_actual_i
                queue_inicio.append(vecino)
                
        # ---PASO 2: Mover el explorador del FIN ---
        nodo_actual_f = queue_fin.popleft()
        
        # Comprobación de encuentro
        if nodo_actual_f in padres_inicio:
            print(f"¡Encuentro en el nodo {nodo_actual_f} (explorado desde FIN)!")
            return reconstruir_camino(padres_inicio, padres_fin, nodo_actual_f)
            
        # Explorar vecinos "hacia atrás" usando el grafo inverso
        for vecino_inverso in grafo_inverso[nodo_actual_f]:
            if vecino_inverso not in padres_fin:
                padres_fin[vecino_inverso] = nodo_actual_f
                queue_fin.append(vecino_inverso)

    return "No se encontró camino"

# --- Ejecución ---

# 1. Creamos el grafo inverso
grafo_inv = crear_grafo_inverso(grafo)
print("Grafo Original:", grafo)
print("Grafo Inverso:", grafo_inv)

# 2. Ejecutamos la búsqueda
print("\n--- Iniciando Búsqueda Bidireccional de 'A' a 'G' ---")
camino_encontrado = busqueda_bidireccional(grafo, grafo_inv, 'A', 'G')

print("\nCamino más corto:", camino_encontrado)

print("\n--- Buscando de 'A' a 'D' ---")
camino_corto = busqueda_bidireccional(grafo, grafo_inv, 'A', 'D')
print("\nCamino más corto:", camino_corto)