

def busqueda_profundidad_limitada_DLS(grafo, nodo_inicial, nodo_objetivo, limite):
    """
    Esta es la función DLS (Depth Limited Search), pero ahora devuelve True si encuentra
    el objetivo dentro del límite, y False si no lo encuentra.
    """
    # La pila guarda tuplas: (nodo, profundidad)
    stack = [(nodo_inicial, 0)]
    
    # Usamos un 'set' para los visitados para evitar ciclos y
    # no visitar el mismo nodo múltiples veces en la misma búsqueda.
    visitados = set()

    while stack:
        nodo_actual, profundidad = stack.pop()

        # --- Verificación del Objetivo ---
        if nodo_actual == nodo_objetivo:
            return True  # ¡Encontrado!

        if nodo_actual not in visitados:
            visitados.add(nodo_actual)

            # Solo exploramos si no hemos llegado al límite
            if profundidad < limite:
                vecinos = grafo[nodo_actual]
                # Invertimos los vecinos para que el stack los saque
                # en el mismo orden que en el algoritmo de ejemplo anterior (B, C, D)
                for vecino in reversed(vecinos): 
                    if vecino not in visitados:
                        stack.append((vecino, profundidad + 1))
                        
    return False # No se encontró en este límite

# --- 2. El nuevo algoritmo IDDFS ---

# El mismo grafo de siempre
grafo = {
    'A': ['B', 'C', 'D'], 'B': [], 'C': ['E', 'F'],
    'D': [], 'E': [], 'F': ['G'], 'G': []
}

def busqueda_profundidad_iterativa_IDDFS(grafo, nodo_inicial, nodo_objetivo):
    """
    Esta es la función principal de IDDFS.
    Llama a DLS en un bucle, aumentando el límite.
    """
    
    # Empezamos a buscar con límite 0, luego 1, 2, ...
    limite = 0
    
    while True: # Este bucle simula el incremento "infinito"
        print(f"--- Intentando con Límite = {limite} ---")
        
        # Llamamos a nuestro DLS
        if busqueda_profundidad_limitada_DLS(grafo, nodo_inicial, nodo_objetivo, limite):
            # Si DLS devuelve True, lo encontramos
            print(f"¡Éxito! Nodo '{nodo_objetivo}' encontrado en la profundidad {limite}.")
            return (nodo_objetivo, limite)
        
        # aquí solo se aumenta el límite.
        if limite > len(grafo): # Una simple guarda para no ir al infinito
             print(f"Nodo '{nodo_objetivo}' no encontrado.")
             return None

        limite += 1 # Incrementamos el límite para la próxima iteración

# --- Ejecutando la búsqueda ---

print("Buscando el nodo 'G'...")
busqueda_profundidad_iterativa_IDDFS(grafo, 'A', 'G')

print("\nBuscando el nodo 'Z' (no existe)...")
busqueda_profundidad_iterativa_IDDFS(grafo, 'A', 'Z')