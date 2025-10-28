# Algoritmo de MANTO DE MARKOV

# Este es un CONCEPTO teórico y una propiedad de los grafos.
# El "algoritmo" encuentra el conjunto de nodos que forman este "manto".
#
# Definición:
# El Manto de Markov de un nodo (variable) 'X' es el conjunto MÍNIMO de nodos que "aísla" o "protege" a 'X' del resto de la red.

# Es la única "burbuja" de información que necesitas para predecir 'X'.

# (La Regla Formal de Independencia):
# P(X | Manto_Markov(X), Resto_de_la_Red) = P(X | Manto_Markov(X))
#
# Componentes del Manto de Markov de un nodo X:
# 1. Los Padres de X (causas directas).
# 2. Los Hijos de X (efectos directos).
# 3. Los "Co-Padres" de X (otras causas de los efectos directos de X).
#
# 
#
# ¿Cómo funciona este programa?
# Escribiremos una función que, dada una red y un nodo, recorra la estructura del grafo para identificar y devolver el *conjunto* de nodos que forman su Manto de Markov.

# Aplicaciones:
# - Es fundamental para los algoritmos de inferencia porque les dice qué nodos son relevantes.
# - En selección de características (Feature Selection), nos dice cuál es el conjunto mínimo de variables predictivas.
#
# Ventajas:
# - Define formalmente la "localidad" de la información en un modelo probabilístico complejo.
#
# Ejemplo de uso (Usando le red para el nodo 'Alarma'):
# 1. Padres de 'Alarma': {'Robo', 'Terremoto'}
# 2. Hijos de 'Alarma': {'JuanLlama', 'MariaLlama'}
# 3. Co-Padres de 'Alarma': Ninguno (ni Juan ni Maria tienen otros padres).
# => Manto_Markov('Alarma') = {'Robo', 'Terremoto', 'JuanLlama', 'MariaLlama'}

# --- P1: Definición de la Red ---
# (Usamos la misma red 'alarma' anterior)
red_alarma = {
    'Robo': {
        'parents': [], 'cpt': {(): 0.001}
    },
    'Terremoto': {
        'parents': [], 'cpt': {(): 0.002}
    },
    'Alarma': {
        'parents': ['Robo', 'Terremoto'],
        'cpt': {
            (True, True): 0.95, (True, False): 0.94,
            (False, True): 0.29, (False, False): 0.001
        }
    },
    'JuanLlama': {
        'parents': ['Alarma'],
        'cpt': {(True,): 0.90, (False,): 0.05}
    },
    'MariaLlama': {
        'parents': ['Alarma'],
        'cpt': {(True,): 0.70, (False,): 0.01}
    }
}

# --- P2: Algoritmo para Encontrar el Manto de Markov ---

def encontrar_manto_markov(red, nodo_X):
    """
    Encuentra y devuelve el conjunto de nodos que
    forman el Manto de Markov de 'nodo_X'.
    """
    
    # 1. Crear el conjunto (set) para guardar los nodos del manto
    #    (Usamos un 'set' para evitar duplicados automáticamente)
    manto = set()
    
    # --- Paso 1: Añadir los PADRES de X ---
    # Obtener el diccionario del nodo (ej. red['Alarma'])
    nodo_info = red[nodo_X]
    # .update() añade todos los ítems de la lista al 'set'
    manto.update(nodo_info['parents']) # Ej: añade 'Robo', 'Terremoto'
    
    # --- Paso 2: Añadir los HIJOS de X ---
    hijos = set() # Un 'set' temporal para guardar los hijos
    
    # Para encontrar a los hijos, debemos recorrer *toda* la red
    for nombre_nodo_v, info_v in red.items():
        # Si 'nodo_X' está en la lista de padres de 'v'...
        if nodo_X in info_v['parents']:
            # ...entonces 'v' es un hijo de 'X'.
            hijos.add(nombre_nodo_v) # Ej: añade 'JuanLlama', 'MariaLlama'
            
    manto.update(hijos) # Añadir los hijos al manto principal
    
    # --- Paso 3: Añadir los CO-PADRES de X ---
    # (Los otros padres de los hijos de X)
    
    for hijo in hijos: # Iterar sobre los hijos que acabamos de encontrar
        # Obtener la información del nodo hijo
        info_hijo = red[hijo]
        # Obtener *sus* padres
        padres_del_hijo = info_hijo['parents'] # Ej: ['Alarma']
        
        # Añadir todos esos padres al manto
        manto.update(padres_del_hijo)
        
    # --- Limpieza Final ---
    # El manto puede incluir accidentalmente al 'nodo_X' mismo
    # (si un hijo tiene a 'X' como padre, lo cual es seguro).
    # El manto *no* se incluye a sí mismo.
    if nodo_X in manto:
        manto.remove(nodo_X)
        
    return manto # Devuelve el conjunto final

# --- P3: Ejecutar el algoritmo ---
print("--- 3. Manto de Markov (Algoritmo de Búsqueda) ---") # Título
print(f"Red definida con {len(red_alarma)} nodos.")

# Ejemplo 1: Encontrar el Manto de Markov de 'Alarma'
nodo_objetivo = 'Alarma'
manto_alarma = encontrar_manto_markov(red_alarma, nodo_objetivo)
print(f"\nNodo Objetivo: '{nodo_objetivo}'")
print(f"Manto de Markov: {manto_alarma}")
# Resultado esperado: {'Robo', 'Terremoto', 'JuanLlama', 'MariaLlama'}

# Ejemplo 2: Encontrar el Manto de Markov de 'Robo'
# Padres: {}
# Hijos: {'Alarma'}
# Co-Padres (padres de 'Alarma'): {'Robo', 'Terremoto'}
nodo_objetivo = 'Robo'
manto_robo = encontrar_manto_markov(red_alarma, nodo_objetivo)
print(f"\nNodo Objetivo: '{nodo_objetivo}'")
print(f"Manto de Markov: {manto_robo}")
# Resultado esperado: {'Alarma', 'Terremoto'} (¡'Robo' se elimina a sí mismo!)