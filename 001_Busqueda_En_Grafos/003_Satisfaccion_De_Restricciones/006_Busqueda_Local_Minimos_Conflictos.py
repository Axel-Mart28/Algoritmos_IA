#Este es el algoritmo de Búsqueda Local de Mínimos Conflictos para problemas de satisfacción de restricciones (CSP).
# Es un algoritmo estocástico (aleatorio) e incompleto (podría no encontrar la solución, aunque exista), pero es extremadamente rápido y efectivo para ciertos tipos de CSP, especialmente los muy grandes.
# Entre las aplicaciones comunes de este algoritmo se encuentran la coloración de mapas, la asignación de horarios y la resolución de puzzles como el Sudoku.
# Entre sus ventajas se encuentran su rapidez y capacidad para manejar grandes problemas, mientras que sus desventajas incluyen su naturaleza incompleta y la posibilidad de quedar atrapado en mínimos locales.

import random # Necesitaremos funciones aleatorias

# --- 1. Definición del CSP de mapa de colores ---

variables = ['WA', 'NT', 'SA', 'Q', 'NSW', 'V', 'T']

domains = { # colores disponibles
    'WA': ['rojo', 'verde', 'azul'],
    'NT': ['rojo', 'verde', 'azul'],
    'SA': ['rojo', 'verde', 'azul'],
    'Q':  ['rojo', 'verde', 'azul'],
    'NSW':['rojo', 'verde', 'azul'],
    'V':  ['rojo', 'verde', 'azul'],
    'T':  ['rojo', 'verde', 'azul']
}

constraints = { # vecinos
    'WA': ['NT', 'SA'],
    'NT': ['WA', 'SA', 'Q'],
    'SA': ['WA', 'NT', 'Q', 'NSW', 'V'],
    'Q':  ['NT', 'SA', 'NSW'],
    'NSW':['Q', 'SA', 'V'],
    'V':  ['SA', 'NSW'],
    'T':  []
}

# --- 2. Funciones Auxiliares para Minimos conflictos ---

def initial_assignment(variables, domains): # Crea una asignación inicial aleatoria
    """ Crea una asignación inicial completa, eligiendo al azar. """
    assignment = {} # diccionario de asignación
    for var in variables: # para cada variable
        # Elige un valor aleatorio del dominio de la variable
        assignment[var] = random.choice(domains[var]) # asignarlo
    return assignment # devolver la asignación inicial

def count_conflicts(variable, value, assignment, constraints): # Cuenta conflictos para una variable y valor dados
    """
    Cuenta cuántos conflictos tendría 'variable' si se le asigna 'value',
    dada la asignación actual del resto.
    """
    count = 0 # contador de conflictos
    for neighbor in constraints[variable]: # Iterar sobre los vecinos
        # Si el vecino ya tiene un valor y es el mismo que el nuestro
        if neighbor in assignment and assignment[neighbor] == value: # conflicto
            count += 1 # incrementar contador
    return count # devolver número de conflictos

def get_conflicted_variables(assignment, constraints): # Obtiene la lista de variables en conflicto
    """ Devuelve una lista de todas las variables que están en conflicto. """
    conflicted = [] # lista de variables en conflicto
    for var in variables: # para cada variable
        # Usamos la función anterior para ver si la asignación ACTUAL de 'var'
        # tiene conflictos.
        if count_conflicts(var, assignment[var], assignment, constraints) > 0: # si hay conflictos
            conflicted.append(var) # añadir a la lista
    return conflicted # devolver la lista

#Implementación del algoritmo de Mínimos Conflictos

def min_conflicts(variables, domains, constraints, max_steps=1000): # Algoritmo de Mínimos Conflictos
    """
    Resuelve el CSP usando el algoritmo de Mínimos-Conflictos.
    'max_steps' es un límite para evitar bucles infinitos.
    """
    
    # 1. Generar una asignación inicial aleatoria completa
    current_assignment = initial_assignment(variables, domains)
    
    print(f"Asignación inicial (aleatoria): {current_assignment}")
    
    #Iterar y reparar para un máximo de 'max_steps' pasos
    for i in range(max_steps):
        
        # 3. Comprobar si es una solución
        conflicted_vars = get_conflicted_variables(current_assignment, constraints)
        
        if not conflicted_vars:
            print(f"\n¡Solución encontrada en {i+1} pasos!")
            return current_assignment # ¡Éxito!
        
        # 4. Elegir una variable en conflicto al azar
        var = random.choice(conflicted_vars)
        
        # 5. Encontrar el valor que minimiza los conflictos
        scores = {} # {valor: num_conflictos}
        for value in domains[var]:
            scores[value] = count_conflicts(var, value, current_assignment, constraints)
            
        # Encontrar la puntuación (conflicto) más baja
        min_score = min(scores.values())
        
        # Obtener todos los valores que tienen esa puntuación mínima
        best_values = [v for v, s in scores.items() if s == min_score]
        
        # 6. Asignar uno de los mejores valores (aleatoriamente si hay empate)
        current_assignment[var] = random.choice(best_values)
        
        # (Opcional: imprimir el progreso)
        print(f"Paso {i}: Re-asignado {var} a {current_assignment[var]}")

    print(f"\nFallo: No se encontró solución después de {max_steps} pasos.")
    return None # Fracaso

# --- 4. Resolver el problema ---

print("Buscando una solución con Mínimos-Conflictos...")

solution = min_conflicts(variables, domains, constraints)

if solution:
    print("\nAsignación final sin conflictos:")
    for variable in sorted(solution.keys()):
        print(f"  {variable}: {solution[variable]}")
else:
    print("\nLa búsqueda local se atascó o agotó los pasos.")