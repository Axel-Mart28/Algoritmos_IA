#Este algoritmo de salto atrás por conflictos (Conflict-Directed Backjumping, CBJ) es una mejora del algoritmo de vuelta atrás tradicional para resolver problemas de satisfacción de restricciones (CSP).
#El CBJ optimiza la búsqueda al permitir "saltar" atrás más allá de la última variable asignada cuando se detecta un conflicto, basándose en la identificación de las variables que realmente causaron el conflicto.
#Entre las aplicaciones comunes de este algoritmo se encuentran la coloración de mapas, la asignación de horarios y la resolución de puzzles como el Sudoku.
#Entre sus ventajas se encuentran la reducción del espacio de búsqueda y la eficiencia en la resolución de conflictos, mientras que sus desventajas incluyen la complejidad adicional en la gestión de conjuntos de conflictos y la posible sobrecarga computacional.

# --- 1. Definición del CSP (igual que antes) ---

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

# --- 2. Algoritmo de Conflict-Directed Backjumping (CBJ) ---

def select_unassigned_variable(variables, assignment): # Selecciona la primera variable no asignada
    """ Selecciona la primera variable no asignada """
    for v in variables: # Itera sobre todas las variables
        if v not in assignment: # si no está asignada
            return v # la retorna
    return None

def find_conflict(var, value, assignment, constraints): # Encuentra conflictos inmediatos
    """
    Comprueba si 'value' para 'var' entra en conflicto con
    alguna variable ya en 'assignment'.
    Devuelve la variable en conflicto si la hay, o None si no.
    """
    for neighbor in constraints[var]: # Iterar sobre los vecinos
        if neighbor in assignment: # Si el vecino ya está asignado
            if assignment[neighbor] == value: # y el valor es el mismo
                return neighbor # ¡Conflicto!
    return None # No hay conflicto

def backtrack_cbj(assignment, variables, domains, constraints): # Implementación del algoritmo CBJ
    """
    Función recursiva para Búsqueda con Conflict-Directed Backjumping.
    Devuelve: (asignación_solución, conjunto_de_conflictos)
    """
    
    # Caso Base: ¿Está la asignación completa?
    if len(assignment) == len(variables): # Si todas las variables están asignadas
        return assignment, set()  # ¡Solución encontrada! Sin conflictos.

    # 1. Seleccionar una variable no asignada
    var = select_unassigned_variable(variables, assignment)
    
    # 2. Conjunto de conflictos para esta variable 'var'
    #    (se irá llenando si los valores fallan)
    current_conflict_set = set()
    
    # 3. Iterar sobre todos los valores del dominio
    for value in domains[var]:
        
        # 4. Comprobar si el valor actual tiene un conflicto *inmediato*
        #    con las variables ya asignadas.
        conflicting_var = find_conflict(var, value, assignment, constraints)
        
        if conflicting_var is None:
            # 4a. No hay conflicto inmediato, asignamos y descendemos
            assignment[var] = value
            
            # Llamada recursiva
            result, child_conflict_set = backtrack_cbj(assignment, variables, domains, constraints)
            
            # Si la recursión encontró una solución
            if result is not None:
                return result, set()
            
            # Si la recursión falló (recibimos un conjunto de conflictos)
            
            # --- Aquí está la lógica clave del CBJ ---
            if var not in child_conflict_set:
                # Si 'var' (esta variable) NO es parte del conflicto,
                # significa que el conflicto es "más antiguo".
                # No necesitamos probar otros valores para 'var'.
                # Simplemente deshacemos la asignación y propagamos
                # el conjunto de conflictos hacia arriba.
                del assignment[var]
                return None, child_conflict_set # ¡El "Salto"!
            else:
                # 'var' SÍ es parte del conflicto.
                # Añadimos los conflictos del hijo (menos 'var')
                # a nuestro conjunto de conflictos actual.
                current_conflict_set.update(child_conflict_set - {var})
                # Continuamos el bucle para probar el siguiente 'value'
            
            # Deshacer la asignación para probar el siguiente valor
            del assignment[var]

        else:
            # 4b. Este 'value' tiene un conflicto inmediato.
            #     Añadimos la variable en conflicto a nuestro conjunto.
            current_conflict_set.add(conflicting_var)
            # Y continuamos el bucle para probar el siguiente 'value'

    # 5. Si el bucle termina, probamos todos los valores para 'var'
    #    y todos fallaron.
    #    Devolvemos 'None' y el conjunto de conflictos que acumulamos.
    return None, current_conflict_set

# --- Resolver el problema ---

print("Buscando una solución con Conflict-Directed Backjumping (CBJ)...")
# Empezamos con una asignación vacía
solution, conflicts = backtrack_cbj({}, variables, domains, constraints)

if solution: # Si se encontró una solución
    print("\n¡Solución encontrada!")
    for variable in sorted(solution.keys()): # Imprime la solución de forma ordenada
        print(f"  {variable}: {solution[variable]}")
else: # Si no se encontró solución
    print(f"\nNo se encontró solución. Conflicto final: {conflicts}")