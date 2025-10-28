#Este es el algoritmo de vuelta atras, el cual se usa para resolver problemas de satisfaccion de restricciones (CSP)
# Este algoritmo consiste en asignar valores a las variables de un problema CSP de manera recursiva, retrocediendo cuando se encuentra una inconsistencia.
# Entre las aplicaciones comunes de este algoritmo se encuentran la coloración de mapas, la asignación de horarios y la resolución de puzzles como el Sudoku.
#Entre sus ventajas se encuentran su simplicidad y efectividad para problemas pequeños a medianos, mientras que sus desventajas incluyen su ineficiencia en problemas grandes debido a la explosión combinatoria.
#Entre sus desventajas se encuentran su ineficiencia en problemas grandes debido a la explosión combinatoria.

#Se usa el mismo problema de colorear un mapa de Australia
variables = ['WA', 'NT', 'SA', 'Q', 'NSW', 'V', 'T'] # las regiones

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

def is_consistent(variable, assignment, constraints): # variable: la variable que acabamos de asignar
    """
    Comprueba si una asignación parcial es consistente.
    """
    assigned_color = assignment[variable] # El color asignado a la variable
    for neighbor in constraints[variable]: # Iterar sobre todos los vecinos de la variable recién asignada
        if neighbor in assignment: # Si el vecino ya tiene un color asignado...
            if assignment[neighbor] == assigned_color: # y el color es el mismo que el nuestro...
                return False # entonces hemos violado la restricción
    return True # Si no hay conflictos, es consistente.

# Implementación del algoritmo de vuelta atrás

def backtrack(assignment, variables, domains, constraints): # Asignación actual, variables, dominios y restricciones
    """
    Función recursiva para la Búsqueda de Vuelta Atrás.
    'assignment' es el diccionario de asignaciones actuales {var: valor}.
    """
    
    # Caso Base: ¿Está la asignación completa?
    # Si el número de variables asignadas es igual al total de variables...
    if len(assignment) == len(variables):
        return assignment  # ¡Solución encontrada!

    # 1. Seleccionar una variable no asignada
    # (Por simplicidad, tomamos la primera que encontremos en la lista)
    var = None # variable a asignar
    for v in variables: # buscar variable no asignada
        if v not in assignment: # si no está asignada
            var = v # la seleccionamos
            break # salir del bucle
    
    # 2. Iterar sobre todos los valores del dominio de la variable
    for value in domains[var]:
        
        # 3. Intentar asignar el valor
        assignment[var] = value
        
        # 4. Comprobar la consistencia
        # Usamos la función que ya teníamos
        if is_consistent(var, assignment, constraints):
            
            # Si es consistente, pasar al siguiente nivel (recursión)
            result = backtrack(assignment, variables, domains, constraints)
            
            # Si la recursión encontró una solución, la propagamos hacia arriba
            if result is not None:
                return result
        
        # 5. Si no es consistente O la recursión falló:
        # "Backtrack" -> Quitar la asignación e intentar el siguiente valor
        # Esta línea es el corazón del "vuelta atrás"
        del assignment[var]

    # Si hemos probado todos los valores para 'var' y ninguno funcionó,
    # significa que la rama actual es un callejón sin salida.
    return None

# --- 3. Resolver el problema ---

print("Buscando una solución con Backtracking...")
# Empezamos con una asignación vacía
solution = backtrack({}, variables, domains, constraints) # llamada inicial

if solution: # Si se encontró una solución
    print("\n¡Solución encontrada!")
    # Imprime la solución de forma ordenada
    for variable in sorted(solution.keys()):
        print(f"  {variable}: {solution[variable]}")
else:
    print("\nNo se encontró solución.")