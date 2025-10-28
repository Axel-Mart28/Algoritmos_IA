# Este es el algoritmo de Acondicionamiento del Corte para resolver problemas de satisfacción de restricciones (CSP).
# Este algoritmo divide el problema en un "cutset" (conjunto de corte) y un "árbol" (subproblema sin ciclos).
# Primero, se asignan valores a las variables del cutset, y luego se resuelve el subproblema del árbol condicionado a esas asignaciones.
# Entre las aplicaciones comunes de este algoritmo se encuentran la coloración de mapas, la asignación de horarios y la resolución de puzzles como el Sudoku.
# Entre sus ventajas se encuentran la reducción del espacio de búsqueda y la capacidad para manejar problemas con ciclos, mientras que sus desventajas incluyen la complejidad adicional en la identificación del cutset y la posible explosión combinatoria en el cutset.

import copy # Necesitaremos 'deepcopy' para copiar los dominios

#Definición del CSP (igual que antes)

variables = ['WA', 'NT', 'SA', 'Q', 'NSW', 'V', 'T'] # las regiones

domains = { # colores disponibles
    'WA': ['rojo', 'verde', 'azul'], # dominio de WA
    'NT': ['rojo', 'verde', 'azul'], # dominio de NT
    'SA': ['rojo', 'verde', 'azul'], # dominio de SA
    'Q':  ['rojo', 'verde', 'azul'], # dominio de Q
    'NSW':['rojo', 'verde', 'azul'], # dominio de NSW
    'V':  ['rojo', 'verde', 'azul'], # dominio de V
    'T':  ['rojo', 'verde', 'azul']  # dominio de T
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

#Implementacion de acondicionamiento del corte
# Lo necesitaremos para resolver el subproblema del "árbol"

def is_consistent_backtrack(variable, assignment, constraints): # Comprueba si la asignación parcial es consistente
    assigned_color = assignment[variable] # El color asignado a la variable
    for neighbor in constraints[variable]: # Iterar sobre todos los vecinos de la variable recién asignada
        if neighbor in assignment: # Si el vecino ya tiene un color asignado
            if assignment[neighbor] == assigned_color: # y el color es el mismo que el nuestro
                return False # entonces hemos violado la restricción
    return True # Si no hay conflictos, es consistente.

def backtrack(assignment, variables, domains, constraints): # Implementación del algoritmo de vuelta atrás
    if len(assignment) == len(variables): # Caso Base: ¿Está la asignación completa?
        return assignment # ¡Solución encontrada!

    var = None # 1. Seleccionar una variable no asignada
    for v in variables: # buscar variable no asignada
        if v not in assignment: # si no está asignada
            var = v # la seleccionamos
            break # salir del bucle
    
    for value in domains[var]: # 2. Iterar sobre todos los valores del dominio de la variable
        assignment[var] = value # Asignación tentativa
        if is_consistent_backtrack(var, assignment, constraints): # 3. Comprobar consistencia
            result = backtrack(assignment, variables, domains, constraints) # 4. Recursión
            if result is not None: # 5. Si se encontró solución, devolverla
                return result # ¡Solución encontrada!
        del assignment[var] # 6. Si no es consistente o no hay solución, desasignar y probar otro valor
    return None # 7. Si ningún valor funciona, devolver None (fracaso)


def cutset_conditioning(variables, domains, constraints): # 1. Identificar el Cutset (S)
    """
    Resuelve el CSP usando Acondicionamiento del Corte.
    """
    
    # 1. Identificar el Cutset (S) - Lo definimos manualmente
    cutset_vars = ['SA']
    
    # 2. Identificar las variables restantes (T) - el "árbol"
    tree_vars = [v for v in variables if v not in cutset_vars]
    
    print(f"Cutset (S): {cutset_vars}")
    print(f"Árbol (T): {tree_vars}")

    # 3. Iterar (Acondicionar) sobre todas las asignaciones posibles para S
    #    (En este caso, solo iterar sobre los valores de 'SA')
    
    for value in domains['SA']: # para cada valor posible de SA
        
        print(f"\n--- Intentando acondicionamiento: SA = {value} ---")
        
        # 4. Crear el subproblema para el árbol T
        
        # 4a. Asignación inicial para el subproblema (vacía)
        tree_assignment = {}
        
        # 4b. Dominios para el subproblema (copia profunda)
        tree_domains = copy.deepcopy(domains)
        
        # 4c. Restricciones para el subproblema (podemos reusar las originales)
        
        # 4d. Propagar las restricciones de S a T
        #    Eliminamos 'value' de los dominios de los vecinos de SA
        
        # Asignamos SA temporalmente para la propagación
        cutset_assignment = {'SA': value}
        
        for tree_var in tree_vars:
            # Si una variable del árbol es vecina de 'SA'...
            if 'SA' in constraints[tree_var]:
                # ...y el valor de 'SA' está en su dominio...
                if value in tree_domains[tree_var]:
                    # ...¡lo eliminamos!
                    tree_domains[tree_var].remove(value)
                    
        print(f"  Dominio podado de 'WA' (vecino de SA): {tree_domains['WA']}")
        print(f"  Dominio podado de 'NT' (vecino de SA): {tree_domains['NT']}")

        # 5. Resolver el subproblema (el árbol T)
        #    Usamos backtracking estándar, que será rápido.
        #    Le pasamos la lista de variables del árbol, los dominios podados
        #    y la asignación inicial vacía.
        
        solution_T = backtrack(tree_assignment, tree_vars, tree_domains, constraints) # Resolver T
        
        if solution_T is not None:
            print("  ¡Éxito! El subproblema del árbol tiene solución.")
            # 6. Combinar la solución del árbol (T) con la del cutset (S)
            solution_T.update(cutset_assignment)
            return solution_T
        else:
            print("  Fallo. El subproblema del árbol no tiene solución.")

    # 7. Si el bucle termina, no se encontró solución
    return None

# --- 4. Resolver el problema ---

print("Buscando una solución con Acondicionamiento del Corte...")
solution = cutset_conditioning(variables, domains, constraints)

if solution: # Si se encontró una solución
    print("\n¡Solución final encontrada!")
    for variable in sorted(solution.keys()): # Imprime la solución de forma ordenada
        print(f"  {variable}: {solution[variable]}")
else:
    print("\nNo se encontró solución.") # Si no se encontró solución.