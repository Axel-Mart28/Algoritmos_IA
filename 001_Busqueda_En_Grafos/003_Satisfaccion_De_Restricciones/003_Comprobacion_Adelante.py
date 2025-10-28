# Algoritmo de Comprobación Adelante (Forward Checking) para problemas de satisfacción de restricciones (CSP).
# Este algoritmo mejora la eficiencia de la búsqueda al eliminar valores inconsistentes de los dominios de las variables no asignadas
# Tan pronto como se realiza una asignación. Esto ayuda a detectar conflictos temprano y reduce el espacio de búsqueda.
# Entre las aplicaciones comunes de este algoritmo se encuentran la coloración de mapas, la asignación de horarios y la resolución de puzzles como el Sudoku.
# Entre sus ventajas se encuentran la reducción del espacio de búsqueda y la detección temprana de conflictos, mientras que sus desventajas incluyen la sobrecarga computacional adicional debido a la gestión de los dominios y la posible incompletitud en ciertos casos.


import copy # Necesitaremos 'deepcopy' para copiar los dominios

# --- 1. Definición del CSP (el mismo problema de colorear un mapa de Australia) ---

variables = ['WA', 'NT', 'SA', 'Q', 'NSW', 'V', 'T'] # las regiones

# Dominios iniciales
initial_domains = {
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

#Implementacion de la función de consistencia (misma que en las partes anteriores)

def select_unassigned_variable(variables, assignment): # Selecciona la primera variable no asignada
    """ Selecciona la primera variable no asignada (helper) """
    for v in variables: # Itera sobre todas las variables
        if v not in assignment: # si no está asignada
            return v # la retorna
    return None # Si todas están asignadas

def forward_check(assignment, variables, domains, constraints): # Implementación de funcion de comprobación hacia adelante
    """
    Función recursiva para Búsqueda con Forward Checking.
    'assignment' es la asignación actual.
    'domains' es el diccionario de dominios *actuales* (que se irá podando).
    """
    
    # Caso Base: ¿Está la asignación completa?
    if len(assignment) == len(variables):
        return assignment  # ¡Solución encontrada!

    # 1. Seleccionar una variable no asignada
    var = select_unassigned_variable(variables, assignment)
    
    # 2. Iterar sobre todos los valores del dominio de la variable
    # (Usamos list() para iterar sobre una copia, ya que el original puede cambiar)
    for value in list(domains[var]):
        
        # Asignación tentativa
        assignment[var] = value
        
        # 3. Paso de "Comprobación Hacia Delante" (Poda)
        
        # Guardamos los valores que vamos a podar, para poder restaurarlos
        # si esta rama falla.
        # {vecino: [valores_podados]}
        pruned_values = {} 
        domain_wipeout = False # Flag para "dominio vacío"

        # Revisar todos los vecinos de la variable que acabamos de asignar
        for neighbor in constraints[var]:
            # Si el vecino aún no está asignado...
            if neighbor not in assignment:
                
                # ...y el 'value' que queremos usar está en su dominio...
                if value in domains[neighbor]:
                    
                    # ...¡lo eliminamos de su dominio!
                    domains[neighbor].remove(value)
                    
                    # Guardamos registro de lo que hicimos
                    if neighbor not in pruned_values:
                        pruned_values[neighbor] = []
                    pruned_values[neighbor].append(value)
                    
                    # ¡Comprobación clave! ¿Dejamos un dominio vacío?
                    if not domains[neighbor]:
                        domain_wipeout = True
                        break # Dejar de podar, esta asignación es un fracaso

        # Decidir si continuar
        
        # Si NO hubo un dominio vacío (la poda fue "segura")...
        if not domain_wipeout:
            # ...continuamos con la recursión
            result = forward_check(assignment, variables, domains, constraints)
            
            # Si la recursión encontró una solución, la propagamos
            if result is not None:
                return result
        
        # 5. Backtrack
        # Si domain_wipeout=True O la recursión falló (result=None),
        # debemos deshacer nuestros cambios.
        
        # Restaurar los dominios que podamos
        for neighbor, values in pruned_values.items():
            domains[neighbor].extend(values)
            
        # Quitar la asignación de 'var'
        del assignment[var]

    # Si hemos probado todos los valores para 'var' y ninguno funcionó,
    # esta rama es un callejón sin salida.
    return None

# ---  Resolver el problema ---

print("Buscando una solución con Comprobación Adelante...")
# Pasamos una copia PROFUNDA de los dominios, para que el original
# no se modifique por si queremos reusarlo.
solution = forward_check({}, variables, copy.deepcopy(initial_domains), constraints)

if solution: # Si se encontró una solución
    print("\n¡Solución encontrada!")
    for variable in sorted(solution.keys()): # Imprime la solución de forma ordenada
        print(f"  {variable}: {solution[variable]}")
else: # Si no se encontró solución
    print("\nNo se encontró solución.")