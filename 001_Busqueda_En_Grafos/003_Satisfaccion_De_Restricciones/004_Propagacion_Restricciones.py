# Algoritmo AC-3 (Propagación de Restricciones) para problemas de satisfacción de restricciones (CSP).
# Este algoritmo mejora la consistencia de los dominios de las variables al asegurar que para cada par de variables conectadas por una restricción,
# los valores en sus dominios sean mutuamente consistentes. Esto ayuda a reducir el espacio de búsqueda y puede detectar inconsistencias temprano.
# Entre las aplicaciones comunes de este algoritmo se encuentran la coloración de mapas, la asignación de horarios y la resolución de puzzles como el Sudoku.
# Entre sus ventajas se encuentran la reducción del espacio de búsqueda y la detección temprana de inconsistencias, mientras que sus desventajas incluyen la sobrecarga computacional adicional debido a la gestión de los dominios.    
# Entre sus desventajas se encuentran la sobrecarga computacional adicional debido a la gestión de los dominios.

import copy # Necesitaremos 'deepcopy' para copiar los dominios

#Definicion del problema

variables = ['WA', 'NT', 'SA'] # Definir las regiones

# Dominios iniciales 
initial_domains = {
    'WA': ['rojo', 'verde'],
    'NT': ['rojo', 'verde'],
    'SA': ['rojo'] 
}

# Restricciones (solo entre estas 3 variables)
constraints = {
    'WA': ['NT', 'SA'],
    'NT': ['WA', 'SA'],
    'SA': ['WA', 'NT']
}

# ---  Algoritmo de Propagación de Restricciones (AC-3) ---

def get_all_arcs(constraints): # Genera una lista de todos los arcos (en ambas direcciones)
    """
    Genera una lista de todos los arcos (en ambas direcciones)
    a partir del diccionario de restricciones (vecinos).
    """
    arcs = [] # lista de arcos
    for var in constraints: # para cada variable
        for neighbor in constraints[var]: # para cada vecino
            arcs.append((var, neighbor)) # añadir arco (var -> vecino)
    return arcs # devolver lista de arcos

def revise(domains, var1, var2): # Revisa la consistencia del arco (var1 -> var2)
    """
    Revisa la consistencia del arco (var1 -> var2).
    Elimina valores del dominio de 'var1' si no tienen
    "soporte" (un valor compatible) en el dominio de 'var2'.
    
    Devuelve True si el dominio de var1 fue modificado, False si no.
    """
    revised = False # bandera para saber si se modificó el dominio
    
    # (Usamos list() para iterar sobre una copia mientras modificamos el original)
    for value_x in list(domains[var1]):
        
        has_support = False
        # Buscar al menos un valor en var2 que sea compatible
        for value_y in domains[var2]:
            
            # ¡Aquí está la restricción! Para nuestro problema, es "no ser iguales"
            if value_x != value_y: 
                has_support = True
                break # Encontramos soporte, pasamos al siguiente value_x
        
        # Si, después de revisar todo var2, no encontramos soporte...
        if not has_support:
            # ...eliminamos value_x del dominio de var1
            domains[var1].remove(value_x) # eliminar value_x
            print(f"    REVISE({var1}, {var2}): Eliminado '{value_x}' de {var1}. Nuevo dominio: {domains[var1]}") # mostrar cambio
            revised = True # marcar que se modificó el dominio
            
    return revised # devolver si se modificó el dominio

def ac3(variables, domains, constraints): # Algoritmo AC-3 principal
    """
    Algoritmo AC-3 para hacer que todos los arcos sean consistentes.
    Modifica el diccionario 'domains' directamente.
    
    Devuelve True si el CSP es consistente (ningún dominio vacío).
    Devuelve False si se encuentra una inconsistencia (un dominio vacío).
    """
    
    # 1. Inicializar la cola con todos los arcos del problema
    queue = get_all_arcs(constraints) # obtener todos los arcos
    print(f"Cola inicial: {queue}\n") # mostrar cola inicial
    
    # 2. Procesar la cola
    while queue: # mientras haya arcos en la cola
        (var1, var2) = queue.pop(0) # Tomar el primer arco
        print(f"Procesando arco: ({var1}, {var2})")
        
        # 3. Revisar el arco
        if revise(domains, var1, var2): # Si 'revise' modificó el dominio de var1
            
            # 4. Si el dominio de var1 quedó vacío, no hay solución
            if not domains[var1]: # Si el dominio de var1 está vacío
                print(f"\n¡INCONSISTENCIA! Dominio de {var1} está vacío.")
                return False # Inconsistencia encontrada
            
            # 5. Si 'revise' quitó algo, debemos re-evaluar a
            #    todos los vecinos de var1 (excepto var2)
            print(f"    -> Dominio de {var1} cambió. Añadiendo arcos vecinos a la cola.")
            for neighbor in constraints[var1]:
                if neighbor != var2:
                    # Añadir el arco (vecino -> var1) de vuelta a la cola
                    queue.append((neighbor, var1))
                    
    print("\nAC-3 completado. El CSP es consistente (no se encontraron dominios vacíos).")
    return True # Todos los arcos son consistentes

# --- 3. Ejecutar el algoritmo AC-3 ---

print("--- Ejecutando AC-3 (Propagación de Restricciones) ---")
print(f"Dominios Iniciales:\n  WA: {initial_domains['WA']}\n  NT: {initial_domains['NT']}\n  SA: {initial_domains['SA']}\n")

# Creamos una copia profunda para que AC-3 la modifique
domains_ac3 = copy.deepcopy(initial_domains)

# Ejecutar el algoritmo
consistency = ac3(variables, domains_ac3, constraints)

print("\n--- Resultado ---")
print(f"¿El problema es consistente? -> {consistency}")
print(f"Dominios Finales (después de AC-3):\n  WA: {domains_ac3['WA']}\n  NT: {domains_ac3['NT']}\n  SA: {domains_ac3['SA']}")