# En el siguiente algoritmo se muestra una implementación básica de un solucionador de problemas de satisfacción de restricciones
# (CSP) para el problema de colorear un mapa. El objetivo es asignar colores a las regiones de un mapa de manera que no haya dos regiones adyacentes con el mismo color.
# --- Definición del Problema CSP ---
# 1. Variables: Representan las regiones del mapa.
# 2. Dominios: Representan los colores disponibles para cada región.
# 3. Restricciones: Definen qué regiones son vecinas y, por lo tanto, no pueden compartir el mismo color.
# En el codigo, el problema que se plantea es colorear un mapa de Australia.

# 1. Definir las Variables (las regiones)
variables = ['WA', 'NT', 'SA', 'Q', 'NSW', 'V', 'T']

# 2. Definir los Dominios (los colores disponibles para cada región)
# Usamos un diccionario donde cada variable mapea a su lista de colores.
domains = { # colores disponibles
    'WA': ['rojo', 'verde', 'azul'],
    'NT': ['rojo', 'verde', 'azul'],
    'SA': ['rojo', 'verde', 'azul'],
    'Q':  ['rojo', 'verde', 'azul'],
    'NSW':['rojo', 'verde', 'azul'],
    'V':  ['rojo', 'verde', 'azul'],
    'T':  ['rojo', 'verde', 'azul']
}

# 3. Definir las Restricciones (qué regiones son vecinas)
# Usamos un diccionario donde cada variable mapea a sus vecinos.
# La restricción implícita es: "mi color no puede ser igual al de un vecino".
constraints = {
    'WA': ['NT', 'SA'],
    'NT': ['WA', 'SA', 'Q'],
    'SA': ['WA', 'NT', 'Q', 'NSW', 'V'],
    'Q':  ['NT', 'SA', 'NSW'],
    'NSW':['Q', 'SA', 'V'],
    'V':  ['SA', 'NSW'],
    'T':  [] # Tasmania no tiene vecinos
}

# --- Función Auxiliar para Comprobar Restricciones ---

def is_consistent(variable, assignment, constraints): # variable: la variable que acabamos de asignar
    """
    Comprueba si una asignación parcial es consistente.
    'variable' es la última variable que acabamos de asignar.
    'assignment' es un diccionario {variable: color} con las asignaciones actuales.
    'constraints' es nuestro diccionario de vecinos.
    """
    # Si la variable no está en la asignación, no podemos comprobarla (aunque esto
    # no debería pasar si la usamos correctamente).
    if variable not in assignment:
        return True

    assigned_color = assignment[variable] # El color asignado a la variable
    
    # Iterar sobre todos los vecinos de la variable recién asignada
    for neighbor in constraints[variable]:
        # Si el vecino ya tiene un color asignado...
        if neighbor in assignment:
            # ...y ese color es el mismo que el nuestro...
            if assignment[neighbor] == assigned_color:
                # ...¡entonces hemos violado la restricción!
                return False
                
    # Si hemos comprobado todos los vecinos y no hay conflictos, es consistente.
    return True

# --- Ejemplo de Uso ---

print(f"Problema CSP a resolver:")
print(f"Variables: {variables}")
print(f"Dominios de 'WA': {domains['WA']}")
print(f"Vecinos (restricciones) de 'SA': {constraints['SA']}")

# Ejemplo de una asignación parcial VÁLIDA:
assignment_ok = { # asignación parcial
    'WA': 'rojo',
    'NT': 'verde'
}
# ¿Es consistente asignar 'verde' a 'NT' si 'WA' es 'rojo'?
# is_consistent comprueba 'NT' contra sus vecinos en 'assignment_ok' (solo 'WA')
print(f"\n¿Es consistente {assignment_ok}? -> {is_consistent('NT', assignment_ok, constraints)}")

# Ejemplo de una asignación parcial INVÁLIDA:
assignment_bad = {
    'WA': 'rojo',
    'NT': 'rojo' # 'NT' es vecino de 'WA'
}
# ¿Es consistente asignar 'rojo' a 'NT' si 'WA' es 'rojo'?
print(f"¿Es consistente {assignment_bad}? -> {is_consistent('NT', assignment_bad, constraints)}")