# Algoritmo de Iteración de Políticas
# Este algoritmo implementa el método de Iteración de Políticas para resolver MDPs.
# Combina evaluación y mejora de políticas de forma iterativa hasta converger a la política óptima.
# Entre sus aplicaciones están: sistemas de control, robótica y toma de decisiones secuenciales.
# Ventajas: más eficiente que iteración de valores en muchos casos, garantiza convergencia.
# Desventajas: requiere múltiples evaluaciones de política que pueden ser costosas.

# Definición del MDP:
estados = ['A', 'B', 'C']  # Conjunto de estados del sistema
acciones = {  # Acciones disponibles por estado
    'A': ['toB'],  # Desde A solo se puede ir a B
    'B': ['toA', 'toC'],  # Desde B se puede ir a A o C
    'C': []  # Estado terminal (sin acciones disponibles)
}
recompensa = {'A': 0, 'B': 0, 'C': 1}  # Recompensas por estado (simplificado)
transiciones = {  # Función de transición P(s'|s,a)
    ('A','toB'): {'B': 1.0},  # Desde A con acción toB: 100% a B
    ('B','toA'): {'A': 0.5, 'C': 0.5},  # Desde B con acción toA: 50% a A, 50% a C
    ('B','toC'): {'C': 1.0},  # Desde B con acción toC: 100% a C
    ('C', None): {'C': 1.0}  # Desde C: 100% permanece en C
}

# Parámetros del algoritmo
gamma = 0.9  # Factor de descuento para recompensas futuras
epsilon = 0.001  # Umbral de convergencia para evaluación de política

# Inicializamos una política arbitraria
politica = {'A':'toB','B':'toA','C':None}  # Política inicial (puede ser cualquiera)
U = {s: 0 for s in estados}  # Inicialización de utilidades en 0

# --- Evaluación de la política ---
def evaluar_politica():
    """Evalúa la política actual calculando las utilidades de cada estado."""
    global U
    while True:
        delta = 0  # Controla la convergencia en la evaluación
        U_nuevo = U.copy()  # Copia de las utilidades actuales
        
        # Actualiza la utilidad para cada estado según la política actual
        for s in estados:
            a = politica[s]  # Acción dictada por la política actual
            if a is None:  # Si es estado terminal
                U_nuevo[s] = recompensa[s]  # Utilidad = recompensa inmediata
                continue
            
            # Calcula la utilidad esperada siguiendo la política actual
            U_nuevo[s] = sum(transiciones[(s,a)][s2] * 
                            (recompensa[s2] + gamma * U[s2]) 
                            for s2 in transiciones[(s,a)])
            
            # Actualiza el delta para control de convergencia
            delta = max(delta, abs(U_nuevo[s] - U[s]))
        
        U.update(U_nuevo)  # Actualiza las utilidades con los nuevos valores
        
        # Verifica criterio de convergencia para la evaluación
        if delta < epsilon:
            break

# --- Mejora de la política ---
def mejorar_politica():
    """Mejora la política actual usando las utilidades calculadas."""
    cambio = False  # Indica si hubo cambios en la política
    
    for s in estados:
        if not acciones[s]:  # Si es estado terminal, salta
            continue
        
        # Busca la mejor acción para el estado s
        mejor_accion = None
        mejor_valor = -float('inf')  # Inicializa con valor muy bajo
        
        for a in acciones[s]:
            # Calcula el valor de la acción a
            valor = sum(transiciones[(s,a)][s2] * 
                       (recompensa[s2] + gamma * U[s2]) 
                       for s2 in transiciones[(s,a)])
            
            # Actualiza si encontramos una acción mejor
            if valor > mejor_valor:
                mejor_valor = valor
                mejor_accion = a
        
        # Si la mejor acción es diferente a la actual, actualiza la política
        if politica[s] != mejor_accion:
            politica[s] = mejor_accion
            cambio = True  # Marca que hubo cambio
    
    return cambio  # Retorna si la política cambió

# --- Iteración de políticas ---
def iteracion_politicas():
    """Algoritmo principal de iteración de políticas."""
    while True:
        evaluar_politica()  # Paso 1: Evalúa la política actual
        if not mejorar_politica():  # Paso 2: Mejora la política
            break  # Si no hay cambios, hemos convergido
    return politica, U

# --- Ejecutar el algoritmo ---
politica_final, U_final = iteracion_politicas()  # Ejecuta iteración de políticas

print("=== ITERACIÓN DE POLÍTICAS ===\n")
print("Utilidades finales:")
for s in estados:
    print(f"U({s}) = {U_final[s]:.4f}")  # Imprime utilidades formateadas

print("\nPolítica óptima:")
for s, a in politica_final.items():
    print(f"π({s}) = {a}")  # Imprime la política óptima encontrada