# Este algoritmo es sobre el Proceso de Decisión de Markov (MDP)
# Este algoritmo implementa un Proceso de Decisión de Markov completo
# Donde se consideran múltiples acciones posibles por estado.
# Se utiliza para encontrar la política óptima que maximiza la recompensa esperada.
# Entre sus aplicaciones están: robótica, sistemas de control, juegos y planificación automática.
# Ventajas: encuentra la política óptima considerando todas las acciones posibles.
# Desventajas: complejidad computacional mayor que en el caso de una sola acción por estado.

# Definición del MDP completo:
estados = ['A', 'B', 'C']  # Lista de estados del sistema
acciones = {  # Diccionario de acciones disponibles por estado
    'A': ['toB'],  # Desde A solo se puede ir a B
    'B': ['toA', 'toC'],  # Desde B se puede ir a A o C
    'C': []  # Estado terminal (sin acciones disponibles)
}
recompensa = {  # Función de recompensa 
    'A': 0,  # Recompensa por ir de A a B con acción toB
    'B': 0,  # Recompensa por ir de B a A con acción toA
    'C': 1   # Recompensa por permanecer en C (estado terminal)
       
}
transiciones = {  # Función de transición P(s'|s,a)
    ('A','toB'): {'B': 1.0},  # Desde A con acción toB: 100% a B
    ('B','toA'): {'A': 0.5, 'C': 0.5},  # Desde B con acción toA: 50% a A, 50% a C
    ('B','toC'): {'C': 1.0},  # Desde B con acción toC: 100% a C
    ('C', None): {'C': 1.0}  # Desde C: 100% permanece en C
}

# Parámetros del algoritmo
gamma = 0.9  # Factor de descuento para recompensas futuras
epsilon = 0.001  # Umbral de convergencia
U = {s: 0 for s in estados}  # Inicialización de utilidades

# --- Iteración de Valores para MDP ---
def iteracion_valores_mdp():
    """Calcula las utilidades óptimas usando iteración de valores para MDP."""
    global U
    while True:
        delta = 0  # Controla la convergencia
        U_nuevo = U.copy()  # Copia de las utilidades actuales
        
        # Actualiza la utilidad para cada estado
        for s in estados:
            if not acciones[s]:  # Si es estado terminal, no tiene acciones
                U_nuevo[s] = recompensa[s]
                continue
            
            # Calcula la utilidad para cada accion disponible
            utilidades_acciones = []
            for a in acciones[s]:
                # Calcula el valor esperado para la acción a
                u = sum(transiciones[(s,a)][s2] * 
                       (recompensa.get((s,a,s2),0) + gamma * U[s2]) 
                       for s2 in transiciones[(s,a)])
                utilidades_acciones.append(u)
            
            # Toma el máximo valor entre todas las acciones (mejor acción)
            U_nuevo[s] = max(utilidades_acciones)
            # Actualiza el delta para control de convergencia
            delta = max(delta, abs(U_nuevo[s] - U[s]))
        
        U = U_nuevo  # Actualiza las utilidades
        
        # Verifica criterio de convergencia
        if delta < epsilon:
            break
    
    return U

# --- Política óptima ---
def politica_optima():
    """Determina la política óptima basada en las utilidades calculadas."""
    politica = {}  # Diccionario para almacenar la política óptima
    
    for s in estados:
        if not acciones[s]:  # Si es estado terminal
            politica[s] = None  # No tiene acción asociada
            continue
        
        # Busca la mejor acción para el estado s
        mejor_accion = None
        mejor_valor = -float('inf')  # Inicializa con valor muy bajo
        
        for a in acciones[s]:
            # Calcula el valor de la acción a
            valor = sum(transiciones[(s,a)][s2] * 
                       (recompensa.get((s,a,s2),0) + gamma * U[s2]) 
                       for s2 in transiciones[(s,a)])
            
            # Actualiza si encontramos una acción mejor
            if valor > mejor_valor:
                mejor_valor = valor
                mejor_accion = a
        
        politica[s] = mejor_accion  # Asigna la mejor acción al estado
    
    return politica

# --- Ejecutar el algoritmo ---
U_final = iteracion_valores_mdp()  # Calcula utilidades óptimas
politica = politica_optima()  # Obtiene política óptima

print("=== MDP: Iteración de Valores ===\n")
print("Utilidades finales:")
for s in estados:
    print(f"U({s}) = {U_final[s]:.4f}")  # Imprime utilidades formateadas

print("\nPolítica óptima:")
for s, a in politica.items():
    print(f"π({s}) = {a}")  # Imprime la política óptima