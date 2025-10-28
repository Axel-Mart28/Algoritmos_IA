# --- 34. APRENDIZAJE POR REFUERZO ACTIVO (ITERACIÓN DE VALORES) ---

# Este algoritmo es sobre Aprendizaje por Refuerzo Activo, implementado mediante la Iteración de Valores (Value Iteration).
# "Activo" significa que el agente NO tiene una política. Su objetivo es *encontrar* la política óptima pi*(s).
# El programa calcula el valor "óptimo" (V*) de cada estado. V*(s) es la máxima utilidad que un agente puede
# obtener a largo plazo si empieza en 's' y actúa de forma óptima.
#
# ¿Cómo funciona?
# Utiliza la ecuación de Bellman *óptima* de forma iterativa:
# V_k+1(s) = R(s) + gamma * max_a [ Sumatoria[ P(s' | s, a) * V_k(s') ] ]
# En español: El valor nuevo de 's' es = (Recompensa) + (descuento) * (El valor MÁXIMO que puedo obtener
# probando TODAS las acciones 'a' y eligiendo la mejor).
#
# Componentes de este algoritmo:
# 1. Estados (S), Recompensas (R(s)), Modelo de Transición (T(s, a, s')), Factor de Descuento (gamma).
# 2. (¡Notar que NO hay una Política 'pi' de entrada!)
#
# El algoritmo tiene dos fases:
# 1. Iteración de Valores: Calcular V*(s) para todos los estados usando la fórmula de arriba.
# 2. Extracción de Política: Una vez que tenemos V*(s), calculamos la política óptima pi*(s) revisando,
#    para cada estado, qué acción 'a' nos lleva al estado con mayor valor V*.
#
# Aplicaciones:
# - Encontrar la política óptima (la mejor estrategia) cuando el modelo del mundo es conocido.
# - Robótica (planificación de rutas), estrategias de juegos, optimización de recursos.
#
# Ventajas:
# - Encuentra la política óptima garantizada (dado un modelo correcto).
#
# Desventajas:
# - Requiere conocer el modelo completo del entorno (T y R), lo cual es raro en problemas reales.
# - El paso de 'max_a' es computacionalmente costoso si hay muchas acciones.
# - No funciona si el agente no conoce las probabilidades (para eso se usa Q-Learning, el siguiente tema).
#
# Ejemplo de uso:
# - En un "Mundo de Rejilla", el agente "piensa" en todas las acciones ('N', 'S', 'E', 'O') en cada casilla
#   para descubrir el camino óptimo hacia la meta, evitando el peligro.

import copy # Necesario para copiar los diccionarios de valores
import math 

# --- P1: Definición del Entorno (MDP - Mundo de Rejilla) ---

class GridWorld: # Define el entorno MDP
    def __init__(self): # Constructor
        self.rows = 3 # Filas
        self.cols = 4 # Columnas
        self.states = [] # Lista de estados
        for r in range(self.rows): # Itera filas
            for c in range(self.cols): # Itera columnas
                self.states.append((r, c)) # Añade estado
        self.wall = (1, 1) # Define la pared
        self.terminal_states = [(0, 3), (1, 3)] # Define meta y peligro
        self.states.remove(self.wall) # Quita la pared de estados
        
        self.rewards = {} # Diccionario de recompensas
        for s in self.states: # Itera estados
            if s == (0, 3): self.rewards[s] = 1.0 # Recompensa meta
            elif s == (1, 3): self.rewards[s] = -1.0 # Recompensa peligro
            else: self.rewards[s] = -0.04 # Costo de vida
                
        self.actions = ['N', 'S', 'E', 'O'] # Acciones posibles
        self.gamma = 0.9 # Factor de descuento

    def get_transitions(self, state, action): # Devuelve (prob, s')
        if state in self.terminal_states: # Si es terminal
            return [(0.0, state)] # No hay transiciones
        transitions = [] # Lista de resultados
        if action == 'N' or action == 'S': slip_actions = ['O', 'E'] # Desvíos
        else: slip_actions = ['N', 'S'] # Desvíos
        transitions.append((0.8, self.calculate_next_state(state, action))) # 80% éxito
        transitions.append((0.1, self.calculate_next_state(state, slip_actions[0]))) # 10% desvío 1
        transitions.append((0.1, self.calculate_next_state(state, slip_actions[1]))) # 10% desvío 2
        return transitions # Devuelve la lista

    def calculate_next_state(self, state, action): # Calcula el resultado de la acción
        r, c = state # Estado actual
        if action == 'N': r = max(0, r - 1) # Mover Norte
        elif action == 'S': r = min(self.rows - 1, r + 1) # Mover Sur
        elif action == 'E': c = min(self.cols - 1, c + 1) # Mover Este
        elif action == 'O': c = max(0, c - 1) # Mover Oeste
        next_state = (r, c) # Siguiente estado
        if next_state == self.wall: # Si es pared
            return state # Se queda quieto
        return next_state # Devuelve el estado calculado

# --- P2: Algoritmo de Aprendizaje Activo (Value Iteration) ---

def value_iteration(grid, k=100): # Función principal del algoritmo
    """
    Implementa el Aprendizaje Activo (Iteración de Valores)
    para encontrar la utilidad óptima V*(s).
    """
    
    # 1. Inicializar V(s) = 0 para todos los estados
    V = {s: 0.0 for s in grid.states} # Diccionario de Utilidad/Valor, todo en 0.0
    
    # 2. Iterar k veces (o hasta la convergencia)
    for i in range(k): # Bucle principal de iteración
        V_new = copy.deepcopy(V) # Copia los valores V_k
        
        for s in grid.states: # Para cada estado en el mundo
            if s in grid.terminal_states: # Si es un estado terminal
                V_new[s] = grid.rewards[s] # Su valor es su recompensa
                continue # Pasar al siguiente estado

            # 3. Calcular el valor para CADA acción posible
            action_values = {} # Diccionario para guardar {acción: utilidad_esperada}
            
            for action in grid.actions: # Probar 'N', 'S', 'E', 'O'
                
                # Calcular la suma esperada: sum[ P(s'|s,a) * V_k(s') ]
                expected_utility = 0.0 # Acumulador para esta acción
                for prob, next_state in grid.get_transitions(s, action): # Para (80%, 10%, 10%)
                    expected_utility += prob * V[next_state] # Suma (Prob * Valor_antiguo_siguiente)
                
                action_values[action] = expected_utility # Guarda el valor de esta acción
            
            # 4. Aplicar la Ecuación de Bellman Óptima (el paso 'max')
            # V_k+1(s) = R(s) + gamma * max_a [ sum(...) ]
            max_future_utility = max(action_values.values()) # Encuentra el valor más alto entre 'N','S','E','O'
            V_new[s] = grid.rewards[s] + grid.gamma * max_future_utility # Actualiza V_new
            
        V = V_new # V_k+1 se convierte en V_k para el siguiente bucle
        
    return V # Devuelve los valores óptimos V*

# --- P3: Extracción de la Política Óptima ---

def extract_policy(grid, V): # Función para descubrir la política pi*
    """
    Extrae la política óptima pi*(s) a partir de los valores V*(s).
    """
    policy = {} # Diccionario para guardar la política óptima
    for s in grid.states: # Itera sobre todos los estados
        if s in grid.terminal_states: # No hay acciones en estados terminales
            continue # Saltar
            
        # Calcular el valor para cada acción (similar a value_iteration)
        action_values = {} # {acción: utilidad_esperada}
        for action in grid.actions: # Probar 'N', 'S', 'E', 'O'
            expected_utility = 0.0 # Acumulador
            for prob, next_state in grid.get_transitions(s, action): # Para (80%, 10%, 10%)
                expected_utility += prob * V[next_state] # Usamos los V* finales
            action_values[action] = expected_utility # Guarda el valor de la acción
            
        # La mejor acción es la que maximiza la utilidad futura esperada
        best_action = max(action_values, key=action_values.get) # Encuentra la *llave* (acción) con el valor máximo
        policy[s] = best_action # Asigna esa mejor acción a la política
        
    return policy # Devuelve el mapa de política óptima

# --- Funciones auxiliares de impresión ---

def print_values_optimal(V, grid): # Imprime los valores V*
    print("Valores/Utilidad Óptimos V*(s):") # Título
    for r in range(grid.rows): # Itera filas
        print("+---------+" * grid.cols) # Borde
        row_str = "" # String de fila
        for c in range(grid.cols): # Itera columnas
            state = (r, c) # Estado
            if state == grid.wall: row_str += "|  PARED  " # Pared
            elif state in V: row_str += f"| {V[state]:.4f} " # Valor
            else: row_str += "|         " # Vacío
        print(row_str + "|") # Imprime fila
    print("+---------+" * grid.cols) # Borde

def print_policy(policy, grid): # Imprime la política pi* con flechas
    print("\nPolítica Óptima pi*(s):") # Título
    action_arrows = {'N': '↑', 'S': '↓', 'E': '→', 'O': '←'} # Mapa de flechas
    for r in range(grid.rows): # Itera filas
        print("+---------+" * grid.cols) # Borde
        row_str = "" # String de fila
        for c in range(grid.cols): # Itera columnas
            state = (r, c) # Estado
            if state == grid.wall: row_str += "|  PARED  " # Pared
            elif state in grid.terminal_states: row_str += "|  FIN    " # Terminal
            elif state in policy: row_str += f"|    {action_arrows[policy[state]]}    " # Flecha
            else: row_str += "|         " # Vacío
        print(row_str + "|") # Imprime fila
    print("+---------+" * grid.cols) # Borde

# --- P4: Ejecutar el algoritmo ---

grid_active = GridWorld() # 1. Crear el entorno

print("--- 34. Aprendizaje Activo (Iteración de Valores) ---") # Título
print("Calculando V*(s) (Valores Óptimos)...") # Mensaje
V_optimal = value_iteration(grid_active, k=100) # 2. Ejecutar la Iteración de Valores
print_values_optimal(V_optimal, grid_active) # 3. Imprimir los V*

print("\nExtrayendo la política óptima pi*(s)...") # Mensaje
policy_optimal = extract_policy(grid_active, V_optimal) # 4. Ejecutar la Extracción de Política
print_policy(policy_optimal, grid_active) # 5. Imprimir la política óptima (las flechas)