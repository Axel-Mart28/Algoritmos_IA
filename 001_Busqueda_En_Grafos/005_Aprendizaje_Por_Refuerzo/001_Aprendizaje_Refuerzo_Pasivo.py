# Algoritmo de APRENDIZAJE POR REFUERZO PASIVO (EVALUACIÓN DE POLÍTICA)

# Este algoritmo es sobre Aprendizaje por Refuerzo Pasivo, específicamente la Evaluación de Política.
# "Pasivo" significa que el agente NO toma decisiones; se le da una política (un "manual") y la sigue ciegamente.
# El objetivo del programa es calcular el "valor" o "utilidad" (V) de cada estado (casilla) del mundo.
# La utilidad V(s) es la recompensa esperada a largo plazo por empezar en el estado 's' y seguir la política fija 'pi'.
# 
# ¿Cómo funciona?
# Utiliza la ecuación de Bellman para una política fija de forma iterativa:
# V_k+1(s) = R(s) + gamma * Sumatoria[ P(s' | s, pi(s)) * V_k(s') ]
# En español: El valor nuevo de 's' es = (Recompensa inmediata en 's') + (descuento) * (la suma ponderada del valor antiguo de los siguientes estados).
# 
# Componentes de este algoritmo:
# 1. Estados (S): Un conjunto de estados (las casillas de la rejilla).
# 2. Recompensas (R(s)): Un valor numérico por estar en un estado (ej. +1, -1, -0.04).
# 3. Modelo de Transición (T(s, a, s')): La probabilidad P(s' | s, a) de ir a 's'' desde 's' tomando la acción 'a'.
# 4. Factor de Descuento (gamma): Un número (0 a 1) que mide la importancia de las recompensas futuras.
# 5. Política Fija (pi(s)): Un mapa que dice qué acción 'a' tomar en cada estado 's'.
#
# Aplicaciones:
# - Evaluar qué tan buena es una estrategia o política existente.
# - Determinar el riesgo o valor de una posición en un juego, dada una forma de jugar.
#
# Ventajas:
# - Es un algoritmo simple y converge de forma garantizada al valor verdadero de la política.
# - Es la base para entender algoritmos más complejos.
#
# Desventajas:
# - NO aprende una política *mejor*. Solo evalúa la política que se le da.
# - Requiere conocer el modelo completo del entorno (las probabilidades de transición T y las recompensas R).
#
# Ejemplo de uso:
# - Calcular el valor de cada casilla en un "Mundo de Rejilla" si el agente siempre intenta ir.

import copy # Necesario para copiar los diccionarios de valores en cada iteración

# --- P1: Definición del Entorno (MDP - Mundo de Rejilla) ---

class GridWorld: # Define el entorno, que es un Proceso de Decisión de Markov (MDP)
    """
    Define el Mundo de Rejilla (MDP)
    Las coordenadas son (fila, columna)
    """
    def __init__(self): # Constructor de la clase
        # Dimensiones de la rejilla
        self.rows = 3 # Número de filas
        self.cols = 4 # Número de columnas
        
        # Definir los estados (todas las casillas)
        self.states = [] # Inicializa una lista para los estados
        for r in range(self.rows): # Itera sobre las filas
            for c in range(self.cols): # Itera sobre las columnas
                self.states.append((r, c)) # Añade la coordenada (fila, col) como un estado
                
        # Definir la casilla de pared (inaccesible)
        self.wall = (1, 1) # Estado (1, 1) es una pared
        
        # Definir los estados terminales (donde termina el juego)
        self.terminal_states = [(0, 3), (1, 3)] # Meta (+1) y Peligro (-1)
        
        # Quitar la pared de los estados válidos
        self.states.remove(self.wall) # Elimina la pared de la lista de estados
        
        # 1. Función de Recompensa R(s)
        self.rewards = {} # Inicializa un diccionario para las recompensas
        for s in self.states: # Itera sobre todos los estados válidos
            if s == (0, 3): # Si el estado es la meta
                self.rewards[s] = 1.0  # Recompensa positiva
            elif s == (1, 3): # Si el estado es el peligro
                self.rewards[s] = -1.0 # Recompensa negativa
            else: # Para cualquier otro estado
                self.rewards[s] = -0.04 # Recompensa negativa pequeña (costo de vida)
                
        # 2. Acciones A (las acciones que el agente puede *intentar*)
        self.actions = ['N', 'S', 'E', 'O'] # Norte, Sur, Este, Oeste
        
        # 3. Modelo de Transición T(s, a, s')
        #    Se define a través de la función get_transitions()
        #    Es estocástico (incierto): 80% éxito, 10% desvío izq, 10% desvío der.
        
        # 4. Factor de Descuento (gamma)
        self.gamma = 0.9 # Mide la importancia de recompensas futuras (cercano a 1 es paciente)

    def get_transitions(self, state, action): # Calcula los posibles siguientes estados
        """
        Devuelve una lista de tuplas: (probabilidad, siguiente_estado)
        """
        if state in self.terminal_states: # Si el estado es terminal
            return [(0.0, state)] # No hay transiciones, se queda ahí sin prob

        transitions = [] # Lista para guardar los resultados de la acción
        
        # Define las acciones "ortogonales" (desvíos)
        if action == 'N' or action == 'S': # Si la acción es vertical
            slip_actions = ['O', 'E'] # Los desvíos son horizontales (Oeste, Este)
        else: # Si la acción es horizontal (E u O)
            slip_actions = ['N', 'S'] # Los desvíos son verticales (Norte, Sur)

        # 1. Acción principal (80% de probabilidad de éxito)
        transitions.append((0.8, self.calculate_next_state(state, action))) # Añade el resultado deseado
        
        # 2. Desvío 1 (10% de probabilidad)
        transitions.append((0.1, self.calculate_next_state(state, slip_actions[0]))) # Añade el desvío 1
        
        # 3. Desvío 2 (10% de probabilidad)
        transitions.append((0.1, self.calculate_next_state(state, slip_actions[1]))) # Añade el desvío 2
        
        return transitions # Devuelve la lista de (prob, estado_siguiente)

    def calculate_next_state(self, state, action): # Calcula dónde termina una acción *determinista*
        """
        Calcula el estado resultante de una acción *determinista*.
        Si choca con pared o borde, se queda en el mismo 'state'.
        """
        r, c = state # Desempaqueta la fila y columna del estado actual
        
        if action == 'N': # Si la acción es Norte
            r = max(0, r - 1) # Moverse arriba (fila - 1), sin salirse del borde (min 0)
        elif action == 'S': # Si la acción es Sur
            r = min(self.rows - 1, r + 1) # Moverse abajo (fila + 1), sin salirse (max rows-1)
        elif action == 'E': # Si la acción es Este
            c = min(self.cols - 1, c + 1) # Moverse derecha (col + 1), sin salirse
        elif action == 'O': # Si la acción es Oeste
            c = max(0, c - 1) # Moverse izquierda (col - 1), sin salirse
            
        next_state = (r, c) # El nuevo estado calculado
        
        # Si el siguiente estado es la pared
        if next_state == self.wall: 
            return state # El agente se queda quieto (regresa al estado original)
        
        return next_state # Devuelve el estado final válido

# --- P2: Definición de la Política Pasiva (PI) ---

# Una política pi(s) -> acción
# Es un diccionario que mapea estados a una acción fija.
# Esta es la política "ingenua" que el agente debe seguir.
policy = { # Diccionario que define la política
    (0, 0): 'E', (0, 1): 'E', (0, 2): 'E', # Fila 0: Ir al Este (hacia la meta)
    (1, 0): 'N',            # (1, 1) es pared
                         (1, 2): 'N', # Fila 1: Ir al Norte (para alejarse del peligro)
    (2, 0): 'N', (2, 1): 'E', (2, 2): 'E', (2, 3): 'O', # Fila 2: Moverse
}

# --- P3: Algoritmo de Evaluación de Política (Policy Evaluation) ---

def policy_evaluation(grid, policy, k=100): # Función principal del algoritmo
    """
    Implementa el Aprendizaje Pasivo (Evaluación de Política)
    usando 'k' iteraciones de la actualización de Bellman.
    """
    
    # 1. Inicializar V(s) = 0 para todos los estados
    V = {s: 0.0 for s in grid.states} # Diccionario de Utilidad/Valor, todo en 0.0
    
    # 2. Iterar k veces (k es el número de "pasos de pensamiento")
    for i in range(k): # Bucle principal de iteración
        
        # Creamos una copia para V_k+1 (V_new)
        # Es crucial actualizar sobre los valores del paso anterior (V_k)
        V_new = copy.deepcopy(V) # Copia el diccionario de valores del paso k
        
        # Iterar sobre cada estado s (excepto terminales)
        for s in grid.states: # Para cada estado en el mundo
            
            if s in grid.terminal_states: # Si es un estado terminal (meta o peligro)
                V_new[s] = grid.rewards[s] # Su valor es simplemente su recompensa
                continue # Pasar al siguiente estado del bucle

            # Obtener la acción fija de nuestra política
            action = policy[s] # Busca la acción que la política ordena para este estado 's'
            
            # Calcular la suma esperada: sum[ P(s'|s,a) * V_k(s') ]
            expected_utility = 0.0 # Inicializa el acumulador de utilidad esperada
            
            # Obtiene las transiciones (80%, 10%, 10%)
            for prob, next_state in grid.get_transitions(s, action): # Para cada posible resultado de la acción
                expected_utility += prob * V[next_state] # Suma (Probabilidad * Valor_antiguo_del_siguiente_estado)
            
            # 3. Aplicar la Ecuación de Bellman (la fórmula clave)
            # V_k+1(s) = R(s) + gamma * sum[ P(s'|s,a) * V_k(s') ]
            V_new[s] = grid.rewards[s] + grid.gamma * expected_utility # Actualiza el valor de V_new
            
        # Actualizar V_k con los nuevos valores V_k+1 para la siguiente iteración
        V = V_new # El nuevo conjunto de valores se convierte en el "antiguo" para el paso i+1
        
    return V # Devuelve el diccionario de valores final después de k iteraciones

def print_values(V, grid): # Función auxiliar para imprimir la rejilla de valores
    """ Función auxiliar para imprimir la rejilla de valores. """
    print("Valores V(s) calculados para la política dada:") # Título
    for r in range(grid.rows): # Itera filas
        print("+---------" * grid.cols + "+") # Imprime borde superior
        row_str = "" # String para la fila actual
        for c in range(grid.cols): # Itera columnas
            state = (r, c) # Estado actual
            if state == grid.wall: # Si es la pared
                row_str += "|  PARED  " # Imprime PARED
            elif state in V: # Si es un estado válido
                row_str += f"| {V[state]:.4f} " # Imprime su valor formateado
            else: # (Esto no debería pasar si se quita la pared)
                 row_str += "|         " # Espacio vacío
        print(row_str + "|") # Imprime la fila completa
    print("+---------" * grid.cols + "+") # Imprime borde inferior

# --- P4: Ejecutar el algoritmo ---

print("--- 33. Aprendizaje Pasivo (Evaluación de Política) ---") # Título
grid = GridWorld() # 1. Crear el entorno
print("Mundo de Rejilla y Política Fija cargados.") # Mensaje

print("\nCalculando V(s) para la política dada...") # Mensaje
V_final = policy_evaluation(grid, policy, k=100) # 2. Ejecutar el algoritmo

print_values(V_final, grid) # 3. Imprimir los resultados