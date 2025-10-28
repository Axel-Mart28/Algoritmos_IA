# Algoritmo de APRENDIZAJE POR REFUERZO ACTIVO (Q-LEARNING)

# Este algoritmo implementa Q-Learning, un pilar del Aprendizaje por Refuerzo.
# Es un algoritmo "Activo" (aprende la política óptima) y "Libre de Modelo" (Model-Free).
# "Libre de Modelo" es la clave: el agente NO conoce las probabilidades de transición T(s, a, s') ni la función de recompensa R(s).
# El agente aprende explorando el mundo (como un bebé) y observando los resultados (s, a, r, s').
#
# ¿Cómo funciona?
# En lugar de aprender el valor de un estado V(s), aprende el valor de un par (estado, acción), llamado Q(s, a).
# Q(s, a) = "La recompensa futura total esperada si tomo la acción 'a' desde el estado 's' y luego actúo de forma óptima".
#
# La fórmula de actualización (Temporal Difference - TD) es el corazón del algoritmo:
# Cuando el agente hace (s, a) -> (r, s'):
# 1. Observa la recompensa 'r' y el nuevo estado 's''.
# 2. Mira el valor máximo que puede obtener desde 's'': max_a'[Q(s', a')]
# 3. Calcula el "valor de muestra" (TD Target): Muestra = r + gamma * max_a'[Q(s', a')]
# 4. Calcula el "error" (TD Error): Error = Muestra - Q(s, a) (Lo que *aprendió* vs. lo que *sabía*)
# 5. Actualiza su conocimiento: Q(s, a) <-- Q(s, a) + alpha * Error
#
# Componentes de este algoritmo:
# 1. Q-Table: Un diccionario o tabla que almacena el valor Q para cada par (s, a).
# 2. alpha (Tasa de Aprendizaje): Un número (0 a 1) que controla qué tan rápido aprende (cuánto confía en la nueva información).
# 3. gamma (Factor de Descuento): Igual que antes, valora recompensas futuras.
# 4. epsilon (Estrategia de Exploración): Un número (0 a 1) para balancear "probar cosas nuevas" (exploración) vs. "hacer lo que sé que funciona" (explotación).
#
# Aplicaciones:
# - Robótica (aprender a caminar o agarrar objetos sin un simulador de física perfecto).
# - Juegos (IA que aprenden a jugar Atari o Mario solo viendo la pantalla).
# - Optimización de rutas, sistemas de recomendación, finanzas algorítmicas.
#
# Ventajas:
# - Es "Model-Free": No necesita conocer las reglas del entorno. Es el escenario más realista.
# - Es "Off-Policy" (Fuera de Política): Puede aprender la política *óptima* (explotar) mientras sigue una política de *exploración* (ej. aleatoria). Esto es muy potente.
#
# Desventajas:
# - Requiere *mucha* exploración (muchos episodios) para converger. Puede ser muy lento.
# - El Q-Table puede volverse gigantesco si hay muchos estados o acciones (se resuelve con Deep Q-Learning).
# - Requiere ajustar "hiperparámetros" (alpha, gamma, epsilon) para que funcione bien.
#
# Ejemplo de uso:
# - Entrenar a un agente en nuestro "Mundo de Rejilla" (GridWorld) sin decirle las probabilidades (80/10/10)
#   ni dónde están las recompensas. El agente debe descubrirlo todo.

import random # Necesario para la exploración aleatoria (epsilon-greedy)
import copy   

# --- P1: Definición del Entorno (MDP - Mundo de Rejilla) ---
# Esta clase SIMULA el mundo. Es la "caja negra" con la que el agente interactúa.
# El agente NO puede ver el código de esta clase. Solo puede llamar a env.step().
class GridWorldEnv: # El Entorno (simulador)
    def __init__(self): # Constructor del entorno
        self.rows = 3 # Filas
        self.cols = 4 # Columnas
        self.wall = (1, 1) # Pared
        self.terminal_states = [(0, 3), (1, 3)] # Meta y Peligro
        
        # El agente NO conoce estas recompensas. El entorno las usa para
        # devolver un número cuando el agente "aterriza" en una casilla.
        self._rewards = {} # Recompensas (privadas, el agente no las ve)
        for r in range(self.rows): # Itera filas
            for c in range(self.cols): # Itera columnas
                s = (r, c) # Estado
                if s == (0, 3): self._rewards[s] = 1.0 # Meta
                elif s == (1, 3): self._rewards[s] = -1.0 # Peligro
                elif s == self.wall: self._rewards[s] = 0.0 # Pared (irrelevante)
                else: self._rewards[s] = -0.04 # Costo de vida
        
        self.all_actions = ['N', 'S', 'E', 'O'] # Acciones posibles

    def get_start_state(self): # Devuelve un estado inicial aleatorio (que no sea pared/terminal)
        while True: # Bucle
            r = random.randint(0, self.rows - 1) # Fila aleatoria
            c = random.randint(0, self.cols - 1) # Columna aleatoria
            s = (r, c) # Estado
            if s != self.wall and s not in self.terminal_states: # Si es válido
                return s # Devuelve este estado inicial

    def step(self, state, action): # Esta es la ÚNICA función que el agente puede llamar
        """
        El agente provee un 'state' y una 'action'.
        El entorno simula el resultado estocástico (80/10/10)
        y devuelve: (next_state, reward)
        """
        # --- Simulación interna del Modelo de Transición (el agente no ve esto) ---
        
        # Primero, calcula el siguiente estado estocásticamente
        if state in self.terminal_states: # Si ya está en un terminal
            return state, 0.0 # Se queda ahí, sin recompensa adicional
            
        # Define las acciones "ortogonales" (desvíos)
        if action == 'N' or action == 'S': slip_actions = ['O', 'E'] # Desvíos
        else: slip_actions = ['N', 'S'] # Desvíos
        
        # Elige el resultado de la acción basado en probabilidades
        rand_num = random.random() # Número aleatorio [0.0, 1.0)
        if rand_num < 0.8: # 80% de probabilidad
            chosen_action = action # La acción tiene éxito
        elif rand_num < 0.9: # 10% de probabilidad
            chosen_action = slip_actions[0] # Desvío 1
        else: # 10% de probabilidad
            chosen_action = slip_actions[1] # Desvío 2

        # Ahora calcula el estado final determinista basado en la acción elegida
        r, c = state # Estado actual
        if chosen_action == 'N': r = max(0, r - 1) # Mover Norte
        elif chosen_action == 'S': r = min(self.rows - 1, r + 1) # Mover Sur
        elif chosen_action == 'E': c = min(self.cols - 1, c + 1) # Mover Este
        elif chosen_action == 'O': c = max(0, c - 1) # Mover Oeste
            
        next_state = (r, c) # El siguiente estado
        
        if next_state == self.wall: # Si choca con la pared
            next_state = state # Se queda en el estado original
        
        # --- Fin de la simulación interna ---
        
        # El entorno mira la recompensa en el *nuevo estado* y la devuelve
        reward = self._rewards[next_state] # Obtiene la recompensa de aterrizar ahí
        
        return next_state, reward # Devuelve el resultado al agente

# --- P2: Algoritmo del Agente (Q-Learning) ---

class QLearningAgent: # El agente (el "cerebro" que aprende)
    def __init__(self, actions, alpha=0.1, gamma=0.9, epsilon=0.1): # Constructor
        self.actions = actions # Lista de acciones posibles ['N', 'S', 'E', 'O']
        self.alpha = alpha     # Tasa de Aprendizaje (qué tan rápido aprende)
        self.gamma = gamma     # Factor de Descuento (cuánto valora el futuro)
        self.epsilon = epsilon   # Tasa de Exploración (cuánto prueba cosas nuevas)
        
        # El Q-Table. Es un diccionario de diccionarios.
        # Ejemplo: self.q_table[(2, 0)]['N'] = 0.5
        # (Usamos un defaultdict para crear diccionarios internos automáticamente)
        from collections import defaultdict
        self.q_table = defaultdict(lambda: {a: 0.0 for a in self.actions}) # Inicializa Q(s,a)=0

    def get_q_value(self, state, action): # Método simple para leer el Q-Table
        return self.q_table[state][action] # Devuelve el valor Q para (s, a)

    def choose_action(self, state): # Decide qué acción tomar (Exploración vs. Explotación)
        """
        Implementa la estrategia "epsilon-greedy"
        """
        if random.random() < self.epsilon: # Si un número aleatorio es menor que epsilon...
            # --- EXPLORACIÓN ---
            return random.choice(self.actions) # ...toma una acción COMPLETAMENTE ALEATORIA.
        else:
            # --- EXPLOTACIÓN ---
            # Elige la mejor acción conocida desde este estado
            q_values_for_state = self.q_table[state] # Obtiene los Q-values {'N':0.1, 'S':0.5, ...}
            # Encuentra la acción ('a') que tiene el valor Q máximo
            best_action = max(q_values_for_state, key=q_values_for_state.get)
            return best_action # Devuelve la mejor acción conocida

    def update(self, state, action, reward, next_state): # El corazón del aprendizaje
        """
        Actualiza el Q-Table usando la fórmula de TD
        Q(s, a) <-- Q(s, a) + alpha * (Muestra - Q(s, a))
        """
        
        # 1. Obtener el valor Q(s, a) antiguo (lo que *sabíamos*)
        old_q_value = self.get_q_value(state, action) # Ej: 0.5
        
        # 2. Calcular el valor máximo futuro: max_a'[Q(s', a')]
        #    (Aquí es donde Q-Learning es "off-policy": mira la *mejor* acción futura,
        #    incluso si no es la que tomará en el siguiente paso)
        q_values_for_next_state = self.q_table[next_state] # Obtiene Q(s', a) para todas las 'a'
        max_future_q = 0.0 # Inicializa
        if q_values_for_next_state: # Si el estado siguiente no es terminal (tiene Q-values)
            max_future_q = max(q_values_for_next_state.values()) # Encuentra el valor más alto
        
        # 3. Calcular la "Muestra" (TD Target): r + gamma * max_a'[Q(s', a')]
        sample = reward + self.gamma * max_future_q # Ej: -0.04 + 0.9 * 0.8 = 0.68
        
        # 4. Calcular el "Error" (TD Error): Muestra - Q(s, a)
        td_error = sample - old_q_value # Ej: 0.68 - 0.5 = 0.18
        
        # 5. Actualizar el Q-Table: Q(s, a) <-- Q(s, a) + alpha * Error
        self.q_table[state][action] = old_q_value + self.alpha * td_error # Ej: 0.5 + 0.1 * 0.18

# --- P3: Bucle de Entrenamiento ---

def train(n_episodes=20000): # Función para entrenar al agente
    print("--- 35. Q-Learning ---") # Título
    print(f"Entrenando al agente por {n_episodes} episodios...") # Mensaje
    
    env = GridWorldEnv() # 1. Crear el entorno (la "caja negra")
    agent = QLearningAgent(env.all_actions) # 2. Crear el agente (el "cerebro")
    
    # Bucle principal de entrenamiento: repetir N episodios (juegos completos)
    for i in range(n_episodes): # Repetir 20,000 veces
        
        state = env.get_start_state() # Empezar en una casilla aleatoria
        
        while state not in env.terminal_states: # Mientras el episodio no termine...
            
            # 1. Agente elige una acción (epsilon-greedy)
            action = agent.choose_action(state) # Ej: 'N'
            
            # 2. Entorno simula el resultado y devuelve (s', r)
            next_state, reward = env.step(state, action) # Ej: ((1,0), -0.04)
            
            # 3. Agente actualiza su Q-Table con esta experiencia (s, a, r, s')
            agent.update(state, action, reward, next_state)
            
            # 4. Moverse al siguiente estado
            state = next_state
        
        if (i + 1) % 2000 == 0: # Imprimir progreso cada 2000 episodios
            print(f"Episodio {i + 1} completado.")
            
    print("¡Entrenamiento completado!") # Mensaje final
    return agent, env # Devuelve el agente entrenado y el entorno

def extract_policy_from_q(q_table, env): # Función para ver la política final
    policy = {} # Diccionario para la política
    for state in env._rewards.keys(): # Itera sobre todos los estados (incluidos terminales y pared)
        if state == env.wall or state in env.terminal_states: # Si es pared o terminal
            continue # No hay acción
        
        q_values_for_state = q_table[state] # Obtiene los Q-values para este estado
        if not q_values_for_state: # Si el estado nunca fue visitado
             policy[state] = '?' # Marcar como desconocido
             continue
             
        best_action = max(q_values_for_state, key=q_values_for_state.get) # Encuentra la acción con max Q
        policy[state] = best_action # Asigna a la política
    return policy # Devuelve el mapa

# --- P4: Ejecutar Entrenamiento e Imprimir Resultados ---

# 1. Entrenar al agente
trained_agent, environment = train()

# 2. Extraer la política final del Q-Table del agente
final_policy = extract_policy_from_q(trained_agent.q_table, environment)

# 3. Imprimir la política
print("\n--- Política Óptima (pi*) aprendida por Q-Learning ---")
action_arrows = {'N': '↑', 'S': '↓', 'E': '→', 'O': '←', '?': '?'} # Mapa de flechas
for r in range(environment.rows): # Itera filas
    print("+---------+" * environment.cols) # Borde
    row_str = "" # String de fila
    for c in range(environment.cols): # Itera columnas
        state = (r, c) # Estado
        if state == environment.wall: row_str += "|  PARED  " # Pared
        elif state in environment.terminal_states: row_str += "|  FIN    " # Terminal
        elif state in final_policy: row_str += f"|    {action_arrows[final_policy[state]]}    " # Flecha
        else: row_str += "|    ?    " # (Nunca visitado)
    print(row_str + "|") # Imprime fila
print("+---------+" * environment.cols) # Borde

# (Opcional: Imprimir una parte del Q-Table para ver los valores)
print(f"\nEjemplo de Q-Values aprendidos para el estado (2, 0):")
print(trained_agent.q_table[(2, 0)])