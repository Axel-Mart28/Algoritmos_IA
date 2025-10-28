#Algoritmo de BÚSQUEDA DE LA POLÍTICA

# Este algoritmo es sobre Búsqueda de la Política (también llamado "Policy Gradient").
# Es un enfoque fundamentalmente *diferente* a la Iteración de Valores o Q-Learning.

# El Dilema:
# - Métodos Basados en Valor (Q-Learning): Aprenden una *función de valor* (el Q-Table) y luego *derivan* una política de ella (eligiendo la acción con el Q-value más alto).
# - Métodos Basados en Política (Policy Search): Aprenden la *política* pi(s) *directamente*. No aprenden un Q-Table.

# ¿Cómo funciona?
# 1. La política 'pi' es una función con "parámetros" (pesos/parámetros, 'theta' $\theta$).
#    pi_theta(a | s) = "La probabilidad de tomar la acción 'a' en el estado 's'
#    dados los parámetros actuales 'theta'".
# 2. El agente "prueba" su política actual en el entorno (juega un episodio completo).
# 3. Observa la recompensa total (el "Retorno" Gt) que obtuvo en ese episodio.
# 4. Usa cálculo (gradiente) para "empujar" (nudge) los parámetros 'theta'.
#    - Si el episodio fue *bueno* (recompensa alta), "empuja" los parámetros para
#      hacer que las acciones que tomó sean *más* probables en el futuro.
#    - Si el episodio fue *malo* (recompensa baja/negativa), "empuja" los parámetros
#      para hacer que esas acciones sean *menos* probables.

# El algoritmo específico que se implementa en este caso se llama REINFORCE (un algoritmo Monte Carlo).
#
# Componentes:
# 1. Política Parametrizada (pi_theta): El "cerebro" del agente. En nuestro caso,
#    será una tabla de "logits" (puntuaciones) para cada par (s, a).
# 2. Función Softmax: Convierte los "logits" (puntuaciones) en probabilidades (ej. 70% N, 10% S...).
# 3. Episodio (Trayectoria): Una lista de (estado, acción, recompensa) de un juego completo.
# 4. Retorno (Gt): La suma de recompensas *descontadas* desde el paso 't' hasta el final.
# 5. Tasa de Aprendizaje (alpha): Controla qué tan grandes son los "empujones" a los parámetros.
#
# Aplicaciones:
# - Robótica (control continuo, ej. "girar el brazo 15.3 grados").
# - Problemas donde la política óptima es aleatoria (ej. Piedra, Papel o Tijera).
#
# Ventajas:
# - Puede aprender políticas estocásticas (aleatorias) de forma natural.
# - Funciona en espacios de acción continuos (donde Q-Learning falla, ya que no puede
#   hacer 'max_a' sobre un número infinito de acciones).
#
# Desventajas:
# - Varianza alta: El aprendizaje es inestable porque un episodio "bueno" puede
#   ocurrir por pura suerte, llevando al agente a una conclusión incorrecta.
# - Es lento: A menudo requiere muchísimos episodios para aprender.
# - Tiende a converger a un "óptimo local" (una política "buena"), no
#   necesariamente al "óptimo global" (la política "perfecta").
#
# Ejemplo de uso:
# - El agente prueba ir (N, E, E, N) y obtiene una recompensa total de +0.8 (bueno).
# - El algoritmo REINFORCE "recompensará" los parámetros que hicieron que
#   pi(N|s1), pi(E|s2), pi(E|s3), pi(N|s4) fueran más probables.

import random # Necesario para elegir acciones basadas en probabilidad
import copy   # Necesario para el entorno
import math   # Necesario para la función softmax (math.exp)
from collections import defaultdict # Para nuestro "cerebro" (la política)

# --- P1: Definición del Entorno (MDP - Mundo de Rejilla) ---
# (Usamos la *misma* clase de entorno que en Q-Learning. El agente
# interactúa con esta "caja negra" llamando a env.step())
class GridWorldEnv:
    def __init__(self):
        self.rows = 3
        self.cols = 4
        self.wall = (1, 1)
        self.terminal_states = [(0, 3), (1, 3)]
        self._rewards = {}
        for r in range(self.rows):
            for c in range(self.cols):
                s = (r, c)
                if s == (0, 3): self._rewards[s] = 1.0
                elif s == (1, 3): self._rewards[s] = -1.0
                elif s == self.wall: self._rewards[s] = 0.0
                else: self._rewards[s] = -0.04
        self.all_actions = ['N', 'S', 'E', 'O']
    def get_start_state(self):
        while True:
            s = (random.randint(0, self.rows - 1), random.randint(0, self.cols - 1))
            if s != self.wall and s not in self.terminal_states:
                return s
    def step(self, state, action):
        if state in self.terminal_states: return state, 0.0
        if action == 'N' or action == 'S': slip_actions = ['O', 'E']
        else: slip_actions = ['N', 'S']
        rand_num = random.random()
        if rand_num < 0.8: chosen_action = action
        elif rand_num < 0.9: chosen_action = slip_actions[0]
        else: chosen_action = slip_actions[1]
        r, c = state
        if chosen_action == 'N': r = max(0, r - 1)
        elif chosen_action == 'S': r = min(self.rows - 1, r + 1)
        elif chosen_action == 'E': c = min(self.cols - 1, c + 1)
        elif chosen_action == 'O': c = max(0, c - 1)
        next_state = (r, c)
        if next_state == self.wall: next_state = state
        reward = self._rewards[next_state]
        return next_state, reward

# --- P2: Algoritmo del Agente (Policy Gradient - REINFORCE) ---

class PolicyGradientAgent:
    def __init__(self, actions, alpha=0.01, gamma=0.9): # Constructor
        self.actions = actions # Lista de acciones ['N', 'S', 'E', 'O']
        self.alpha = alpha     # Tasa de Aprendizaje (qué tan grande es el "empujón")
        self.gamma = gamma     # Factor de Descuento
        
        # El "cerebro" del agente (los parámetros 'theta').
        # Es un diccionario de "logits" (puntuaciones).
        # self.policy_logits[(2,0)]['N'] = 0.5 (una puntuación, no una probabilidad)
        # Inicializamos todas las puntuaciones en 0.0, lo que significa que al
        # principio, todas las acciones son igualmente probables (prob = 25%).
        self.policy_logits = defaultdict(lambda: {a: 0.0 for a in self.actions})

    def _softmax(self, logits): # Función auxiliar para convertir logits en probabilidades
        """ Convierte un diccionario de puntuaciones (logits) en probabilidades """
        exp_scores = {a: math.exp(s) for a, s in logits.items()} # e^puntuación
        sum_exp_scores = sum(exp_scores.values()) # Suma de todas las e^puntuación
        # La probabilidad es (e^puntuación_accion) / (suma de todas las e^puntuación)
        probs = {a: s / sum_exp_scores for a, s in exp_scores.items()}
        return probs # Devuelve ej: {'N': 0.7, 'S': 0.1, 'E': 0.1, 'O': 0.1}

    def choose_action(self, state): # Decide qué acción tomar
        """
        Elige una acción *muestreando* (sampling) de la política de probabilidad.
        """
        # 1. Obtener los logits (puntuaciones) para este estado
        logits = self.policy_logits[state]
        
        # 2. Convertir los logits en probabilidades usando softmax
        probs = self._softmax(logits)
        
        # 3. Muestrear una acción basada en esas probabilidades
        # (ej. si 'N' tiene 70% de prob, será elegido el 70% de las veces)
        actions = list(probs.keys()) # ['N', 'S', 'E', 'O']
        probabilities = list(probs.values()) # [0.7, 0.1, 0.1, 0.1]
        
        # random.choices devuelve una *lista* de 1 elemento, así que tomamos el [0]
        return random.choices(actions, weights=probabilities, k=1)[0]

    def update(self, episode_memory): # El corazón del aprendizaje (se llama al FINAL del episodio)
        """
        Actualiza los parámetros de la política (logits) usando REINFORCE.
        'episode_memory' es una lista de tuplas: [(s, a, r), (s, a, r), ...]
        """
        
        # 1. Calcular los Retornos (Gt) para CADA paso del episodio
        #    Gt = r_t + gamma*r_{t+1} + gamma^2*r_{t+2} ...
        #    Lo calculamos "hacia atrás" para eficiencia.
        
        returns = [] # Lista para guardar los Retornos (Gt)
        discounted_return = 0.0 # Empezamos desde el final
        
        # Iterar sobre la memoria del episodio EN REVERSA
        for (state, action, reward) in reversed(episode_memory):
            # Gt = r_t + gamma * G_{t+1}
            discounted_return = reward + self.gamma * discounted_return
            returns.insert(0, discounted_return) # Insertar al *principio* de la lista
        
        # (Opcional: "Normalizar" los retornos. Esto estabiliza el aprendizaje)
        # (Lo omitimos por simplicidad, pero es crucial en la práctica)
        
        # 2. Actualizar los parámetros (logits) para cada paso del episodio
        
        # Iterar sobre la memoria (hacia adelante) y los retornos al mismo tiempo
        for i, (state, action, reward) in enumerate(episode_memory):
            
            # Obtener el Retorno (Gt) para este paso
            G_t = returns[i]
            
            # Obtener las probabilidades actuales para este estado
            logits = self.policy_logits[state]
            probs = self._softmax(logits)
            
            # --- Aquí está la "magia" del Policy Gradient ---
            # La regla de actualización es:
            # logit(s,a) <-- logit(s,a) + alpha * Gt * (1 - prob(a|s))
            # (Y para todas las *otras* acciones 'b' != 'a'):
            # logit(s,b) <-- logit(s,b) - alpha * Gt * (prob(b|s))
            #
            # En esencia:
            # - Si Gt es positivo (buen resultado), aumenta el logit de la acción
            #   que tomamos (y reduce el de las otras).
            # - Si Gt es negativo (mal resultado), *reduce* el logit de la acción
            #   que tomamos (y aumenta el de las otras).
            
            for a in self.actions: # Iterar sobre 'N', 'S', 'E', 'O'
                prob = probs[a] # Probabilidad de esta acción 'a'
                
                if a == action:
                    # Esta es la acción que SÍ tomamos
                    # "Empuja" el logit en proporción a (1 - probabilidad)
                    gradient_log_prob = (1 - prob)
                else:
                    # Esta es una acción que NO tomamos
                    # "Empuja" el logit en la dirección opuesta
                    gradient_log_prob = -prob
                    
                # Aplicar la actualización
                # Actualización = tasa_aprendizaje * Retorno * (gradiente_log_prob)
                update = self.alpha * G_t * gradient_log_prob
                self.policy_logits[state][a] += update

# --- P3: Bucle de Entrenamiento ---

def train_policy_gradient(n_episodes=20000): # Función para entrenar al agente
    print("--- 37. Búsqueda de la Política (REINFORCE) ---") # Título
    print(f"Entrenando al agente por {n_episodes} episodios...") # Mensaje
    
    env = GridWorldEnv() # 1. Crear el entorno
    agent = PolicyGradientAgent(env.all_actions, alpha=0.01) # 2. Crear el agente
    
    # Bucle principal de entrenamiento
    for i in range(n_episodes): # Repetir 20,000 veces
        
        episode_memory = [] # Lista para guardar (s, a, r) de este episodio
        state = env.get_start_state() # Empezar en una casilla aleatoria
        
        while state not in env.terminal_states: # Mientras el episodio no termine
            
            # 1. Agente elige una acción (muestreando de su política)
            action = agent.choose_action(state)
            
            # 2. Entorno simula el resultado
            next_state, reward = env.step(state, action)
            
            # 3. Guardar la experiencia en la memoria del episodio
            episode_memory.append((state, action, reward))
            
            # 4. Moverse al siguiente estado
            state = next_state
        
        # 5. ¡Episodio terminado!
        #    El agente ahora aprende de la *experiencia completa*
        agent.update(episode_memory)
        
        if (i + 1) % 2000 == 0: # Imprimir progreso
            print(f"Episodio {i + 1} completado.")
            
    print("¡Entrenamiento completado!") # Mensaje final
    return agent, env # Devuelve el agente entrenado

def extract_policy_from_logits(policy_logits, env): # Función para ver la política final
    policy = {} # Diccionario para la política
    for state in env._rewards.keys(): # Itera sobre todos los estados
        if state == env.wall or state in env.terminal_states: # Si es pared o terminal
            continue # No hay acción
        
        logits_for_state = policy_logits[state] # Obtiene los logits para este estado
        if not logits_for_state: # Si el estado nunca fue visitado
             policy[state] = '?'
             continue
             
        best_action = max(logits_for_state, key=logits_for_state.get) # Encuentra la acción con max logit
        policy[state] = best_action # Asigna a la política
    return policy # Devuelve el mapa

# --- P4: Ejecutar Entrenamiento e Imprimir Resultados ---

# 1. Entrenar al agente
# (Nota: Policy Gradient es sensible a 'alpha' y 'n_episodes'.
# Puede necesitar más episodios que Q-Learning para converger)
trained_agent, environment = train_policy_gradient(n_episodes=50000) # Entrenar por 50k episodios

# 2. Extraer la política final (la más probable) de los logits
final_policy = extract_policy_from_logits(trained_agent.policy_logits, environment)

# 3. Imprimir la política
print("\n--- Política Óptima (pi*) aprendida por Policy Gradient ---")
action_arrows = {'N': '↑', 'S': '↓', 'E': '→', 'O': '←', '?': '?'}
for r in range(environment.rows):
    print("+---------+" * environment.cols)
    row_str = ""
    for c in range(environment.cols):
        state = (r, c)
        if state == environment.wall: row_str += "|  PARED  "
        elif state in environment.terminal_states: row_str += "|  FIN    "
        elif state in final_policy: row_str += f"|    {action_arrows[final_policy[state]]}    "
        else: row_str += "|    ?    "
    print(row_str + "|")
print("+---------+" * environment.cols)

# (Opcional: Imprimir logits y probabilidades para un estado)
test_state = (2, 0)
logits = trained_agent.policy_logits[test_state]
probs = trained_agent._softmax(logits)
print(f"\nEjemplo de 'cerebro' para el estado {test_state}:")
print(f"  Logits (Puntuaciones): {logits}")
print(f"  Probabilidades (Softmax): {probs}")