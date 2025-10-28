# Algoritmo de EXPLORACIÓN vs. EXPLOTACIÓN

# Este tema no es un algoritmo nuevo, sino un *concepto fundamental* y un *dilema* que debe resolverse en el Aprendizaje por Refuerzo Activo (como Q-Learning).
# El Dilema:
# 1. Explotación (Exploitation): Usar el conocimiento actual para tomar la acción que creemos que es la mejor (maximizar la recompensa a corto plazo).
#    Ej: "Mi Q-Table dice que 'Norte' tiene el valor 0.8, así que iré al Norte".

# 2. Exploración (Exploration): Probar acciones nuevas o que parecen "malas" para ganar más conocimiento y descubrir si *en realidad* son mejores (maximizar la recompensa a largo plazo).
#    Ej: "Nunca he probado ir al 'Sur' desde aquí. ¿Qué pasa si lo hago?"

# ¿Por qué es un dilema?
# - Demasiada explotación: El agente se atasca en un "óptimo local" (un camino "bueno" pero no "el mejor") y nunca descubre la verdadera solución.
# - Demasiada exploración: El agente nunca usa lo que aprendió y se comporta de forma aleatoria, obteniendo una recompensa baja.

# ¿Cómo funciona?
# Se implementa una "política de exploración", como Epsilon-Greedy ($\epsilon$-greedy).
# Esta es la regla que el agente de Q-Learning (#35) usaba para decidir qué acción tomar.
#
# Estrategia Epsilon-Greedy:
# - Se define un parámetro 'epsilon' ($\epsilon$), usualmente un valor pequeño como 0.1 (10%).
# - En cada paso, se genera un número aleatorio (entre 0 y 1).
# - Si el número es MENOR que epsilon (pasa el 10% del tiempo): El agente "Explora".
# - Si el número es MAYOR que epsilon (pasa el 90% del tiempo): El agente "Explota".
#
# Componentes:
# 1. Epsilon ($\epsilon$): La probabilidad de explorar.
# 2. Una fuente de aleatoriedad (random).
# 3. El Q-Table (conocimiento actual) para poder explotar.
#
# Aplicaciones:
# - Esencial en Q-Learning y otros algoritmos "online" (que aprenden mientras actúan).
# - Problemas de "Multi-Armed Bandit" (decidir qué palanca de máquina tragamonedas jalar).
#
# Ventajas:
# - Muy simple de implementar.
# - Garantiza que el agente (eventualmente) explorará todos los estados y acciones, lo que es necesario para que Q-Learning converja a la solución óptima.
#
# Desventajas:
# - Es "tonto": cuando explora, elige *cualquier* acción al azar, incluso la obviamente peor. (Estrategias más avanzadas, como "softmax", son más inteligentes).
# - Un epsilon fijo puede ser ineficiente (al principio se quiere explorar mucho, al final se quiere explotar más). Esto se resuelve con un "epsilon decreciente".

# Ejemplo de uso:
# - El agente está en el estado (2,0) y su Q-Table dice:
#   {'N': 0.8, 'S': 0.1, 'E': 0.3, 'O': 0.2}
# - (Explotar): Elegiría 'N' (valor 0.8).
# - (Explorar): Elegiría 'N', 'S', 'E', u 'O' con la misma probabilidad (25% c/u).
# - Epsilon-Greedy (con $\epsilon=0.1$) elegirá 'N' el 90% de las veces, y
#   ('N'/'S'/'E'/'O' al azar) el 10% de las veces.

import random # Necesario para generar el número aleatorio y elegir acciones

# --- P1: Implementación de la Estrategia (Epsilon-Greedy) ---

# Esta es la *misma* clase del algoritmo de Q_Learning, pero nos enfocamos solo en el constructor y en el método 'choose_action'.
class AgentWithEpsilonGreedy: # Un agente que usa esta estrategia
    
    def __init__(self, actions, epsilon=0.1): # Constructor
        self.actions = actions     # Lista de acciones ['N', 'S', 'E', 'O']
        self.epsilon = epsilon   # Tasa de Exploración (probabilidad de explorar)
        
        # El Q-Table (conocimiento actual del agente)
        # Lo llenamos con valores de ejemplo para esta demostración
        # (En un agente real, este Q-Table estaría vacío al inicio y se llenaría)
        self.q_table = {
            # Estado (2, 0)
            (2, 0): {'N': 0.82, 'S': 0.73, 'E': 0.76, 'O': 0.76},
            # Estado (2, 1)
            (2, 1): {'N': 0.65, 'S': 0.50, 'E': 0.60, 'O': 0.78} 
        }
        print(f"Agente inicializado con Epsilon = {self.epsilon} ({self.epsilon*100}% de exploración)")

    def choose_action(self, state): # Decide qué acción tomar
        """
        Implementa la estrategia "epsilon-greedy"
        """
        
        # 1. Generar un número aleatorio entre 0.0 y 1.0
        rand_num = random.random()
        
        if rand_num < self.epsilon: # (pasa ~10% de las veces, si epsilon=0.1)
            # --- FASE DE EXPLORACIÓN ---
            print(f"  -> Decisión (Rand {rand_num:.2f} < {self.epsilon}): EXPLORAR") # Mensaje
            chosen_action = random.choice(self.actions) # Elige una acción al azar
            return chosen_action # Devuelve la acción aleatoria
        
        else: # (pasa ~90% de las veces)
            # --- FASE DE EXPLOTACIÓN ---
            print(f"  -> Decisión (Rand {rand_num:.2f} >= {self.epsilon}): EXPLOTAR") # Mensaje
            
            # Obtiene los Q-values conocidos para este estado
            # .get() es por si el estado es nuevo; si no, devuelve un Q=0 por defecto
            q_values_for_state = self.q_table.get(state, {a: 0.0 for a in self.actions})
            
            # Encuentra la acción ('a') que tiene el valor Q máximo
            # (key=q_values_for_state.get) le dice a max() que mire los *valores*
            # del diccionario, no las llaves ('N', 'S', ...).
            best_action = max(q_values_for_state, key=q_values_for_state.get)
            
            return best_action # Devuelve la mejor acción conocida

# --- P2: Demostración de la Estrategia ---
print("--- 36. Exploración vs. Explotación (Demo) ---")
    
# 1. Crear las acciones y el agente
available_actions = ['N', 'S', 'E', 'O'] # Acciones que el agente conoce
agent = AgentWithEpsilonGreedy(available_actions, epsilon=0.1) # 10% exploración

# 2. Simular 10 decisiones desde el estado (2, 0)
current_state = (2, 0) # Estado de prueba
print(f"\nQ-Table para estado {current_state}: {agent.q_table[current_state]}") # Muestra conocimiento
print(f"  (Mejor acción conocida: 'N' con Q=0.82)") # Indica la acción óptima
print(f"\nSimulando 10 decisiones desde {current_state}:")
    
for i in range(10): # Bucle de 10 intentos
    action_taken = agent.choose_action(current_state) # Preguntar al agente
    print(f"  Intento {i+1}: Agente eligió '{action_taken}'") # Imprimir resultado

# 3. Simular 10 decisiones desde el estado (2, 1)
current_state = (2, 1) # Segundo estado de prueba
print(f"\nQ-Table para estado {current_state}: {agent.q_table[current_state]}") # Muestra conocimiento
print(f"  (Mejor acción conocida: 'O' con Q=0.78)") # Indica la acción óptima
print(f"\nSimulando 10 decisiones desde {current_state}:")
    
for i in range(10): # Bucle de 10 intentos
    action_taken = agent.choose_action(current_state) # Preguntar al agente
    print(f"  Intento {i+1}: Agente eligió '{action_taken}'") # Imprimir resultado