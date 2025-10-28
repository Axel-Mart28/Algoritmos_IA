# Algoritmo de MDP parcialmente observable (POMDP) simple
# Este algoritmo implementa un Proceso de Decisión de Markov Parcialmente Observable (POMDP)
# Los POMDPs extienden los MDPs al considerar que el agente no conoce el estado real del sistema, sino que mantiene una creencia (belief) sobre los estados posibles basada en observaciones.
# Entre sus aplicaciones están: robótica (localización), diagnósticos médicos, y sistemas de recomendación donde la información del estado completo no está disponible.
# Ventajas: modela situaciones realistas con información incompleta y ruidosa.
# Desventajas: computacionalmente muy complejo, el espacio de creencias es continuo.
#En este caso, se aplica a un ejemplo de acciones, las cuales son llevar paraguas o no, dependiendo ei llueve o no

import numpy as np  # Biblioteca para cálculos numéricos
# Definición del POMDP:

# Estados y acciones
estados = ['Lluvia', 'Sol']  # Estados reales del mundo (no directamente observables)
acciones = ['Llevar paraguas', 'No llevar paraguas']  # Acciones disponibles para el agente
observaciones = ['Nublado', 'Soleado']  # Observaciones que el agente puede percibir (ruidosas)

# Recompensas R(s,a) - función de recompensa que depende del estado real y la acción
recompensa = {
    ('Lluvia','Llevar paraguas'): 5,      # Recompensa por llevar paraguas cuando llueve
    ('Lluvia','No llevar paraguas'): -5,  # Recompensa negativa por no llevar paraguas cuando llueve
    ('Sol','Llevar paraguas'): 1,         # Recompensa pequeña por llevar paraguas cuando hace sol
    ('Sol','No llevar paraguas'): 4       # Recompensa positiva por no llevar paraguas cuando hace sol
}

# Probabilidades de observación P(o|s) - probabilidad de observar o dado el estado real s
P_obs = {
    ('Lluvia','Nublado'): 0.8,  # Probabilidad de ver nublado cuando realmente llueve
    ('Lluvia','Soleado'): 0.2,  # Probabilidad de ver soleado cuando realmente llueve (error)
    ('Sol','Nublado'): 0.1,     # Probabilidad de ver nublado cuando realmente hace sol (error)
    ('Sol','Soleado'): 0.9      # Probabilidad de ver soleado cuando realmente hace sol
}

# Probabilidades de transición P(s'|s,a) - evolución del estado real (en este ejemplo, la acción no afecta la transición)
P_trans = {
    ('Lluvia','Lluvia'): 0.7,  # Probabilidad de que siga lloviendo si está lloviendo
    ('Lluvia','Sol'): 0.3,     # Probabilidad de que salga el sol si está lloviendo
    ('Sol','Lluvia'): 0.2,     # Probabilidad de que llueva si hace sol
    ('Sol','Sol'): 0.8         # Probabilidad de que siga haciendo sol si hace sol
}

# Factor de descuento
gamma = 0.9  # Factor de descuento para recompensas futuras

# Estado inicial incierto (belief) - distribución de probabilidad sobre los estados
belief = {'Lluvia': 0.5, 'Sol': 0.5}  # Creencia inicial: 50% probabilidad de lluvia o sol

# --- Función para actualizar belief después de acción y observación ---
def actualizar_belief(belief, accion, observacion):
    """Actualiza la creencia (belief) usando el filtro de Bayes después de una acción y observación."""
    b_new = {}  # Nuevo belief a calcular
    
    # Para cada estado posible s_prime en el nuevo tiempo
    for s_prime in estados:
        # P(o|s') * sum_s[P(s'|s) * b(s)] - fórmula de actualización Bayesiana
        prob_o_s = P_obs[(s_prime, observacion)]  # P(observación|s_prime)
        prob_s = sum(P_trans[(s, s_prime)] * belief[s] for s in estados)  # sum_s[P(s'|s) * b(s)]
        b_new[s_prime] = prob_o_s * prob_s  # P(o|s') * sum_s[P(s'|s) * b(s)]
    
    # Normalizar para que las probabilidades sumen 1
    total = sum(b_new.values())
    for s in b_new:
        b_new[s] /= total
    
    return b_new

# --- Calcular utilidad esperada de una acción dado belief ---
def utilidad_esperada(belief, accion):
    """Calcula la utilidad esperada inmediata de una acción dado el belief actual."""
    return sum(belief[s] * recompensa[(s, accion)] for s in estados)  # sum_s[b(s) * R(s,a)]

# --- Ejemplo de decisión ---
accion1 = 'Llevar paraguas'    # Primera acción a evaluar
accion2 = 'No llevar paraguas' # Segunda acción a evaluar
UE1 = utilidad_esperada(belief, accion1)  # Utilidad esperada de llevar paraguas
UE2 = utilidad_esperada(belief, accion2)  # Utilidad esperada de no llevar paraguas

print("=== POMDP SIMPLE ===\n")
print(f"Utilidad esperada '{accion1}': {UE1:.2f}")  # Muestra utilidad de acción 1
print(f"Utilidad esperada '{accion2}': {UE2:.2f}")  # Muestra utilidad de acción 2

# Determina la mejor acción basada en las utilidades esperadas
mejor_accion = accion1 if UE1 > UE2 else accion2
print(f"\nMejor acción según belief: {mejor_accion}")  # Muestra la mejor acción