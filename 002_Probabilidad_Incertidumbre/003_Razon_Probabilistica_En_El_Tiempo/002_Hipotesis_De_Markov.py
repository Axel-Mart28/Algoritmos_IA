# Algoritmo de Hipotesis de Markov

# Este es un CONCEPTO y una suposición fundamental, no un algoritmo complejo.
# El "algoritmo" aquí es una demostración de la diferencia entre un proceso que es estacionario y uno que no lo es.

# Definición:
# Un proceso (un sistema que cambia con el tiempo) es "estacionario" si sus reglas de transición (sus probabilidades) NO cambian con el tiempo.

# En términos matemáticos:
# La probabilidad de P(X_{t+1} | X_t) es la *misma* para cualquier valor de 't'.

#Ejemplo:
# - Un dado *cargado* (ej. 50% de sacar '6') es ESTACIONARIO.
#   Sus reglas son injustas, pero son *consistentes*. La probabilidad de sacar '6' en el lanzamiento 100 es la misma que en el lanzamiento 1.
# - Un dado *hecho de hielo que se derrite* es NO-ESTACIONARIO.
#   Su forma cambia, por lo que sus probabilidades de transición
#   (P(Resultado | Lanzamiento)) cambian con cada lanzamiento (tiempo).
#
# ¿Cómo funciona este programa?
# Vamos a simular un proceso de "caminata aleatoria" simple (ej. un robot moviéndose) bajo dos condiciones:
# 1. Un mundo estacionario: La prob. de avanzar es siempre 70%.
# 2. Un mundo no-estacionario: La prob. de avanzar *decae* con el tiempo (ej. las baterías del robot se gastan).
#
# Aplicaciones:
# - Esta *suposición* es necesaria para casi todos los modelos de  esta sección (HMMs, Filtros de Kalman).
#
# Ventajas (de la suposición):
# - Simplifica el problema enormemente. Solo necesitamos aprender un modelo de transición.

# Desventajas (de la suposición):
# - El mundo real a menudo *no* es estacionario (ej. el clima cambia con las estaciones, la economía cambia con los años).
# - El modelo puede volverse obsoleto.

import random # Para simular las transiciones

# --- P1: Definición de los Procesos ---

def transicion_estacionaria(estado_actual):
    """
    Simula una transición con reglas FIJAS.
    P(Avanzar) = 0.7
    P(Quedarse) = 0.3
    """
    prob_avanzar = 0.7 # Esta probabilidad NUNCA cambia
    
    if random.random() < prob_avanzar:
        return estado_actual + 1 # Avanza
    else:
        return estado_actual # Se queda

def transicion_no_estacionaria(estado_actual, tiempo_t):
    """
    Simula una transición con reglas CAMBIANTES.
    La probabilidad de avanzar decae con el tiempo 't'.
    """
    # La probabilidad de avanzar depende del tiempo
    # (ej. 0.99^t -> 0.7, 0.693, 0.686, ...)
    prob_avanzar = 0.7 * (0.99 ** tiempo_t) 
    
    if random.random() < prob_avanzar:
        return estado_actual + 1 # Avanza
    else:
        return estado_actual # Se queda

# --- P2: Simulación del Proceso Estacionario ---
print("Demostración de Proceso Estacionario") # Título
N_PASOS = 20 # Número de pasos en el tiempo
estado = 0 # Posición inicial
print("Simulando proceso ESTACIONARIO (reglas fijas)...")
print(f"  t=0: Estado={estado}")

for t in range(1, N_PASOS + 1): # Simular N_PASOS
    # Llamar a la función de transición estacionaria
    estado = transicion_estacionaria(estado)
    print(f"  t={t}: Estado={estado}")

print(f"Posición final (Estacionario): {estado}")

# --- P3: Simulación del Proceso No-Estacionario ---
print("\n--- Demostración de Proceso de Markov ---") # Título
estado = 0 # Reiniciar posición inicial
print("Simulando proceso NO-ESTACIONARIO (reglas cambiantes)...")
print(f"  t=0: Estado={estado}")

for t in range(1, N_PASOS + 1): # Simular N_PASOS
    # Llamar a la función, pasando el tiempo 't'
    estado = transicion_no_estacionaria(estado, t) 
    print(f"  t={t}: Estado={estado}")

print(f"Posición final (No-Estacionario): {estado}")

print("\nConclusión:")
print("En el proceso Estacionario, el robot avanza de forma constante.")
print("En el No-Estacionario, el robot avanza bien al principio,")
print("pero deja de avanzar a medida que 't' aumenta y su prob. decae.")
print("(Ejecuta el código varias veces para ver diferentes resultados)")