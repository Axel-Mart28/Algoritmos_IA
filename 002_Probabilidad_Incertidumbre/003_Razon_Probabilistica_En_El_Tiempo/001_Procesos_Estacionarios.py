# Algoritmo de PROCESOS ESTACIONARIOS

# Este es un CONCEPTO y una *suposición* fundamental, no un algoritmo complejo.
# El algoritmo aquí es una demostración de la diferencia entre un proceso que es estacionario y uno que no lo es.

# Definición:
# Un proceso (un sistema que cambia con el tiempo) es "estacionario" si sus reglas de transición (sus probabilidades) NO cambian con el tiempo.

# En términos matemáticos:
# La probabilidad de P(X_{t+1} | X_t) es la *misma* para cualquier valor de 't'.

# ¿Cómo funciona este programa?
# 1. Mundo Estacionario: Simulamos un robot donde P(Avanzar) = 0.7 (siempre).
# 2. Mundo No-Estacionario: Simulamos un robot donde P(Avanzar) *decae* con el tiempo (ej. las baterías se gastan).

# Aplicaciones:
# - Esta *suposición* es necesaria para casi todos los modelos de esta sección (HMMs, Filtros de Kalman).
#
# Ventajas (de la suposición):
# - Simplifica el problema: Solo necesitamos aprender *un* modelo de transición.
#
# Desventajas (de la suposición):
# - El mundo real a menudo *no* es estacionario.

import random # Para simular las transiciones aleatorias

# --- P1: Definición de los Procesos ---

def transicion_estacionaria(estado_actual): # Función para el mundo con reglas fijas
    """
    Simula una transición con reglas FIJAS.
    P(Avanzar) = 0.7
    P(Quedarse) = 0.3
    """
    prob_avanzar = 0.7 # Esta probabilidad NUNCA cambia con el tiempo
    
    if random.random() < prob_avanzar: # Genera un float [0.0, 1.0) y compara
        return estado_actual + 1 # Resultado 1: Avanza
    else: # Si el float es > 0.7 (pasa el 30% de las veces)
        return estado_actual # Resultado 2: Se queda

def transicion_no_estacionaria(estado_actual, tiempo_t): # Función con reglas cambiantes
    """
    Simula una transición con reglas CAMBIANTES.
    La probabilidad de avanzar decae con el tiempo 't'.
    """
    # La probabilidad de avanzar depende del tiempo 't'
    # (ej. 0.99^t -> 0.7, 0.693, 0.686, ...)
    # A medida que 't' aumenta, 'prob_avanzar' disminuye
    prob_avanzar = 0.7 * (0.99 ** tiempo_t) 
    
    if random.random() < prob_avanzar: # Compara con la probabilidad decreciente
        return estado_actual + 1 # Resultado 1: Avanza (menos probable con el tiempo)
    else:
        return estado_actual # Resultado 2: Se queda (más probable con el tiempo)

# --- P2: Simulación del Proceso Estacionario ---
print("Algoritmo de Proceso Estacionario") # Título
N_PASOS = 20 # Número de pasos en el tiempo
estado = 0 # Posición inicial del robot
print("Simulando proceso ESTACIONARIO (reglas fijas)...") # Mensaje
print(f"  t=0: Estado={estado}") # Imprime el estado inicial

for t in range(1, N_PASOS + 1): # Bucle desde t=1 hasta 20
    # Llama a la función de transición estacionaria
    estado = transicion_estacionaria(estado) # No necesita 't'
    print(f"  t={t}: Estado={estado}") # Imprime el nuevo estado

print(f"Posición final (Estacionario): {estado}") # Imprime el resultado final

# --- P3: Simulación del Proceso No-Estacionario ---
print("\n--- Demostración de Proceso NO-Estacionario ---") # Título
estado = 0 # Reiniciar posición inicial
print("Simulando proceso NO-ESTACIONARIO (reglas cambiantes)...") # Mensaje
print(f"  t=0: Estado={estado}") # Imprime el estado inicial

for t in range(1, N_PASOS + 1): # Bucle desde t=1 hasta 20
    # Llama a la función, pasando el tiempo 't' como argumento
    estado = transicion_no_estacionaria(estado, t) 
    print(f"  t={t}: Estado={estado}") # Imprime el nuevo estado

print(f"Posición final (No-Estacionario): {estado}") # Imprime el resultado final

print("\nConclusión:") # Resumen
print("En el proceso Estacionario, el robot avanza de forma constante.")
print("En el No-Estacionario, el robot avanza bien al principio,")
print("pero deja de avanzar a medida que 't' aumenta y su prob. decae.")