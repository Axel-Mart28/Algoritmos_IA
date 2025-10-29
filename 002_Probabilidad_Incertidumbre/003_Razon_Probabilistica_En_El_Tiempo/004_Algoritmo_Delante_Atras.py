# ALGORITMO HACIA DELANTE-ATRÁS (FORWARD-BACKWARD) 

# Este es un algoritmo de inferencia exacta para Modelos Ocultos de Markov (HMMs).

# Definición:
# Es un algoritmo de programación dinámica que calcula las distribuciones de probabilidad de los estados ocultos para toda una secuencia de tiempo, dada una secuencia de evidencia.
#
# Es la solución a las tareas de Filtrado y Suavizado.
#
# ¿Cómo funciona?
# Se compone de dos "pasadas" (passes) a través de los datos:
#
# 1. La Pasada HACIA ADELANTE (Forward Pass) [Tarea: FILTRADO]
#    - Calcula un mensaje "forward", f_t = P(X_t, e_1:t)
#    - Este mensaje es la probabilidad del estado *actual* (X_t)
#      y *toda la evidencia pasada* (e_1:t).
#    - Es una recursión que va desde t=1 hasta T (el final).
#    - Fórmula: f_t = alpha * P(e_t | X_t) * Sum[ P(X_t | X_{t-1}) * f_{t-1} ]
#    - Si normalizamos f_t en cualquier punto, P(X_t | e_1:t),
#      ¡hemos resuelto el FILTRADO!
#
# 2. La Pasada HACIA ATRÁS (Backward Pass)
#    - Calcula un mensaje "backward", b_t = P(e_{t+1:T} | X_t)
#    - Este mensaje es la probabilidad de *toda la evidencia futura*
#      dado el estado *actual* (X_t).
#    - Es una recursión que va desde t=T (el final) hacia atrás, a 1.
#    - Fórmula: b_t = Sum[ P(e_{t+1} | X_{t+1}) * P(X_{t+1} | X_t) * b_{t+1} ]
#
# 3. La Combinación [Tarea: SUAVIZADO]
#    - El algoritmo combina los dos mensajes para resolver el SUAVIZADO.
#    - P(X_k | e_1:T) = alpha * f_k * b_k  (multiplicación elemento a elemento)
#    - Esto nos da la probabilidad "corregida" del estado pasado X_k,
#      usando *toda* la evidencia (pasada y futura).
#
# 
#
# Componentes (Un Modelo Oculto de Markov - HMM):
# 1. Estados Ocultos (X): El estado real (ej. 'Lloviendo', 'Sol').
# 2. Evidencia (e): La observación del sensor (ej. 'Paraguas', 'SinParaguas').
# 3. Prob. Inicial (P(X_0)): Dónde empezamos (ej. 50/50).
# 4. Modelo de Transición (P(X_t | X_{t-1})): Las reglas de cómo cambia el estado.
# 5. Modelo del Sensor (P(e_t | X_t)): Qué tan preciso es el sensor.
#
# Ventajas:
# - Es *exacto* (da la respuesta probabilística perfecta).
# - Es *eficiente* (tiempo lineal con la longitud de la secuencia, O(T)).
#   Usa programación dinámica para no repetir cálculos.
#
# Desventajas:
# - Requiere conocer el modelo (P(Transición) y P(Sensor)) de antemano.
# - El código es más complejo que el muestreo simple.
#
# Ejemplo de uso:
# - Evidencia: ('Paraguas', 'Paraguas', 'SinParaguas')
# - Tarea de Filtrado: ¿Cuál es P(Clima_3 | 'P', 'P', 'S')?
# - Tarea de Suavizado: ¿Cuál es P(Clima_1 | 'P', 'P', 'S')?
#   (Corregir nuestra creencia sobre el día 1, ahora que vimos 'SinParaguas' el día 3).

import math 
import copy   

# --- P1: Definición del Modelo Oculto de Markov (HMM) ---
# (El escenario del "Paraguas")

# 0. Definición de Estados y Evidencia
estados = ('Lloviendo', 'Sol') # Estados ocultos (X_t)
evidencias = ('Paraguas', 'SinParaguas') # Observaciones (e_t)

# 1. Probabilidad Inicial P(X_0)
# P(Clima en día 0) = 50% Lluvia, 50% Sol
P_X0 = {
    'Lloviendo': 0.5,
    'Sol': 0.5
}

# 2. Modelo de Transición P(X_t | X_{t-1})
#    (Proceso Estacionario de Markov)
P_Xt_Xt_1 = {
    # Si ayer (X_{t-1}) fue 'Lloviendo':
    'Lloviendo': { 'Lloviendo': 0.7, 'Sol': 0.3 }, # P(Hoy=Lluvia|Ayer=Lluvia), P(Hoy=Sol|Ayer=Lluvia)
    # Si ayer (X_{t-1}) fue 'Sol':
    'Sol':       { 'Lloviendo': 0.3, 'Sol': 0.7 }  # P(Hoy=Lluvia|Ayer=Sol), P(Hoy=Sol|Ayer=Sol)
}

# 3. Modelo del Sensor (Emisión) P(e_t | X_t)
P_e_X = {
    # Si hoy (X_t) es 'Lloviendo':
    'Lloviendo': { 'Paraguas': 0.9, 'SinParaguas': 0.1 }, # P(Paraguas|Lluvia), P(SinParaguas|Lluvia)
    # Si hoy (X_t) es 'Sol':
    'Sol':       { 'Paraguas': 0.2, 'SinParaguas': 0.8 }  # P(Paraguas|Sol), P(SinParaguas|Sol)
}

# --- P2: Funciones Auxiliares (Normalización) ---

def normalizar(puntuaciones): # (Función que ya hemos usado)
    """ Normaliza un diccionario de {etiqueta: puntuacion} """
    total = sum(puntuaciones.values()) # Suma todas las puntuaciones
    if total == 0: return {e: 0.0 for e in puntuaciones} # Evita división por cero
    return {e: p / total for e, p in puntuaciones.items()} # Devuelve {etiqueta: prob}

# --- P3: Algoritmo Hacia ADELANTE (Forward) ---

def forward_pass(evidencia_seq, estados, P_X0, P_Xt_Xt_1, P_e_X):
    """
    Realiza la pasada hacia adelante (Filtrado).
    Calcula f_t = P(X_t, e_1:t) para toda la secuencia.
    """
    
    # Lista para almacenar *todos* los mensajes forward, uno por cada paso
    mensajes_f = []
    
    # --- Paso 0: Inicialización (t=0) ---
    # f_0 = P(X_0, e_0)
    # (Asumimos que e_0 es la primera evidencia, t=0)
    e_0 = evidencia_seq[0] # Primera evidencia (ej. 'Paraguas')
    f_0 = {} # Diccionario para el primer mensaje
    for s in estados: # 'Lloviendo', 'Sol'
        # f_0(s) = P(e_0 | X_0=s) * P(X_0=s)
        f_0[s] = P_e_X[s][e_0] * P_X0[s]
    
    mensajes_f.append(normalizar(f_0)) # Guardamos el mensaje (normalizado es P(X_0|e_0))
    
    # --- Paso t=1 en adelante: Recursión ---
    # f_t = alpha * P(e_t | X_t) * Sum[ P(X_t | X_{t-1}) * f_{t-1} ]
    
    # Iterar sobre el resto de la evidencia, desde t=1
    for t in range(1, len(evidencia_seq)):
        e_t = evidencia_seq[t] # Evidencia actual (ej. 'SinParaguas')
        f_t_anterior = mensajes_f[t-1] # Mensaje f_{t-1} que calculamos antes
        f_t_nuevo = {} # Diccionario para el mensaje f_t
        
        for s_t in estados: # Para cada estado actual (X_t = 'Lloviendo', 'Sol')
            
            # 1. Calcular la Sumatoria: Sum[ P(X_t | X_{t-1}) * f_{t-1} ]
            suma_transicion = 0.0
            for s_t_1 in estados: # Para cada estado anterior (X_{t-1} = 'Lloviendo', 'Sol')
                # P(X_t=s_t | X_{t-1}=s_t_1)
                prob_trans = P_Xt_Xt_1[s_t_1][s_t]
                # suma += P(Trans) * f_{t-1}(s_t_1)
                suma_transicion += prob_trans * f_t_anterior[s_t_1]
                
            # 2. Multiplicar por el Modelo del Sensor P(e_t | X_t)
            # f_t(s_t) = P(e_t | X_t=s_t) * (suma_transicion)
            f_t_nuevo[s_t] = P_e_X[s_t][e_t] * suma_transicion
            
        # 3. Guardar el nuevo mensaje (normalizado)
        mensajes_f.append(normalizar(f_t_nuevo))
        
    return mensajes_f # Devuelve la lista de *todos* los mensajes forward

# --- P4: Algoritmo Hacia ATRÁS (Backward) ---

def backward_pass(evidencia_seq, estados, P_X0, P_Xt_Xt_1, P_e_X):
    """
    Realiza la pasada hacia atrás.
    Calcula b_t = P(e_{t+1:T} | X_t) para toda la secuencia.
    """
    
    # Lista para almacenar los mensajes backward
    # (La inicializamos con el tamaño correcto)
    T = len(evidencia_seq) # Longitud total
    mensajes_b = [{} for _ in range(T)] # [{}, {}, {}, ...]
    
    # --- Paso T: Inicialización (t=T) ---
    # b_T se define como un vector de 1s (P(e_futura|X_T) = P(nada) = 1)
    b_T = {}
    for s in estados:
        b_T[s] = 1.0 # {'Lloviendo': 1.0, 'Sol': 1.0}
    mensajes_b[T-1] = b_T # Guardar en la *última* posición
    
    # --- Paso t=T-1 hacia atrás a 0: Recursión ---
    # b_t = Sum[ P(e_{t+1} | X_{t+1}) * P(X_{t+1} | X_t) * b_{t+1} ]
    
    # Iterar *hacia atrás*, desde T-2 hasta 0
    for t in range(T - 2, -1, -1):
        e_t_1 = evidencia_seq[t+1] # Evidencia *futura* (e_{t+1})
        b_t_1 = mensajes_b[t+1]    # Mensaje futuro (b_{t+1})
        b_t_nuevo = {} # Diccionario para el mensaje b_t
        
        for s_t in estados: # Para cada estado actual (X_t = 'Lloviendo', 'Sol')
            
            # 1. Calcular la Sumatoria:
            suma_ponderada = 0.0
            for s_t_1 in estados: # Para cada estado futuro (X_{t+1} = 'Lloviendo', 'Sol')
                # P(e_{t+1} | X_{t+1}=s_t_1)
                prob_sensor = P_e_X[s_t_1][e_t_1]
                # P(X_{t+1}=s_t_1 | X_t=s_t)
                prob_trans = P_Xt_Xt_1[s_t][s_t_1]
                # suma += P(Sensor) * P(Trans) * b_{t+1}(s_t_1)
                suma_ponderada += prob_sensor * prob_trans * b_t_1[s_t_1]
                
            b_t_nuevo[s_t] = suma_ponderada # b_t(s_t) = suma_ponderada
            
        # Guardar el nuevo mensaje (NO es necesario normalizar b)
        mensajes_b[t] = b_t_nuevo 
        
    return mensajes_b # Devuelve la lista de *todos* los mensajes backward

# --- P5: Algoritmo Hacia Delante-Atrás (Combinación) ---

def forward_backward_smoothing(evidencia_seq, estados, P_X0, P_Xt_Xt_1, P_e_X):
    """
    Resuelve la tarea de SUAVIZADO combinando f y b.
    Calcula P(X_k | e_1:T) para todos los k.
    """
    
    # 1. Ejecutar la pasada Forward (Filtro)
    mensajes_f = forward_pass(evidencia_seq, estados, P_X0, P_Xt_Xt_1, P_e_X)
    
    # 2. Ejecutar la pasada Backward
    mensajes_b = backward_pass(evidencia_seq, estados, P_X0, P_Xt_Xt_1, P_e_X)
    
    # 3. Combinar los mensajes para el Suavizado
    distribuciones_suavizadas = [] # Lista para P(X_k | e_1:T)
    T = len(evidencia_seq) # Longitud total
    
    for k in range(T): # Iterar sobre cada paso de tiempo k
        f_k = mensajes_f[k] # Mensaje f_k
        b_k = mensajes_b[k] # Mensaje b_k
        
        # P(X_k | e_1:T) = alpha * f_k * b_k
        prob_X_k = {}
        for s in estados:
            prob_X_k[s] = f_k[s] * b_k[s] # Multiplicación elemento a elemento
            
        # 4. Normalizar el resultado
        distribuciones_suavizadas.append(normalizar(prob_X_k))
        
    return distribuciones_suavizadas, mensajes_f # Devolver ambas listas

# --- P6: Ejecutar el cálculo ---
print("Algoritmo Hacia Delante-Atrás") # Título

# 1. Definir la secuencia de evidencia observada
#    Día 1: Paraguas, Día 2: Paraguas, Día 3: SinParaguas
evidencia_secuencia = ['Paraguas', 'Paraguas', 'SinParaguas']
T = len(evidencia_secuencia) # Longitud T=3

print(f"Secuencia de Evidencia (e_1:T): {evidencia_secuencia}")

# 2. Llamar al algoritmo
suavizado, filtrado = forward_backward_smoothing(
    evidencia_secuencia, estados, P_X0, P_Xt_Xt_1, P_e_X
)

# --- 3. Imprimir Resultados de FILTRADO ---
# (El filtrado es solo la salida de 'mensajes_f')
print("\n--- Tarea de FILTRADO P(X_t | e_1:t) ---")
print("  (Creencia *mientras* ocurren los eventos)")
for t in range(T):
    print(f"  t={t+1} (dado e_1:{t+1}): {filtrado[t]}")
    
# --- 4. Imprimir Resultados de SUAVIZADO ---
print("\n--- Tarea de SUAVIZADO P(X_k | e_1:T) ---")
print("  (Creencia *corregida* después de ver TODA la evidencia)")
for k in range(T):
    print(f"  k={k+1} (dado e_1:{T}): {suavizado[k]}")

print("\nConclusión:")
print(f"Filtrado (t=1): {filtrado[0]}")
print(f"Suavizado (k=1): {suavizado[0]}")
print("¡Nota la diferencia! Al filtrar en t=1, P(Lluvia) era alta (81%).")
print("Pero después de ver 'SinParaguas' en t=3, el Suavizado")
print("corrige esa creencia, bajando P(Lluvia) del día 1 a (58%).")