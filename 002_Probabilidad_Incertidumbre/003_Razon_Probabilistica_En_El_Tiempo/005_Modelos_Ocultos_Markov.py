# Algoritmo de MODELOS OCULTOS DE MARKOV

# Este algoritmo es el "Algoritmo de Viterbi".
# Es el segundo algoritmo principal para resolver un Modelo Oculto de Markov (HMM).
# El Problema:
# Mientras que 'Hacia Delante-Atrás' (#4) calcula la probabilidad en un *punto* específico del tiempo (ej. P(Clima_2 | e_1, e_2, e_3) = 0.58),
# Viterbi resuelve la tarea de "Explicación" (#3).
#
# Definición:
# Viterbi encuentra la *secuencia única* de estados ocultos (el "camino") que es *más probable* que haya generado la secuencia de evidencia que observamos.
#
#Pregunta: "¿Cuál fue el *camino completo* más probable?"
# Ej: Dado ('Paraguas', 'Paraguas', 'SinParaguas'),
#     ¿Qué es más probable?
#     - ('Lluvia', 'Lluvia', 'Sol')
#     - ('Sol', 'Lluvia', 'Sol')
#     - ('Lluvia', 'Lluvia', 'Lluvia') <-- (quizás el sensor falló)
# Viterbi encuentra cuál de estas secuencias tiene la prob. más alta.
#
# ¿Cómo funciona?
# Es un algoritmo de Programación Dinámica, casi *idéntico* al
# algoritmo "Hacia Delante", pero con una diferencia clave:
#
# 1. 'Hacia Delante' usa una SUMA (Suma[ P(X_t | X_{t-1}) * f_{t-1} ])
#    para sumar las probabilidades de *todas* las rutas posibles
#    que llegan a un estado.
#
# 2. 'Viterbi' usa un MAX (max[ P(X_t | X_{t-1}) * v_{t-1} ])
#    para guardar *únicamente* la probabilidad del *mejor camino*
#    (el más probable) que llega a un estado.
#
# 3. También almacena "punteros" (back-pointers) en cada paso para
#    recordar *qué* estado anterior llevó a ese camino "max".
#
# 4. Al final, sigue los punteros hacia atrás para reconstruir el camino.
#
# 
#
# Componentes (El mismo HMM de antes):
# 1. Estados, Evidencia, P_X0, P_Transición, P_Sensor.
# 2. Una tabla 'viterbi_t' para guardar la prob. del mejor camino (el 'max').
# 3. Una tabla 'punteros_t' para guardar la decisión tomada (el 'mejor_estado_anterior').
#
# Aplicaciones:
# - Reconocimiento del Habla (¡el siguiente tema de tu lista!).
# - Bioinformática (alineación de secuencias de ADN).
# - Autocorrección de texto (¿qué *secuencia de palabras* es más probable?).
#
# Ventajas:
# - Es eficiente (programación dinámica, O(T * |S|^2), donde T es
#   el tiempo y S es el número de estados).
# - Da la respuesta *exacta* al problema de "Explicación".
#
# Desventajas:
# - No da probabilidades (como P(Lluvia)=0.58). Solo da *la mejor* secuencia.

import math 
import copy 

# --- P1: Definición del Modelo Oculto de Markov (HMM) ---
# (Usamos el mismo modelo "Paraguas" del algoritmo anterior)

# 0. Definición de Estados y Evidencia
estados = ('Lloviendo', 'Sol') # Estados ocultos (X_t)
# (Evidencia se define en la secuencia)

# 1. Probabilidad Inicial P(X_0)
P_X0 = {
    'Lloviendo': 0.5,
    'Sol': 0.5
}

# 2. Modelo de Transición P(X_t | X_{t-1})
P_Xt_Xt_1 = {
    'Lloviendo': { 'Lloviendo': 0.7, 'Sol': 0.3 }, # P(Hoy|Ayer=Lluvia)
    'Sol':       { 'Lloviendo': 0.3, 'Sol': 0.7 }  # P(Hoy|Ayer=Sol)
}

# 3. Modelo del Sensor (Emisión) P(e_t | X_t)
P_e_X = {
    'Lloviendo': { 'Paraguas': 0.9, 'SinParaguas': 0.1 }, # P(Evidencia|Lluvia)
    'Sol':       { 'Paraguas': 0.2, 'SinParaguas': 0.8 }  # P(Evidencia|Sol)
}

# --- P2: Algoritmo de Viterbi ---

def algoritmo_viterbi(evidencia_seq, estados, P_X0, P_Xt_Xt_1, P_e_X):
    """
    Encuentra el camino (secuencia) más probable usando Viterbi.
    """
    
    # --- 1. Inicialización (t=0) ---
    
    # 'viterbi' es una lista de diccionarios.
    # viterbi[t][s] = Probabilidad del camino más probable que
    #                 termina en el estado 's' en el tiempo 't'.
    viterbi = [{}] # viterbi[0]
    
    # 'punteros' almacena el "mejor estado anterior"
    punteros = [{}] # punteros[0]
    
    e_0 = evidencia_seq[0] # Primera evidencia (ej. 'Paraguas')
    
    for s in estados: # 'Lloviendo', 'Sol'
        # Prob(Camino en t=0) = P(e_0 | X_0=s) * P(X_0=s)
        prob_e0 = P_e_X[s][e_0] # Prob. del sensor
        prob_x0 = P_X0[s]         # Prob. inicial
        viterbi[0][s] = prob_e0 * prob_x0 # Guardar la prob. del camino
        punteros[0][s] = None # No hay estado anterior en t=0
        
    # --- 2. Paso Recursivo (t=1 hasta T) ---
    
    T = len(evidencia_seq) # Longitud total de la evidencia
    
    for t in range(1, T): # Iterar sobre el resto del tiempo
        viterbi.append({}) # Añadir un nuevo dict vacío para viterbi[t]
        punteros.append({}) # Añadir un nuevo dict vacío para punteros[t]
        
        e_t = evidencia_seq[t] # Evidencia actual (ej. 'SinParaguas')
        
        # Para cada estado *actual* posible s_t ('Lloviendo', 'Sol')
        for s_t in estados: 
            
            # (max_prob, mejor_estado_anterior) = 
            #   max_{s_t_1} [ P(X_t=s_t | X_{t-1}=s_t_1) * viterbi[t-1][s_t_1] ]
            (max_prob, mejor_estado_anterior) = (0.0, None)
            
            # Iterar sobre cada estado *anterior* posible s_t_1
            for s_t_1 in estados:
                # Obtener P(Transición)
                prob_trans = P_Xt_Xt_1[s_t_1][s_t] # P(s_t | s_t_1)
                # Obtener la prob. del *mejor camino anterior*
                prob_camino_anterior = viterbi[t-1][s_t_1] # v_{t-1}(s_t_1)
                
                # Prob. de *este* camino (pasando por s_t_1)
                prob_total = prob_camino_anterior * prob_trans
                
                # Si este camino es mejor que el máximo que hemos visto...
                if prob_total > max_prob:
                    max_prob = prob_total # ...actualizar el máximo
                    mejor_estado_anterior = s_t_1 # ...recordar de dónde venía
            
            # Ahora que tenemos el 'max', multiplicamos por P(Sensor)
            # v_t(s_t) = P(e_t | X_t=s_t) * max_prob
            prob_sensor = P_e_X[s_t][e_t]
            viterbi[t][s_t] = prob_sensor * max_prob
            
            # Guardar el puntero
            punteros[t][s_t] = mejor_estado_anterior
            
    # --- 3. Terminación (Encontrar el camino) ---
    
    # 3a. Encontrar la probabilidad del *mejor camino de todos*
    # (Mirando la prob. más alta en el último paso de tiempo)
    prob_camino_total = max(viterbi[T-1].values())
    
    # 3b. Encontrar el *nombre* del estado final en ese mejor camino
    #     (usando .get como clave para la función max)
    mejor_estado_final = max(viterbi[T-1], key=viterbi[T-1].get)
    
    # 3c. Reconstruir el camino (Backtracking)
    camino_mas_probable = [] # Lista para guardar el camino
    camino_mas_probable.append(mejor_estado_final) # Añadir el último estado
    
    estado_actual = mejor_estado_final # Empezar desde el final
    
    # Iterar *hacia atrás* desde T-1 hasta 1
    for t in range(T - 1, 0, -1):
        # Obtener el estado anterior siguiendo el puntero
        estado_anterior = punteros[t][estado_actual]
        # Insertar el estado anterior al *principio* de la lista
        camino_mas_probable.insert(0, estado_anterior)
        # Movernos un paso atrás
        estado_actual = estado_anterior
        
    # Devolver el camino y su probabilidad
    return camino_mas_probable, prob_camino_total

# --- P3: Ejecutar el cálculo ---
print("Algoritmo de Modelos Ocultos de Markov") # Título

# 1. Definir la secuencia de evidencia observada
#    Día 1: Paraguas, Día 2: Paraguas, Día 3: SinParaguas
evidencia_secuencia = ['Paraguas', 'Paraguas', 'SinParaguas']
T = len(evidencia_secuencia) # Longitud T=3

print(f"Secuencia de Evidencia (e_1:T): {evidencia_secuencia}")
print("Calculando el camino más probable...")

# 2. Llamar al algoritmo
camino, prob_camino = algoritmo_viterbi(
    evidencia_secuencia, estados, P_X0, P_Xt_Xt_1, P_e_X
)

# --- 3. Imprimir el Resultado ---
print("\n--- Resultado (Explicación Más Probable) ---")
print(f"Secuencia de estados ocultos: {camino}")
print(f"Probabilidad de esa secuencia: {prob_camino:.6f} (o {prob_camino:e})")

# Desglose de la respuesta:
# ('Lluvia', 'Lluvia', 'Sol')
# t=0 (Lluvia): P(Lluvia_0)*P(Paraguas|Lluvia_0) = 0.5 * 0.9 = 0.45
# t=1 (Lluvia): P(Paraguas|Lluvia_1) * max[ P(L_1|L_0)*v_0(L_0), P(L_1|S_0)*v_0(S_0) ]
#               P(Paraguas|Lluvia_1) * max[ 0.7*0.45, 0.3*(0.5*0.2) ]
#               0.9 * max[ 0.315, 0.03 ] = 0.9 * 0.315 = 0.2835
# t=2 (Sol):    P(SinParaguas|Sol_2) * max[ P(S_2|L_1)*v_1(L_1), P(S_2|S_1)*v_1(S_1) ]
#               0.8 * max[ 0.3*0.2835, 0.7*... ] = 0.8 * 0.08505 = 0.06804
# (Los números exactos dependen de todos los cálculos, pero el camino es el punto)

print("\nConclusión:")
print("El algoritmo Hacia Delante-Atrás nos dijo P(Lluvia_3) = 0.17 (prob. puntual).")
print("Viterbi nos da el *camino completo*: el mundo *más probable*")
print("es uno donde llovió los primeros 2 días y paró el tercero,")
print(f"lo cual explica perfectamente la evidencia.")