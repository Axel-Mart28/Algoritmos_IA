# --- 8. RECONOCIMIENTO DEL HABLA (Aplicación de HMMs) ---

# Este "algoritmo" es en realidad una *aplicación* que combina un Modelo Oculto de Markov (HMM) con el Algoritmo de Viterbi.
#
# Definición:
# Es el proceso de tomar una señal de audio (la evidencia) y encontrar la secuencia de palabras (los estados ocultos) que es más probable que haya generado ese audio.
#
# 💡 Es *exactamente* la tarea de "Explicación":
# P(Palabras | Audio) -> Queremos encontrar la *secuencia* de
# palabras más probable que explique la *secuencia* de audio.
#
# ¿Cómo funciona? (El Modelo HMM):
# 1. Estados Ocultos (X_t): Las *palabras* (o fonemas)
#    que el hablante *intenta* decir (ej. 'Hola', 'Mundo').
#    No podemos verlas directamente; están ocultas en el cerebro
#    del hablante.
#
# 2. Evidencia (e_t): Los "cepstral coefficients" (MFCCs).
#    Son un conjunto de números (un vector) que describen
#    el "timbre" del audio en un pequeño fragmento (ej. 10ms).
#    Esta es la *señal acústica* que observamos.
#
# 3. Modelo de Transición P(X_t | X_{t-1}) (El "Modelo de Lenguaje"):
#    - ¿Cuál es la probabilidad de decir "Mundo" después
#      de haber dicho "Hola"? (¡Alta!)
#    - ¿Cuál es la probabilidad de decir "Queso" después
#      de haber dicho "Hola"? (¡Baja!)
#    - Esto le da "gramática" y "contexto" al modelo.
#
# 4. Modelo del Sensor P(e_t | X_t) (El "Modelo Acústico"):
#    - ¿Cómo "suena" la palabra "Hola"?
#    - Dada la palabra "Hola" (estado oculto), ¿cuál es la
#      probabilidad de observar este vector de audio (evidencia)?
#    - Este modelo captura el acento, el ruido de fondo, etc.
#
# ¿Cómo funciona este programa?
# Vamos a crear una versión *extremadamente* simplificada de esto.
# - Estados (Palabras): 'Hola', 'Mundo'
# - Evidencia (Audio): 'sonido_O', 'sonido_A', 'sonido_UN', 'sonido_DO'
#
# Usaremos el Algoritmo de Viterbi para encontrar
# la secuencia de palabras más probable (el "camino") que explique la secuencia de sonidos observada.

import math 
import copy 

# --- P1: Definición del HMM (Modelo de Habla Simplificado) ---

# 1. Estados Ocultos (Las palabras que queremos reconocer)
estados = ('Hola', 'Mundo')

# (Evidencia se define en la secuencia)

# 2. Probabilidad Inicial P(X_0)
#    (Con qué palabra es más probable empezar una frase)
P_X0 = {
    'Hola': 0.8, # 80% de prob. de empezar con 'Hola'
    'Mundo': 0.2  # 20% de prob. de empezar con 'Mundo'
}

# 3. Modelo de Transición P(X_t | X_{t-1}) (Modelo de Lenguaje)
#    (Qué palabra sigue a qué palabra)
P_Xt_Xt_1 = {
    # Si la palabra anterior (X_{t-1}) fue 'Hola':
    'Hola': { 'Hola': 0.1, 'Mundo': 0.9 }, # P(Hoy=Hola|Ayer=Hola), P(Hoy=Mundo|Ayer=Hola)
    # Si la palabra anterior (X_{t-1}) fue 'Mundo':
    'Mundo': { 'Hola': 0.0, 'Mundo': 1.0 }  # (No se puede decir 'Hola' después de 'Mundo',
                                          #  y 'Mundo' tiende a repetirse)
}

# 4. Modelo del Sensor P(e_t | X_t) (Modelo Acústico)
#    (Cómo "suena" cada palabra)
P_e_X = {
    # Si la palabra (X_t) es 'Hola':
    'Hola': { 'sonido_O': 0.7, 'sonido_A': 0.2, 'sonido_UN': 0.0, 'sonido_DO': 0.1 },
    # Si la palabra (X_t) es 'Mundo':
    'Mundo': { 'sonido_O': 0.1, 'sonido_A': 0.0, 'sonido_UN': 0.5, 'sonido_DO': 0.4 }
}

# --- P2: Algoritmo de Viterbi ---
# (Copiamos la *misma* función del tema #5, no cambia nada)

def algoritmo_viterbi(evidencia_seq, estados, P_X0, P_Xt_Xt_1, P_e_X):
    """
    Encuentra el camino (secuencia de palabras) más probable.
    (Código idéntico al del tema #5)
    """
    viterbi = [{}] # viterbi[0]
    punteros = [{}] # punteros[0]
    e_0 = evidencia_seq[0] # Primer sonido
    
    # 1. Inicialización (t=0)
    for s in estados: # 'Hola', 'Mundo'
        prob_e0 = P_e_X[s][e_0] # P(sonido_0 | Palabra_0)
        prob_x0 = P_X0[s]         # P(Palabra_0)
        viterbi[0][s] = prob_e0 * prob_x0
        punteros[0][s] = None
        
    # 2. Paso Recursivo (t=1 hasta T)
    T = len(evidencia_seq) # Longitud de la secuencia de audio
    for t in range(1, T):
        viterbi.append({})
        punteros.append({})
        e_t = evidencia_seq[t] # Sonido actual
        
        for s_t in estados: # Para cada palabra *actual*
            (max_prob, mejor_estado_anterior) = (0.0, None)
            
            for s_t_1 in estados: # Para cada palabra *anterior*
                prob_trans = P_Xt_Xt_1[s_t_1][s_t] # P(Palabra_t | Palabra_{t-1})
                prob_camino_anterior = viterbi[t-1][s_t_1]
                prob_total = prob_camino_anterior * prob_trans
                
                if prob_total > max_prob: # Encontrar el MAX
                    max_prob = prob_total
                    mejor_estado_anterior = s_t_1
            
            prob_sensor = P_e_X[s_t][e_t] # P(Sonido_t | Palabra_t)
            viterbi[t][s_t] = prob_sensor * max_prob
            punteros[t][s_t] = mejor_estado_anterior
            
    # 3. Terminación (Reconstruir el camino)
    prob_camino_total = max(viterbi[T-1].values())
    mejor_estado_final = max(viterbi[T-1], key=viterbi[T-1].get)
    camino_mas_probable = []
    camino_mas_probable.append(mejor_estado_final)
    estado_actual = mejor_estado_final
    
    for t in range(T - 1, 0, -1):
        estado_anterior = punteros[t][estado_actual]
        camino_mas_probable.insert(0, estado_anterior)
        estado_actual = estado_anterior
        
    return camino_mas_probable, prob_camino_total

# --- P3: Ejecutar el "Reconocimiento" ---
print("---Reconocimiento del Habla ---") # Título

# 1. Definir la secuencia de EVIDENCIA (audio) que observamos
#    Imaginemos que el audio que entra es:
#    "O-A-UN-DO"
#    (El sonido 'A' es un error/ruido, 'Hola' no suena así,
#     pero 'sonido_O' (de Hola) es 0.7 y 'sonido_O' (de Mundo) es 0.1,
#     así que Viterbi debería detectarlo)
evidencia_audio = ['sonido_O', 'sonido_A', 'sonido_UN', 'sonido_DO']

print(f"Secuencia de audio observada (Evidencia): {evidencia_audio}")
print("Calculando la secuencia de palabras más probable...")

# 2. Llamar al algoritmo de Viterbi
secuencia_palabras, prob_secuencia = algoritmo_viterbi(
    evidencia_audio, estados, P_X0, P_Xt_Xt_1, P_e_X
)

# --- 3. Imprimir el Resultado ---
print("\n--- Resultado del Reconocimiento ---")
print(f"Secuencia de palabras reconocida: {secuencia_palabras}")
print(f"(Probabilidad del camino: {prob_secuencia:e})")

# Desglose de la respuesta esperada: ['Hola', 'Hola', 'Mundo', 'Mundo']
# ¿Por qué ['Hola', 'Hola', ...]?
# El Modelo Acústico P('sonido_A' | 'Hola') es 0.2
# El Modelo Acústico P('sonido_A' | 'Mundo') es 0.0 (imposible)
# El Modelo de Lenguaje P('Mundo' | 'Hola') es 0.9 (muy probable)
# Viterbi debe decidir si el 0.0 de 'Mundo' es peor que
# el bajo 0.2 de 'Hola' y el bajo 0.1 de P('Hola'|'Hola').
# El algoritmo encuentra el balance óptimo entre el
# "Modelo Acústico" (cómo suena) y el "Modelo de Lenguaje" (la gramática).
# (Dependiendo de los números, podría ser ['Hola', 'Mundo', 'Mundo', 'Mundo'])

# Vamos a probar otra secuencia
evidencia_audio_2 = ['sonido_O', 'sonido_UN', 'sonido_DO', 'sonido_DO']
print(f"\nNueva secuencia de audio: {evidencia_audio_2}")
secuencia_palabras_2, _ = algoritmo_viterbi(
    evidencia_audio_2, estados, P_X0, P_Xt_Xt_1, P_e_X
)
print(f"Secuencia de palabras reconocida: {secuencia_palabras_2}")
# (Resultado esperado: ['Hola', 'Mundo', 'Mundo', 'Mundo'])

print("\nConclusión:")
print("El Algoritmo de Viterbi encontró la *secuencia de palabras* (estados ocultos)")
print("que *mejor explica* la *secuencia de sonidos* (evidencia),")
print("balanceando la 'probabilidad acústica' (sensor) y la 'gramática' (transición).")