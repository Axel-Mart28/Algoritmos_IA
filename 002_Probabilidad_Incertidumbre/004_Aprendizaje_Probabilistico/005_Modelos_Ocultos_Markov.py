# ALGORITMO DE MODELOS OCULTOS DE MARKOV 

# Este algoritmo es sobre cómo *aprender* los parámetros de un HMM
# (P_Inicial, P_Transición, P_Emisión) cuando *no* los conocemos,
# pero sí tenemos secuencias de *observaciones* (evidencia).
#
# Es un problema de Aprendizaje No Supervisado porque no sabemos
# cuál era la secuencia de estados "correcta" que generó la evidencia.
#
# Definición (El Problema):
# Dada una o más secuencias de evidencia (ej. ['Paraguas', 'Paraguas', 'SinParaguas']),
# encontrar los parámetros (P_X0, P_Xt_Xt_1, P_e_X) del HMM que
# *mejor explican* (maximizan la probabilidad de) esas secuencias.
#
# El Algoritmo (Baum-Welch, un caso de EM):
# Es un algoritmo *iterativo* que mejora gradualmente una estimación
# inicial de los parámetros. Consta de dos pasos que se repiten:
#
# 1. PASO E (Expectation - Expectativa):
#    - Usa los parámetros *actuales* del HMM (la estimación actual)
#      y el algoritmo "Hacia Delante-Atrás" (tema #4 de la sección anterior)
#      para calcular las "responsabilidades":
#      - gamma_t(i): La probabilidad esperada de estar en el estado oculto 'i'
#                    en el tiempo 't', dados los datos y los parámetros actuales.
#                    (Es el resultado del 'Suavizado').
#      - xi_t(i, j): La probabilidad esperada de transitar del estado 'i'
#                    al estado 'j' entre el tiempo 't' y 't+1', dados los
#                    datos y los parámetros actuales.
#    - Estas son nuestras "expectativas" sobre los estados ocultos.
#
# 2. PASO M (Maximization - Maximización):
#    - Re-estima los parámetros del HMM para *maximizar* la probabilidad
#      de las expectativas calculadas en el Paso E.
#    - Es como "contar" eventos, pero usando las probabilidades esperadas:
#      - Nueva P_X0(i) = gamma_1(i) (Prob. esperada de empezar en 'i').
#      - Nueva P_Xt_Xt_1(i -> j) = (Suma_t xi_t(i, j)) / (Suma_t gamma_t(i))
#                                  (Frecuencia esperada de i->j / Frecuencia esperada de salir de i).
#      - Nueva P_e_X(k | j) = (Suma_{t donde e_t=k} gamma_t(j)) / (Suma_t gamma_t(j))
#                             (Frecuencia esperada de estar en j y ver k / Frecuencia esperada de estar en j).
#
# 3. REPETIR: Se repiten los pasos E y M hasta que los parámetros
#    dejen de cambiar significativamente (convergencia).
#
# 
#
# Componentes:
# 1. Una (o más) secuencias de evidencia.
# 2. Una *estimación inicial* de los parámetros del HMM (puede ser aleatoria).
# 3. El algoritmo Hacia Delante-Atrás (para el Paso E).
#
# Aplicaciones:
# - ¡Aprender los modelos para Reconocimiento de Habla! (tema #8 anterior).
# - Bioinformática (modelar secuencias de ADN).
# - Finanzas (modelar el comportamiento del mercado).
#
# Ventajas:
# - Es el método estándar para aprender HMMs.
# - Garantiza mejorar (o mantener) la probabilidad de los datos en cada iteración.
#
# Desventajas:
# - Puede converger a un "óptimo local" (una solución "buena" pero no
#   necesariamente la "mejor posible"), especialmente si la
#   inicialización es mala.
# - Es computacionalmente costoso (requiere ejecutar Hacia Delante-Atrás
#   en cada iteración).
#
# Ejemplo de uso:
# - Le damos al algoritmo ['P', 'P', 'S'] y parámetros iniciales aleatorios.
# - Después de varias iteraciones E-M, debería *aprender* algo similar
#   a las tablas P_Transición y P_Emisión que usamos en el ejemplo del Paraguas.


import math
import copy
from collections import defaultdict
import random # <--- ¡LA LÍNEA QUE FALTABA!

# --- P1: Funciones Auxiliares (Asumimos que existen) ---

# (Necesitaríamos las funciones 'forward_pass', 'backward_pass' y 'normalizar'
#  definidas en el algoritmo #4. Las omitimos aquí por brevedad,
#  pero serían necesarias para una ejecución real.)

# (Función de normalización del tema #3b de Probabilidad)
def normalizar(puntuaciones):
    """ Normaliza un diccionario de {etiqueta: puntuacion} """
    total = sum(puntuaciones.values()) # Suma todas las puntuaciones
    if total == 0:
        num_items = len(puntuaciones)
        if num_items > 0:
            return {e: 1.0 / num_items for e in puntuaciones}
        else:
            return {}
    return {e: p / total for e, p in puntuaciones.items()} # Devuelve {etiqueta: prob}


def calcular_gamma_y_xi(evidencia_seq, estados, P_X0, P_Xt_Xt_1, P_e_X):
    """
    Simula el PASO E: Calcula las probabilidades gamma y xi.
    (Esta función usaría 'forward_pass' y 'backward_pass' internamente).
    """
    print("    (Simulando Paso E: Calculando gamma y xi con Forward-Backward...)") # Mensaje cambiado
    T = len(evidencia_seq) # Longitud de la secuencia

    # --- SIMULACIÓN de resultados de Forward-Backward ---
    # (En un código real, aquí irían las llamadas a las funciones F-B)
    # Generamos valores 'falsos' pero con la estructura correcta
    mensajes_f = [{s: random.random() for s in estados} for _ in range(T)]
    mensajes_b = [{s: random.random() for s in estados} for _ in range(T)]
    # Normalizar f (porque F-B lo devuelve normalizado)
    mensajes_f = [normalizar(f) for f in mensajes_f]
    # --- FIN DE SIMULACIÓN ---

    # 2. Calcular gamma_t(i) = P(X_t=i | e_1:T)
    gamma = [{} for _ in range(T)] # Lista para guardar gamma[t][estado]
    for t in range(T):
        prob_no_norm = {s: mensajes_f[t][s] * mensajes_b[t][s] for s in estados}
        gamma[t] = normalizar(prob_no_norm)

    # 3. Calcular xi_t(i, j) = P(X_t=i, X_{t+1}=j | e_1:T)
    xi = [defaultdict(dict) for _ in range(T - 1)] # Lista para xi[t][estado_i][estado_j]
    for t in range(T - 1):
        prob_no_norm_t = {} # Puntuaciones no normalizadas para este tiempo t
        e_t_1 = evidencia_seq[t+1] # Evidencia en t+1

        for i in estados: # Para cada estado i en t
            for j in estados: # Para cada estado j en t+1
                f_i = mensajes_f[t][i]
                # Manejar KeyError si la transición no está definida (debería estarlo)
                trans_ij = P_Xt_Xt_1.get(i, {}).get(j, 0.0)
                # Manejar KeyError si la emisión no está definida (debería estarlo)
                emis_j = P_e_X.get(j, {}).get(e_t_1, 0.0)
                b_j = mensajes_b[t+1][j]

                puntuacion_ij = f_i * trans_ij * emis_j * b_j
                prob_no_norm_t[(i, j)] = puntuacion_ij # Guardar

        # Normalizar todas las puntuaciones (i,j) para este tiempo t
        prob_norm_t = normalizar(prob_no_norm_t)

        # Guardar en la estructura xi
        for (i, j), prob in prob_norm_t.items():
             # Asegurarse de que las claves existen antes de asignar
            if i not in xi[t]:
                xi[t][i] = {}
            xi[t][i][j] = prob

    return gamma, xi # Devolver las expectativas calculadas

# --- P2: Algoritmo de Baum-Welch (El Bucle EM) ---
# (El resto del código es igual, pero ahora la llamada a random.random() funcionará)

def baum_welch_aprendizaje(evidencia_seq, estados, n_iteraciones):
    """
    Aprende los parámetros de un HMM usando Baum-Welch (EM).
    """
    T = len(evidencia_seq) # Longitud de la secuencia
    n_estados = len(estados) # Número de estados

    # --- 1. Inicialización Aleatoria (o con conocimiento previo) ---
    print("Inicializando parámetros del HMM aleatoriamente...")
    # P_X0: Probabilidades iniciales (deben sumar 1)
    rand_x0 = [random.random() for _ in estados] # ¡Ahora funciona!
    P_X0 = normalizar(dict(zip(estados, rand_x0)))

    # P_Xt_Xt_1: Matriz de transición (cada fila debe sumar 1)
    P_Xt_Xt_1 = {}
    for i in estados:
        rand_trans = [random.random() for _ in estados] # ¡Ahora funciona!
        P_Xt_Xt_1[i] = normalizar(dict(zip(estados, rand_trans)))

    # P_e_X: Matriz de emisión (cada fila debe sumar 1)
    posibles_evidencias = set(evidencia_seq)
    P_e_X = {}
    for i in estados:
        rand_emis = [random.random() for _ in posibles_evidencias] # ¡Ahora funciona!
        P_e_X[i] = normalizar(dict(zip(posibles_evidencias, rand_emis)))

    print("Parámetros iniciales (ejemplo):")
    print(f" P_X0: {P_X0}")
    print(f" P_Xt_Xt_1['Lloviendo']: {P_Xt_Xt_1.get('Lloviendo', {})}")
    print(f" P_e_X['Lloviendo']: {P_e_X.get('Lloviendo', {})}")

    # --- 2. Bucle Iterativo E-M ---
    print(f"\nIniciando {n_iteraciones} iteraciones de Baum-Welch (EM)...")
    for iter_num in range(n_iteraciones):
        print(f"  Iteración {iter_num + 1}:")

        # --- PASO E: Calcular Expectativas ---
        gamma, xi = calcular_gamma_y_xi(evidencia_seq, estados, P_X0, P_Xt_Xt_1, P_e_X)

        # --- PASO M: Maximizar (Re-estimar Parámetros) ---
        print("    (Ejecutando Paso M: Re-estimando parámetros...)")

        # 1. Re-estimar P_X0
        P_X0 = gamma[0]

        # 2. Re-estimar P_Xt_Xt_1
        suma_gamma_i = {s: 0.0 for s in estados}
        suma_xi_ij = defaultdict(lambda: defaultdict(float))

        for t in range(T - 1):
            for i in estados:
                suma_gamma_i[i] += gamma[t][i]
                for j in estados:
                    # Acceder a xi de forma segura
                    suma_xi_ij[i][j] += xi[t].get(i, {}).get(j, 0.0)


        for i in estados:
            if suma_gamma_i[i] == 0: continue
            for j in estados:
                 # Asegurarse de que la clave i existe
                if i not in P_Xt_Xt_1: P_Xt_Xt_1[i] = {}
                P_Xt_Xt_1[i][j] = suma_xi_ij[i][j] / suma_gamma_i[i]

        # 3. Re-estimar P_e_X
        suma_gamma_j = {s: sum(gamma[t][s] for t in range(T)) for s in estados}
        suma_gamma_j_y_k = defaultdict(lambda: defaultdict(float))

        for t in range(T):
            e_t = evidencia_seq[t]
            for j in estados:
                suma_gamma_j_y_k[j][e_t] += gamma[t][j]

        for j in estados:
            if suma_gamma_j[j] == 0: continue
            for k in posibles_evidencias:
                # Asegurarse de que la clave j existe
                if j not in P_e_X: P_e_X[j] = {}
                P_e_X[j][k] = suma_gamma_j_y_k[j][k] / suma_gamma_j[j]

        if (iter_num + 1) % 5 == 0:
             print(f"    P_Xt_Xt_1['Lloviendo'] después de iter {iter_num+1}: {P_Xt_Xt_1.get('Lloviendo', {})}")


    print("¡Aprendizaje completado!")
    return P_X0, P_Xt_Xt_1, P_e_X

# --- P3: Ejecutar el Aprendizaje ---
# (El resto del código es igual)

evidencia_secuencia = ['Paraguas', 'Paraguas', 'SinParaguas', 'Paraguas'] * 10
estados = ('Lloviendo', 'Sol')
N_ITER = 20

print("Modelos de Markov Ocultos")
print(f"Secuencia de Evidencia (primeros 10): {evidencia_secuencia[:10]}...")
print(f"Longitud total: {len(evidencia_secuencia)}")

P_X0_aprendido, P_Trans_aprendido, P_Emis_aprendido = baum_welch_aprendizaje(
    evidencia_secuencia, estados, N_ITER
)

print("\n--- Parámetros Aprendidos del HMM ---")
print(f"P(Inicial) P_X0: {P_X0_aprendido}")
print("\nP(Transición) P_Xt_Xt_1:")
for s, dist in P_Trans_aprendido.items():
    print(f"  Desde '{s}': {dist}")
print("\nP(Emisión) P_e_X:")
for s, dist in P_Emis_aprendido.items():
    print(f"  Desde '{s}': {dist}")