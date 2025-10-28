# Algoritmo de MONTE CARLO PARA CADENAS DE MARKOV (MCMC)

# Este es un algoritmo de inferencia aproximada, como los de muestreo anteriores, pero funciona de manera muy diferente.

# Definición:
# MCMC es un algoritmo que estima P(X|e) generando *una sola muestra* (un "estado") y luego "haciéndola pasear" aleatoriamente por el espacio de estados, modificándola paso a paso.

# El Problema de los Muestreadores Anteriores:
# - Rechazo: Desperdicia >99% de las muestras si la evidencia 'e' es rara.
# - Ponderación: Genera muestras inútiles (con peso ~0) si la evidencia 'e' está "al final" de la red.
#
# ¿Cómo funciona (Muestreo de Gibbs)?:
# 1. Inicia un "estado actual": un evento completo (ej. {R:T, T:F, A:T...}).
# 2. IMPORTANTE: "Fija" las variables de evidencia a sus valores (ej. 'JuanLlama' *siempre* será True en nuestro estado).
# 3. Repite N veces (ej. 100,000 iteraciones):
# 4.   Elige una variable *oculta* Z al azar (ej. 'Alarma').
# 5.   "Remuestrea" esa variable Z:
# 6.   Calcula la probabilidad P(Z | "todo lo demás").
# 7.   CLAVE: Gracias al "Manto de Markov" (tema #3), P(Z | "todo lo demás") = P(Z | Manto_Markov(Z))
# 8.   Calculamos P(Z | Manto_Markov(Z)) y elegimos un nuevo  valor para Z (ej. 'Alarma' cambia de T a F).
# 9.   El "estado actual" ha sido modificado. Volvemos al paso 3.

# ¿Cómo responde la consulta P(X|e)?
# - El algoritmo tiene un "período de calentamiento" (burn-in), donde la cadena "pasea" para encontrar una zona probable.
# - Después del burn-in, empezamos a *contar*.
# - Si la consulta es P(Robo|e), contamos cuántas iteraciones la variable 'Robo' estuvo en True vs. False.
# - P(Robo=T|e) ~= (Conteo de Robo=T) / (Conteo Total)
#
# Componentes:
# 1. La Red Bayesiana y la Evidencia 'e'.
# 2. Un "estado" (muestra) que se modifica en cada paso.
# 3. La función para encontrar el Manto de Markov (tema #3).
# 4. La función de "remuestreo" (el corazón de Gibbs).
#
# Ventajas:
# - Extremadamente potente. Es el método de inferencia *de facto*
#   para problemas probabilísticos muy grandes y complejos.
# - Evita los problemas del Muestreo por Ponderación.
#
# Desventajas:
# - Es lento. Necesita un "burn-in" y muchas iteraciones.
# - Las muestras están *autocorrelacionadas* (cada muestra es muy similar a la anterior), lo que las hace menos eficientes que las muestras independientes.
# - Puede "atascarse" en un óptimo local (una zona probable pero no la más probable).
#
# Ejemplo de uso:
# Estimar P(Robo | JuanLlama=True, MariaLlama=True).

import random # Para elegir variables y muestrear
import math   # Para normalizar
from collections import defaultdict # Para Manto de Markov

# --- P1: Definición de la Red y TODAS las funciones auxiliares ---
# (Esta vez incluimos TODAS las dependencias para evitar errores)

red_alarma = {
    'Robo': {'parents': [], 'cpt': {(): 0.001}},
    'Terremoto': {'parents': [], 'cpt': {(): 0.002}},
    'Alarma': {
        'parents': ['Robo', 'Terremoto'],
        'cpt': {
            (True, True): 0.95, (True, False): 0.94,
            (False, True): 0.29, (False, False): 0.001
        }
    },
    'JuanLlama': {
        'parents': ['Alarma'],
        'cpt': {(True,): 0.90, (False,): 0.05}
    },
    'MariaLlama': {
        'parents': ['Alarma'],
        'cpt': {(True,): 0.70, (False,): 0.01}
    }
}

def normalizar(puntuaciones):
    """ Normaliza un diccionario de {etiqueta: puntuacion} """
    total = sum(puntuaciones.values())
    if total == 0:
        num = len(puntuaciones)
        return {e: 1.0/num if num > 0 else 0.0 for e in puntuaciones}
    return {e: p / total for e, p in puntuaciones.items()}

def get_prob_cpt(red, variable, valor, evidencia):
    """ Obtiene P(variable=valor | evidencia) de la CPT """
    nodo = red[variable]
    padres = nodo['parents']
    if not padres:
        clave_cpt = ()
    else:
        # Crea la tupla de clave (ej. (True, False))
        if not all(p in evidencia for p in padres):
             # Esto puede pasar durante la inicialización
             return 0.5 # Devolver un valor neutral
        clave_cpt = tuple([evidencia[padre] for padre in padres])
    prob_true = nodo['cpt'][clave_cpt]
    return prob_true if valor == True else (1.0 - prob_true)

# (Función del tema #3)
def encontrar_manto_markov(red, nodo_X):
    """ Encuentra el Manto de Markov de 'nodo_X'. """
    manto = set()
    # 1. Padres
    manto.update(red[nodo_X]['parents'])
    # 2. Hijos
    hijos = set()
    for v_name, v_info in red.items():
        if nodo_X in v_info['parents']:
            hijos.add(v_name)
    manto.update(hijos)
    # 3. Co-Padres
    for h in hijos:
        manto.update(red[h]['parents'])
    # Limpieza
    if nodo_X in manto:
        manto.remove(nodo_X)
    return manto

# --- P2: Algoritmo de Muestreo de Gibbs (el "corazón") ---

def _remuestrear_variable(var_Z, current_state, red):
    """
    Calcula P(Z | Manto_Markov(Z)) y devuelve un nuevo
    valor (True/False) muestreado de esa distribución.
    """
    
    # La fórmula es: P(Z | Manto) = alpha * P(Z | Padres(Z)) * Producto[ P(Hijo | Padres(Hijo)) ]
    
    # 1. Obtener los hijos de Z (necesarios para la fórmula)
    hijos_de_Z = set()
    for v_name, v_info in red.items():
        if var_Z in v_info['parents']:
            hijos_de_Z.add(v_name)
            
    # --- 2. Calcular Puntuación para Z = True ---
    #     (No necesitamos el Manto completo, solo Padres e Hijos)
    
    # P(Z=T | Padres(Z))
    prob_Z_true = get_prob_cpt(red, var_Z, True, current_state)
    
    # Producto[ P(Hijo | Padres(Hijo)) ] donde Z=True
    prob_hijos_true = 1.0
    # Creamos un estado "hipotético" donde Z es True
    state_true = current_state.copy()
    state_true[var_Z] = True
    for H in hijos_de_Z:
        prob_hijos_true *= get_prob_cpt(red, H, state_true[H], state_true)
        
    puntuacion_true = prob_Z_true * prob_hijos_true # Puntuación no normalizada

    # --- 3. Calcular Puntuación para Z = False ---
    
    # P(Z=F | Padres(Z))
    prob_Z_false = get_prob_cpt(red, var_Z, False, current_state)
    
    # Producto[ P(Hijo | Padres(Hijo)) ] donde Z=False
    prob_hijos_false = 1.0
    # Creamos un estado "hipotético" donde Z es False
    state_false = current_state.copy()
    state_false[var_Z] = False
    for H in hijos_de_Z:
        prob_hijos_false *= get_prob_cpt(red, H, state_false[H], state_false)

    puntuacion_false = prob_Z_false * prob_hijos_false # Puntuación no normalizada

    # --- 4. Normalizar y Muestrear ---
    dist = normalizar({'True': puntuacion_true, 'False': puntuacion_false})
    
    # "Lanzar la moneda" con la nueva probabilidad
    if random.random() < dist['True']:
        return True
    else:
        return False

# --- P3: Algoritmo Principal MCMC (Gibbs Ask) ---

def gibbs_ask(query_X, evidence_e, red, N_samples=10000, burn_in=1000):
    """
    Estima P(X|e) usando Muestreo de Gibbs (MCMC).
    """
    
    # 1. Inicializar el "estado actual"
    current_state = {}
    for var in red.keys():
        if var in evidence_e:
            current_state[var] = evidence_e[var] # Fijar evidencia
        else:
            current_state[var] = random.choice([True, False]) # Inicializar aleatoriamente
            
    # 2. Obtener la lista de variables a "pasear" (las no-evidencia)
    non_evidence_vars = [v for v in red.keys() if v not in evidence_e]
    
    # 3. Inicializar contadores para la variable de consulta
    conteo_X = {True: 0.0, False: 0.0} # Usamos floats
    
    # 4. Bucle principal (Burn-in + Muestras)
    print(f"  Ejecutando MCMC por {N_samples + burn_in} iteraciones...")
    for i in range(N_samples + burn_in):
        
        # 5. Elegir una variable oculta al azar para actualizar
        var_Z = random.choice(non_evidence_vars)
        
        # 6. Remuestrear esa variable basado en su Manto de Markov
        nuevo_valor_Z = _remuestrear_variable(var_Z, current_state, red)
        
        # 7. Actualizar el estado
        current_state[var_Z] = nuevo_valor_Z
        
        # 8. Si ya pasó el "burn-in", registrar la muestra
        if i >= burn_in:
            valor_X_actual = current_state[query_X]
            conteo_X[valor_X_actual] += 1
            
    # 9. Normalizar los conteos finales
    return normalizar(conteo_X)

# --- P4: Ejecutar el cálculo ---
print("MCMC (Monte Carlo para cadenas De Markov)") # Título

# La consulta: P(Robo | JuanLlama=True, MariaLlama=True)
consulta_X = 'Robo'
evidencia = {
    'JuanLlama': True,
    'MariaLlama': True
}
# (Esta consulta es difícil para MCMC porque la evidencia está
# "lejos" de la consulta, pero funcionará)

# Usamos menos muestras que en Rechazo, pero
# cada "muestra" (iteración) es más costosa
N_GIBBS = 50000
BURN_IN = 5000

print(f"\nConsulta: P({consulta_X} | {evidencia})")
print(f"Generando {N_GIBBS} muestras MCMC (con {BURN_IN} de burn-in)...")

# Llamar a la función de inferencia
dist_gibbs = gibbs_ask(consulta_X, evidencia, red_alarma, N_GIBBS, BURN_IN)

print("\n--- Resultado (Distribución Aproximada) ---")
print(f"{dist_gibbs}")

print("\nConclusión:")
print(f"La estimación es P(Robo=True) ~= {dist_gibbs[True]:.4f}")
print(f"El resultado *exacto* (de Eliminación de Variables) era ~0.284.")
print("Este algoritmo converge al resultado correcto, 'paseando' por")
print("el espacio de estados en lugar de generar muestras 'frescas'.")