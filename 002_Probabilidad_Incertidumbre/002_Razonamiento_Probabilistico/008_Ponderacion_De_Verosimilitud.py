# Algoritmo de PONDERACIÓN DE VEROSIMILITUD

# Este es un algoritmo de inferencia *aproximada*, como el Muestreo por Rechazo, pero es muchísimo más eficiente.

# Definición:
# Es un algoritmo que estima P(X|e) (consulta | evidencia) sin nunca rechazar (tirar) una muestra.

# El Problema del Muestreo por Rechazo:
# Si la evidencia 'e' es rara (ej. P(e) = 0.001), rechazamos el 99.9% de las muestras. Esto es un desperdicio computacional masivo.

# ¿Cómo funciona (La Solución "Ponderada")?:
# 1. El algoritmo *fuerza* que la muestra sea consistente con la evidencia.
# 2. Genera la muestra de arriba a abajo (como el Muestreo Directo).
# 3. Mantiene una "ponderación" (peso), que empieza en w = 1.0.
# 4. CUANDO LLEGA A UNA VARIABLE DE EVIDENCIA 'V' (ej. e = {JuanLlama:True}):
# 5.   *No* lanza una moneda.
# 6.   *Fija* el valor de la muestra a la evidencia (ej. muestra['JuanLlama'] = True).
# 7.   Actualiza el peso: multiplica 'w' por la probabilidad de esa evidencia dados los padres que ya se muestrearon. w = w * P(JuanLlama=True | Alarma=False) (ej. w = w * 0.05)
# 8. CUANDO LLEGA A UNA VARIABLE OCULTA 'Y' (que no es evidencia):
# 9.   Lanza la moneda y la muestrea normalmente (como Muestreo Directo).
#      El peso 'w' *no* cambia.
# 10. Al final, cada muestra tiene un peso (ej. 0.005, 0.4, 0.01...).
#
# ¿Cómo responde la consulta P(X|e)?
# - En lugar de *contar* las muestras, *suma* sus pesos.
# - Puntuación(X=True) = Suma de los pesos 'w' de todas las muestras donde X=True
# - Puntuación(X=False) = Suma de los pesos 'w' de todas las muestras donde X=False
# - Finalmente, normaliza estas puntuaciones.
#
# 
#
# Componentes:
# 1. La Red Bayesiana (`red_alarma`).
# 2. Una consulta X (ej. 'Robo') y una evidencia 'e' (ej. {'JuanLlama':True}).
# 3. Un peso 'w' que se acumula para cada muestra.
#
# Ventajas:
# - Mucho más eficiente Nunca rechaza una muestra. Si se piden 100,000 muestras, se obtienen 100,000 muestras *útiles*.
# - Converge a la respuesta correcta mucho más rápido que el Rechazo.
#
# Desventajas:
# - El resultado puede ser inestable si los pesos varían mucho (ej. si la evidencia 'e' es *extremadamente* improbable, casi todas las muestras tendrán un peso cercano a cero, y unas pocas "afortunadas" tendrán un peso alto, distorsionando la estimación).
#
# Ejemplo de uso:
# Estimar P(Robo | JuanLlama=True, MariaLlama=True).
# El algoritmo generará N muestras, *todas* con J=True y M=True,
# cada una con un peso diferente (ej. 0.063, 0.0001, etc.).

import random # Necesario para muestrear variables ocultas
import copy   

# --- P1: Definición de la Red y Funciones Auxiliares ---
# (Necesitamos la red, get_prob_cpt, y normalizar)

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

def get_prob_cpt(red, variable, valor, evidencia={}):
    """ Obtiene P(variable=valor | evidencia) de la CPT """
    nodo = red[variable]
    padres = nodo['parents']
    if not padres:
        clave_cpt = ()
    else:
        # Crea la tupla de clave (ej. (True, False))
        # Maneja el caso de que un padre aún no esté en la evidencia
        if not all(p in evidencia for p in padres):
             # Esto no debería pasar si el orden es topológico
             raise ValueError(f"Padre(s) de {variable} no encontrado en evidencia")
        clave_cpt = tuple([evidencia[padre] for padre in padres])
    prob_true = nodo['cpt'][clave_cpt]
    return prob_true if valor == True else (1.0 - prob_true)

def normalizar(puntuaciones):
    """ Normaliza un diccionario de {etiqueta: puntuacion} """
    total = sum(puntuaciones.values())
    if total == 0:
        num_items = len(puntuaciones)
        if num_items > 0: return {e: 1.0 / num_items for e in puntuaciones}
        else: return {}
    return {e: p / total for e, p in puntuaciones.items()}

# --- P2: Algoritmo de Ponderación de Verosimilitud (1 Muestra) ---

def ponderacion_verosimilitud_una_muestra(red, evidencia):
    """
    Genera UN evento completo (muestra) y su peso (weight).
    La muestra es forzada a ser consistente con la 'evidencia'.
    """
    
    # 1. Inicializar el peso de la muestra en 1.0
    peso_w = 1.0
    
    # 2. Inicializar la muestra (copiando la evidencia)
    muestra = evidencia.copy() # Empezamos con la evidencia ya fijada
    
    # 3. Iterar sobre las variables en orden topológico
    variables = red.keys() # ['Robo', 'Terremoto', 'Alarma', ...]
    
    for var in variables:
        
        if var in evidencia:
            # --- Caso A: La variable es EVIDENCIA ---
            
            # 1. Obtener la probabilidad de esta evidencia
            #    P(var=evidencia[var] | Padres(var))
            #    Los padres ya deben estar en la 'muestra'
            #    (porque el orden es topológico).
            prob_evidencia = get_prob_cpt(red, var, evidencia[var], muestra)
            
            # 2. Multiplicar el peso por esta probabilidad (likelihood)
            peso_w *= prob_evidencia
            
            # 3. El valor de la muestra ya está fijado (lo copiamos al inicio)
            #    muestra[var] = evidencia[var]
            
        else:
            # --- Caso B: La variable es OCULTA ---
            
            # 1. Muestrear normalmente, como en Muestreo Directo
            #    Obtener P(var=True | Padres(var))
            prob_true = get_prob_cpt(red, var, True, muestra)
            
            # 2. "Lanzar la moneda"
            if random.random() < prob_true:
                muestra[var] = True
            else:
                muestra[var] = False
            
            # 3. El peso NO cambia
    
    # 4. Devolver la muestra (que contiene la evidencia) y su peso
    return muestra, peso_w

# --- P3: Algoritmo Principal de Ponderación de Verosimilitud ---

def muestreo_ponderado_verosimilitud(query_X, evidencia_e, red, N):
    """
    Estima P(X|e) usando Ponderación de Verosimilitud con N muestras.
    """
    
    # 1. Inicializar los *pesos* acumulados para la consulta X
    #    (No son conteos, son sumas de pesos)
    W_X = {True: 0.0, False: 0.0}
    
    # 2. Bucle N veces (generar N muestras *útiles*)
    for i in range(N):
        
        # 3. Generar una muestra ponderada
        #    (Cada llamada genera una muestra y su peso)
        muestra, peso = ponderacion_verosimilitud_una_muestra(red, evidencia_e)
        
        # 4. Obtener el valor de la variable de consulta X de la muestra
        valor_X = muestra[query_X] # Ej: True o False
        
        # 5. Añadir el *peso* de esta muestra al acumulador apropiado
        W_X[valor_X] += peso
        
    # 6. Normalizar las sumas de pesos
    #    Esto nos da la distribución de probabilidad final
    return normalizar(W_X)

# --- P4: Ejecutar el cálculo ---
print("--- 7. Ponderación de Verosimilitud (Likelihood Weighting) ---") # Título
N_PONDERADO = 100000 # 100k muestras. Es mucho más rápido que el Rechazo

# La consulta: P(Robo | JuanLlama=True, MariaLlama=True)
consulta_X = 'Robo'
evidencia = {
    'JuanLlama': True,
    'MariaLlama': True
}

print(f"\nConsulta: P({consulta_X} | {evidencia})")
print(f"Generando {N_PONDERADO} muestras ponderadas (ninguna será rechazada)...")

# Llamar a la función de inferencia
dist_ponderada = muestreo_ponderado_verosimilitud(consulta_X, evidencia, red_alarma, N_PONDERADO)

print("\n--- Resultado (Distribución Aproximada) ---")
print(f"{dist_ponderada}")

print("\nConclusión:")
print(f"La estimación es P(Robo=True) ~= {dist_ponderada[True]:.4f}")
print(f"El resultado *exacto* (de Eliminación de Variables) era ~0.284.")
print("Este algoritmo es mucho más eficiente que el Rechazo porque")
print(f"las {N_PONDERADO} muestras generadas fueron *todas* utilizadas en el cálculo.")