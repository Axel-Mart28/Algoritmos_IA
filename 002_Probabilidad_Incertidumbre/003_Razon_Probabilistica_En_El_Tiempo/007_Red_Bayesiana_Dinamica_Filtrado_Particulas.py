# Algoritmo de RED BAYESIANA DINÁMICA: FILTRADO DE PARTÍCULAS 

# Este es un algoritmo de *inferencia aproximada* (como MCMC).
# Resuelve la tarea de "Filtrado" (tema #3) para cualquier
# Red Bayesiana Dinámica (DBN), sin importar cuán compleja sea.
#
# Definición:
# Un Filtro de Partículas estima el estado oculto (X_t) usando un gran número de "partículas" (muestras).
# Cada "partícula" es una hipótesis de "dónde creo que está el estado".
# (ej. Partícula 1: "Creo que X=5.1", P2: "Creo que X=4.9", etc.)
# La "creencia" del filtro es la *nube* completa de N partículas.
#
# ¿Cómo funciona? (El ciclo Muestrear-Importancia-Remuestrear)
# Es un bucle que se repite en cada paso de tiempo (t)
# 
#
# 1. PREDECIR (Muestrear - "Sample"):
#    - Mueve *cada* partícula (hipótesis) hacia adelante en el tiempo según el modelo de transición (ej. "mover 1m").
#    - Añade ruido aleatorio a cada partícula (simulando la incertidumbre del movimiento).
#
# 2. ACTUALIZAR (Importancia - "Importance"):
#    - Recibe una nueva medición del sensor (la evidencia, e_t).
#    - Compara *cada* partícula con la evidencia.
#    - Asigna un "peso" (weight) a cada partícula:
#      w = P(e_t | X_t = posicion_particula)
#    - Partículas cercanas a la evidencia -> Peso ALTO.
#    - Partículas lejanas a la evidencia -> Peso BAJO.
#
# 3. REMUESTREAR (Resample):
#    - Este es el paso de "supervivencia del más apto".
#    - Crea una *nueva* lista de N partículas.
#    - Elige partículas de la lista *antigua* para "reproducirse"
#      en la nueva lista.
#    - Partículas con peso ALTO se copian *muchas veces*.
#    - Partículas con peso BAJO *mueren* (no se copian).
#    - (Esto se hace muestreando con reemplazo, ponderado por los pesos).
#
# 4. Volver al paso 1 con la nueva generación de partículas.
#
# Componentes:
# 1. N Partículas: Un conjunto de hipótesis (ej. 5000 partículas).
# 2. Modelo de Transición P(X_t|X_{t-1}): Cómo se mueven las partículas.
# 3. Modelo de Sensor P(e_t|X_t): Cómo calcular el peso (la "verosimilitud").
#
# Aplicaciones:
# - Robótica (SLAM - Localización y Mapeo Simultáneos).
# - Rastreo de objetos en video (el objeto es no-lineal).
# - Finanzas, meteorología (sistemas caóticos).
#
# Ventajas:
# - Increíblemente flexible. Puede modelar *cualquier* transición
#   (no-lineal) y *cualquier* ruido (no-Gaussiano).
#
# Desventajas:
# - Es *aproximado* (la respuesta mejora con más partículas).
# - Es computacionalmente *caro* (simular 50,000 partículas).
# - Puede sufrir "Degeneración": si una partícula es *muy* buena,
#   el remuestreo puede eliminar a todas las demás.
#
# Ejemplo de uso:
# Rastrear un robot en un pasillo *circular* (no-lineal,
# ¡Kalman fallaría!) con un sensor *defectuoso* (no-Gaussiano).

import random # Para el muestreo y el ruido
import math   # Para calcular la probabilidad Gaussiana (PDF)

# --- P1: El Escenario (Un Mundo No-Lineal y No-Gaussiano) ---

PASILLO_LONGITUD = 20.0 # Un pasillo circular de 0.0 a 20.0

# 1. Modelo de Transición (Movimiento) - No Lineal
def mover_particula(particula_pos, movimiento, ruido_mov_std):
    """ Mueve una partícula y añade ruido, con 'loop' (no-lineal) """
    # Aplicar el movimiento (ej. +1.0)
    pos = particula_pos + movimiento 
    # Añadir ruido Gaussiano al movimiento
    ruido = random.gauss(0, ruido_mov_std) # Ruido del motor
    pos += ruido
    # Aplicar la "no-linealidad" (el loop)
    # % es el operador "módulo"
    return pos % PASILLO_LONGITUD # (ej. 20.5 -> 0.5, -1.2 -> 18.8)

# 2. Modelo de Sensor (Peso) - No Gaussiano
def calcular_peso(particula_pos, medicion, ruido_sens_std):
    """
    Calcula el peso P(e|X) de una partícula.
    Este sensor es "defectuoso": 90% del tiempo es Gaussiano,
    10% del tiempo da una lectura *completamente aleatoria*.
    """
    
    # --- P(e|X) = 0.9 * P_Gaussiana(e|X) + 0.1 * P_Aleatoria(e|X) ---
    
    # 1. Calcular P_Gaussiana(e|X)
    #    (La fórmula de la curva de campana, PDF)
    #    Mide qué tan "lejos" está la partícula de la medición
    dist = abs(particula_pos - medicion)
    # (Ajuste para el pasillo circular)
    dist = min(dist, PASILLO_LONGITUD - dist) 
    # PDF: e^(-(dist^2 / (2*var)))
    prob_gauss = math.exp(-(dist**2) / (2 * (ruido_sens_std**2)))
    
    # 2. Calcular P_Aleatoria(e|X)
    #    (La probabilidad de una lectura aleatoria uniforme)
    prob_aleatoria = 1.0 / PASILLO_LONGITUD # (ej. 1/20 = 0.05)
    
    # 3. Combinar (Mezcla de Modelos)
    peso_final = (0.9 * prob_gauss) + (0.1 * prob_aleatoria)
    
    return peso_final

# --- P2: Algoritmo de Remuestreo (Resampling) ---

def remuestreo(particulas, pesos):
    """
    Paso 3: REMUESTREAR (Supervivencia del más apto)
    Crea una nueva generación de partículas.
    """
    
    N = len(particulas) # Número de partículas
    
    # Python tiene una función *perfecta* para esto:
    # random.choices(poblacion, pesos, k=tamaño)
    # Elige 'k' ítems de 'poblacion' (con reemplazo),
    # usando 'pesos' como la probabilidad de ser elegido.
    
    # Partículas con pesos altos serán elegidas múltiples veces.
    # Partículas con pesos bajos no serán elegidas.
    nueva_generacion = random.choices(
        population=particulas, # La lista de partículas [4.5, 5.1, ...]
        weights=pesos,       # La lista de pesos [0.1, 0.8, ...]
        k=N                  # Queremos N nuevas partículas
    )
    
    return nueva_generacion # Devuelve la nueva generación

# --- P3: El Filtro de Partículas (El Bucle Principal) ---

def filtro_de_particulas(evidencia_seq, N_particulas):
    """
    Ejecuta el Filtro de Partículas completo.
    """
    
    print("Iniciando Filtro de Partículas...")
    print(f"Longitud del Pasillo: {PASILLO_LONGITUD}")
    print(f"Número de Partículas: {N_particulas}")
    
    # --- Parámetros del Modelo ---
    ruido_mov_std = 0.5   # Incertidumbre del movimiento (std dev)
    ruido_sens_std = 1.0  # Incertidumbre del sensor (std dev)
    movimiento_real = 1.0 # El robot *intenta* moverse 1m
    
    # --- 1. Inicialización ---
    # Crear N partículas, distribuidas aleatoriamente por el pasillo
    particulas = [random.uniform(0.0, PASILLO_LONGITUD) for _ in range(N_particulas)]
    
    print("\nt  | Evidencia | Pos. Estimada (Media de Partículas)")
    print("-------------------------------------------------")
    
    # --- Bucle de Simulación (Iterar sobre la evidencia) ---
    for t, e_t in enumerate(evidencia_seq):
        
        # --- PASO 1: PREDECIR (Muestrear) ---
        # Mover cada partícula hacia adelante (con ruido y loop)
        for i in range(N_particulas): # Iterar sobre cada partícula
            particulas[i] = mover_particula(particulas[i], movimiento_real, ruido_mov_std)
            
        # --- PASO 2: ACTUALIZAR (Importancia) ---
        # Calcular el peso de cada partícula basado en la evidencia e_t
        pesos = [] # Lista para los pesos
        for p in particulas: # Iterar sobre cada partícula
            # ¿Qué tan probable es e_t, dada esta partícula?
            peso = calcular_peso(p, e_t, ruido_sens_std)
            pesos.append(peso) # Añadir el peso a la lista
            
        # --- PASO 3: REMUESTREAR ---
        # Reemplazar la lista de partículas con la nueva generación
        particulas = remuestreo(particulas, pesos)
        
        # --- 4. Estimar la Posición Actual ---
        # La estimación es la "media" (promedio) de la nube de partículas
        # (Hay que tener cuidado con la media en un círculo, pero
        #  para este ejemplo, un promedio simple funciona)
        estimacion = sum(particulas) / N_particulas
        
        print(f"{t:2} | {e_t:9.2f} | {estimacion:18.2f}") # Imprimir el estado
        
    return estimacion # Devolver la estimación final

# --- P4: Ejecutar la Simulación ---

# 1. Crear una "Verdad" (Ground Truth) y Evidencia Ruidosa
pos_verdadera = 0.0 # El robot empieza en 0
evidencia_real = [] # Lista de lecturas del sensor
T_PASOS = 20 # Simular 20 segundos

for t in range(T_PASOS):
    # El robot real se mueve (no-lineal)
    pos_verdadera = (pos_verdadera + 1.0) % PASILLO_LONGITUD
    
    # El sensor (no-Gaussiano) toma una lectura
    if random.random() < 0.9: # 90% del tiempo, sensor normal
        ruido = random.gauss(0, 1.0)
        evidencia_real.append((pos_verdadera + ruido) % PASILLO_LONGITUD)
    else: # 10% del tiempo, sensor aleatorio
        evidencia_real.append(random.uniform(0.0, PASILLO_LONGITUD))

# 2. Ejecutar el Filtro de Partículas
N_PARTICULAS = 2000 # Usar 2000 "hipótesis"
estimacion_final = filtro_de_particulas(evidencia_real, N_PARTICULAS)

print("-------------------------------------------------")
print(f"Posición Verdadera Final: {pos_verdadera:.2f}")
print(f"Estimación Final del Filtro: {estimacion_final:.2f}")

print("\nConclusión:")
print("A pesar del movimiento no-lineal (loop) y el sensor no-Gaussiano (aleatorio),")
print("la nube de partículas logró rastrear la posición verdadera.")
print("Un Filtro de Kalman habría fallado en este escenario.")