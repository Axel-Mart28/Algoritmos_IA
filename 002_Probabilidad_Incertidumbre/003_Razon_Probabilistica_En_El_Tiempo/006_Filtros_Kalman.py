# Algoritmo de FILTROS DE KALMAN

# Este es el algoritmo de FILTRADO (tema #3) por excelencia para datos del mundo real, que son continuos y ruidosos.

# Definición:
# Un Filtro de Kalman es un algoritmo que estima el *estado oculto*
# (ej. la *verdadera* posición y velocidad de un misil) a partir de una serie de *mediciones ruidosas* (ej. lecturas de un radar).
#
# Es un HMM para variables Gaussianas (continuas).
#
# ¿Cómo funciona?
# El algoritmo es un ciclo de dos pasos que se repite en cada
# paso de tiempo (t):
#
# 1. PREDECIR (PREDICT):
#    - Basado en el estado *anterior* (X_{t-1}) y un "modelo de movimiento"
#      (ej. leyes de la física), el filtro *predice* dónde
#      debería estar el estado *ahora* (X_t).
#    - Esta predicción *aumenta* la incertidumbre (la varianza crece).
#
# 2. ACTUALIZAR (UPDATE):
#    - El filtro recibe una nueva "medición" (evidencia) de un sensor
#      (ej. una lectura de GPS, e_t), que también tiene ruido.
#    - "Fusiona" la Predicción (creencia previa) con la Medición (evidencia nueva).
#    - Calcula una "Ganancia de Kalman" (Kalman Gain) que decide
#      cuánto confiar en la predicción vs. la medición.
#    - El resultado es una nueva estimación, *mejor* que ambas.
#    - Esta actualización *disminuye* la incertidumbre (la varianza se reduce).
#
# 
#
# El "Estado" y la "Medición" no son números únicos, sino
# *Distribuciones Gaussianas* (curvas de campana), definidas por:
# - Una media (mu, $\mu$): El valor más probable.
# - Una varianza (sigma^2, $\sigma^2$): La incertidumbre (qué tan ancha es la campana).
#
# Aplicaciones:
# - Naves espaciales (misión Apolo, SpaceX).
# - Navegación (GPS, drones, submarinos, misiles).
# - Robótica (para fusionar sensores, ej. ruedas y cámara).
# - Economía (para suavizar series de tiempo).
#
# Ventajas:
# - Es *óptimo* si las suposiciones son correctas (ruido Gaussiano, modelo lineal).
# - Increíblemente rápido y eficiente. No necesita guardar el historial.
#
# Desventajas:
# - Requiere un modelo matemático (ej. de física) *correcto*.
# - Falla si el proceso no es lineal o el ruido no es Gaussiano.
#   (Para eso se usa el "Filtrado de Partículas", el siguiente tema).
#
# Ejemplo de uso:
# Vamos a simular un robot moviéndose en 1D.
# - Estado Oculto (X_t): Posición Verdadera (ej. 1, 2, 3...)
# - Evidencia (e_t): Lectura del Sensor (ej. 0.9, 2.1, 2.9...)
# El Filtro de Kalman debe estimar X_t usando solo e_t.

import math 
import random # Para simular el ruido
import numpy as np 

# --- P1: Las Dos Ecuaciones Centrales (en 1D) ---
# (Estas son las "matemáticas" de fusionar Curvas de Campana)

def update_step(media_creencia, var_creencia, media_medicion, var_medicion):
    """
    Paso 2: ACTUALIZAR. Fusiona la creencia (predicción)
    con la medición (sensor) para obtener una nueva creencia.
    """
    
    # 1. Calcular la Ganancia de Kalman (K)
    #    K = incertidumbre_creencia / (incertidumbre_creencia + incertidumbre_medicion)
    #    Si el sensor es bueno (var_medicion baja), K es alto (confía más en el sensor).
    #    Si la creencia es buena (var_creencia baja), K es bajo (confía más en la predicción).
    K = var_creencia / (var_creencia + var_medicion) # La Ganancia
    
    # 2. Calcular la nueva Media (la nueva estimación de posición)
    #    media_nueva = media_vieja + K * (error_de_medicion)
    #    error_de_medicion = (lo que el sensor vio - lo que creíamos que vería)
    media_nueva = media_creencia + K * (media_medicion - media_creencia)
    
    # 3. Calcular la nueva Varianza (la nueva incertidumbre)
    #    var_nueva = (1 - K) * var_vieja
    #    Como K está entre 0 y 1, la varianza *siempre* se reduce.
    var_nueva = (1 - K) * var_creencia
    
    return (media_nueva, var_nueva) # Devuelve el estado actualizado

def predict_step(media_estado, var_estado, media_movimiento, var_movimiento):
    """
    Paso 1: PREDECIR. Aplica el movimiento al estado actual
    para predecir el siguiente estado.
    """
    
    # Al "sumar" dos gaussianas (estado + movimiento):
    
    # 1. La nueva media es la suma de las medias
    media_nueva = media_estado + media_movimiento
    
    # 2. La nueva varianza es la suma de las varianzas
    #    (La incertidumbre *siempre* crece en la predicción)
    var_nueva = var_estado + var_movimiento
    
    return (media_nueva, var_nueva)

# --- P2: Simulación del Filtro de Kalman ---
print("Filtro de Kalman") # Título

# --- Parámetros de la Simulación ---

# 1. Nuestro modelo de "ruido"
var_medicion = 4.0   # Incertidumbre del Sensor (sigma^2). ¡Alta! (ej. +/- 2.0m)
var_movimiento = 0.1 # Incertidumbre del Movimiento (sigma^2). Baja (ej. +/- 0.3m)

# 2. Nuestro modelo de "movimiento" (Control)
#    (El robot intenta moverse 1.0m cada segundo)
movimiento = 1.0

# 3. Estado inicial
posicion_verdadera = 0.0 # El robot empieza en 0
kf_estado = (0.0, 20.0)  # Creencia inicial: (media=0.0, var=20.0)
                         # (Empezamos con alta incertidumbre)

N_PASOS = 15 # Simular por 15 segundos

print(f"Estado Inicial KF: media={kf_estado[0]}, var={kf_estado[1]}") # Imprime inicio
print("Simulando...")
print("----------------------------------------------------------------------")
print("t | Pos_Verdadera | Medición (ruidosa) | KF_Media (estimada) | KF_Var (incertidumbre)")
print("----------------------------------------------------------------------")

# --- Bucle de Simulación ---
for t in range(1, N_PASOS + 1): # Iterar por cada paso de tiempo
    
    # --- 1. PASO DE PREDICCIÓN (¿Dónde cree el filtro que estará?) ---
    # (El filtro usa su estado anterior y el modelo de movimiento)
    kf_estado = predict_step(kf_estado[0], kf_estado[1], movimiento, var_movimiento)
    # (En este punto, kf_estado[1] (varianza) ha aumentado)
    
    
    # --- 2. SIMULACIÓN DEL MUNDO REAL (El robot se mueve) ---
    #    (El filtro NO ve esto)
    posicion_verdadera += movimiento # El robot se mueve exactamente 1.0m
    
    #    (El filtro SÍ ve esto)
    #    El sensor genera una lectura RUIDOSA
    ruido_medicion = random.gauss(0, math.sqrt(var_medicion)) # Ruido Gaussiano
    medicion = posicion_verdadera + ruido_medicion # (pos_verdadera + ruido)
    
    
    # --- 3. PASO DE ACTUALIZACIÓN (El filtro corrige su predicción) ---
    # (El filtro fusiona su predicción con la medición ruidosa)
    kf_estado = update_step(kf_estado[0], kf_estado[1], medicion, var_medicion)
    # (En este punto, kf_estado[1] (varianza) ha disminuido)
    
    
    # --- 4. Imprimir resultados ---
    kf_media, kf_var = kf_estado # Desempaquetar el estado final
    print(f"{t:2} | {posicion_verdadera:13.3f} | {medicion:18.3f} | {kf_media:19.3f} | {kf_var:20.3f}")

print("----------------------------------------------------------------------")
print("\nConclusión:")
print("Observa la columna 'Medición (ruidosa)' y compárala con 'KF_Media'.")
print("Aunque el sensor (Medición) salta erráticamente, la estimación")
print("del Filtro de Kalman (KF_Media) sigue a la 'Pos_Verdadera'")
print("de forma mucho más suave y precisa.")
print("También observa cómo la 'KF_Var' (incertidumbre) se reduce")
print("rápidamente de 20.0 a un valor bajo y estable.")