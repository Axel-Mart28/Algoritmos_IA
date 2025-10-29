# Algoritmo de FILTRADO, PREDICCIÓN, SUAVIZADO Y EXPLICACIÓN

# Este no es un algoritmo, sino un CONCEPTO. Define las 4 TAREAS (las 4 preguntas) que queremos resolver en un "Razonamiento Probabilístico en el Tiempo".

# Definición:
# Asumimos que tenemos un "proceso oculto" (ej. la *verdadera* posición de un robot, X_t) y "evidencia" (ej. la *lectura del sensor*, e_t).
# El estado X_t es "oculto" porque no podemos verlo directamente, el sensor 'e_t' es "ruidoso" (impreciso).

# El tiempo actual es 't'. Hemos visto evidencia e_1, e_2, ..., e_t.
# Las 4 tareas son:
#
# 1. Filtrado (Filtering): P(X_t | e_1:t)
#    - Pregunta: "¿Dónde estoy *ahora*?"
#    - Propósito: Estimar el estado *presente* (X_t) usando toda la evidencia hasta el *presente* (e_1...e_t).
#
# 2. Predicción (Prediction): P(X_{t+k} | e_1:t) (donde k > 0)
#    - Pregunta: "¿Dónde estaré *en el futuro*?"
#    - Propósito: Estimar un estado *futuro* (X_{t+k}) usando toda la evidencia hasta el *presente* (e_1...e_t).
#
# 3. Suavizado (Smoothing): P(X_k | e_1:t) (donde k < t)
#    - Pregunta: "¿Dónde estuve *en el pasado*?"
#    - Propósito: Estimar un estado *pasado* (X_k) usando toda la evidencia hasta el *presente* (e_1...e_t).
#    - Esto es "mirar atrás" para *corregir* nuestra creencia pasada, usando evidencia más nueva.
#
# 4. Explicación (Explanation / Most Likely Path): argmax P(X_1:t | e_1:t)
#    - Pregunta: "¿Cuál fue el *camino completo* más probable?"
#    - Propósito: Encontrar la *secuencia entera* de estados (X_1...X_t) que mejor explica la *secuencia entera* de evidencia (e_1...e_t).
#
# ¿Cómo funciona este programa?
# Este código NO calcula las respuestas. Solo usaremos
# print() para *plantear* las 4 preguntas (tareas)
# en el contexto de un ejemplo.
#
# Aplicaciones:
# - Filtrado: Rastrear un objeto en un radar (¿dónde está AHORA?).
# - Predicción: Pronóstico del tiempo (¿lloverá MAÑANA?).
# - Suavizado: Análisis de video (¿dónde estaba el objeto en el frame 50, visto el video completo de 100 frames?).
# - Explicación: Reconocimiento de voz (¿Qué *secuencia de palabras* explica mejor esta *secuencia de audio*?).
#
# Ejemplo de uso:
# Un robot se mueve por un pasillo. X_t = Posición. e_t = Sensor (ruidoso).


# --- P1: Definición del Escenario ---

# Variable Oculta (Estado) X_t: PosiciónVerdadera del robot en el pasillo (ej. casilla 5)
# Variable de Evidencia e_t: LecturaDelSensor en el tiempo t (ej. "Veo-Pared")

# Supongamos que estamos en el "Presente", tiempo t=5
t_presente = 5 # El "Presente"

# Hemos recolectado 5 lecturas de sensor (evidencia)
evidencia_recolectada = {
    'e_1': 'No-Pared',  # t=1
    'e_2': 'No-Pared',  # t=2
    'e_3': 'Pared',     # t=3 <-- ¡Una lectura ruidosa! El robot no chocó.
    'e_4': 'No-Pared',  # t=4
    'e_5': 'No-Pared'   # t=5
}

print("Filtrado, Prediccion y Suavizado") # Título
print(f"Estamos en el tiempo t={t_presente}.") # Mensaje
print(f"Evidencia recolectada (e_1...e_{t_presente}): {evidencia_recolectada}") # Mensaje

# --- P2: Tarea 1 - FILTRADO ---
# "¿Cuál es la prob. de mi posición *ahora* (t=5),
# dados los 5 sensores (e_1...e_5)?"
print("\n1. Tarea de FILTRADO (¿Dónde estoy AHORA?)") # Tarea 1
print("  PREGUNTA: P(X_5 | e_1, e_2, e_3, e_4, e_5)") # La fórmula

# --- P3: Tarea 2 - PREDICCIÓN ---
# "¿Cuál es la prob. de mi posición en 2 segundos (t=7),
# dados los 5 sensores (e_1...e_5)?"
k_futuro = 2 # Cuántos pasos en el futuro
print("\n2. Tarea de PREDICCIÓN (¿Dónde estaré DESPUÉS?)") # Tarea 2
print(f"  PREGUNTA: P(X_{t_presente + k_futuro} | e_1, e_2, e_3, e_4, e_5)") # La fórmula

# --- P4: Tarea 3 - SUAVIZADO ---
# "¿Cuál era mi posición en t=3, dados los 5 sensores?"
# Esto es útil para *corregir* nuestra creencia sobre el pasado.
# En t=3, solo con e_1,e_2,e_3, pudimos pensar que chocamos
# (P(X_3='Chocado') era alta).
# Pero ahora, viendo e_4 y e_5 (No-Pared), podemos "suavizar" esa
# creencia y concluir que el sensor en t=3 probablemente fue un error.
k_pasado = 3 # El punto en el pasado que queremos corregir
print("\n3. Tarea de SUAVIZADO (¿Dónde estuve ANTES?)") # Tarea 3
print(f"  PREGUNTA: P(X_{k_pasado} | e_1, e_2, e_3, e_4, e_5)") # La fórmula
print(f"  (Corrigiendo nuestra creencia sobre el pasado t={k_pasado} usando evidencia futura)") # Explicación

# --- P5: Tarea 4 - EXPLICACIÓN (Camino Más Probable) ---
# "¿Cuál es la *secuencia completa* de posiciones (X_1 a X_5)
# que *mejor explica* la secuencia de sensores (e_1 a e_5)?"
# ej. ('Casilla1', 'Casilla2', 'Casilla3', 'Casilla4', 'Casilla5')
print("\n4. Tarea de EXPLICACIÓN (¿Cuál fue el CAMINO más probable?)") # Tarea 4
print("  PREGUNTA: argmax_{X_1...X_5} [ P(X_1...X_5 | e_1...e_5) ]") # La fórmula
print("  (Esto lo resuelve el algoritmo de Viterbi, no el Hacia Delante-Atrás)") # Aclaración