# --- ALGORITMO DE TRADUCCIÓN AUTOMÁTICA ESTADÍSTICA (SMT - Simulación) ---

# Concepto: Traducir encontrando la oración destino más probable
#           P(destino|fuente) ∝ P(fuente|destino) * P(destino),
#           usando un Modelo de Traducción y un Modelo de Lenguaje.
# Objetivo: Simular cómo se puntúan traducciones candidatas.

import math # Para usar logaritmos (más estable numéricamente)
from collections import Counter # Para n-gramas

# --- P1: Modelos Simplificados Predefinidos ---

# 1a. Modelo de Lenguaje (LM) - Bigramas (para Inglés Destino)
#     P(palabra_i | palabra_{i-1})
#     (Valores inventados para el ejemplo)
#     Usaremos log-probabilidades (log(P)) para evitar underflow y sumar en lugar de multiplicar.
#     Un valor MENOS NEGATIVO es MEJOR (más probable).
lm_log_probs = {
    ('<s>', 'the'): math.log(0.8), # Prob de empezar con 'the'
    ('<s>', 'cat'): math.log(0.1),
    ('<s>', 'dog'): math.log(0.1),
    ('the', 'cat'): math.log(0.5),
    ('the', 'dog'): math.log(0.3),
    ('the', 'cheese'): math.log(0.1), # 'the cheese' es menos común
    ('cat', 'eats'): math.log(0.6),
    ('cat', 'chases'): math.log(0.3),
    ('dog', 'chases'): math.log(0.7),
    ('dog', 'eats'): math.log(0.2),
    ('eats', 'cheese'): math.log(0.9), # 'eats cheese' es probable
    ('chases', 'the'): math.log(0.8),
    ('cheese', '</s>'): math.log(0.9), # Prob de terminar después de 'cheese'
    ('cat', '</s>'): math.log(0.1),
    ('dog', '</s>'): math.log(0.1),
    # ... (necesitaríamos muchos más bigramas)
}
# Probabilidad por defecto si un bigrama no se encuentra (muy baja)
lm_log_prob_default = math.log(1e-6)

# 1b. Modelo de Traducción (TM) - Palabra por Palabra (Español -> Inglés)
#     P(palabra_fuente | palabra_destino)
#     (Valores inventados)
tm_log_probs = {
    # P(español | inglés)
    ('gato', 'cat'): math.log(0.9),
    ('el', 'the'): math.log(0.8),
    ('queso', 'cheese'): math.log(0.95),
    ('come', 'eats'): math.log(0.85),
    ('persigue', 'chases'): math.log(0.7),
    ('perro', 'dog'): math.log(0.8),
    # Añadir algunas traducciones "malas" con baja probabilidad
    ('gato', 'dog'): math.log(0.05),
    ('queso', 'cat'): math.log(0.01),
    # ... (necesitaríamos muchas más alineaciones)
}
# Probabilidad por defecto si una traducción no se encuentra
tm_log_prob_default = math.log(1e-6)

print("---Traducción Automática Estadística (SMT - Simulación) ---")
print("Modelos LM (Bigrama Inglés) y TM (Palabra Esp->Ing) simplificados definidos.")

# --- P2: Función de Puntuación (Simula el Decodificador) ---

def puntuar_traduccion(oracion_fuente_tokens, oracion_destino_tokens, lm_probs, tm_probs):
    """
    Calcula una puntuación (log-probabilidad) para una traducción candidata.
    Puntuación = log P(destino | LM) + log P(fuente | destino, TM)
    """

    # --- 1. Calcular Puntuación del Modelo de Lenguaje (LM) ---
    log_prob_lm = 0.0
    # Añadir marcadores de inicio (<s>) y fin (</s>)
    tokens_lm = ['<s>'] + oracion_destino_tokens + ['</s>']
    # Iterar sobre los bigramas (pares de palabras)
    for i in range(len(tokens_lm) - 1):
        bigrama = (tokens_lm[i], tokens_lm[i+1]) # Ej: ('<s>', 'the'), ('the', 'cat'), ...
        # Obtener la log-prob del bigrama, usar default si no existe
        log_prob_lm += lm_probs.get(bigrama, lm_log_prob_default)

    # --- 2. Calcular Puntuación del Modelo de Traducción (TM) ---
    #    (¡Simplificación ENORME! Asumimos alineación 1 a 1)
    log_prob_tm = 0.0
    if len(oracion_fuente_tokens) != len(oracion_destino_tokens):
        # Penalizar fuertemente si las longitudes no coinciden (muy simple)
        log_prob_tm = math.log(1e-10)
    else:
        # Sumar log P(fuente_i | destino_i) para cada palabra
        for palabra_f, palabra_d in zip(oracion_fuente_tokens, oracion_destino_tokens):
            par_traduccion = (palabra_f, palabra_d) # Ej: ('gato', 'cat')
            log_prob_tm += tm_probs.get(par_traduccion, tm_log_prob_default)

    # --- 3. Puntuación Total ---
    puntuacion_total = log_prob_lm + log_prob_tm # Sumar log-probs
    return puntuacion_total, log_prob_lm, log_prob_tm # Devolver puntajes

# --- P3: Simular la Traducción de una Frase ---

# Oración fuente en español (ya tokenizada)
fuente = ['el', 'gato', 'come', 'queso']
print(f"\nOración Fuente (Español): {' '.join(fuente)}")

# Dos traducciones candidatas al inglés
candidata1_tokens = ['the', 'cat', 'eats', 'cheese'] # Buena traducción
candidata2_tokens = ['the', 'dog', 'eats', 'cat']   # Mala traducción

print(f"Candidata 1 (Inglés): {' '.join(candidata1_tokens)}")
print(f"Candidata 2 (Inglés): {' '.join(candidata2_tokens)}")

# Puntuar ambas candidatas
score1, lm1, tm1 = puntuar_traduccion(fuente, candidata1_tokens, lm_log_probs, tm_log_probs)
score2, lm2, tm2 = puntuar_traduccion(fuente, candidata2_tokens, lm_log_probs, tm_log_probs)

print("\n--- Puntuaciones (Log-Probabilidad) ---")
print(f"Candidata 1:")
print(f"  Puntuación LM: {lm1:.3f}")
print(f"  Puntuación TM: {tm1:.3f}")
print(f"  Puntuación Total: {score1:.3f}")

print(f"\nCandidata 2:")
print(f"  Puntuación LM: {lm2:.3f}") # Puede ser razonable ('the dog eats')
print(f"  Puntuación TM: {tm2:.3f}") # Será muy baja (P('gato'|'dog'), P('queso'|'cat'))
print(f"  Puntuación Total: {score2:.3f}")

print("\n--- Decisión ---")
if score1 > score2: # Recordar: menos negativo es mejor
    print("La Candidata 1 ('the cat eats cheese') es la traducción más probable.")
else:
    print("La Candidata 2 ('the dog eats cat') es la traducción más probable.")

print("\nConclusión:")
print("SMT combina un Modelo de Lenguaje (fluidez en destino) y un Modelo")
print("de Traducción (fidelidad a la fuente) para encontrar la mejor traducción.")
print("Este ejemplo simplificado muestra cómo se puntúan las candidatas.")
print("Un sistema real usa modelos mucho más complejos y un decodificador")
print("sofisticado para buscar entre miles de millones de posibilidades.")