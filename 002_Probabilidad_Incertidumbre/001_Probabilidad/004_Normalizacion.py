# Algoritmo de NORMALIZACION

# Este es un "truco" matemático o un paso de limpieza que es crucial en probabilidad.

# Definición:
# Es el proceso de tomar un conjunto de "puntuaciones" o "pesos" (que no necesariamente suman 1) y escalarlos para que *sí* sumen 1.
# Esto convierte un conjunto de "creencias relativas" en una *Distribución de Probabilidad* válida.

# ¿Cómo funciona?:
# 1. Tiene un conjunto de puntuaciones: V = {v1, v2, v3}
# 2. Se suman todas las puntuaciones: Total = v1 + v2 + v3
# 3. Divide cada puntuación individual por el Total:
#    - v1_normalizada = v1 / Total
#    - v2_normalizada = v2 / Total
#    - v3_normalizada = v3 / Total
# 4. El nuevo conjunto {v1_norm, v2_norm, v3_norm} ahora suma 1.
#
# Componentes:
# 1. Un vector (lista o diccionario) de puntuaciones o valores no normalizados.
# 2. La suma total de esos valores (llamada la "constante de normalización").
#
# Aplicaciones:
# - El paso final de la Regla de Bayes! A veces es fácil calcular P(A|B) * P(B) y P(no_A|B) * P(B), pero difícil calcular P(B).
# - En lugar de dividir por P(B), simplemente sumamos los resultados y normalizamos. El total *es* P(B).
# - En Machine Learning, para convertir las salidas de una red neuronal (logits) en probabilidades (la función Softmax es una normalización).
#
# Ventajas:
# - Un paso simple que garantiza que nuestros números obedezcan las reglas de la probabilidad.
# - Nos permite "saltarnos" el cálculo de P(B) en la Regla de Bayes.

# --- Algoritmo de Normalización ---

def normalizar(puntuaciones):
    """
    Toma un diccionario de {etiqueta: puntuacion} y lo normaliza
    para que todos los valores sumen 1.0.
    """
    
    # 1. Calcular la suma total de todas las puntuaciones
    # (puntuaciones.values() obtiene una lista de los valores: [0.4, 0.2, 0.1])
    total = sum(puntuaciones.values()) # total = 0.4 + 0.2 + 0.1 = 0.7
    
    # Manejar el caso de división por cero (si todas las puntuaciones son 0)
    if total == 0:
        print("Error: El total de puntuaciones es 0, no se puede normalizar.")
        # Podríamos devolver 0 para todo, o una distribución uniforme
        num_items = len(puntuaciones)
        if num_items > 0:
            return {etiqueta: 1.0 / num_items for etiqueta in puntuaciones}
        else:
            return {}
            
    # 2. Crear un nuevo diccionario para las probabilidades normalizadas
    probabilidades_normalizadas = {}
    
    # 3. Iterar sobre cada (etiqueta, puntuacion) en el diccionario original
    for etiqueta, puntuacion in puntuaciones.items():
        
        # 4. Dividir la puntuación individual por el total
        probabilidad = puntuacion / total
        
        # Guardar en el nuevo diccionario
        probabilidades_normalizadas[etiqueta] = probabilidad
        
    # 5. Devolver el diccionario de probabilidades válidas
    return probabilidades_normalizadas

# --- Ejecutar el cálculo ---
print("\n--- 3b. Normalización ---") # Título

# Las "puntuaciones" (probabilidades no normalizadas) del ejemplo del doctor
puntuaciones_no_normalizadas = {
    'Enfermedad 1': 0.4,
    'Enfermedad 2': 0.2,
    'Enfermedad 3': 0.1
}

print(f"Puntuaciones (no normalizadas): {puntuaciones_no_normalizadas}")
print(f"Suma de puntuaciones: {sum(puntuaciones_no_normalizadas.values())}") # Suma 0.7

# Llamar a la función de normalización
distribucion_prob = normalizar(puntuaciones_no_normalizadas)

print(f"\nProbabilidades Normalizadas: {distribucion_prob}")

# Comprobar que la nueva suma es 1
print(f"Nueva suma (debe ser 1.0): {sum(distribucion_prob.values())}")

# Resultados esperados:
# Enfermedad 1: 0.4 / 0.7 = 0.571...
# Enfermedad 2: 0.2 / 0.7 = 0.285...
# Enfermedad 3: 0.1 / 0.7 = 0.142...
# Suma: 1.0