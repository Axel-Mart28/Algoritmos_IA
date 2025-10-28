# Algoritmo de  REGLA DE BAYES

# Este es el algoritmo de inferencia probabilística más fundamental.
#
# Definición:
# Es una fórmula que nos permite "invertir" la probabilidad condicionada.
# Nos deja calcular P(H|E) (la probabilidad de una Hipótesis dada la Evidencia)
# usando P(E|H) (la probabilidad de la Evidencia dada la Hipótesis).
#
# ¿Por qué es esto tan importante?
# Porque en el mundo real, P(E|H) es *mucho* más fácil de saber que P(H|E).
# Ejemplo (Diagnóstico Médico):
# - P(E|H) = P(Test Positivo | Tienes Enfermedad): Es la "sensibilidad" del test.
#   Un laboratorio puede medir esto fácilmente (toman 100 enfermos y ven cuántos dan positivo).
# - P(H|E) = P(Tienes Enfermedad | Test Positivo): Esto es lo que el paciente
#   quiere saber. Es *muy* difícil de medir directamente.
#
# La Regla de Bayes nos permite calcular lo segundo usando lo primero.
#
#La Fórmula:
#
#              P(E | H) * P(H)
#    P(H | E) = -----------------
#                   P(E)
#
# Componentes (Los Nombres "Bayesianos"):
# - P(H | E) -> Posterior: La probabilidad *actualizada* de la hipótesis después
#               de ver la evidencia. Es lo que queremos calcular.
# - P(H) -> Prior (A Priori): Nuestra creencia *inicial* en la hipótesis (del tema #2).
# - P(E | H) -> Likelihood (Verosimilitud): Qué tan probable es la evidencia si la
#               hipótesis es verdadera.
# - P(E) -> Marginal Likelihood: La probabilidad total de la evidencia.
#           Actúa como la *Constante de Normalización* (del tema #3b).
#
# ¿Cómo funciona este programa?
# Implementaremos la Regla de Bayes usando el "truco" de la normalización,
# que es como se usa casi siempre en IA.
# 1. Calculamos el "Prior" P(H) y su opuesto P(¬H) (¬ = "no").
# 2. Calculamos los "Likelihoods" P(E|H) y P(E|¬H).
# 3. Calculamos una "puntuación" (probabilidad no normalizada) para cada caso:
#    - puntuacion_H   = P(E | H) * P(H)
#    - puntuacion_no_H = P(E | ¬H) * P(¬H)
# 4. Normalizamos estas puntuaciones (del tema #3b). El resultado es la
#    distribución de probabilidad posterior P(H|E) y P(¬H|E).
#
# Aplicaciones:
# - Diagnóstico médico, filtros de SPAM, sistemas de recomendación,
#   visión por computadora... básicamente, toda la IA moderna.
#
# Ventajas:
# - Nos da un método matemáticamente sólido para actualizar creencias.
#
# Desventajas:
# - Requiere conocer las probabilidades a priori (P(H)) y los likelihoods (P(E|H)),
#   los cuales pueden ser difíciles de estimar con precisión en el mundo real.

# Importamos la función de normalizacion
def normalizar(puntuaciones):
    """
    Toma un diccionario de {etiqueta: puntuacion} y lo normaliza
    para que todos los valores sumen 1.0.
    """
    total = sum(puntuaciones.values()) # Suma las puntuaciones
    if total == 0: # Evitar división por cero
        return {etiqueta: 0.0 for etiqueta in puntuaciones}
    # Devuelve el diccionario normalizado
    return {etiqueta: puntuacion / total for etiqueta, puntuacion in puntuaciones.items()}

# --- Algoritmo de la Regla de Bayes (con Normalización) ---

def regla_de_bayes(prior_H, likelihood_E_dado_H, likelihood_E_dado_no_H):
    """
    Calcula P(H|E) usando la Regla de Bayes y normalización.
    
    Argumentos:
    prior_H (float): P(H) - La prob. a priori de la hipótesis.
    likelihood_E_dado_H (float): P(E|H) - El "likelihood" (ej. sensibilidad del test).
    likelihood_E_dado_no_H (float): P(E|¬H) - El "likelihood" del opuesto (ej. tasa de falsos positivos).
    """
    
    # 1. Calcular el prior del opuesto
    # P(¬H) = 1 - P(H)
    prior_no_H = 1.0 - prior_H # Ej: 1.0 - 0.01 = 0.99
    
    # 2. Calcular las puntuaciones (probabilidades no normalizadas)
    # Esto es el *numerador* de la Regla de Bayes
    
    # Puntuación para H (Hipótesis Verdadera) = P(E|H) * P(H)
    puntuacion_H = likelihood_E_dado_H * prior_H # Ej: 0.99 * 0.01
    
    # Puntuación para ¬H (Hipótesis Falsa) = P(E|¬H) * P(¬H)
    puntuacion_no_H = likelihood_E_dado_no_H * prior_no_H # Ej: 0.05 * 0.99
    
    # 3. Crear el diccionario de puntuaciones
    puntuaciones = {
        'Hipótesis Verdadera (H)': puntuacion_H,
        'Hipótesis Falsa (¬H)': puntuacion_no_H
    }
    
    # 4. Normalizar las puntuaciones para obtener la distribución posterior
    #    La función normalizar() suma las puntuaciones (calculando P(E) por nosotros)
    #    y divide cada puntuación por ese total.
    distribucion_posterior = normalizar(puntuaciones)
    
    return distribucion_posterior # Devuelve el diccionario {P(H|E), P(¬H|E)}

# --- Ejecutar el cálculo (Ejemplo del Diagnóstico Médico) ---
print("--- 6. Regla de Bayes (Ejemplo de Test Médico) ---") # Título

# Hipótesis (H): El paciente tiene la enfermedad.
# Evidencia (E): El paciente dio positivo en el test.
# PREGUNTA: ¿Cuál es la P(H|E)? (Prob. de tener la enfermedad, DADO el positivo)

# --- Valores Conocidos (Nuestros "Inputs") ---

# 1. Prior: P(H)
# La creencia inicial. 1% de la población general tiene la enfermedad.
P_H = 0.01

# 2. Likelihood (Sensibilidad): P(E|H)
# Si *tienes* la enfermedad, el test da positivo el 99% de las veces.
P_E_dado_H = 0.99

# 3. Likelihood (Falso Positivo): P(E|¬H)
# Si *no* tienes la enfermedad, el test *aun así* da positivo el 5% de las veces.
P_E_dado_no_H = 0.05

# --- Imprimir los datos de entrada ---
print(f"Probabilidad a Priori, P(H) = {P_H * 100}% (Creencia inicial de tener la enfermedad)")
print(f"Sensibilidad, P(E|H) = {P_E_dado_H * 100}% (Test acierta si estás enfermo)")
print(f"Tasa Falso Positivo, P(E|¬H) = {P_E_dado_no_H * 100}% (Test falla si estás sano)")

# --- Llamar a la función de la Regla de Bayes ---
print("\nCalculando P(H|E)...")
distribucion_final = regla_de_bayes(P_H, P_E_dado_H, P_E_dado_no_H)

# --- Imprimir el Resultado ---
print("\n--- Resultado (Probabilidades Posteriores) ---")
prob_H_dado_E = distribucion_final['Hipótesis Verdadera (H)']
prob_no_H_dado_E = distribucion_final['Hipótesis Falsa (¬H)']

print(f"P(H|E) = P(Enfermedad | Positivo): {prob_H_dado_E:.4f} (o {prob_H_dado_E * 100:.2f}%)")
print(f"P(¬H|E) = P(No Enfermedad | Positivo): {prob_no_H_dado_E:.4f} (o {prob_no_H_dado_E * 100:.2f}%)")

print("\nConclusión:")
print("¡Aunque el test dio positivo, solo hay un 16.64% de probabilidad de")
print("tener la enfermedad! Esto se debe a que la tasa de falsos positivos (5%)")
print("es alta en comparación con la rareza de la enfermedad (1%).")