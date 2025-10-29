# --- ALGORITMO DE HEBB ---

# Este NO es una arquitectura de red, sino una *REGLA DE APRENDIZAJE*
# fundamental en neurociencia y redes neuronales.
# Es un tipo de aprendizaje *No Supervisado*.
#
# Definición (Postulado de Hebb):
# "Cuando un axón de la célula A está lo suficientemente cerca como para excitar
#  a la célula B y participa repetida o persistentemente en su disparo,
#  ocurre algún proceso de crecimiento o cambio metabólico en una o ambas
#  células, de modo que la eficiencia de A, como una de las células que
#  disparan a B, aumenta."

# La Regla Matemática Simple:
# El cambio en el peso de la conexión (w_ij) entre la neurona 'i' (presináptica)
# y la neurona 'j' (postsináptica) es proporcional al producto de sus activaciones.
#
# delta_w_ij = tasa_aprendizaje * activacion_i * activacion_j
#
# ¿Cómo funciona este programa?
# Simularemos esta regla simple para *una sola conexión* (un solo peso)
# y veremos cómo cambia basándose en diferentes combinaciones de
# activación de las neuronas pre y postsináptica.

# --- P1: Implementación de la Regla de Hebb ---

def regla_hebb(peso_actual, activacion_pre, activacion_post, tasa_aprendizaje):
    """
    Calcula el nuevo peso usando la regla básica de Hebb.
    """
    # delta_w = eta * pre * post
    delta_peso = tasa_aprendizaje * activacion_pre * activacion_post # Calcular cambio
    # w_nuevo = w_viejo + delta_w
    nuevo_peso = peso_actual + delta_peso # Calcular nuevo peso
    return nuevo_peso # Devolver

# --- P2: Simulación de la Regla ---
print("\n--- Algoritmo de HEBB ---") # Título

# Parámetros iniciales
peso_inicial = 0.5 # Peso inicial de la conexión
eta = 0.1         # Tasa de aprendizaje

print(f"Peso Inicial: {peso_inicial}") # Imprimir inicio
print(f"Tasa de Aprendizaje (eta): {eta}") # Imprimir eta

# --- Caso 1: Ambas neuronas activas (+1) ---
print("\nCaso 1: Neurona Pre (+1) y Post (+1) disparan juntas")
pre = 1.0 # Activación presináptica
post = 1.0 # Activación postsináptica
nuevo_peso_1 = regla_hebb(peso_inicial, pre, post, eta) # Calcular
delta_1 = nuevo_peso_1 - peso_inicial # Calcular cambio
print(f"  Delta Peso = {eta} * {pre} * {post} = {delta_1:.2f}") # Imprimir cambio
print(f"  Nuevo Peso = {peso_inicial} + {delta_1:.2f} = {nuevo_peso_1:.2f}") # Imprimir resultado
print("  -> ¡La conexión se FORTALECE!") # Conclusión

# --- Caso 2: Neurona Pre activa (+1), Post inactiva (-1 o 0) ---
# (Usaremos -1 para bipolar, aunque con 0 también funcionaría diferente)
print("\nCaso 2: Neurona Pre (+1) dispara, Post (-1) no dispara")
pre = 1.0
post = -1.0 # Neurona postsináptica inhibida o inactiva
nuevo_peso_2 = regla_hebb(nuevo_peso_1, pre, post, eta) # Calcular
delta_2 = nuevo_peso_2 - nuevo_peso_1 # Calcular cambio
print(f"  Delta Peso = {eta} * {pre} * {post} = {delta_2:.2f}") # Imprimir cambio
print(f"  Nuevo Peso = {nuevo_peso_1:.2f} + {delta_2:.2f} = {nuevo_peso_2:.2f}") # Imprimir resultado
print("  -> ¡La conexión se DEBILITA!") # Conclusión

# --- Caso 3: Ambas neuronas inactivas (-1) ---
# (En el modelo bipolar, esto también fortalece si están correlacionadas)
print("\nCaso 3: Neurona Pre (-1) y Post (-1) están inactivas juntas")
pre = -1.0
post = -1.0
nuevo_peso_3 = regla_hebb(nuevo_peso_2, pre, post, eta) # Calcular
delta_3 = nuevo_peso_3 - nuevo_peso_2 # Calcular cambio
print(f"  Delta Peso = {eta} * {pre} * {post} = {delta_3:.2f}") # Imprimir cambio
print(f"  Nuevo Peso = {nuevo_peso_2:.2f} + {delta_3:.2f} = {nuevo_peso_3:.2f}") # Imprimir resultado
print("  -> ¡La conexión se FORTALECE (correlación)! ") # Conclusión

print("\nConclusión:")
print("La Regla de Hebb ajusta los pesos basándose en la correlación")
print("entre las activaciones de las neuronas pre y postsináptica.")
print("Es la base del aprendizaje 'no supervisado' y asociativo,")
print("y se usa (en formas modificadas) en redes como Hopfield.")