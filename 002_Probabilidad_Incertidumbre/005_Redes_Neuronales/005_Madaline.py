# ALGORITMO DE MADALINE (MULTIPLE ADALINE) 

# Propuesto también por Widrow y Hoff, como una extensión de ADALINE.
# Es una de las *primeras redes neuronales MULTICAPA*.
#
# Definición:
# MADALINE conecta múltiples neuronas ADALINE en paralelo (primera capa)
# a una o más neuronas de salida (segunda capa), a menudo con lógica
# fija (ej. una puerta AND o OR) en la salida.
#
# Objetivo:
# Intentar superar la limitación de la separabilidad lineal del
# Perceptrón y ADALINE. ¡MADALINE SÍ puede resolver XOR!
#
# ¿Cómo funciona?
# - Capa Oculta: Múltiples unidades ADALINE, cada una aprendiendo
#   una frontera lineal diferente.
# - Capa de Salida: Combina las salidas de la capa oculta.
#   Originalmente, esta capa era fija (ej. una puerta AND).
#
# ¿Cómo aprende? (MRI - Minimum Disturbance Principle):
# - El algoritmo de entrenamiento original era complejo y heurístico.
# - Si la salida final era incorrecta, intentaba "voltear" (cambiar el peso)
#   la unidad ADALINE de la capa oculta que estaba *más cerca* de su
#   umbral (la que tenía la suma ponderada 'z' más cercana a 0),
#   con la esperanza de que eso corrigiera la salida final.
# - Era un intento temprano, antes de Backpropagation.
#
# Aplicaciones:
# - Histórico, demostró el potencial de las redes multicapa.
# - Filtros adaptativos.
#
# Ventajas:
# - Podía resolver problemas no linealmente separables (como XOR).
#
# Desventajas:
# - El algoritmo de entrenamiento original (MRI) no era muy eficiente
#   ni garantizaba encontrar la mejor solución.
# - Fue eclipsado por el desarrollo de Backpropagation para
#   redes multicapa (MLPs).

import numpy as np # Para operaciones vectoriales

# --- P1: Simulación de una Unidad ADALINE (Salida Lineal) ---
# (Esta vez no necesitamos la clase completa, solo la predicción lineal)

def adaline_output(X, pesos, sesgo): # Calcula la salida lineal z = w*x + b
    """ Calcula la salida lineal (suma ponderada + sesgo) de una ADALINE """
    return np.dot(X, pesos) + sesgo # Producto punto + sesgo

# --- P2: Estructura de MADALINE (para XOR) ---

# Pesos y sesgos PREDEFINIDOS para dos ADALINEs que *podrían* ayudar a resolver XOR.
# (En un sistema real, estos serían aprendidos por un algoritmo como MRI o Backprop).

# ADALINE 1 (Podría aprender a separar (0,0) del resto, activándose para (0,1), (1,0), (1,1))
pesos_1 = np.array([1.0, 1.0]) # w1=1, w2=1
sesgo_1 = -0.5                 # Umbral > 0.5

# ADALINE 2 (Podría aprender a separar (1,1) del resto, activándose para (0,0), (0,1), (1,0))
pesos_2 = np.array([-1.0, -1.0]) # w1=-1, w2=-1
sesgo_2 = 1.5                   # Umbral < 1.5

# Función de activación escalón (la misma de antes)
def funcion_escalon(z):
    return 1 if z >= 0 else 0

# Función de predicción de MADALINE
def madaline_predict_xor(X):
    """
    Simula la predicción de una MADALINE simple para XOR.
    Usa 2 ADALINEs en la capa oculta y una puerta AND en la salida.
    """
    # 1. Calcular salida lineal de cada ADALINE
    z1 = adaline_output(X, pesos_1, sesgo_1) # Salida de ADALINE 1
    z2 = adaline_output(X, pesos_2, sesgo_2) # Salida de ADALINE 2

    # 2. Aplicar función escalón a cada salida ADALINE (capa oculta)
    h1 = funcion_escalon(z1) # Salida binaria de ADALINE 1
    h2 = funcion_escalon(z2) # Salida binaria de ADALINE 2

    # 3. Combinar con una puerta AND (capa de salida fija)
    #    La salida es 1 solo si h1 es 1 Y h2 es 1.
    #    - Para [0,0]: z1=-0.5(h1=0), z2=1.5(h2=1) -> AND(0,1)=0 (Correcto)
    #    - Para [0,1]: z1=0.5 (h1=1), z2=0.5(h2=1) -> AND(1,1)=1 (Correcto)
    #    - Para [1,0]: z1=0.5 (h1=1), z2=0.5(h2=1) -> AND(1,1)=1 (Correcto)
    #    - Para [1,1]: z1=1.5 (h1=1), z2=-0.5(h2=0) -> AND(1,0)=0 (Correcto)
    output = 1 if (h1 == 1 and h2 == 1) else 0 # Puerta AND

    return output

# --- P3: Ejecutar la Simulación para XOR ---
print(" MADALINE") # Título

# Datos de la puerta XOR
X_xor = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y_xor = np.array([0, 1, 1, 0]) # Salidas esperadas

print("Pesos predefinidos para resolver XOR:")
print(f" ADALINE 1: Pesos={pesos_1}, Sesgo={sesgo_1}")
print(f" ADALINE 2: Pesos={pesos_2}, Sesgo={sesgo_2}")
print(" Capa de Salida: Puerta AND")

print("\nPredicciones de MADALINE para XOR:")
correctas = 0
for xi, yi_verdadera in zip(X_xor, y_xor): # Iterar sobre datos XOR
    prediccion = madaline_predict_xor(xi) # Obtener predicción
    print(f"  Entrada: {xi} -> Predicción MADALINE: {prediccion} (Esperado: {yi_verdadera})")
    if prediccion == yi_verdadera: # Contar aciertos
        correctas += 1

print(f"\nPrecisión: {correctas / len(y_xor) * 100:.0f}%") # Imprimir precisión

print("\nConclusión:")
print("Con los pesos adecuados (que podrían ser aprendidos),")
print("la estructura MADALINE (múltiples ADALINEs + lógica de salida)")
print("SÍ puede resolver problemas no linealmente separables como XOR,")
print("a diferencia del Perceptrón o ADALINE simple.")