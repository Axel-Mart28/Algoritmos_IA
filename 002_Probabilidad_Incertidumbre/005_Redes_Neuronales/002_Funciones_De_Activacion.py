# ALGORITMO DE FUNCIONES DE ACTIVACIÓN 

# Este tema describe las diferentes funciones matemáticas que una
# neurona artificial puede usar para transformar su "suma ponderada" (z) en su "salida final" (y). y = f(z).

# Definición:
# Una función de activación decide si una neurona debe "activarse"
# (disparar una señal) o no, basándose en la suma total de sus entradas.
# Introduce *no linealidad* en la red.
#
# ¿Por qué son importantes (No Linealidad)?
# - Si *no* usáramos funciones de activación (o usáramos solo
#   funciones lineales, como f(z)=z), una red neuronal multicapa
#   sería matemáticamente equivalente a una *única* capa lineal.
# - ¡No podría aprender patrones complejos! Sería solo una regresión lineal glorificada.
# - Las funciones de activación *no lineales* permiten a la red
#   aprender fronteras de decisión curvas y relaciones complejas.
#
# ¿Cómo funciona este programa?
# Implementaremos y visualizaremos varias funciones de activación comunes para entender sus propiedades.

import math 
import numpy as np # Para generar rangos de valores para graficar
import matplotlib.pyplot as plt # Para visualizar las funciones

# --- P1: Implementación de Funciones de Activación Comunes ---

# 1. Función Escalon (Step Function) - (La que usamos antes)
def funcion_escalon(z):
    """ Devuelve 1 si z >= 0, sino 0. """
    return 1 if z >= 0 else 0
    # Propiedades: No lineal, salida binaria, ¡derivada 0 (mala para backprop)!.

# 2. Función Sigmoide (Logistic)
def funcion_sigmoide(z):
    """ Devuelve 1 / (1 + e^(-z)). Escala la salida entre 0 y 1. """
    return 1.0 / (1.0 + math.exp(-z))
    # Propiedades: No lineal, salida (0, 1) (interpretable como prob.),
    #              ¡problema del "gradiente evanescente"! (derivada casi 0 lejos de z=0).
    # 

# 3. Función Tangente Hiperbólica (Tanh)
def funcion_tanh(z):
    """ Devuelve (e^z - e^(-z)) / (e^z + e^(-z)). Escala la salida entre -1 y 1. """
    return math.tanh(z)
    # Propiedades: No lineal, salida (-1, 1) (centrada en cero, a veces mejor que Sigmoid),
    #              ¡también sufre de gradiente evanescente!
    # 

# 4. Función ReLU (Rectified Linear Unit) - ¡La más popular!
def funcion_relu(z):
    """ Devuelve max(0, z). """
    return max(0, z)
    # Propiedades: No lineal (debido al "codo" en z=0), muy rápida computacionalmente,
    #              ¡evita el gradiente evanescente para z>0!,
    #              puede sufrir el problema de "neuronas muertas" (si z siempre es <0).
    # 

# 5. Función Leaky ReLU (Variante de ReLU)
def funcion_leaky_relu(z, alpha=0.01):
    """ Devuelve z si z > 0, sino alpha * z. """
    return z if z > 0 else alpha * z
    # Propiedades: Intenta solucionar el problema de "neuronas muertas"
    #              permitiendo un pequeño gradiente negativo.

# 6. Función Softmax (¡Especial! Se usa en la CAPA DE SALIDA)
#    (No la implementaremos aquí como función simple, ya que opera
#     sobre un *vector* de salidas para convertirlas en probabilidades)
#    - Propiedades: Convierte un vector de puntuaciones (logits) en una
#                   distribución de probabilidad (suma 1). Ideal para
#                   clasificación multiclase.

# --- P2: Visualización de las Funciones ---
print("Funciones de Activación Comunes") # Título

# Generar un rango de valores para 'z' (la suma ponderada)
# np.linspace(-5, 5, 100) crea 100 puntos entre -5 y 5
z_valores = np.linspace(-5, 5, 100)

# Calcular las salidas 'y' para cada función
y_escalon = [funcion_escalon(z) for z in z_valores]
y_sigmoide = [funcion_sigmoide(z) for z in z_valores]
y_tanh = [funcion_tanh(z) for z in z_valores]
y_relu = [funcion_relu(z) for z in z_valores]
y_leaky_relu = [funcion_leaky_relu(z) for z in z_valores]

# Crear el gráfico
plt.figure(figsize=(10, 6)) # Tamaño de la figura

# Graficar cada función
plt.plot(z_valores, y_escalon, label='Escalon (Step)', linestyle='--')
plt.plot(z_valores, y_sigmoide, label='Sigmoide (Logistic)')
plt.plot(z_valores, y_tanh, label='Tanh')
plt.plot(z_valores, y_relu, label='ReLU')
plt.plot(z_valores, y_leaky_relu, label='Leaky ReLU', linestyle=':')

# Añadir títulos y leyenda
plt.title("Funciones de Activación Comunes") # Título del gráfico
plt.xlabel("Suma Ponderada (z)") # Etiqueta Eje X
plt.ylabel("Salida de la Neurona (y)") # Etiqueta Eje Y
plt.grid(True) # Añadir rejilla
plt.legend() # Mostrar leyenda
plt.ylim(-1.5, 1.5) # Ajustar límites del eje Y para mejor visualización
plt.axhline(0, color='black', linewidth=0.5) # Eje X
plt.axvline(0, color='black', linewidth=0.5) # Eje Y

# Mostrar el gráfico
plt.show() # Mostrar la ventana del gráfico

print("\nConclusión:")
print("Cada función de activación tiene propiedades diferentes.")
print("La elección afecta cómo aprende la red y qué tan bien funciona.")
print("ReLU es la más común en capas ocultas hoy en día por su eficiencia")
print("y por mitigar el problema del gradiente evanescente.")