# ALGORITMO DE  COMPUTACIÓN NEURONAL (NEURONAL COMPUTATION)

# Este es el CONCEPTO fundamental: la idea de imitar (de forma muy simplificada) cómo funciona una neurona biológica para realizar cálculos.
# Es el bloque de construcción de todas las Redes Neuronales Artificiales (ANNs).
#
# Definición:
# Una "neurona artificial" (también llamada "unidad" o "nodo") es una unidad matemática que recibe una o más *entradas*, realiza un cálculo simple, y produce una *salida*.
# Elementos:
# 1. ENTRADAS (Inputs, x): La neurona recibe señales (números) de
#    otras neuronas o directamente de los datos de entrada. (ej. x1, x2, x3)
#
# 2. PESOS (Weights, w): Cada conexión de entrada tiene un "peso" asociado.
#    Este peso representa la *importancia* o *fuerza* de esa conexión.
#    Un peso alto significa que la entrada es muy influyente; un peso bajo
#    o negativo significa que es menos influyente o inhibitoria.
#    (ej. w1, w2, w3). ¡Estos pesos son los parámetros que la red *aprende*!
#
# 3. SUMA PONDERADA (Weighted Sum, z): La neurona multiplica cada entrada
#    por su peso correspondiente y suma todos los resultados. A menudo
#    se añade un término extra llamado "sesgo" (bias, b).
#    z = (x1 * w1) + (x2 * w2) + (x3 * w3) + b
#
# 4. FUNCIÓN DE ACTIVACIÓN (Activation Function, f): La suma ponderada 'z'
#    se pasa a través de una función (generalmente no lineal) que decide
#    la *salida final* (y) de la neurona. Esta función introduce
#    complejidad y permite a la red aprender patrones no lineales.
#    (Veremos ejemplos en el siguiente tema).
#    y = f(z)
#
# 5. SALIDA (Output, y): El resultado final de la neurona, que se envía
#    a otras neuronas en la siguiente capa o es la predicción final.
#
# ¿Cómo funciona este programa?
# Vamos a simular una *única* neurona artificial simple con 3 entradas.
# Usaremos una función de activación básica (la función escalón/step).

import math # (No es estrictamente necesario aquí)

# --- P1: Definición de la Neurona Simple (con Función Escalon) ---

def funcion_activacion_escalon(suma_ponderada):
    """
    Función de activación simple:
    - Devuelve 1 si la entrada es >= 0
    - Devuelve 0 si la entrada es < 0
    (Simula si una neurona "se dispara" o no)
    """
    if suma_ponderada >= 0: # Si la suma alcanza el umbral (0 en este caso)
        return 1 # La neurona se "activa"
    else:
        return 0 # La neurona permanece "inactiva"

def neurona_artificial(entradas, pesos, sesgo):
    """
    Simula el cálculo de una neurona artificial.
    """
    
    # 1. Comprobar que el número de entradas y pesos coincida
    if len(entradas) != len(pesos): # Validar dimensiones
        raise ValueError("Número de entradas y pesos debe ser igual") # Error
        
    # 2. Calcular la SUMA PONDERADA (z)
    suma_ponderada = 0.0 # Inicializar la suma
    # Iterar sobre cada par (entrada, peso)
    for x_i, w_i in zip(entradas, pesos): 
        suma_ponderada += x_i * w_i # Acumular x_i * w_i
        
    # Añadir el sesgo (bias)
    suma_ponderada += sesgo # z = (x1*w1 + x2*w2 + ...) + b
    
    # 3. Aplicar la FUNCIÓN DE ACTIVACIÓN
    salida = funcion_activacion_escalon(suma_ponderada) # y = f(z)
    
    # 4. Devolver la SALIDA
    return salida

# --- P2: Ejecutar la Simulación de la Neurona ---
print("---Computación Neuronal ---") # Título

# Definir las entradas (ej. características de un dato)
entradas_x = [0.5, 1.0, -0.2] 

# Definir los pesos (¡estos se APRENDERÍAN en una red real!)
pesos_w = [0.8, -0.5, 1.5]

# Definir el sesgo (bias)
sesgo_b = 0.1

print(f"Entradas (x): {entradas_x}") # Imprimir entradas
print(f"Pesos (w): {pesos_w}") # Imprimir pesos
print(f"Sesgo (b): {sesgo_b}") # Imprimir sesgo

# Calcular la suma ponderada manualmente para verificar:
# z = (0.5 * 0.8) + (1.0 * -0.5) + (-0.2 * 1.5) + 0.1
# z = 0.4 - 0.5 - 0.3 + 0.1
# z = -0.3
suma_calculada = (entradas_x[0]*pesos_w[0] + 
                  entradas_x[1]*pesos_w[1] + 
                  entradas_x[2]*pesos_w[2] + sesgo_b)
print(f"\nSuma Ponderada (z) = Sum(xi*wi) + b = {suma_calculada:.2f}") # Imprimir z

# Llamar a la función de la neurona
salida_y = neurona_artificial(entradas_x, pesos_w, sesgo_b) # Calcular la salida

# Calcular la activación manualmente para verificar:
# y = escalon(-0.3) = 0
print(f"Función de Activación: Escalon (Step)") # Indicar activación
print(f"Salida (y) = f(z) = {salida_y}") # Imprimir y (debería ser 0)

print("\nConclusión:")
print("La neurona artificial combinó linealmente sus entradas usando")
print("los pesos, añadió un sesgo, y aplicó una función de activación")
print("para producir una salida final (0 o 1 en este caso).")
print("Las Redes Neuronales conectan muchas de estas unidades.")