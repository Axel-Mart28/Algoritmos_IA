# ---ALGORITMO DE RED DE HAMMING (HAMMING NETWORK) ---

# Este es un tipo de red neuronal utilizada para *reconocimiento de patrones*,
# específicamente para encontrar cuál de un conjunto de patrones *memorizados*
# es el *más cercano* (en distancia de Hamming) a un patrón de entrada.
# Es útil para corregir errores en patrones binarios.
#
# Definición:
# Una Red de Hamming compara un patrón de entrada (generalmente binario bipolar: +1, -1)
# con un conjunto de patrones prototipo almacenados. Identifica el prototipo
# que tiene la menor distancia de Hamming (menor número de bits diferentes)
# respecto a la entrada.
#
# Distancia de Hamming: Número de posiciones en las que dos vectores binarios difieren.
#
# ¿Cómo funciona? (Arquitectura de dos capas):
# [Image of a Hamming Network structure with feedforward matching layer and recurrent competitive layer]
# 1. CAPA DE ENTRADA/COINCIDENCIA (Feedforward - Similar al Perceptrón):
#    - Cada neurona representa un patrón prototipo almacenado.
#    - Los *pesos* de la neurona 'j' son *exactamente* el patrón prototipo 'j'.
#    - Calcula una "puntuación de coincidencia" entre la entrada 'x' y cada prototipo 'p_j'.
#      La puntuación es a menudo (x · p_j + n) / 2, donde 'n' es la longitud del vector.
#      Esto es proporcional a (n - Distancia_Hamming). Una puntuación *alta*
#      significa *baja* distancia.
#
# 2. CAPA COMPETITIVA (Recurrente - Winner-Take-All):
#    - Cada neurona en esta capa corresponde a una neurona de la capa anterior.
#    - Las neuronas *inhiben* a todas las demás neuronas en esta capa.
#    - A través de iteraciones, solo la neurona que recibió la *mayor*
#      puntuación de coincidencia (la más cercana) permanecerá activa.
#    - La salida final indica qué prototipo fue el ganador.
#
# ¿Cómo funciona este programa?
# Simularemos los cálculos de la *primera capa* (coincidencia) y luego
# simplemente encontraremos el *máximo* para simular la capa competitiva.
# Usaremos patrones bipolares (+1 para bit '1', -1 para bit '0').

import numpy as np # Para operaciones vectoriales

# --- P1: Definir los Patrones Prototipo ---
# (Los patrones que la red ha "memorizado")
# Usaremos vectores bipolares (+1, -1) de longitud 4.
prototipos = {
    'P1': np.array([+1, +1, -1, -1]),
    'P2': np.array([-1, -1, +1, +1]),
    'P3': np.array([+1, -1, +1, -1])
}
n = 4 # Longitud de los vectores

print("-ALGORITMO DE HAMMING-") # Título
print("Patrones Prototipo Almacenados:") # Mensaje
for nombre, p in prototipos.items(): # Iterar sobre prototipos
    print(f"  {nombre}: {p}") # Imprimir prototipo

# --- P2: Simular la Red de Hamming ---

def hamming_network_predict(entrada, prototipos):
    """
    Simula la Red de Hamming para encontrar el prototipo más cercano.
    """
    n_local = len(entrada) # Longitud del vector de entrada
    puntuaciones = {} # Diccionario para guardar {nombre_prototipo: puntuacion}
    
    # --- 1. Calcular Puntuaciones de Coincidencia (Capa 1) ---
    print("\nCalculando puntuaciones de coincidencia (proporcional a n - HammingDist):")
    for nombre, p in prototipos.items(): # Iterar sobre cada prototipo
        # Puntuación = (Entrada · Prototipo + n) / 2
        # (El producto punto x·p es alto si son similares)
        producto_punto = np.dot(entrada, p) # Calcular producto punto
        puntuacion = (producto_punto + n_local) / 2.0 # Calcular puntuación
        puntuaciones[nombre] = puntuacion # Guardar puntuación
        print(f"  Entrada vs {nombre}: {puntuacion:.1f}") # Imprimir puntuación
        
    # --- 2. Encontrar el Ganador (Simula Capa 2 Competitiva) ---
    # Encontrar el nombre del prototipo con la puntuación MÁS ALTA
    ganador = max(puntuaciones, key=puntuaciones.get) # Encontrar clave con valor máximo
    
    return ganador, puntuaciones[ganador] # Devolver nombre y puntuación del ganador

# --- P3: Ejecutar con una Entrada (con un error) ---

# Entrada: Es P1 ([+1,+1,-1,-1]) pero con un error en el último bit.
entrada_test = np.array([+1, +1, -1, +1])

print(f"\nEntrada de prueba: {entrada_test}") # Imprimir entrada
print(" (Es como P1 pero con 1 error -> Dist Hamming = 1)") # Explicación

# Llamar a la función de la red
prototipo_ganador, puntuacion_ganadora = hamming_network_predict(entrada_test, prototipos) # Ejecutar

print("\n--- Resultado ---") # Título resultado
print(f"El prototipo más cercano (ganador) es: {prototipo_ganador}") # Imprimir ganador
print(f"Puntuación de coincidencia máxima: {puntuacion_ganadora:.1f}") # Imprimir puntuación
# (P1 debería ganar con puntuación 3.0: ((1*1 + 1*1 + -1*-1 + 1*-1) + 4) / 2 = (1+1+1-1+4)/2 = 6/2 = 3)
# (P2 debería tener puntuación 1.0)
# (P3 debería tener puntuación 1.0)

print("\nConclusión:")
print("La Red de Hamming calculó qué prototipo almacenado era el más")
print("similar (menor distancia de Hamming) a la entrada ruidosa,")
print("realizando efectivamente una corrección de errores simple.")