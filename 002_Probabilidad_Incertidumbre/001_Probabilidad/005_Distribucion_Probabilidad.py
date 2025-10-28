# Algoritmo de DISTRIBUCIÓN DE PROBABILIDAD

# Este "algoritmo" es en realidad una "estructura de datos".
# Definición:
# Es una estructura (generalmente un diccionario o tabla) que describe todos los posibles resultados* de una variable aleatoria y la probabilidad asociada a *cada uno* de esos resultados.

# ¿Cómo funciona (Reglas)?:
# Una distribución de probabilidad (discreta) debe seguir dos reglas:
# 1. Cada probabilidad individual P(x) debe estar entre 0 y 1.
# 2. La *suma* de todas las probabilidades de todos los resultados posibles debe ser exactamente 1.0 (100%).

# ¿Cómo funciona este programa?:
# Escribiremos una función que toma una lista de datos crudos (como el historial del clima) y calcula la *distribución de probabilidad completa* para esa variable, contando la frecuencia de cada resultado.

# Componentes:
# 1. Variable Aleatoria (X): La variable que estamos midiendo (ej. "Clima").
# 2. Espacio Muestral: El conjunto de todos los resultados (ej. {'Soleado', 'Lluvioso', 'Nublado'}).
# 3. Función de Probabilidad: El mapeo de cada resultado a su prob. (ej. 'Soleado' -> 0.6).

# Aplicaciones:
# - Describir cualquier fenómeno aleatorio: el resultado de un dado, el clima de mañana, el próximo ganador de la lotería.
# - Es la estructura de datos de entrada para casi todos los algoritmos de inferencia probabilística.

# Ventajas:
# - Proporciona una imagen completa y resumida de la incertidumbre.
#
# Desventajas:
# - Puede ser grande si la variable tiene millones de resultados posibles.
# - En el mundo real, la *verdadera* distribución nunca se conoce exactamente; solo podemos *estimarla* (como haremos aquí).
#
# Ejemplo de uso:
# - Para un dado justo: P(Dado) = {1: 0.166, 2: 0.166, 3: 0.166, 4: 0.166, 5: 0.166, 6: 0.166}
# - El  programa creará esto para la variable "Clima".

import math # Lo usaremos para verificar la suma (manejo de decimales)
from collections import Counter # Una forma muy eficiente de contar ítems en una lista

# --- Datos de ejemplo (nuestro "historial de observaciones") ---
# Los mismos datos que en el tema #2
historial_clima = ['Soleado', 'Lluvioso', 'Nublado', 'Soleado', 'Soleado', 
                   'Lluvioso', 'Soleado', 'Nublado', 'Soleado', 'Soleado']

# --- Algoritmo para Generar una Distribución de Probabilidad ---

def calcular_distribucion_probabilidad(datos):
    """
    Toma una lista de datos crudos y devuelve un diccionario
    que representa la distribución de probabilidad completa.
    """
    
    # 1. Contar cuántas veces apareció cada resultado único
    #    Counter(datos) crea un diccionario como:
    #    {'Soleado': 6, 'Lluvioso': 2, 'Nublado': 2}
    conteo_eventos = Counter(datos) # Diccionario de conteos
    
    # 2. Obtener el número total de observaciones
    total_observaciones = len(datos) # Longitud de la lista (10)
    
    # Manejar el caso de una lista vacía
    if total_observaciones == 0:
        return {} # Devuelve una distribución vacía
        
    # 3. Crear el diccionario de distribución de probabilidad
    distribucion = {} # Inicializar el diccionario final
    
    # 4. Iterar sobre el diccionario de conteos
    for evento, conteo in conteo_eventos.items():
        
        # 5. Calcular la probabilidad (frecuencia) para este evento
        probabilidad = conteo / total_observaciones # Ej: 6 / 10 = 0.6
        
        # 6. Asignar la probabilidad al evento en el diccionario final
        distribucion[evento] = probabilidad
        
    return distribucion # Devuelve el diccionario completo

# --- Función Auxiliar para Verificar una Distribución ---

def verificar_distribucion(distribucion):
    """
    Comprueba si un diccionario es una distribución de probabilidad válida.
    Devuelve (True/False, suma_total)
    """
    
    # 1. Obtener todos los valores de probabilidad
    probabilidades = distribucion.values() # [0.6, 0.2, 0.2]
    
    # 2. Comprobar la Regla 1: ¿Están todos entre 0 y 1?
    #    all() comprueba si cada ítem en la lista es Verdadero
    if not all(0 <= p <= 1 for p in probabilidades):
        return False, sum(probabilidades) # No es válida si alguna prob está fuera de rango
        
    # 3. Comprobar la Regla 2: ¿Suman (casi) 1.0?
    suma_total = sum(probabilidades) # Suma todos los valores
    
    # Usamos math.isclose() para manejar pequeños errores de punto flotante
    # (ej. 0.9999999999999 es lo mismo que 1.0)
    if not math.isclose(suma_total, 1.0):
        return False, suma_total # No es válida si la suma no es 1
        
    return True, suma_total # Si pasa ambas pruebas, es válida

# --- Ejecutar el cálculo ---
print("--- 4. Distribución de Probabilidad ---") # Título
print(f"Historial de datos crudos: {historial_clima}") # Muestra los datos

# 1. Llamar a la función para generar la distribución
dist_clima = calcular_distribucion_probabilidad(historial_clima)

print(f"\nDistribución de Probabilidad (P(Clima)):") # Título
# Imprimir de forma bonita
for evento, prob in dist_clima.items():
    print(f"  P({evento}) = {prob:.2f}") # Imprime P(Soleado) = 0.60, etc.

# 2. Verificar si la distribución que creamos es válida
es_valida, suma = verificar_distribucion(dist_clima)

print(f"\nVerificación:")
print(f"  Suma total de probabilidades: {suma}") # Imprime 1.0
print(f"  ¿Es una distribución válida? {es_valida}") # Imprime True