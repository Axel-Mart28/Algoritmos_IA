# ALGORITMO K-NEAREST NEIGHBORS (k-NN) 

# Este es un algoritmo de *Aprendizaje Supervisado* utilizado para Clasificación (y a veces Regresión). Es uno de los algoritmos más simples e intuitivos.
#
# Definición:
# k-NN clasifica un *nuevo* punto de datos basándose en la "mayoría de votos"
# de sus 'k' vecinos más cercanos en el conjunto de datos de *entrenamiento*.
#
# Si quieres saber a qué partido político pertenece una persona nueva,
# miras a sus 'k' vecinos más cercanos en el mapa. Si la mayoría
# de ellos son del partido 'A', predices que la persona nueva
# también es del partido 'A'.
#
# ¿Cómo funciona?
# 1. FASE DE "ENTRENAMIENTO" (Memorización):
#    - k-NN no "aprende" un modelo en el sentido tradicional.
#    - Simplemente *memoriza* TODOS los datos de entrenamiento (puntos y sus etiquetas).
#
# 2. FASE DE PREDICCIÓN (Para un nuevo punto 'x'):
#    a. Calcular la Distancia: Calcula la distancia entre el nuevo punto 'x'
#       y *todos* los puntos en el conjunto de entrenamiento memorizado.
#       (La distancia más común es la Euclidiana).
#    b. Encontrar los k Vecinos: Identifica los 'k' puntos de entrenamiento
#       que están *más cerca* del nuevo punto 'x'. ('k' es un número que elegimos, ej. k=3).
#    c. Votar: Mira las etiquetas (clases) de esos 'k' vecinos.
#    d. Decidir: La clase que aparece *más veces* entre los 'k' vecinos
#       es la predicción para el nuevo punto 'x'.
#       (Si hay empate, se puede romper al azar o con alguna regla).
#
# 
#
# Componentes:
# 1. Los Datos de Entrenamiento (X_train, y_train): Puntos con etiquetas conocidas.
# 2. El parámetro 'k': El número de vecinos a considerar.
# 3. Una Métrica de Distancia: Cómo medir la "cercanía" (ej. Euclidiana).
#
# Aplicaciones:
# - Sistemas de recomendación ("Clientes que compraron X también compraron Y").
# - Reconocimiento de patrones (ej. OCR de dígitos escritos a mano).
# - Clasificación simple cuando no se quiere entrenar un modelo complejo.
#
# Ventajas:
# - Muy simple de entender e implementar.
# - No requiere entrenamiento (solo memorización).
# - Flexible a diferentes métricas de distancia.
#
# Desventajas:
# - Lento en la predicción Debe calcular la distancia a *todos* los puntos
#   de entrenamiento para cada nueva predicción. (Inviable para datasets gigantes).
# - Sensible a la elección de 'k'.
# - Sensible a características irrelevantes (que pueden "distorsionar" la distancia).
# - Sensible a la escala de los datos (ej. si 'edad' va de 0-100 y 'salario'
#   de 0-1,000,000, el salario dominará la distancia. Se necesita *normalizar* los datos).
#
# Ejemplo de uso:
# - Tenemos datos de flores (largo_pétalo, ancho_pétalo) y sus especies ('setosa', 'versicolor').
# - Llega una flor nueva sin etiqueta.
# - Calculamos su distancia a todas las flores conocidas.
# - Si k=5, miramos las 5 más cercanas. Si 4 son 'setosa' y 1 es 'versicolor',
#   predecimos que la nueva flor es 'setosa'.

import math # Para calcular la distancia euclidiana (sqrt)
from collections import Counter # Para contar los votos de los vecinos

# --- P1: Función de Distancia Euclidiana ---

def distancia_euclidiana(punto1, punto2):
    """ Calcula la distancia euclidiana entre dos puntos (listas o tuplas) """
    # Asegurarse de que los puntos tengan la misma dimensión
    if len(punto1) != len(punto2): # Comprobar longitudes
        raise ValueError("Los puntos deben tener la misma dimensión") # Error si no
        
    # Calcular la suma de los cuadrados de las diferencias
    suma_cuadrados = 0.0 # Inicializar suma
    for i in range(len(punto1)): # Iterar sobre cada dimensión (ej. x, y)
        diferencia = punto1[i] - punto2[i] # Calcular diferencia (ej. x1 - x2)
        suma_cuadrados += diferencia * diferencia # Sumar el cuadrado (dif^2)
        
    # Devolver la raíz cuadrada de la suma
    return math.sqrt(suma_cuadrados) # Devuelve sqrt( (x1-x2)^2 + (y1-y2)^2 + ... )

# --- P2: Algoritmo k-NN (Clasificación) ---

class KNearestNeighbors: # Clase para el clasificador k-NN
    
    def __init__(self, k=3): # Constructor
        # 'k' es el número de vecinos a considerar
        self.k = k # Almacenar k
        # Variables para almacenar los datos de entrenamiento ("memorización")
        self.X_train = None # Puntos de entrenamiento
        self.y_train = None # Etiquetas de entrenamiento

    def fit(self, X_train, y_train): # El método de "Entrenamiento" (solo memoriza)
        """ Memoriza los datos de entrenamiento """
        self.X_train = X_train # Guarda los puntos
        self.y_train = y_train # Guarda las etiquetas

    def predict(self, X_test): # El método de "Clasificación" para nuevos puntos
        """ Predice las etiquetas para una lista de nuevos puntos X_test """
        # Usar una list comprehension para predecir cada punto en X_test
        return [self._predict_one(x) for x in X_test]

    def _predict_one(self, x): # Función auxiliar para clasificar un *solo* punto nuevo 'x'
        """ Predice la etiqueta para un único punto 'x' """
        
        # 1. Calcular las distancias entre 'x' y *todos* los puntos de entrenamiento
        distancias = [] # Lista para guardar (distancia, etiqueta)
        for i, x_train_point in enumerate(self.X_train): # Iterar sobre los puntos memorizados
            # Calcular la distancia euclidiana
            dist = distancia_euclidiana(x, x_train_point)
            # Guardar la distancia junto con la etiqueta original del punto de entrenamiento
            distancias.append((dist, self.y_train[i]))
            
        # 2. Ordenar las distancias (de menor a mayor)
        #    sorted() ordena la lista de tuplas basándose en el primer elemento (la distancia)
        distancias_ordenadas = sorted(distancias)
        
        # 3. Obtener las etiquetas de los 'k' vecinos más cercanos
        #    Tomar los primeros 'k' elementos de la lista ordenada
        k_vecinos = distancias_ordenadas[:self.k] # Ej: [(0.5, 'A'), (0.7, 'B'), (0.9, 'A')] si k=3
        
        # Extraer solo las etiquetas de esos k vecinos
        etiquetas_vecinos = [etiqueta for dist, etiqueta in k_vecinos] # Ej: ['A', 'B', 'A']
        
        # 4. Votar: Encontrar la etiqueta más común
        #    Counter(etiquetas_vecinos) -> {'A': 2, 'B': 1}
        #    .most_common(1) -> [('A', 2)] (Lista con la tupla más común)
        #    [0][0] -> Extraer el primer elemento ('A') de la primera tupla
        mas_comun = Counter(etiquetas_vecinos).most_common(1)[0][0]
        
        # 5. Devolver la etiqueta más común como predicción
        return mas_comun

# --- P3: Ejecutar el Clasificador (Ejemplo Simple 2D) ---
print("k-Nearest Neighbors (k-NN)") # Título

# 1. Datos de Entrenamiento (X_train, y_train)
#    Imaginemos puntos en un plano 2D con dos clases ('A', 'B')
X_train = [
    [1, 1], [1, 2], [2, 2], # Clase A (abajo a la izquierda)
    [5, 5], [5, 6], [6, 6]  # Clase B (arriba a la derecha)
]
y_train = ['A', 'A', 'A', 'B', 'B', 'B']

# 2. Crear y "Entrenar" (memorizar) el clasificador
k = 3 # Usaremos 3 vecinos
knn_classifier = KNearestNeighbors(k=k) # Crear la instancia
knn_classifier.fit(X_train, y_train)    # Memorizar los datos

print(f"Clasificador k-NN entrenado (memorizado) con k={k}.") # Mensaje

# 3. Datos de Prueba (Nuevos puntos sin etiqueta)
X_test = [
    [1.5, 1.8], # Punto cerca del grupo A
    [5.5, 5.8], # Punto cerca del grupo B
    [3, 3]      # Punto en medio
]

print(f"\nClasificando {len(X_test)} nuevos puntos:") # Mensaje

# 4. Obtener Predicciones
predicciones = knn_classifier.predict(X_test) # Llamar al método predict

# 5. Imprimir resultados
for punto, pred in zip(X_test, predicciones): # Emparejar punto con su predicción
    print(f"  Punto: {punto} ==> Predicción: Clase '{pred}'") # Imprimir resultado


print("\nConclusión:")
print("k-NN clasificó los nuevos puntos basándose en la mayoría")
print("de votos de sus vecinos más cercanos en el espacio de características.")