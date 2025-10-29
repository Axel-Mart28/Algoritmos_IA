# --- ALGORITMO DE MAPAS AUTOORGANIZADOS DE KOHONEN (SELF-ORGANIZING MAPS - SOM) ---

# Este es un algoritmo de *Aprendizaje No Supervisado* y un tipo
# especial de Red Neuronal.
#
# Definición:
# Un SOM es una red neuronal que "aprende" a mapear datos de entrada de
# alta dimensión a una rejilla de neuronas de baja dimensión (generalmente 2D),
# de tal manera que las neuronas *cercanas* en la rejilla respondan a
# datos de entrada *similares*. Preserva la topología.
# ¿Cómo funciona? (Aprendizaje Competitivo):
# 
# 1. INICIALIZACIÓN:
#    - Se crea una rejilla de neuronas (ej. 10x10).
#    - Cada neurona tiene un "vector de pesos" con la *misma dimensión*
#      que los datos de entrada. Estos pesos se inicializan aleatoriamente
#      (o con alguna estrategia).
#
# 2. BUCLE DE ENTRENAMIENTO (repetir para cada dato de entrada):
#    a. COMPETENCIA:
#       - Se toma un vector de datos de entrada (ej. un color RGB).
#       - Se calcula la distancia (ej. Euclidiana) entre este vector de entrada
#         y el *vector de pesos* de *cada* neurona en la rejilla.
#       - Se encuentra la neurona cuyo vector de pesos es *más cercano*
#         a la entrada. Esta es la "Neurona Ganadora" o "Best Matching Unit" (BMU).
#
#    b. COOPERACIÓN (Actualización de Vecindario):
#       - No solo se actualiza la BMU, sino también sus *vecinas* en la rejilla.
#       - Se define una "función de vecindad" (generalmente Gaussiana) centrada
#         en la BMU. Las neuronas cercanas a la BMU se ven más afectadas;
#         las lejanas, menos.
#       - El "radio" de esta vecindad *disminuye* con el tiempo (al principio
#         se actualizan grandes zonas, al final solo la BMU y sus vecinas muy cercanas).
#
#    c. ADAPTACIÓN (Actualización de Pesos):
#       - Los pesos de la BMU y sus vecinas se "mueven" *hacia* el vector de entrada.
#       - La cantidad que se mueven depende de:
#         - La Tasa de Aprendizaje (learning rate, alpha): También disminuye con el tiempo.
#         - La Función de Vecindad: Qué tan cerca está la neurona de la BMU.
#       - Fórmula: w_nuevo = w_viejo + alpha * vecindad * (entrada - w_viejo)
#
# 3. RESULTADO:
#    - Después del entrenamiento, las neuronas en la rejilla se habrán
#      "especializado" en diferentes tipos de datos de entrada.
#    - Neuronas cercanas en la rejilla representarán entradas similares.
#    - El SOM ha creado un "mapa topológico" de los datos.
#
# Componentes:
# 1. Rejilla de Neuronas (ej. 10x10).
# 2. Vectores de Pesos (uno por neurona, dimensión = dim_entrada).
# 3. Tasa de Aprendizaje (alpha, decreciente).
# 4. Radio de Vecindad (sigma, decreciente).
# 5. Función de Distancia.
# 6. Función de Vecindad (ej. Gaussiana).
#
# Aplicaciones:
# - Visualización de datos de alta dimensión (ej. agrupar documentos por tema).
# - Reducción de dimensionalidad.
# - Clustering (aunque k-Means suele ser más directo para clustering puro).
# - Detección de novedades.
#
# Ventajas:
# - Excelente para visualizar la estructura de los datos.
# - Preserva las relaciones topológicas (proximidad).
# - No Supervisado (no necesita etiquetas).
#
# Desventajas:
# - Requiere elegir el tamaño de la rejilla, la tasa de aprendizaje, etc.
# - No produce clusters "duros" como k-Means, sino un mapa continuo.
# - El entrenamiento puede ser lento.

# --- P1: Importar Bibliotecas ---
import numpy as np # Para operaciones numéricas
import matplotlib.pyplot as plt # Para visualización

# --- P2: Implementación Simplificada del SOM ---
# (Nota: Una implementación robusta es más compleja, a menudo se usan bibliotecas como MiniSom)

class SimpleSOM:
    def __init__(self, input_dim, map_size=(10, 10), sigma=1.0, learning_rate=0.5, random_seed=42):
        """ Constructor del SOM """
        self.input_dim = input_dim # Dimensión de los datos de entrada (ej. 3 para RGB)
        self.map_size = map_size # Tamaño de la rejilla (filas, columnas)
        self.n_neuronas = map_size[0] * map_size[1] # Número total de neuronas

        # Parámetros iniciales que decrecerán con el tiempo
        self.init_sigma = sigma # Radio de vecindad inicial
        self.init_learning_rate = learning_rate # Tasa de aprendizaje inicial

        # Inicializar los pesos aleatoriamente
        # Pesos: array de [n_neuronas, input_dim]
        np.random.seed(random_seed) # Para reproducibilidad
        self.pesos = np.random.rand(self.n_neuronas, self.input_dim)

        # Crear coordenadas para cada neurona en la rejilla (para calcular vecindad)
        # ej. [[0,0], [0,1], ..., [9,9]]
        self.coords_rejilla = np.array([[i, j] for i in range(map_size[0]) for j in range(map_size[1])])

    def _encontrar_bmu(self, vector_entrada):
        """ Encuentra el índice de la Neurona Ganadora (BMU) """
        # Calcular distancias euclidianas entre la entrada y TODOS los pesos
        # np.linalg.norm(vector_entrada - self.pesos, axis=1) -> calcula la distancia para cada neurona
        distancias = np.linalg.norm(vector_entrada - self.pesos, axis=1)
        # Encontrar el índice de la neurona con la distancia mínima
        bmu_idx = np.argmin(distancias)
        return bmu_idx

    def _funcion_vecindad(self, bmu_idx, t, max_iter):
        """ Calcula la influencia de la BMU en otras neuronas (Gaussiana decreciente) """
        # --- Parámetros Decrecientes ---
        # Radio (sigma) disminuye linealmente
        sigma = self.init_sigma * (1.0 - t / max_iter)
        # (Asegurar que sigma no sea cero para evitar división por cero)
        sigma = max(sigma, 1e-5)

        # Coordenadas de la BMU en la rejilla 2D
        bmu_coord = self.coords_rejilla[bmu_idx]

        # Calcular distancia al cuadrado entre todas las neuronas y la BMU en la *rejilla*
        dist_cuadrado_rejilla = np.sum((self.coords_rejilla - bmu_coord)**2, axis=1)

        # Aplicar la función Gaussiana
        # influencia = exp(-(dist^2) / (2 * sigma^2))
        influencia = np.exp(-dist_cuadrado_rejilla / (2 * (sigma**2)))
        return influencia

    def _actualizar_pesos(self, vector_entrada, bmu_idx, t, max_iter):
        """ Adapta los pesos de la BMU y sus vecinos """
        # --- Parámetros Decrecientes ---
        # Tasa de aprendizaje (alpha) disminuye linealmente
        alpha = self.init_learning_rate * (1.0 - t / max_iter)

        # Calcular la influencia de la vecindad
        influencia = self._funcion_vecindad(bmu_idx, t, max_iter) # Vector [n_neuronas]

        # Calcular el delta (cambio) para TODOS los pesos
        # delta = alpha * influencia * (entrada - peso_actual)
        # 'influencia[:, np.newaxis]' convierte el vector [n,] en [n, 1] para broadcasting
        delta_pesos = alpha * influencia[:, np.newaxis] * (vector_entrada - self.pesos)

        # Actualizar los pesos
        self.pesos += delta_pesos

    def fit(self, X, num_iteraciones):
        """ Entrena el SOM con los datos X """
        print(f"Entrenando SOM por {num_iteraciones} iteraciones...")
        max_iter = float(num_iteraciones) # Para los cálculos decrecientes

        for t in range(num_iteraciones):
            # 1. Elegir un vector de entrada al azar
            idx_azar = np.random.randint(len(X))
            vector_entrada = X[idx_azar]

            # 2. Encontrar la BMU (Competencia)
            bmu_idx = self._encontrar_bmu(vector_entrada)

            # 3. Actualizar pesos (Cooperación y Adaptación)
            self._actualizar_pesos(vector_entrada, bmu_idx, t, max_iter)

            # Imprimir progreso (opcional)
            if (t + 1) % (num_iteraciones // 10) == 0:
                print(f"  Iteración {t + 1}/{num_iteraciones} completada.")

        print("¡Entrenamiento completado!")

    def obtener_mapa_pesos(self):
        """ Devuelve los pesos organizados en la forma de la rejilla """
        # Reorganiza el array plano de pesos [n_neuronas, dim]
        # en un array [map_rows, map_cols, dim]
        return self.pesos.reshape(self.map_size[0], self.map_size[1], self.input_dim)

# --- P3: Ejecutar SOM (Ejemplo con Colores RGB) ---
print("--- 7. Mapas Autoorganizados de Kohonen (SOM) - Ejemplo Colores ---")

# 1. Generar Datos de Colores Aleatorios (Entrada)
#    (Colores RGB: R, G, B van de 0.0 a 1.0)
n_colores = 500
input_dim = 3 # R, G, B
datos_colores = np.random.rand(n_colores, input_dim)
print(f"Generados {n_colores} colores aleatorios (dim={input_dim}).")

# 2. Crear y Entrenar el SOM
map_size = (20, 20) # Rejilla de 20x20 neuronas
sigma_inicial = map_size[0] / 2.0 # Radio inicial = mitad del tamaño del mapa
learning_rate_inicial = 0.5
iteraciones = 10000

som = SimpleSOM(input_dim, map_size, sigma_inicial, learning_rate_inicial)
som.fit(datos_colores, iteraciones)

# 3. Obtener y Visualizar el Mapa de Pesos
print("\nVisualizando el mapa de pesos del SOM...")
mapa_pesos = som.obtener_mapa_pesos() # Obtener pesos organizados [20, 20, 3]

# Graficar el mapa de pesos como una imagen de colores
plt.figure(figsize=(8, 8)) # Tamaño
# imshow() muestra un array 2D como imagen. Nuestros pesos son [20, 20, 3],
# que es exactamente el formato que espera para una imagen RGB.
plt.imshow(mapa_pesos, interpolation='nearest')
plt.title("Mapa de Pesos del SOM Entrenado (Mapa de Colores)")
plt.axis('off') # Ocultar ejes
plt.show() # Mostrar gráfico
# 

print("\nConclusión:")
print("El SOM aprendió a organizar los colores aleatorios.")
print("Observa cómo colores similares (ej. rojos, verdes, azules)")
print("terminan agrupados en regiones cercanas del mapa 2D,")
print("mostrando la preservación de la topología.")