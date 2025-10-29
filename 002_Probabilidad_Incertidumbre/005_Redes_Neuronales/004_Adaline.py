# ALGORITMO DE ADALINE (ADAPTIVE LINEAR NEURON) ---

# Propuesto por Widrow y Hoff poco después del Perceptrón.
# Es muy similar, pero con una diferencia clave en el aprendizaje.
# Definición:
# ADALINE también es un clasificador binario de una sola neurona.
#
# La Diferencia Clave (Regla Delta):
# - Perceptrón: Usa la *salida de la función escalón* (0 o 1) para calcular
#   el error y actualizar los pesos. (Error = y_verdadera - y_predicha_escalon)
# - ADALINE: Usa la *salida de la suma ponderada* (un valor continuo, 'z')
#   *antes* de la función escalón para calcular el error y actualizar los pesos.
#   Define el error como el Error Cuadrático Medio (MSE) entre la salida
#   verdadera y la *salida lineal* ('z').
#   Actualiza los pesos usando "Descenso de Gradiente" para minimizar este MSE.
#   (Error = y_verdadera - z)
#   (Actualización = alpha * error * x) <--- Regla Delta / Widrow-Hoff
#
# ¿Qué significa esto?
# - ADALINE actualiza los pesos basándose en qué tan "lejos" está la suma 'z'
#   del valor objetivo (0 o 1), no solo si la clasificación fue "correcta" o "incorrecta".
# - Esto le permite aprender incluso si los datos *no* son perfectamente separables
#   (encontrará la "mejor línea posible" en términos de MSE).
# - La función escalón solo se usa *después* del entrenamiento, para la clasificación final.
#
# Componentes:
# - Igual que Perceptrón, pero la *función de activación durante el aprendizaje*
#   es la función identidad (f(z)=z).
# - La regla de actualización es la Regla Delta.
#
# Aplicaciones:
# - Histórico, precursor de redes más modernas.
# - Se usó en filtros adaptativos (cancelación de ruido).
#
# Ventajas:
# - Converge a la solución de mínimo error cuadrático (incluso si no es separable).
# - El uso del gradiente sentó bases para Backpropagation.
#
# Desventajas:
# - Sigue siendo una sola neurona, por lo que solo puede encontrar fronteras lineales.

import numpy as np # Usaremos NumPy para operaciones de vectores
import math # (No estrictamente necesario aquí)

class AdalineGD: # GD = Gradient Descent (Descenso de Gradiente)
    def __init__(self, tasa_aprendizaje=0.01, n_iter=50, random_state=1): # Constructor
        self.tasa_aprendizaje = tasa_aprendizaje # Tasa de aprendizaje (usualmente más pequeña)
        self.n_iter = n_iter               # Épocas
        self.random_state = random_state   # Semilla aleatoria
        self.pesos = None                  # Pesos w
        self.sesgo = None                  # Sesgo b
        self.costo_por_epoca = []          # Para guardar el Error Cuadrático Medio (MSE)

    def fit(self, X, y): # Método de entrenamiento
        """ Entrena ADALINE usando Descenso de Gradiente """
        rgen = np.random.RandomState(self.random_state) # Generador aleatorio
        self.pesos = rgen.normal(loc=0.0, scale=0.01, size=X.shape[1]) # Inicializar pesos
        self.sesgo = 0.0 # Inicializar sesgo

        self.costo_por_epoca = [] # Limpiar historial de costo

        for i in range(self.n_iter): # Bucle de épocas (cambiado de _ a i para imprimir)
            # 1. Calcular la salida LINEAL (z) para TODAS las muestras
            salida_lineal = self._suma_ponderada(X) # z = X*w + b (para todo X)

            # 2. Calcular el Error (y_verdadera - z) para TODAS las muestras
            errores = y - salida_lineal

            # 3. Actualizar Pesos (Regla Delta / Descenso de Gradiente)
            #    w = w + alpha * X.T.dot(errores)
            self.pesos += self.tasa_aprendizaje * X.T.dot(errores) # Actualiza todos los pesos

            # 4. Actualizar Sesgo
            #    b = b + alpha * Sum(error)
            self.sesgo += self.tasa_aprendizaje * errores.sum() # Actualiza el sesgo

            # 5. Calcular el Costo (MSE) para esta época y guardarlo
            costo = (errores**2).sum() / 2.0 # Suma de errores al cuadrado / 2
            self.costo_por_epoca.append(costo) # Guardar costo

            # Opcional: Imprimir costo para ver el progreso
            # print(f"  Época {i+1}: Costo MSE = {costo:.4f}")

        return self

    def _suma_ponderada(self, X): # Calcula z = w*x + b
        """ Calcula la entrada neta (salida lineal) """
        # Asegurarse de que X sea al menos 1D si es una sola muestra
        if X.ndim == 1:
             return np.dot(X, self.pesos) + self.sesgo
        else:
             return np.dot(X, self.pesos) + self.sesgo


    def predict(self, X): # Predicción final (usa la función escalón)
        """ Devuelve la etiqueta de clase (0 o 1) usando escalón """
        # La predicción final SÍ usa la función escalón
        return np.where(self._suma_ponderada(X) >= 0.0, 1, 0)

# --- Ejecutar ADALINE (Aprendiendo la puerta AND) ---

print("\n--- ADALINE ---") # Título

# --- ¡LA CORRECCIÓN! Definir los datos ANTES de usarlos ---
# Datos de la puerta AND (Linealmente Separable)
X_and = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y_and = np.array([0, 0, 0, 1])
# --- Fin de la corrección ---

# Crear y entrenar ADALINE
# (Aumentamos n_iter un poco para asegurar convergencia con eta=0.01)
adaline = AdalineGD(tasa_aprendizaje=0.01, n_iter=100)
adaline.fit(X_and, y_and) # Entrenar

print(f"\nPesos aprendidos: {adaline.pesos}") # Imprimir w
print(f"Sesgo aprendido: {adaline.sesgo}")   # Imprimir b
# Descomenta si quieres ver el costo bajar
# print(f"Costo (MSE) por época (últimos 10): {adaline.costo_por_epoca[-10:]}")

# Probar predicciones
print("\nPredicciones:")
for xi in X_and:
    # Asegurarse de pasar un array 2D si _suma_ponderada lo espera
    # O ajustar _suma_ponderada para manejar 1D (hecho arriba)
    print(f"  Entrada: {xi} -> Predicción: {adaline.predict(xi)}") # Imprimir

print("\nConclusión ADALINE:")
print("ADALINE también aprendió AND. La clave es que actualiza pesos")
print("usando la salida *lineal* y Descenso de Gradiente.")