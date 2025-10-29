# ALGORITMO DE PERCEPTRON 

# Este es el *primer* modelo de red neuronal artificial, propuesto por Frank Rosenblatt en 1957.
# Es una *única neurona* (como la del tema #1) que usa la función escalón.
#
# Definición:
# Un Perceptrón es un clasificador binario (produce salida 0 o 1) para datos linealmente separables.
#
# Linealmente Separable:
# Significa que puedes dibujar una *única línea recta* (o plano/hiperplano) para separar perfectamente los puntos de las dos clases.
# ¿Cómo funciona? (El Modelo):
# Es exactamente la neurona que vimos en el tema #1:
# 1. Recibe entradas (x1, x2, ...).
# 2. Tiene pesos (w1, w2, ...).
# 3. Calcula la suma ponderada: z = Sum(xi*wi) + b
# 4. Aplica la función escalón: y = 1 si z >= 0, sino 0.
#
# ¿Cómo aprende? (La Regla de Aprendizaje del Perceptrón):
# Es un algoritmo *supervisado* e *iterativo*.
# 1. Inicializa los pesos (w) y el sesgo (b) aleatoriamente o a cero.
# 2. Para cada ejemplo de entrenamiento (x, y_verdadera):
# 3.   Calcula la predicción de la neurona (y_predicha).
# 4.   Calcula el error: error = y_verdadera - y_predicha (será 0, 1, o -1).
# 5.   Si hay error (error != 0):
# 6.     *Actualiza* cada peso: w_i = w_i + tasa_aprendizaje * error * x_i
# 7.     *Actualiza* el sesgo: b = b + tasa_aprendizaje * error
# 8. Repite los pasos 2-7 muchas veces (épocas) hasta que no haya errores.
#
# Componentes:
# 1. Entradas binarias o reales.
# 2. Pesos y Sesgo.
# 3. Función de activación Escalon (Step).
# 4. Tasa de Aprendizaje (Learning Rate, alpha): Controla el tamaño de los ajustes.
#
# Aplicaciones:
# - Principalmente histórico y educativo. Fue el inicio de todo.
# - Puede aprender puertas lógicas simples (AND, OR, NAND, NOR).
#
# Ventajas:
# - Muy simple.
# - Garantiza converger *si* los datos son linealmente separables.
#
# Desventajas:
# - Solo funciona si los datos son linealmente separables
#  - No puede aprender la puerta XOR. Esta limitación causó el primer "invierno de la IA".
# - La función escalón no es diferenciable, impidiendo usar
#   métodos de gradiente más avanzados (como backpropagation).

import numpy as np # Usaremos NumPy para operaciones de vectores

class Perceptron: # Clase para el Perceptrón
    def __init__(self, tasa_aprendizaje=0.1, n_iter=50, random_state=1): # Constructor
        self.tasa_aprendizaje = tasa_aprendizaje # Qué tan grandes son los pasos de ajuste
        self.n_iter = n_iter               # Número de épocas (pasadas por los datos)
        self.random_state = random_state   # Para inicialización aleatoria reproducible
        self.pesos = None                  # Vector de pesos w
        self.sesgo = None                  # Sesgo b
        self.errores_por_epoca = []        # Para ver cómo aprende

    def fit(self, X, y): # Método de entrenamiento
        """ Entrena el Perceptrón con datos X y etiquetas y """
        # Inicializar pesos y sesgo
        rgen = np.random.RandomState(self.random_state) # Generador aleatorio
        # Pesos pequeños aleatorios (uno por cada característica en X)
        self.pesos = rgen.normal(loc=0.0, scale=0.01, size=X.shape[1])
        self.sesgo = 0.0 # Inicializar sesgo a 0

        self.errores_por_epoca = [] # Limpiar historial de errores

        for _ in range(self.n_iter): # Bucle de épocas
            errores_epoca = 0 # Contador de errores en esta época
            # Iterar sobre cada muestra (xi) y su etiqueta (yi)
            for xi, yi_verdadera in zip(X, y):
                # Calcular la predicción
                prediccion = self.predict(xi)
                # Calcular el error
                error = yi_verdadera - prediccion
                # Actualizar pesos y sesgo *si* hay error
                if error != 0:
                    # w = w + alpha * error * x
                    actualizacion = self.tasa_aprendizaje * error
                    self.pesos += actualizacion * xi
                    # b = b + alpha * error
                    self.sesgo += actualizacion
                    errores_epoca += 1 # Contar el error
            self.errores_por_epoca.append(errores_epoca) # Guardar errores de la época
            # Opcional: Parar si no hubo errores
            if errores_epoca == 0:
                print(f"  Convergencia alcanzada en la época {_ + 1}")
                #break
        return self

    def _suma_ponderada(self, X): # Calcula la suma z = w*x + b
        """ Calcula la entrada neta (suma ponderada + sesgo) """
        # np.dot(X, self.pesos) es el producto punto (x1*w1 + x2*w2 + ...)
        return np.dot(X, self.pesos) + self.sesgo

    def predict(self, X): # Hace la predicción final
        """ Devuelve la etiqueta de clase (0 o 1) """
        # Aplica la función escalón a la suma ponderada
        # np.where(condición, valor_si_true, valor_si_false)
        return np.where(self._suma_ponderada(X) >= 0.0, 1, 0)

# --- Ejecutar Perceptrón (Aprendiendo la puerta AND) ---
print("--- PERCEPTRON ---") # Título

# Datos de la puerta AND (Linealmente Separable)
# Entradas X: [[0,0], [0,1], [1,0], [1,1]]
# Salidas y:  [  0,    0,    0,    1  ]
X_and = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y_and = np.array([0, 0, 0, 1])

# Crear y entrenar el Perceptrón
percep = Perceptron(tasa_aprendizaje=0.1, n_iter=10) # Crear instancia
percep.fit(X_and, y_and) # Entrenar

print(f"\nPesos aprendidos: {percep.pesos}") # Imprimir w
print(f"Sesgo aprendido: {percep.sesgo}")   # Imprimir b
print(f"Errores por época: {percep.errores_por_epoca}") # Ver historial de aprendizaje

# Probar predicciones
print("\nPredicciones:")
for xi in X_and:
    print(f"  Entrada: {xi} -> Predicción: {percep.predict(xi)}") # Imprimir

print("\nConclusión Perceptrón:")
print("El Perceptrón aprendió exitosamente la función AND porque")
print("es linealmente separable.")