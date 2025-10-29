# ALGORITMO DE RETROPROPAGACIÓN DEL ERROR (BACKPROPAGATION) 

# Este programa implementa Backpropagation desde cero usando NumPy
# para entrenar un MLP simple (2 entradas -> 2 neuronas ocultas -> 1 salida)
# en el problema XOR.
#
# Objetivo:
# Demostrar los pasos matemáticos del cálculo de gradientes (Backward Pass)
# y la actualización de pesos (Descenso de Gradiente).

import numpy as np # Fundamental para operaciones matriciales eficientes

# --- P1: Funciones de Activación y Pérdida ---

def sigmoid(x):
    """ Función de activación Sigmoide """
    # Previene overflow con valores muy negativos
    return 1 / (1 + np.exp(-np.clip(x, -500, 500)))

def sigmoid_derivative(x):
    """ Derivada de la función Sigmoide """
    # Calcula la derivada a partir de la *salida* de la sigmoide (eficiente)
    fx = sigmoid(x)
    return fx * (1 - fx)

def mean_squared_error(y_true, y_pred):
    """ Función de Pérdida: Error Cuadrático Medio (MSE) """
    return np.mean((y_true - y_pred)**2)

# --- P2: Clase de la Red Neuronal (MLP Simple) ---

class SimpleMLP:
    def __init__(self, input_size, hidden_size, output_size, learning_rate=0.1):
        # Constructor: Define la arquitectura e inicializa los pesos
        self.input_size = input_size   # Neuronas en capa de entrada (ej. 2 para XOR)
        self.hidden_size = hidden_size # Neuronas en capa oculta (ej. 2)
        self.output_size = output_size # Neuronas en capa de salida (ej. 1 para XOR)
        self.learning_rate = learning_rate # Tasa de aprendizaje para el descenso de gradiente

        # --- Inicialización de Pesos y Sesgos ---
        # Pesos entre Entrada y Oculta (matriz input_size x hidden_size)
        self.W_h = np.random.randn(self.input_size, self.hidden_size) * 0.1
        # Sesgos de la Capa Oculta (vector 1 x hidden_size)
        self.b_h = np.zeros((1, self.hidden_size))

        # Pesos entre Oculta y Salida (matriz hidden_size x output_size)
        self.W_o = np.random.randn(self.hidden_size, self.output_size) * 0.1
        # Sesgos de la Capa de Salida (vector 1 x output_size)
        self.b_o = np.zeros((1, self.output_size))

    def forward_pass(self, X):
        """ Realiza el Paso Hacia Adelante """
        # --- Capa Oculta ---
        # 1. Suma ponderada: z_h = X @ W_h + b_h
        self.z_h = np.dot(X, self.W_h) + self.b_h
        # 2. Activación: a_h = sigmoid(z_h)
        self.a_h = sigmoid(self.z_h)

        # --- Capa de Salida ---
        # 3. Suma ponderada: z_o = a_h @ W_o + b_o
        self.z_o = np.dot(self.a_h, self.W_o) + self.b_o
        # 4. Activación: a_o = sigmoid(z_o) (Predicción final)
        self.a_o = sigmoid(self.z_o)

        return self.a_o # Devuelve la predicción

    def backward_pass(self, X, y):
        """ Realiza el Paso Hacia Atrás (Backpropagation) y calcula gradientes """
        # --- Errores (Deltas) ---
        # (Usando la Regla de la Cadena: dL/dW = dL/da * da/dz * dz/dW)

        # 1. Error en la Capa de Salida (delta_o)
        #    delta_o = dL/da_o * da_o/dz_o
        #    dL/da_o para MSE = (a_o - y)
        #    da_o/dz_o = sigmoid'(z_o)
        error_output = self.a_o - y # (predicción - verdadero)
        delta_o = error_output * sigmoid_derivative(self.z_o) # Error ponderado por derivada

        # 2. Error en la Capa Oculta (delta_h)
        #    delta_h = (delta_o @ W_o.T) * da_h/dz_h
        #    (Propagar error hacia atrás) * (derivada activación oculta)
        error_hidden = np.dot(delta_o, self.W_o.T) # Propagar error
        delta_h = error_hidden * sigmoid_derivative(self.z_h) # Error ponderado

        # --- Calcular Gradientes ---
        # (Gradiente = Activación_Capa_Anterior.T @ Delta_Capa_Actual)

        # 3. Gradientes para Pesos Oculta -> Salida (W_o)
        #    grad_W_o = a_h.T @ delta_o
        self.grad_W_o = np.dot(self.a_h.T, delta_o)
        #    Gradiente para Sesgo de Salida (b_o)
        self.grad_b_o = np.sum(delta_o, axis=0, keepdims=True)

        # 4. Gradientes para Pesos Entrada -> Oculta (W_h)
        #    grad_W_h = X.T @ delta_h
        self.grad_W_h = np.dot(X.T, delta_h)
        #    Gradiente para Sesgo Oculto (b_h)
        self.grad_b_h = np.sum(delta_h, axis=0, keepdims=True)

    def update_weights(self):
        """ Actualiza los pesos usando Descenso de Gradiente """
        # W = W - tasa_aprendizaje * gradiente_W
        self.W_o -= self.learning_rate * self.grad_W_o
        self.b_o -= self.learning_rate * self.grad_b_o
        self.W_h -= self.learning_rate * self.grad_W_h
        self.b_h -= self.learning_rate * self.grad_b_h

    def fit(self, X, y, epochs=10000):
        """ Entrena la red neuronal """
        print(f"Entrenando por {epochs} épocas...")
        for epoch in range(epochs):
            # 1. Paso Hacia Adelante
            y_pred = self.forward_pass(X)

            # 2. Calcular Pérdida (solo para monitorear)
            loss = mean_squared_error(y, y_pred)

            # 3. Paso Hacia Atrás (Calcular Gradientes)
            self.backward_pass(X, y)

            # 4. Actualizar Pesos
            self.update_weights()

            # Imprimir progreso cada 1000 épocas
            if (epoch + 1) % 1000 == 0:
                print(f"  Época {epoch+1}/{epochs}, Pérdida (MSE): {loss:.6f}")

        print("¡Entrenamiento completado!")

    def predict(self, X):
        """ Hace predicciones (solo forward pass y redondeo) """
        probabilities = self.forward_pass(X)
        # Redondear a 0 o 1 para clasificación binaria
        return np.round(probabilities)

# --- P3: Preparar Datos (XOR) ---
print("--- RETROPROPAGACION DEL ERROR ---")

# Datos de entrada para XOR
X_xor = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
# Etiquetas de salida para XOR (reshape a columna)
y_xor = np.array([[0], [1], [1], [0]])

# --- P4: Crear y Entrenar la Red ---
# (2 entradas, 2 neuronas ocultas, 1 salida)
mlp_xor = SimpleMLP(input_size=2, hidden_size=2, output_size=1, learning_rate=0.5)
mlp_xor.fit(X_xor, y_xor, epochs=20000) # Entrenar por más épocas

# --- P5: Probar Predicciones ---
print("\nProbando predicciones en datos XOR:")
predictions = mlp_xor.predict(X_xor)

for i in range(len(X_xor)):
    print(f"  Entrada: {X_xor[i]} -> Predicción: {int(predictions[i][0])} (Esperado: {y_xor[i][0]})")

print("\nConclusión:")
print("La red MLP, entrenada con Backpropagation, aprendió a ajustar")
print("sus pesos (W_h, b_h, W_o, b_o) para mapear las entradas XOR")
print("a las salidas correctas, resolviendo el problema no lineal.")