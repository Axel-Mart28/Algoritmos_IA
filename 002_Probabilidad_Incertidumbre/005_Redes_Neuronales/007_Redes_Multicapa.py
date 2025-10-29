# --- 5. REDES MULTICAPA (MULTILAYER NETWORKS / MLP) ---

# Definición:
# Una Red Multicapa es una red neuronal artificial que tiene
# al menos *una* capa de neuronas "oculta" (hidden layer)
# entre la capa de entrada (input layer) y la capa de salida (output layer).
#
# ¿Por qué son necesarias?
# - Como vimos, las redes de una sola capa solo pueden aprender
#   fronteras de decisión *lineales*.
# - ¡Las Redes Multicapa pueden aprender fronteras de decisión
#   *no lineales* y complejas! Pueden resolver problemas como XOR.
#
# ¿Cómo funciona este programa?
# Este código es conceptual. Reafirmaremos la estructura y nos
# referiremos al ejemplo de `MLPClassifier` que usamos en el
# de "Aprendizaje Probabilístico" (Deep Learning), ya que
# ese *era* un ejemplo de Red Multicapa.

# --- P1: Importar Bibliotecas ---
from sklearn.neural_network import MLPClassifier # El clasificador MLP
from sklearn.datasets import make_moons # Generador de datos no lineales
from sklearn.model_selection import train_test_split # Para dividir datos
from sklearn.preprocessing import StandardScaler # Para escalar datos
from sklearn.metrics import accuracy_score # Para evaluar el rendimiento
import matplotlib.pyplot as plt # Para graficar
import numpy as np # Para operaciones numéricas

# --- P2: Generar y Preparar Datos ---
print("---Ejemplo de Red Multicapa (MLP) ---") # Título

# Generar datos de "lunas" (no linealmente separables)
X, y = make_moons(n_samples=200, noise=0.2, random_state=42) # 200 puntos, 2 clases

# Dividir en conjuntos de entrenamiento (70%) y prueba (30%)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# Escalar los datos (media 0, varianza 1)
print("Escalando características...") # Mensaje
scaler = StandardScaler() # Crear el objeto escalador
# Ajustar el escalador SÓLO con los datos de entrenamiento
# y luego transformar ambos conjuntos
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test) # Usar la misma transformación

# --- P3: Definir y Entrenar el MLP ---
print("Definiendo y entrenando el MLP...") # Mensaje

# Crear la instancia del clasificador MLP
mlp = MLPClassifier(
    hidden_layer_sizes=(10,), # Arquitectura: UNA capa oculta con 10 neuronas
    activation='relu',       # Función de activación ReLU
    solver='adam',           # Optimizador Adam
    max_iter=500,            # Número máximo de épocas
    random_state=42,         # Semilla para reproducibilidad
    early_stopping=False     # (Desactivado para este ejemplo simple)
)
# Nota: (10,) significa una capa oculta. (10, 5) serían dos capas.

# Entrenar la red neuronal con los datos escalados
mlp.fit(X_train_scaled, y_train) # ¡El aprendizaje ocurre aquí!

print("¡Entrenamiento completado!") # Mensaje

# --- P4: Realizar Predicciones y Evaluar ---
print("\nRealizando predicciones...") # Mensaje

# Predecir las etiquetas para el conjunto de prueba (escalado)
y_pred = mlp.predict(X_test_scaled) # Obtener las predicciones (0 o 1)

# Calcular la precisión (accuracy)
accuracy = accuracy_score(y_test, y_pred) # Comparar predicciones vs. etiquetas reales
print(f"Precisión del MLP en el conjunto de prueba: {accuracy * 100:.2f}%") # Imprimir resultado

# --- P5: Visualizar la Frontera de Decisión ---
# (Usaremos la misma función auxiliar de ejemplos anteriores)
def plot_decision_boundary_mlp(X, y, model, scaler, title): # Función para graficar
    plt.figure(figsize=(8, 6)) # Crear figura
    # Graficar puntos originales (sin escalar)
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap='viridis', edgecolors='k', s=50, alpha=0.7)

    # Crear malla (con límites de datos originales)
    ax = plt.gca()
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    xx, yy = np.meshgrid(np.linspace(xlim[0], xlim[1], 100),
                           np.linspace(ylim[0], ylim[1], 100))

    # Escalar la malla
    mesh_scaled = scaler.transform(np.c_[xx.ravel(), yy.ravel()])
    # Predecir en la malla escalada
    Z = model.predict(mesh_scaled)
    Z = Z.reshape(xx.shape)

    # Graficar regiones de decisión
    plt.contourf(xx, yy, Z, cmap='viridis', alpha=0.4)
    plt.title(title)
    plt.xlabel("Característica 1")
    plt.ylabel("Característica 2")
    plt.show()

print("Generando visualización...") # Mensaje
# 
plot_decision_boundary_mlp(X, y, mlp, scaler, "MLP (1 Capa Oculta) - Frontera de Decisión")

print("\nConclusión:")
print("La Red Multicapa (MLP) aprendió una frontera de decisión NO LINEAL")
print("capaz de separar las 'lunas', algo que un Perceptrón o ADALINE no podría.")
print("Esto es gracias a la capa oculta y la función de activación no lineal.")