# ALGORITMO DE APRENDIZAJE PROFUNDO (DEEP LEARNING)

# Este no es un *único* algoritmo, sino una *familia* de algoritmos
# basados en Redes Neuronales Artificiales (ANNs) con múltiples capas.
#
# Definición:
# Deep Learning utiliza redes neuronales con muchas capas ("profundas")
# para aprender representaciones jerárquicas de los datos.
# Las capas iniciales aprenden características simples (ej. bordes en una imagen),
# las capas intermedias combinan esas características (ej. formas, texturas),
# y las capas finales aprenden conceptos complejos (ej. objetos, caras).
#
# 
#
# ¿Cómo funciona? (La Idea General):
# 1. ARQUITECTURA: Se define una estructura de red con capas de "neuronas".
#    Cada neurona recibe entradas, calcula una suma ponderada, aplica una
#    "función de activación" (no lineal), y pasa la salida a la siguiente capa.
#    Arquitecturas comunes: MLP (Perceptrón Multicapa), CNN (Convolucionales),
#    RNN (Recurrentes).
#
# 2. ENTRENAMIENTO (Backpropagation y Descenso de Gradiente):
#    a. Forward Pass: Se pasan los datos de entrada a través de la red
#       para obtener una predicción.
#    b. Calcular Pérdida (Loss): Se compara la predicción con la etiqueta
#       real usando una "función de pérdida" (ej. Error Cuadrático Medio,
#       Entropía Cruzada).
#    c. Backward Pass (Backpropagation): Se calcula el "gradiente" de la
#       pérdida con respecto a *cada* peso (parámetro) de la red,
#       propagando el error hacia atrás.
#    d. Actualizar Pesos (Descenso de Gradiente): Se ajustan los pesos
#       en la dirección que *reduce* la pérdida, usando un "optimizador"
#       (ej. Adam, SGD).
#    e. Repetir: Se repiten estos pasos muchas veces (épocas) con
#       lotes (batches) de datos.
#
# 
#
# ¿Cómo funciona este programa?
# Implementaremos una red neuronal *muy simple* (un Perceptrón Multicapa - MLP)
# usando la biblioteca `scikit-learn`.
# 1. Generaremos datos de ejemplo (las "lunas" no lineales).
# 2. Crearemos una instancia del `MLPClassifier`.
# 3. Entrenaremos el modelo con `.fit()`.
# 4. Haremos predicciones con `.predict()`.
# 5. Evaluaremos el resultado.
# (Nota: Para Deep Learning "real", se usan bibliotecas como TensorFlow o PyTorch).
#
# Componentes (en scikit-learn):
# 1. `MLPClassifier`: La clase del clasificador.
# 2. `hidden_layer_sizes`: Define la arquitectura (ej. (10, 5) -> 2 capas ocultas).
# 3. `activation`: La función de activación ('relu', 'tanh', 'logistic').
# 4. `solver`: El optimizador ('adam', 'sgd').
# 5. `max_iter`: Número máximo de épocas de entrenamiento.
#
# Aplicaciones:
# - Casi todo en IA moderna Visión por Computadora (reconocimiento de objetos),
#   Procesamiento de Lenguaje Natural (traducción, etc)
#   Reconocimiento de Voz, Juegos (AlphaGo), Medicina (diagnóstico).
#
# Ventajas:
# - Rendimiento de vanguardia (State-of-the-art) en muchas tareas complejas.
# - Aprendizaje automático de características ("representation learning").
# - Flexibilidad increíble en las arquitecturas.
#
# Desventajas:
# - Requiere *muchísimos* datos.
# - Requiere *mucho* poder computacional (GPUs).
# - Son "cajas negras": difíciles de interpretar (¿por qué tomó esa decisión?).
# - Muy sensibles a la elección de la arquitectura y los hiperparámetros.
# - Propenso al "sobreajuste" (overfitting) si no se regulariza bien.

# --- P1: Importar Bibliotecas ---
from sklearn.neural_network import MLPClassifier # El clasificador MLP
from sklearn.datasets import make_moons # Usaremos los datos no lineales
from sklearn.model_selection import train_test_split # Para dividir datos
from sklearn.metrics import accuracy_score # Para evaluar
from sklearn.preprocessing import StandardScaler # ¡Importante! Escalar datos
import matplotlib.pyplot as plt # Para visualizar
import numpy as np # Para manejo numérico

# --- P2: Generar y Preparar Datos ---

print("---Aprendizaje Profundo (Deep Learning) ---") # Título

# Generar los datos no lineales (lunas)
X_moon, y_moon = make_moons(n_samples=200, noise=0.2, random_state=42) # Más puntos

# Dividir en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(
    X_moon, y_moon, test_size=0.3, random_state=42
)

# --- ¡PASO CRUCIAL para Redes Neuronales: Escalar los Datos! ---
# Las redes neuronales funcionan mejor si las características tienen
# media 0 y desviación estándar 1.
print("\nEscalando los datos...") # Mensaje
scaler = StandardScaler() # Crear el objeto escalador
X_train_scaled = scaler.fit_transform(X_train) # Ajustar y transformar entrenamiento
X_test_scaled = scaler.transform(X_test)       # Solo transformar prueba (con la escala de entrena.)

# --- P3: Crear y Entrenar el Modelo MLP ---

print("Creando y entrenando el Perceptrón Multicapa (MLP)...") # Mensaje

# Crear la instancia del MLPClassifier
mlp = MLPClassifier(
    hidden_layer_sizes=(10, 5), # Arquitectura: 2 capas ocultas, 10 neuronas y 5 neuronas
    activation='relu',       # Función de activación ReLU (común)
    solver='adam',           # Optimizador Adam (común)
    max_iter=500,            # Entrenar por 500 épocas (pasadas por los datos)
    random_state=42,         # Para reproducibilidad
    early_stopping=True,     # Detener si no mejora (evita sobreajuste)
    n_iter_no_change=20      # Paciencia para early_stopping
)

# Entrenar el modelo con los datos *escalados*
mlp.fit(X_train_scaled, y_train) # Entrenar

print("¡MLP entrenado!") # Mensaje

# --- P4: Hacer Predicciones y Evaluar ---

print("\nHaciendo predicciones en el conjunto de prueba...") # Mensaje
# Predecir usando los datos de prueba *escalados*
y_pred = mlp.predict(X_test_scaled) # Predecir

# Evaluar la precisión
accuracy = accuracy_score(y_test, y_pred) # Comparar
print(f"Precisión (Accuracy) del MLP: {accuracy * 100:.2f}%") # Imprimir resultado

# --- P5: Visualización (Opcional) ---
# (Usaremos una función similar a la de SVM para visualizar)
def plot_decision_boundary_mlp(X, y, model, scaler, title): # Función para graficar
    plt.figure(figsize=(8, 6)) # Crear figura
    # Graficar los puntos de datos (usando los datos *originales* para verlos bien)
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap='viridis', edgecolors='k', s=50)

    # Crear la malla (usando límites de los datos *originales*)
    ax = plt.gca()
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    xx, yy = np.meshgrid(np.linspace(xlim[0], xlim[1], 100),
                           np.linspace(ylim[0], ylim[1], 100))

    # ¡Importante! Escalar la malla *antes* de predecir
    mesh_scaled = scaler.transform(np.c_[xx.ravel(), yy.ravel()])

    # Predecir en la malla escalada
    Z = model.predict(mesh_scaled)
    Z = Z.reshape(xx.shape)

    # Graficar las regiones de decisión
    plt.contourf(xx, yy, Z, cmap='viridis', alpha=0.4)
    plt.title(title)
    plt.xlabel("Característica 1 (Original)")
    plt.ylabel("Característica 2 (Original)")
    plt.show()

print("\nGenerando visualización de la frontera de decisión...")
plot_decision_boundary_mlp(X_moon, y_moon, mlp, scaler, "MLP (Deep Learning) - Frontera de Decisión")

print("\nConclusión:")
print("La red neuronal (MLP) aprendió una frontera de decisión compleja")
print("para separar los datos no lineales (lunas), logrando una alta precisión.")
print("Este es un ejemplo muy simple; las redes reales tienen millones de parámetros.")