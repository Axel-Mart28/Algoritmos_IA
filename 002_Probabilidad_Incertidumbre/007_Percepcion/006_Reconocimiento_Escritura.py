# --- 6. RECONOCIMIENTO DE ESCRITURA (Dígitos MNIST con k-NN) ---

# Concepto: Convertir imágenes de escritura a mano en texto digital.
# Objetivo: Clasificar imágenes de dígitos escritos a mano (0-9)
# Método: Aprendizaje Supervisado.
# Dataset: MNIST (un dataset estándar de 70,000 imágenes de 28x28 píxeles).


import numpy as np # Para manejo numérico
import matplotlib.pyplot as plt # Para visualizar las imágenes
from sklearn.datasets import fetch_openml # Para descargar MNIST
from sklearn.model_selection import train_test_split # Para dividir datos
from sklearn.neighbors import KNeighborsClassifier # El clasificador k-NN
from sklearn.metrics import accuracy_score # Para evaluar


print("-Reconocimiento de Escritura -")

# --- P1: Cargar el Dataset MNIST ---
# fetch_openml descarga datasets populares. MNIST tiene ID 554.
# Puede tardar un poco la primera vez que se descarga.
print("Cargando dataset MNIST (puede tardar)...")
try:
    # 'parser='auto'' es importante para versiones recientes de sklearn
    mnist = fetch_openml('mnist_784', version=1, cache=True, parser='auto')
    print("Dataset MNIST cargado.")
except Exception as e:
    print(f"Error al cargar MNIST: {e}")
    print("Asegúrate de tener conexión a internet o verifica el estado del dataset.")
    exit() # Salir si no se puede cargar

# Los datos vienen como:
# - mnist.data: Un array [70000, 784] (70000 imágenes aplanadas de 28x28=784 píxeles)
# - mnist.target: Un array [70000] con las etiquetas verdaderas ('0' a '9')
X = mnist.data.astype('float32') # Píxeles (0-255)
y = mnist.target.astype('int')   # Etiquetas (0-9)

# Reducir el tamaño del dataset para que k-NN sea más rápido (opcional)
# Usaremos solo una fracción de los datos para este ejemplo
print("Reduciendo tamaño del dataset para el ejemplo...")
X_subset, _, y_subset, _ = train_test_split(X, y, train_size=10000, stratify=y, random_state=42) # 10k para entrenar/probar
X = X_subset
y = y_subset

print(f"Usando un subconjunto de {len(X)} imágenes.")

# --- P2: Preprocesamiento y División de Datos ---

# 1. Normalizar los valores de los píxeles (de 0-255 a 0-1)
#    Esto ayuda a que el cálculo de distancia en k-NN funcione mejor.
X = X / 255.0

# 2. Dividir en conjuntos de entrenamiento y prueba
#    train_test_split mezcla y divide los datos.
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42 # 70% entrenamiento, 30% prueba
)
print(f"Datos divididos: {len(X_train)} para entrenar, {len(X_test)} para probar.")

# (Opcional: Escalar con StandardScaler, a veces no mejora mucho en MNIST con k-NN)
# scaler = StandardScaler()
# X_train = scaler.fit_transform(X_train)
# X_test = scaler.transform(X_test)

# --- P3: Entrenar ("Memorizar") el Clasificador k-NN ---
# Usaremos k=5 vecinos
k = 5
print(f"\nCreando y 'entrenando' el clasificador k-NN con k={k}...")

knn = KNeighborsClassifier(n_neighbors=k, n_jobs=-1) # n_jobs=-1 usa todos los cores CPU
# El "entrenamiento" de k-NN es solo almacenar los datos
knn.fit(X_train, y_train)

print("k-NN listo (datos memorizados).")

# --- P4: Realizar Predicciones en el Conjunto de Prueba ---
print("\nRealizando predicciones en el conjunto de prueba...")
# ¡Este paso puede ser LENTO con k-NN! Calcula distancias a todos los puntos de entrenamiento.
y_pred = knn.predict(X_test)
print("Predicciones completadas.")

# --- P5: Evaluar el Rendimiento ---
accuracy = accuracy_score(y_test, y_pred) # Comparar predicciones vs. verdad
print(f"\nPrecisión (Accuracy) del k-NN en MNIST (subconjunto): {accuracy * 100:.2f}%")
# (Con k=5 y 10k datos, deberías obtener >90% de precisión)

# --- P6: Visualizar Algunas Predicciones ---
print("\nMostrando algunas imágenes de prueba y sus predicciones:")

# Convertir a NumPy arrays (evita errores con pandas)
X_test = np.array(X_test)
y_test = np.array(y_test)

num_imagenes_a_mostrar = 5
indices_aleatorios = np.random.choice(len(X_test), num_imagenes_a_mostrar, replace=False)

plt.figure(figsize=(12, 4)) # Crear figura
for i, idx in enumerate(indices_aleatorios):
    imagen = X_test[idx].reshape(28, 28) # Reorganizar el vector a imagen 28x28
    etiqueta_real = y_test[idx]          # Etiqueta real
    etiqueta_pred = y_pred[idx]          # Predicción

    plt.subplot(1, num_imagenes_a_mostrar, i + 1)
    plt.imshow(imagen, cmap='gray')
    plt.title(f"Real: {etiqueta_real}\nPred: {etiqueta_pred}")
    plt.axis('off')

plt.tight_layout()
plt.show()


print("\nConclusión:")
print("Se usó k-NN para clasificar dígitos MNIST basándose en la similitud")
print("de píxeles con los ejemplos de entrenamiento memorizados.")
print("Aunque simple, k-NN puede ser efectivo para este problema.")
print("Modelos más avanzados (CNNs, RNNs) logran mayor precisión y son")
print("necesarios para reconocer escritura más compleja (leras, cursiva).")