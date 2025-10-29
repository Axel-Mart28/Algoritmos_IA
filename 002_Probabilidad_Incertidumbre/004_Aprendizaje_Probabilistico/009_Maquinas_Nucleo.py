# ALGORITMO DE  MÁQUINAS DE VECTORES SOPORTE (SUPPORT VECTOR MACHINES - SVM) 

# Este es un algoritmo de *Aprendizaje Supervisado* muy potente,
# principalmente utilizado para *Clasificación*.
#
# Definición:
# SVM busca encontrar el *hiperplano óptimo* que separa los puntos
# de datos de diferentes clases con el *margen máximo* posible.
#
# Conceptos Clave:
# 1. Hiperplano: Es la "frontera" de decisión. En 2D es una línea,
#    en 3D es un plano, y en dimensiones superiores es un hiperplano.
# 2. Margen: Es la "calle" o el espacio vacío entre el hiperplano
#    y los puntos de datos más cercanos de *cada* clase. SVM intenta
#    hacer esta calle lo más ancha posible. Un margen ancho
#    generaliza mejor a datos nuevos.
# 3. Vectores Soporte (Support Vectors): Son los puntos de datos que
#    están *exactamente* en el borde del margen. Son los puntos
#    "críticos" que "soportan" o definen el hiperplano. Si mueves
#    un vector soporte, el hiperplano podría cambiar. Los puntos
#    lejos del margen no importan para definir la frontera.
#
# 
#
# ¿Cómo funciona? (La Idea):
# SVM resuelve un problema de optimización:
# - Encontrar el hiperplano (definido por pesos 'w' y sesgo 'b')
# - Tal que MAXIMICE la distancia (el margen) a los puntos más cercanos
# - Sujeto a la restricción de que todos los puntos estén
#   clasificados correctamente (o lo más correctamente posible).
#
#El "Truco del Núcleo":
# ¿Qué pasa si los datos no se pueden separar con una línea recta?
# SVM usa "kernels" (núcleos) para manejar datos no lineales.
# Un kernel es una función que, *implícitamente*, mapea los datos
# a un espacio de *mayor dimensión* donde SÍ podrían ser separables
# por un hiperplano. ¡Lo hace sin calcular explícitamente las
# nuevas coordenadas, lo cual es muy eficiente!
# Kernels comunes: 'linear', 'poly', 'rbf' (Radial Basis Function).
#
# 
#
# ¿Cómo funciona este programa?
# Implementaremos SVM usando la biblioteca `scikit-learn`, que es
# el estándar en Python. No implementaremos el optimizador desde cero.
# 1. Generaremos datos de ejemplo (lineales y no lineales).
# 2. Crearemos instancias del clasificador `SVC` (Support Vector Classifier).
# 3. Entrenaremos el modelo con `.fit()`.
# 4. Haremos predicciones con `.predict()`.
# 5. Mostraremos cómo usar diferentes kernels ('linear', 'rbf').
#
# Componentes (en scikit-learn):
# 1. `SVC`: La clase del clasificador.
# 2. `kernel`: Parámetro para elegir el tipo de separación ('linear', 'rbf', etc.).
# 3. `C`: Parámetro de Regularización. Controla el balance entre
#    maximizar el margen y minimizar los errores de clasificación.
#    (C bajo = margen más ancho, más errores permitidos;
#     C alto = margen más estrecho, menos errores permitidos).
#
# Aplicaciones:
# - Clasificación de imágenes (ej. detección de caras).
# - Clasificación de texto (ej. análisis de sentimiento).
# - Bioinformática (ej. clasificación de genes).
# - Detección de valores atípicos (outliers).
#
# Ventajas:
# - Muy efectivo en espacios de alta dimensión (muchas características).
# - Efectivo cuando el número de dimensiones es mayor que el de muestras.
# - Eficiente en memoria porque solo usa los vectores soporte.
# - Versátil gracias a los diferentes kernels.
#
# Desventajas:
# - No funciona bien con datasets *muy* grandes (entrenamiento O(n^2) a O(n^3)).
# - Sensible a la elección del kernel y sus parámetros (ej. 'C', 'gamma').
# - No proporciona probabilidades directamente (aunque se pueden estimar).

# --- P1: Importar Bibliotecas ---
from sklearn.svm import SVC # El clasificador SVM
from sklearn.datasets import make_blobs, make_moons # Para generar datos de ejemplo
from sklearn.model_selection import train_test_split # Para dividir datos
from sklearn.metrics import accuracy_score # Para evaluar el modelo
import matplotlib.pyplot as plt # Para visualizar (opcional)
import numpy as np # Para manejo numérico y visualización

# --- P2: Generar Datos de Ejemplo ---

print("--- 7. Máquinas de Vectores Soporte (SVM) ---") # Título

# --- 2a. Datos Linealmente Separables ---
print("\nGenerando datos linealmente separables...") # Mensaje
# make_blobs crea grupos ("gotas") de puntos
X_linear, y_linear = make_blobs(
    n_samples=100,      # Número total de puntos
    centers=2,          # Número de grupos (clases)
    random_state=42,    # Para reproducibilidad
    cluster_std=1.0     # Desviación estándar de los grupos
)
# Dividir en entrenamiento y prueba
X_train_lin, X_test_lin, y_train_lin, y_test_lin = train_test_split(
    X_linear, y_linear, test_size=0.3, random_state=42
)

# --- 2b. Datos No Linealmente Separables ---
print("Generando datos no linealmente separables (forma de luna)...") # Mensaje
# make_moons crea dos "lunas" entrelazadas
X_moon, y_moon = make_moons(
    n_samples=100,      # Número total de puntos
    noise=0.1,          # Añadir un poco de ruido
    random_state=42     # Para reproducibilidad
)
# Dividir en entrenamiento y prueba
X_train_moon, X_test_moon, y_train_moon, y_test_moon = train_test_split(
    X_moon, y_moon, test_size=0.3, random_state=42
)

# --- P3: Entrenar y Predecir con Kernel Lineal ---
print("\n--- SVM con Kernel Lineal ---") # Título sección

# 1. Crear el modelo SVM con kernel lineal
#    'C=1.0' es un valor común de regularización
svm_linear = SVC(kernel='linear', C=1.0) # Crear instancia

# 2. Entrenar el modelo con los datos lineales
print("Entrenando SVM lineal...") # Mensaje
svm_linear.fit(X_train_lin, y_train_lin) # Entrenar

# 3. Hacer predicciones en los datos de prueba lineales
print("Haciendo predicciones...") # Mensaje
y_pred_lin = svm_linear.predict(X_test_lin) # Predecir

# 4. Evaluar el modelo
accuracy_lin = accuracy_score(y_test_lin, y_pred_lin) # Comparar predicciones con etiquetas reales
print(f"Precisión (Accuracy) en datos lineales: {accuracy_lin * 100:.2f}%") # Imprimir resultado

# --- P4: Entrenar y Predecir con Kernel RBF (No Lineal) ---
print("\n--- SVM con Kernel RBF (Radial Basis Function) ---") # Título sección

# 1. Crear el modelo SVM con kernel RBF
#    'gamma='scale'' es un valor por defecto común para el parámetro gamma del RBF
svm_rbf = SVC(kernel='rbf', C=1.0, gamma='scale') # Crear instancia

# 2. Entrenar el modelo con los datos NO lineales (lunas)
print("Entrenando SVM RBF...") # Mensaje
svm_rbf.fit(X_train_moon, y_train_moon) # Entrenar

# 3. Hacer predicciones en los datos de prueba no lineales
print("Haciendo predicciones...") # Mensaje
y_pred_moon = svm_rbf.predict(X_test_moon) # Predecir

# 4. Evaluar el modelo
accuracy_moon = accuracy_score(y_test_moon, y_pred_moon) # Comparar
print(f"Precisión (Accuracy) en datos no lineales (lunas): {accuracy_moon * 100:.2f}%") # Imprimir

# --- P5: Visualización (Opcional, requiere matplotlib) ---
def plot_decision_boundary(X, y, model, title): # Función para graficar
    plt.figure(figsize=(8, 6)) # Crear figura
    # Graficar los puntos de datos
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap='viridis', edgecolors='k', s=50)

    # Crear una malla para graficar el fondo de decisión
    ax = plt.gca() # Obtener ejes actuales
    xlim = ax.get_xlim() # Límites X
    ylim = ax.get_ylim() # Límites Y
    xx, yy = np.meshgrid(np.linspace(xlim[0], xlim[1], 50), # Coordenadas X de la malla
                           np.linspace(ylim[0], ylim[1], 50)) # Coordenadas Y de la malla

    # Predecir en cada punto de la malla
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()]) # Predecir en todos los puntos
    Z = Z.reshape(xx.shape) # Reorganizar para graficar

    # Graficar las regiones de decisión y el margen (si es lineal)
    plt.contourf(xx, yy, Z, cmap='viridis', alpha=0.4) # Fondo de color
    if hasattr(model, "support_vectors_"): # Si el modelo tiene info de vectores soporte
        # Graficar el hiperplano y los márgenes (solo para lineal es fácil)
        if model.kernel == 'linear':
            w = model.coef_[0] # Vector normal
            b = model.intercept_[0] # Sesgo
            x_plot = np.linspace(xlim[0], xlim[1]) # Eje X
            # Ecuación de la línea: w0*x + w1*y + b = 0 => y = (-w0*x - b) / w1
            y_plot = (-w[0] * x_plot - b) / w[1] # Línea central
            margin = 1 / np.sqrt(np.sum(model.coef_ ** 2)) # Ancho del margen
            y_down = y_plot - np.sqrt(1 + (w[0]/w[1])**2) * margin # Margen inferior
            y_up = y_plot + np.sqrt(1 + (w[0]/w[1])**2) * margin # Margen superior
            plt.plot(x_plot, y_plot, 'k-') # Hiperplano
            plt.plot(x_plot, y_down, 'k--') # Margen
            plt.plot(x_plot, y_up, 'k--') # Margen

        # Resaltar los vectores soporte
        plt.scatter(model.support_vectors_[:, 0], model.support_vectors_[:, 1], s=100,
                    linewidth=1, facecolors='none', edgecolors='red')

    plt.title(title) # Título del gráfico
    plt.xlabel("Característica 1") # Etiqueta Eje X
    plt.ylabel("Característica 2") # Etiqueta Eje Y
    plt.show() # Mostrar el gráfico

# Graficar los resultados
print("\nGenerando visualizaciones...")
plot_decision_boundary(X_linear, y_linear, svm_linear, "SVM Lineal (Datos Lineales)")
plot_decision_boundary(X_moon, y_moon, svm_rbf, "SVM con Kernel RBF (Datos No Lineales)")

print("\nConclusión:")
print("SVM encontró la línea óptima (máximo margen) para los datos lineales.")
print("Usando el kernel RBF, SVM pudo encontrar una frontera compleja")
print("para separar los datos no lineales (forma de luna).")
print("Los puntos rodeados en rojo son los 'Vectores Soporte'.")