# Algoritmo de CLASIFICADOR NAÏVE-BAYES

# Este es un algoritmo de *Aprendizaje Supervisado* para *Clasificación*.
#
# Definición:
# Es un clasificador que utiliza la Regla de Bayes (tema #1) para
# determinar la probabilidad de que una muestra de datos (ej. un email)
# pertenezca a una cierta clase (ej. 'spam' o 'no spam').
#
# La Gran Suposición "Naïve" (Ingenua):
# El algoritmo hace una suposición *ingenua* (y casi siempre incorrecta) para que las matemáticas sean súper simples y rápidas:
#
# Supone que todas las "características" (features) son condicionalmente independientes* entre sí, dada la clase.
#
# En español:
# Asume que la palabra "Viagra" y la palabra "gratis" aparecen en un email de 'spam' de forma totalmente independiente, (lo cual es falso, pero el algoritmo funciona sorprendentemente bien de todos modos).
#
# Fórmula Clave:
# P(Clase | Características) ∝ P(Clase) * P(Características | Clase)
#
# Gracias a la suposición "ingenua", P(Características | Clase) se
# convierte en un simple producto de probabilidades individuales:
#
# P(Clase | f1, f2, f3) ∝ P(Clase) * P(f1|Clase) * P(f2|Clase) * P(f3|Clase)
#
# ¿Cómo funciona este programa?
# 1. FASE DE ENTRENAMIENTO (función `fit`):
#    - El algoritmo "lee" todos los emails de entrenamiento.
#    - Calcula el "Prior" P(Clase): ¿Qué % de emails son 'spam'?
#    - Calcula el "Likelihood" P(Palabra | Clase): ¿Qué % de
#      palabras en emails 'spam' son "Viagra"?
#    - (Usa "Suavizado de Laplace" para evitar probabilidades de 0).
#
# 2. FASE DE PREDICCIÓN (función `predict`):
#    - Recibe un email *nuevo* y *sin etiqueta*.
#    - Calcula una "puntuación" para cada clase posible.
#    - Score('spam') = log(P('spam')) + log(P('palabra1'|'spam')) + ...
#    - Score('no spam') = log(P('no spam')) + log(P('palabra1'|'no spam')) + ...
#    - (Usamos logaritmos para evitar que la multiplicación de
#      números pequeños (ej. 0.001 * 0.0001) se vuelva 0 (underflow)).
#    - La clase con la *puntuación más alta* es la predicción.
#
# Componentes:
# 1. P(Clase): Probabilidades a Priori (ej. P('spam')).
# 2. P(Palabra | Clase): Probabilidades de Verosimilitud (Likelihoods).
# 3. k (Suavizado de Laplace): Un número (ej. 1) para evitar
#    el "problema de frecuencia cero" (ver una palabra que no
#    estaba en el entrenamiento).
#
# Aplicaciones:
# - El filtro de SPAM original.
# - Análisis de Sentimiento (¿Esta reseña es 'positiva' o 'negativa'?).
# - Clasificación de documentos.
#
# Ventajas:
# - Increíblemente rápido (para entrenar y predecir).
# - Funciona muy bien con *pocos datos* de entrenamiento.
# - Es el "rey" de la clasificación de texto.
#
# Desventajas:
# - Su suposición de independencia es obviamente incorrecta,
#   por lo que la *probabilidad* que calcula (ej. 99.99%) no es confiable, aunque la *clasificación* (la elección) sí lo es.

from math import log # Usamos logaritmos para estabilidad numérica
from collections import Counter, defaultdict # Para contar palabras eficientemente

class NaiveBayesClassifier: # Define la clase para nuestro clasificador
    
    def __init__(self, k=1): # Constructor
        # 'k' es el factor de Suavizado de Laplace (Laplace Smoothing)
        # (Usar k=1 se llama "Laplace smoothing")
        # (Usar k<1 se llama "Lidstone smoothing")
        self.k = k # Almacena 'k'
        
        # Diccionarios para almacenar las probabilidades aprendidas
        self.priors = {}       # Almacenará P(Clase)
        self.likelihoods = {}  # Almacenará P(Palabra | Clase)
        self.vocab = set()     # Almacenará todas las palabras únicas vistas
        self.classes = set()   # Almacenará las clases (ej. 'spam', 'no spam')

    def fit(self, X_train, y_train): # El método de "Entrenamiento"
        """
        Entrena el clasificador.
        X_train: una lista de "documentos" (ej. strings de email)
        y_train: una lista de "etiquetas" (ej. 'spam', 'no spam')
        """
        
        # --- 1. Calcular Probabilidades a Priori P(Clase) ---
        
        num_documentos = len(X_train) # Total de documentos
        self.classes = set(y_train) # Encuentra las clases únicas (ej. {'spam', 'no spam'})
        conteo_clases = Counter(y_train) # Cuenta cuántos de cada clase (ej. {'spam': 20, 'no spam': 80})

        for label in self.classes: # Iterar sobre 'spam', 'no spam'
            # P(Clase) = (Documentos de esta Clase) / (Total Documentos)
            self.priors[label] = conteo_clases[label] / num_documentos
            
        # --- 2. Calcular Conteos de Palabras P(Palabra | Clase) ---
        
        # self.word_counts['spam'] = Counter({'viagra': 10, 'oferta': 15, ...})
        self.word_counts = {label: Counter() for label in self.classes}
        # self.class_totals['spam'] = 1500 (total de palabras en todos los docs 'spam')
        self.class_totals = {label: 0 for label in self.classes}
        self.vocab = set() # El vocabulario global
        
        # Bucle de conteo
        for doc, label in zip(X_train, y_train): # Emparejar cada doc con su etiqueta
            words = doc.lower().split() # Convertir "Oferta Gratis" -> ['oferta', 'gratis']
            for word in words: # Iterar sobre cada palabra
                self.word_counts[label][word] += 1 # Incrementar conteo (ej. word_counts['spam']['oferta']++)
                self.class_totals[label] += 1      # Incrementar total de palabras de la clase
                self.vocab.add(word)               # Añadir palabra al vocabulario global
        
        len_vocab = len(self.vocab) # Tamaño del vocabulario (ej. 5000 palabras únicas)
        
        # --- 3. Calcular Verosimilitudes (Likelihoods) con Suavizado ---
        #    P(Palabra | Clase) = (conteo(Palabra, Clase) + k) / (total_palabras(Clase) + k * |V|)
        
        self.likelihoods = {label: {} for label in self.classes} # Inicializar
        
        for label in self.classes: # 'spam', 'no spam'
            total_words_in_class = self.class_totals[label] # ej. 1500
            # Este es el denominador de Laplace
            denominador = total_words_in_class + self.k * len_vocab
            
            # Debemos calcular una prob. para *cada* palabra del vocabulario
            for word in self.vocab: 
                # Conteo(palabra, clase)
                conteo_palabra = self.word_counts[label][word] # (Será 0 si la palabra no está)
                
                # Aplicar la fórmula
                prob = (conteo_palabra + self.k) / denominador
                
                # Guardar la probabilidad
                self.likelihoods[label][word] = prob

    def predict(self, X_test): # El método de "Clasificación"
        """
        Clasifica una lista de nuevos documentos.
        X_test: una lista de strings de documentos a clasificar.
        """
        # Usar una "list comprehension" para predecir cada documento
        return [self._predict_one(doc) for doc in X_test]

    def _predict_one(self, doc): # Función auxiliar para clasificar un solo documento
        """ Calcula la puntuación (score) para un documento """
        
        scores = {} # Diccionario para guardar la puntuación de cada clase
        words = doc.lower().split() # Tokenizar el nuevo documento
        
        # Iterar sobre cada clase posible (ej. 'spam', 'no spam')
        for label in self.classes:
            
            # 1. Empezar con el Prior (en logaritmo)
            #    Score = log(P(Clase))
            log_prior = log(self.priors[label])
            
            # 2. Sumar el Likelihood (en logaritmo)
            #    Score += log(P(Palabra1|Clase)) + log(P(Palabra2|Clase)) + ...
            log_likelihood = 0.0 # Inicializar
            
            for word in words: # Iterar sobre las palabras del *nuevo* email
                if word in self.vocab: # *Solo* si la palabra estaba en el entrenamiento
                    # Sumar el log-likelihood
                    log_likelihood += log(self.likelihoods[label][word])
                    
            # 3. Puntuación final para esta clase
            scores[label] = log_prior + log_likelihood
            
        # 4. Devolver la clase que tuvo la puntuación más alta
        #    max(scores, key=scores.get) encuentra la *llave*
        #    que tiene el *valor* más alto en el diccionario.
        return max(scores, key=scores.get)

# --- P3: Ejecutar el Clasificador (Ejemplo de Spam) ---
print("Clasificador Naïve-Bayes") # Título

# 1. Datos de Entrenamiento (X_train, y_train)
#    (Pocos datos, pero Naive Bayes puede manejarlos)
X_train = [
    "oferta increíble dinero rápido",           # spam
    "reunión de trabajo mañana",               # no spam
    "gratis ganar dinero ahora",               # spam
    "reporte de proyecto enviado",             # no spam
    "dinero fácil sin esfuerzo",               # spam
    "revisión del reporte de ventas"           # no spam
]
y_train = [
    "spam", "no spam", "spam", "no spam", "spam", "no spam"
]

# 2. Crear y Entrenar el clasificador
#    (Usamos k=1 para el suavizado)
nb_classifier = NaiveBayesClassifier(k=1) # Crear la instancia
nb_classifier.fit(X_train, y_train)       # Entrenar con los datos

print("¡Clasificador entrenado!") # Mensaje
print(f"Clases: {nb_classifier.classes}") # Imprime {'spam', 'no spam'}
print(f"Prior P(spam): {nb_classifier.priors['spam']:.2f}") # Imprime 0.50
print(f"Prior P(no spam): {nb_classifier.priors['no spam']:.2f}") # Imprime 0.50
print(f"Tamaño del Vocabulario: {len(nb_classifier.vocab)}") # Imprime ~18

# Imprimir un ejemplo de Likelihood (con suavizado)
print(f"Likelihood P('dinero'|'spam'): {nb_classifier.likelihoods['spam']['dinero']:.4f}")
print(f"Likelihood P('dinero'|'no spam'): {nb_classifier.likelihoods['no spam']['dinero']:.4f}")

# 3. Datos de Prueba (Nuevos, sin etiqueta)
X_test = [
    "ganar dinero fácil mañana",           # (Debería ser spam)
    "reunión sobre el reporte de dinero",  # (Debería ser no spam)
    "oferta de trabajo"                    # (Debería ser no spam, aunque 'oferta' es de spam)
]

print(f"\nClasificando {len(X_test)} nuevos documentos...") # Mensaje

# 4. Obtener Predicciones
predicciones = nb_classifier.predict(X_test) # Llamar al método predict

# 5. Imprimir resultados
for doc, pred in zip(X_test, predicciones): # Emparejar documento con su predicción
    print(f"  Documento: '{doc}'\n  Predicción: ==> {pred}") # Imprimir resultado

print("\nConclusión:")
print("El algoritmo calculó P(spam | 'ganar dinero...') vs P(no spam | 'ganar dinero...')")
print("y eligió la clase con la puntuación (log-probabilidad) más alta.")