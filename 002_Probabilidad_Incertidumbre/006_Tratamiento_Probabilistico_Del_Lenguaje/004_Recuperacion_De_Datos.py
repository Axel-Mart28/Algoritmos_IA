# --- ALGORITMO DE RECUPERACIÓN DE DATOS ---

# Concepto: Es el campo dedicado a buscar y recuperar información relevante
#           (generalmente documentos de texto) desde una gran colección,
#           en respuesta a una consulta (query) del usuario.
#           Piensa en Google Search.
#
# Objetivo: Rankear los documentos de la colección según su *relevancia*
#           para la consulta dada.
#
# ¿Cómo se usan las Probabilidades?
# Los modelos probabilísticos en IR intentan estimar P(Relevante | Documento, Consulta).
# Modelos Clásicos:
# 1. Modelo Booleano: Simple (contiene/no contiene palabras), no probabilístico.
# 2. Modelo Vectorial (Vector Space Model): Representa documentos y consultas
#    como vectores en un espacio de alta dimensión. Usa métricas como
#    la similitud del coseno. TF-IDF es una parte clave de este modelo.
# 3. Modelos Probabilísticos (BM25, Language Models for IR): Intentan
#    modelar explícitamente la probabilidad de relevancia. Son más
#    avanzados pero a menudo superan a los modelos más simples.
#
# ¿Cómo funciona este programa?
# Implementaremos una versión muy simplificada del **Modelo Vectorial**
# usando **TF-IDF** (Term Frequency-Inverse Document Frequency).
# - TF (Term Frequency): ¿Qué tan frecuente es una palabra *dentro* de un documento?
#   (Más frecuente -> más importante para *ese* doc).
# - IDF (Inverse Document Frequency): ¿Qué tan *rara* es una palabra en *toda* la colección?
#   (Más rara -> más *distintiva* y útil para buscar).
# - TF-IDF = TF * IDF: Da una puntuación alta a palabras que son frecuentes
#   en un documento pero raras en general.
#
# Pasos:
# 1. PREPROCESAR: Limpiar y tokenizar los documentos y la consulta.
# 2. CALCULAR TF: Contar frecuencias de palabras en cada documento.
# 3. CALCULAR IDF: Contar en cuántos documentos aparece cada palabra.
# 4. CALCULAR TF-IDF: Combinar TF e IDF para crear vectores para cada doc.
# 5. CALCULAR SIMILITUD: Representar la consulta como un vector TF-IDF y
#    calcular la similitud (ej. coseno) entre la consulta y cada documento.
# 6. RANKEAR: Ordenar los documentos por similitud descendente.
#
# 

import math # Para logaritmos (en IDF)
import re # Para preprocesamiento
from collections import Counter, defaultdict # Para conteos

# --- P1: Mini-Corpus (Colección de Documentos) ---
documentos = {
    'Doc1': "El gato persigue al ratón rápido.",
    'Doc2': "El perro ladra fuerte al gato.",
    'Doc3': "El ratón rápido come queso delicioso.",
    'Doc4': "Perro y gato son animales."
}
N_docs = len(documentos) # Número total de documentos

print("--- 4. Recuperación de Datos (TF-IDF Básico) ---")
print(f"Colección de {N_docs} documentos:")
for name, text in documentos.items():
    print(f"  {name}: {text}")

# --- P2: Preprocesamiento y Tokenización ---
def preprocesar(texto): # (La misma función de antes)
    """ Minúsculas, quita puntuación básica, tokeniza """
    texto_lower = texto.lower()
    texto_limpio = re.sub(r'[^\w\s]', '', texto_lower)
    tokens = texto_limpio.split()
    return tokens

# Tokenizar todos los documentos
docs_tokenizados = {name: preprocesar(text) for name, text in documentos.items()}
# Crear un vocabulario (todas las palabras únicas)
vocabulario = set(word for tokens in docs_tokenizados.values() for word in tokens)
print(f"\nVocabulario ({len(vocabulario)} palabras): {sorted(list(vocabulario))}")

# --- P3: Calcular TF (Term Frequency) ---
# TF(palabra, doc) = (Número de veces que 'palabra' aparece en 'doc') / (Total de palabras en 'doc')
tf_docs = defaultdict(lambda: defaultdict(float)) # tf_docs['Doc1']['gato'] = 0.2
for name, tokens in docs_tokenizados.items():
    conteo_palabras = Counter(tokens) # Contar palabras en este doc
    total_palabras_doc = len(tokens) # Total de palabras en este doc
    for palabra, conteo in conteo_palabras.items():
        tf_docs[name][palabra] = conteo / total_palabras_doc

# print("\nTF (Term Frequency - ejemplo Doc1):")
# print(tf_docs['Doc1'])

# --- P4: Calcular IDF (Inverse Document Frequency) ---
# IDF(palabra) = log( (Total de Documentos) / (1 + Documentos que contienen 'palabra') )
# (Sumamos 1 en el denominador para evitar división por cero si una palabra es nueva)
idf = defaultdict(float) # idf['gato'] = log(4 / (1 + 3))
doc_freq = Counter() # Contar en cuántos docs aparece cada palabra
for palabra in vocabulario:
    for tokens in docs_tokenizados.values():
        if palabra in tokens:
            doc_freq[palabra] += 1

for palabra in vocabulario:
    idf[palabra] = math.log(N_docs / (1 + doc_freq[palabra]))

# print("\nIDF (Inverse Document Frequency - ejemplo):")
# print(f"  idf['gato']: {idf['gato']:.3f} (común)")
# print(f"  idf['queso']: {idf['queso']:.3f} (raro)")

# --- P5: Calcular Vectores TF-IDF para Documentos ---
tfidf_vectores = defaultdict(lambda: defaultdict(float)) # tfidf_vectores['Doc1']['gato'] = tf * idf
for name in documentos.keys():
    for palabra in vocabulario:
        # Puntuación TF-IDF = TF * IDF
        tfidf_vectores[name][palabra] = tf_docs[name][palabra] * idf[palabra]

# --- P6: Consulta y Ranking ---
consulta = "gato rápido"
print(f"\nConsulta del usuario: '{consulta}'")

# 1. Preprocesar y calcular TF-IDF para la Consulta
tokens_consulta = preprocesar(consulta)
conteo_consulta = Counter(tokens_consulta)
total_palabras_consulta = len(tokens_consulta)
tfidf_consulta = defaultdict(float)
for palabra in vocabulario: # Usar el vocabulario global
    tf_q = conteo_consulta[palabra] / total_palabras_consulta if total_palabras_consulta > 0 else 0
    tfidf_consulta[palabra] = tf_q * idf[palabra] # Usar el IDF ya calculado

# 2. Calcular Similitud del Coseno entre la consulta y cada documento
#    Sim(q, d) = (q · d) / (||q|| * ||d||)
#    q · d = Producto punto de los vectores TF-IDF
#    ||q|| = Magnitud (norma Euclidiana) del vector TF-IDF

def calcular_sim_coseno(vec1, vec2, vocab):
    """ Calcula la similitud del coseno entre dos vectores TF-IDF """
    producto_punto = 0.0
    norma1_sq = 0.0
    norma2_sq = 0.0
    for palabra in vocab: # Iterar sobre todas las dimensiones (palabras)
        val1 = vec1.get(palabra, 0.0) # Usar .get con default 0 por si falta la palabra
        val2 = vec2.get(palabra, 0.0)
        producto_punto += val1 * val2 # Acumular q_i * d_i
        norma1_sq += val1**2 # Acumular q_i^2
        norma2_sq += val2**2 # Acumular d_i^2

    # Magnitudes ||q|| y ||d||
    norma1 = math.sqrt(norma1_sq)
    norma2 = math.sqrt(norma2_sq)

    # Evitar división por cero
    if norma1 == 0 or norma2 == 0:
        return 0.0
    else:
        return producto_punto / (norma1 * norma2) # Devolver similitud

# Calcular similitud para cada documento
similitudes = {}
for name in documentos.keys():
    similitudes[name] = calcular_sim_coseno(tfidf_consulta, tfidf_vectores[name], vocabulario)

# 3. Rankear los documentos
#    Ordenar por similitud descendente
documentos_rankeados = sorted(similitudes.items(), key=lambda item: item[1], reverse=True)
# sorted(...) ordena una lista de tuplas ('Doc1', 0.8)
# key=lambda item: item[1] le dice que ordene por el *segundo* elemento (la similitud)
# reverse=True es para orden descendente (el más relevante primero)

print("\n--- Ranking de Documentos Relevantes ---")
for name, score in documentos_rankeados:
    print(f"  Doc: {name}, Similitud Coseno: {score:.4f}")
    # Doc1 ('gato', 'ratón', 'rápido') -> Muy relevante
    # Doc3 ('ratón', 'rápido') -> Relevante
    # Doc2 ('gato') -> Algo relevante
    # Doc4 ('gato') -> Algo relevante

print("\nConclusión:")
print("Usando TF-IDF y Similitud del Coseno, hemos rankeado los documentos")
print("según su relevancia estimada para la consulta 'gato rápido'.")
print("Documentos con las palabras de la consulta (especialmente si son raras)")
print("obtienen puntuaciones más altas.")