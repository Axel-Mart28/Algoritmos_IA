# --- 1. MODELO PROBABILÍSTICO DEL LENGUAJE: CORPUS ---

# Concepto: Un Corpus es la colección de textos usada para entrenar modelos.
# Objetivo: Preprocesar texto y calcular frecuencias de palabras.

import re # Para expresiones regulares (limpieza simple)
from collections import Counter # Para contar frecuencias eficientemente

# --- P1: Nuestro Mini-Corpus de Ejemplo ---
corpus_texto = [
    "El gato persigue al ratón.",
    "El perro ladra al gato.",
    "El ratón come queso."
]

print("--- 1. Corpus  ---")
print(f"Corpus Original (Lista de frases):\n{corpus_texto}")

# --- P2: Preprocesamiento Básico ---

def preprocesar(texto):
    """ Convierte a minúsculas, quita puntuación básica y tokeniza. """
    texto_lower = texto.lower() # Convertir a minúsculas
    # Quitar caracteres que no sean letras, números o espacios (simple)
    texto_limpio = re.sub(r'[^\w\s]', '', texto_lower)
    # Dividir el texto en palabras (tokens)
    tokens = texto_limpio.split()
    return tokens

# Aplicar preprocesamiento a todo el corpus
corpus_tokenizado = [] # Lista para guardar todos los tokens
for frase in corpus_texto:
    tokens_frase = preprocesar(frase)
    corpus_tokenizado.extend(tokens_frase) # Añadir los tokens a la lista grande

print(f"\nCorpus Tokenizado (Lista de palabras):\n{corpus_tokenizado}")

# --- P3: Calcular Frecuencias (Modelo Unigrama Básico) ---
# Un modelo "Unigrama" simple solo se basa en la frecuencia de palabras individuales.

# Contar cuántas veces aparece cada palabra
frecuencias = Counter(corpus_tokenizado)
total_palabras = len(corpus_tokenizado) # Número total de tokens

print(f"\nFrecuencias de Palabras (Conteos):")
for palabra, conteo in frecuencias.most_common(): # .most_common() ordena por frecuencia
    print(f"  '{palabra}': {conteo}")

# Calcular probabilidades (Modelo Unigrama)
probabilidades_unigrama = {}
print(f"\nProbabilidades Unigrama P(palabra):")
for palabra, conteo in frecuencias.items():
    prob = conteo / total_palabras # P(palabra) = Freq(palabra) / Total
    probabilidades_unigrama[palabra] = prob
    print(f"  P('{palabra}') = {prob:.3f}")

print("\nConclusión:")
print("Un corpus es la base de datos textual. Preprocesarlo y")
print("calcular frecuencias de palabras (o secuencias de palabras - n-gramas)")
print("es el primer paso para construir modelos probabilísticos del lenguaje.")