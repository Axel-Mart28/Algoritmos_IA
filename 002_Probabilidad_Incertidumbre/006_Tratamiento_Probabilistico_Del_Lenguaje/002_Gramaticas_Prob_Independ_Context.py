# --- 2. GRAMÁTICAS PROBAB. INDEPEND. DEL CONTEXTO (PCFG) ---

# Concepto: Reglas gramaticales con probabilidades asociadas.
# Objetivo: Calcular la probabilidad de un árbol de análisis sintáctico.

import math # Para logaritmos si calculáramos log-probabilidades

# --- P1: Definición de una PCFG Simple ---
# (Ejemplo muy básico para "el gato come")

# Formato: 'NoTerminal': [ (Probabilidad, ['Expansion', 'Regla']), ... ]
pcfg_gramatica = {
    'S': [ # S -> NP VP (Oración -> Sintagma Nominal + Sintagma Verbal)
        (1.0, ['NP', 'VP'])
    ],
    'NP': [ # Sintagma Nominal
        (0.6, ['Det', 'N']),   # NP -> Det N (el gato)
        (0.4, ['N'])           # NP -> N (queso) - (simplificado)
    ],
    'VP': [ # Sintagma Verbal
        (0.7, ['V', 'NP']),   # VP -> V NP (come queso)
        (0.3, ['V'])           # VP -> V (ladra) - (no usado aquí)
    ],
    'Det': [ # Determinante
        (1.0, ['el'])
    ],
    'N': [ # Nombre
        (0.5, ['gato']),
        (0.3, ['ratón']),
        (0.2, ['queso'])
    ],
    'V': [ # Verbo
        (0.6, ['persigue']),
        (0.4, ['come'])
    ]
}

print("--- 2. Gramática Probabilística Independiente del Contexto (PCFG) ---")
print("Gramática definida (ejemplo simplificado).")

# --- P2: Árbol de Análisis Sintáctico (Parse Tree) ---
# Representaremos un árbol como una lista anidada o tupla.
# Este árbol corresponde a la frase "el gato come queso"
# (Asumimos que un parser ya nos dio este árbol)
arbol_analisis = \
    ('S',
        ('NP',
            ('Det', 'el'),
            ('N', 'gato')
        ),
        ('VP',
            ('V', 'come'),
            ('NP',
                ('N', 'queso')
            )
        )
     )

print("\nÁrbol de Análisis para 'el gato come queso':")
# 
# (Una visualización del árbol ayuda a entender la estructura)
print(arbol_analisis) # Imprime la estructura de tupla

# --- P3: Calcular la Probabilidad del Árbol ---

def buscar_prob_regla(gramatica, no_terminal, expansion):
    """ Encuentra la probabilidad de una regla específica en la gramática """
    if no_terminal not in gramatica: # Comprobar si el no terminal existe
        return 0.0 # Regla no encontrada
    for prob, regla in gramatica[no_terminal]: # Buscar la regla
        # Comparar la expansión (lista de símbolos)
        # Si la regla es terminal (ej. ['gato']), 'expansion' será solo 'gato'
        # Si la regla es no terminal (ej. ['NP', 'VP']), 'expansion' será ('NP', 'VP')
        # Necesitamos manejar ambos casos
        if isinstance(expansion, tuple): # Si es una tupla de no terminales
             if regla == list(expansion): # Comparar como lista
                 return prob
        elif isinstance(expansion, str): # Si es una palabra terminal
             if regla == [expansion]: # Comparar como lista de un elemento
                 return prob
    return 0.0 # Regla no encontrada

def calcular_prob_arbol(arbol, gramatica):
    """ Calcula la probabilidad total de un árbol de análisis (producto de probs de reglas) """
    
    # El primer elemento es el símbolo (ej. 'S', 'NP', 'el')
    simbolo = arbol[0]
    
    # Caso Base: Si el árbol es solo una palabra (nodo hoja)
    if len(arbol) == 2 and isinstance(arbol[1], str):
        # La expansión es la palabra en sí
        expansion = arbol[1]
        # Buscar la probabilidad de la regla terminal (ej. N -> 'gato')
        prob_regla = buscar_prob_regla(gramatica, simbolo, expansion)
        return prob_regla
        
    # Caso Recursivo: Si el árbol tiene subárboles
    else:
        # La expansión son los símbolos de los hijos
        expansion_hijos = tuple(hijo[0] for hijo in arbol[1:]) # Ej: ('NP', 'VP')
        # Buscar la probabilidad de esta regla de producción (ej. S -> NP VP)
        prob_regla_actual = buscar_prob_regla(gramatica, simbolo, expansion_hijos)
        
        # Multiplicar por la probabilidad de los subárboles (recursión)
        prob_subarboles = 1.0
        for subarbol in arbol[1:]: # Iterar sobre los hijos ('NP', 'VP')
            prob_subarboles *= calcular_prob_arbol(subarbol, gramatica) # Llamada recursiva
            
        # Probabilidad total = P(Regla Actual) * P(Subárbol 1) * P(Subárbol 2) * ...
        return prob_regla_actual * prob_subarboles

# Calcular la probabilidad del árbol específico
probabilidad = calcular_prob_arbol(arbol_analisis, pcfg_gramatica)

print(f"\nCalculando la probabilidad del árbol...")
# P = P(S->NP VP) * P(NP->Det N) * P(Det->el) * P(N->gato) * P(VP->V NP) * P(V->come) * P(NP->N) * P(N->queso)
# P = 1.0 * 0.6 * 1.0 * 0.5 * 0.7 * 0.4 * 0.4 * 0.2
# P = 0.00672
print(f"Probabilidad del árbol P(árbol): {probabilidad:.6f}")

print("\nConclusión:")
print("Una PCFG asigna probabilidades a las reglas gramaticales.")
print("La probabilidad de un árbol de análisis completo es el producto")
print("de las probabilidades de todas las reglas utilizadas en él.")
print("Los algoritmos de parsing (como CKY) encuentran el árbol más probable.")