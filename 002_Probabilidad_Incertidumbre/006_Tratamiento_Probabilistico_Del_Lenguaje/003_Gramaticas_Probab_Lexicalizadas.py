# --- 3. GRAMÁTICAS PROBABILÍSTICAS LEXICALIZADAS (Algoritmo Demo) ---

# Concepto: PCFGs donde las probabilidades dependen de palabras clave (headwords).
# Objetivo: Demostrar cómo se calculan las probabilidades con reglas lexicalizadas.

import math 

# --- P1: Definición de una LPCFG Simple ---
# Usaremos un formato donde la clave incluye el headword: 'NoTerminal(headword)'
# La expansión también incluye headwords: ['Hijo1(head1)', 'Hijo2(head2)']

lpcfg_gramatica = {
    # Reglas para S (Oración), headword viene de VP
    'S(come)': [
        (1.0, ['NP(gato)', 'VP(come)']) # P(S(come) -> NP(gato) VP(come))
    ],
    'S(persigue)': [
         (1.0, ['NP(gato)', 'VP(persigue)']) # P(S(persigue) -> NP(gato) VP(persigue))
    ],

    # Reglas para NP (Sintagma Nominal), headword viene de N
    'NP(gato)': [
        (1.0, ['Det(el)', 'N(gato)']) # P(NP(gato) -> Det(el) N(gato))
    ],
    'NP(queso)': [
        (1.0, ['N(queso)'])          # P(NP(queso) -> N(queso))
    ],
    'NP(ratón)': [
        (1.0, ['Det(el)', 'N(ratón)']) # P(NP(ratón) -> Det(el) N(ratón))
    ],

    # Reglas para VP (Sintagma Verbal), headword viene de V
    'VP(come)': [
        (0.8, ['V(come)', 'NP(queso)']), # P(VP(come) -> V(come) NP(queso)) = Alta
        (0.2, ['V(come)', 'NP(gato)'])   # P(VP(come) -> V(come) NP(gato)) = Baja (gatos no comen gatos)
    ],
     'VP(persigue)': [
        (0.9, ['V(persigue)', 'NP(ratón)']),# P(VP(persigue) -> V(persigue) NP(ratón)) = Alta
        (0.1, ['V(persigue)', 'NP(queso)']) # P(VP(persigue) -> V(persigue) NP(queso)) = Baja (no persigue queso)
    ],

    # Reglas Terminales (Lexicales) - Headword es la palabra misma
    'Det(el)': [(1.0, ['el'])],
    'N(gato)': [(1.0, ['gato'])],
    'N(queso)': [(1.0, ['queso'])],
    'N(ratón)': [(1.0, ['ratón'])],
    'V(come)': [(1.0, ['come'])],
    'V(persigue)': [(1.0, ['persigue'])]
}

print("--- 3. Gramática Probabilística Lexicalizada (LPCFG Demo) ---")
print("Gramática Lexicalizada definida (ejemplo simplificado).")

# --- P2: Árbol de Análisis Lexicalizado ---
# Este árbol ya tiene la información de los headwords en cada nodo.
# (Un parser LPCFG real se encargaría de generar esto)

# Árbol para "el gato come queso"
arbol_lexicalizado_1 = \
    ('S(come)',
        ('NP(gato)',
            ('Det(el)', 'el'),
            ('N(gato)', 'gato')
        ),
        ('VP(come)',
            ('V(come)', 'come'),
            ('NP(queso)',
                ('N(queso)', 'queso')
            )
        )
     )

# Árbol para "el gato persigue queso" (menos probable)
arbol_lexicalizado_2 = \
    ('S(persigue)',
        ('NP(gato)',
            ('Det(el)', 'el'),
            ('N(gato)', 'gato')
        ),
        ('VP(persigue)',
            ('V(persigue)', 'persigue'),
            ('NP(queso)',              # <-- Esta parte es improbable según la LPCFG
                ('N(queso)', 'queso')
            )
        )
     )

print("\nÁrbol Lexicalizado 1 ('el gato come queso'):")
# 
print(arbol_lexicalizado_1)
print("\nÁrbol Lexicalizado 2 ('el gato persigue queso'):")
print(arbol_lexicalizado_2)


# --- P3: Calcular Probabilidad del Árbol Lexicalizado ---

def buscar_prob_regla_lex(gramatica, regla_lex):
    """ Encuentra la probabilidad de una regla lexicalizada específica """
    # regla_lex es una tupla: ( 'Padre(headP)', ['Hijo1(head1)', ...] o 'palabra' )
    simbolo_padre_lex = regla_lex[0] # Ej: 'VP(come)'
    expansion_lex = regla_lex[1]     # Ej: ['V(come)', 'NP(queso)'] o 'come'

    if simbolo_padre_lex not in gramatica:
        # print(f"Advertencia: Símbolo padre '{simbolo_padre_lex}' no encontrado.")
        return 0.0 # Padre no está en la gramática

    # Buscar la expansión exacta en las reglas del padre
    for prob, regla_gram in gramatica[simbolo_padre_lex]:
        if regla_gram == expansion_lex:
            return prob # Encontrado!

    # print(f"Advertencia: Regla '{simbolo_padre_lex} -> {expansion_lex}' no encontrada.")
    return 0.0 # Regla no encontrada

def calcular_prob_arbol_lex(arbol_lex, gramatica):
    """ Calcula la probabilidad total de un árbol lexicalizado """

    # El primer elemento es el símbolo lexicalizado (ej. 'S(come)')
    simbolo_padre_lex = arbol_lex[0]

    # Caso Base: Nodo hoja (terminal)
    if len(arbol_lex) == 2 and isinstance(arbol_lex[1], str):
        palabra = arbol_lex[1] # La palabra terminal (ej. 'come')
        # La regla es ('V(come)', ['come'])
        prob_regla = buscar_prob_regla_lex(gramatica, (simbolo_padre_lex, [palabra]))
        # print(f"  Terminal: {simbolo_padre_lex} -> {palabra} (Prob: {prob_regla})")
        return prob_regla

    # Caso Recursivo: Nodo interno
    else:
        # La expansión son los símbolos lexicalizados de los hijos
        expansion_hijos_lex = [hijo[0] for hijo in arbol_lex[1:]] # Ej: ['NP(gato)', 'VP(come)']
        # La regla es ('S(come)', ['NP(gato)', 'VP(come)'])
        prob_regla_actual = buscar_prob_regla_lex(gramatica, (simbolo_padre_lex, expansion_hijos_lex))
        # print(f"  Regla: {simbolo_padre_lex} -> {expansion_hijos_lex} (Prob: {prob_regla_actual})")

        # Multiplicar por la probabilidad de los subárboles
        prob_subarboles = 1.0
        for subarbol in arbol_lex[1:]:
            prob_subarboles *= calcular_prob_arbol_lex(subarbol, gramatica) # Llamada recursiva

        return prob_regla_actual * prob_subarboles

# Calcular probabilidad para el primer árbol
print("\nCalculando probabilidad del Árbol 1 ('el gato come queso')...")
prob_1 = calcular_prob_arbol_lex(arbol_lexicalizado_1, lpcfg_gramatica)
print(f"Probabilidad Árbol 1: {prob_1:.6f}")
# P = P(S(c)->NP(g)VP(c)) * P(NP(g)->D(e)N(g)) * P(D(e)->e) * P(N(g)->g) * P(VP(c)->V(c)NP(q)) * P(V(c)->c) * P(NP(q)->N(q)) * P(N(q)->q)
# P = 1.0 * 1.0 * 1.0 * 1.0 * 0.8 * 1.0 * 1.0 * 1.0 = 0.8

# Calcular probabilidad para el segundo árbol
print("\nCalculando probabilidad del Árbol 2 ('el gato persigue queso')...")
prob_2 = calcular_prob_arbol_lex(arbol_lexicalizado_2, lpcfg_gramatica)
print(f"Probabilidad Árbol 2: {prob_2:.6f}")
# P = P(S(p)->NP(g)VP(p)) * P(NP(g)->D(e)N(g)) * P(D(e)->e) * P(N(g)->g) * P(VP(p)->V(p)NP(q)) * P(V(p)->p) * P(NP(q)->N(q)) * P(N(q)->q)
# P = 1.0 * 1.0 * 1.0 * 1.0 * 0.1 * 1.0 * 1.0 * 1.0 = 0.1

print("\nConclusión:")
print("La LPCFG asigna probabilidades diferentes a estructuras sintácticas")
print("basándose en las palabras involucradas.")
print("El árbol para 'el gato come queso' (0.8) es mucho más probable")
print("que el de 'el gato persigue queso' (0.1) según esta gramática,")
print("reflejando mejor las dependencias del mundo real.")