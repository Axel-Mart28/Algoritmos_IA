# Algoritmo de INFERENCIA POR ENUMERACION

# Este es el algoritmo de inferencia más simple y fundamental para las Redes Bayesianas. Es la forma directa de responder a una consulta.

# Definición:
# Es un algoritmo que calcula la distribución de probabilidad posterior para una variable de consulta (X), dada una evidencia (e). P(X | e)

# ¿Cómo funciona?
# 1. Se basa en la fórmula de la probabilidad condicional:
#    P(X | e) = P(X, e) / P(e)
#
# 2. Usa el "truco" de la normalización:
#    P(X | e) = alpha * P(X, e)
#    Donde 'alpha' es la constante de normalización (1 / P(e)).
#
# 3. ¿Cómo calcula P(X, e)?
#    Sumando hacia afuera (marginalizando) todas las variables ocultas
#    Si 'y' son todas las variables que NO son X ni e (las ocultas)
#    P(X, e) = Sumatoria sobre 'y' de [ P(X, e, y) ]
#
# 4. ¿Y cómo calcula P(X, e, y)? Usando la Regla de la Cadena
#
# En resumen, el algoritmo hace:
# 1. Itera sobre cada valor de la variable de consulta X (ej. True, False).
# 2. Para cada uno, itera sobre todas las combinaciones posibles de las variables ocultas 'y'.
# 3. Para cada combinación, calcula la probabilidad conjunta total usando la Regla de la Cadena.
# 4. Suma todos esos resultados.
# 5. Al final, normaliza las sumas.

# Componentes:
# 1. X: La variable de Consulta (ej. 'Robo').
# 2. e: La Evidencia, un dict (ej. {'JuanLlama': True}).
# 3. y: Las variables Ocultas (el resto de la red).
# 4. La Red Bayesiana (nuestra `red_alarma`).
#
# Aplicaciones:
# - El método base para cualquier consulta en una Red Bayesiana.
#
# Ventajas:
# - Es "conceptualmente simple" y fácil de implementar.
# - Garantiza la respuesta correcta.
#
# Desventajas:
# - Extremadamente lento Es exponencial en el número de variables ocultas 'y'. Si hay 20 variables ocultas, debe calcular 2^20 (más de 1 millón) de probabilidades conjuntas.
# - Es la razón por la que existen algoritmos más inteligentes (como Eliminación de Variables).

red_alarma = {
    'Robo': {
        'parents': [], 'cpt': {(): 0.001}
    },
    'Terremoto': {
        'parents': [], 'cpt': {(): 0.002}
    },
    'Alarma': {
        'parents': ['Robo', 'Terremoto'],
        'cpt': {
            (True, True): 0.95, (True, False): 0.94,
            (False, True): 0.29, (False, False): 0.001
        }
    },
    'JuanLlama': {
        'parents': ['Alarma'],
        'cpt': {(True,): 0.90, (False,): 0.05}
    },
    'MariaLlama': {
        'parents': ['Alarma'],
        'cpt': {(True,): 0.70, (False,): 0.01}
    }
}

# (Función del tema #1)
def get_prob_cpt(red, variable, valor, evidencia={}):
    """ Obtiene P(variable=valor | evidencia) de la CPT """
    nodo = red[variable]
    padres = nodo['parents']
    if not padres:
        clave_cpt = ()
    else:
        clave_cpt = tuple([evidencia[padre] for padre in padres])
    prob_true = nodo['cpt'][clave_cpt]
    return prob_true if valor == True else (1.0 - prob_true)

# (Función del tema #3b de la sección anterior)
def normalizar(puntuaciones):
    """ Normaliza un diccionario de {etiqueta: puntuacion} """
    total = sum(puntuaciones.values())
    if total == 0: return {e: 0.0 for e in puntuaciones}
    return {e: p / total for e, p in puntuaciones.items()}

# --- P2: Algoritmo de Inferencia por Enumeración ---

def enumerate_all(variables, evidencia, red):
    """
    Función recursiva que calcula P(variables | evidencia)
    sumando sobre las variables ocultas.
    """
    
    # --- Caso Base ---
    if not variables:
        # Si no hay más variables, hemos llegado al final de una rama
        return 1.0
        
    # --- Paso Recursivo ---
    
    # 1. Tomar la primera variable Y de la lista
    Y = variables[0]
    # 'rest' son las variables restantes
    rest = variables[1:]
    
    # 2. Comprobar si Y está en la evidencia
    if Y in evidencia:
        # --- Caso 2a: Y es una variable de evidencia ---
        # No necesitamos sumar.
        # Simplemente multiplicamos P(Y=valor | Padres(Y)) por el
        # resto de la recursión.
        
        # Obtener el valor de Y de la evidencia
        valor_Y = evidencia[Y]
        
        # Obtener P(Y | Padres(Y)) (los padres de Y ya están en 'evidencia')
        prob_Y = get_prob_cpt(red, Y, valor_Y, evidencia)
        
        # Multiplicar y continuar la recursión
        return prob_Y * enumerate_all(rest, evidencia, red)
        
    else:
        # --- Caso 2b: Y es una variable OCULTA ---
        # Debemos "sumar sobre" todos los valores de Y (True y False).
        
        total_sum = 0.0 # Inicializar el acumulador
        
        # Iterar sobre los posibles valores de Y (True, False)
        for valor_Y in [True, False]:
            
            # Calcular P(Y=valor_Y | Padres(Y))
            prob_Y = get_prob_cpt(red, Y, valor_Y, evidencia)
            
            # Crear una nueva evidencia extendida para la recursión
            evidencia_extendida = evidencia.copy()
            evidencia_extendida[Y] = valor_Y # Añadir Y=valor_Y
            
            # Acumular la suma:
            # total_sum += P(Y=valor_Y | Padres(Y)) * P(rest | Y=valor_Y, e)
            total_sum += prob_Y * enumerate_all(rest, evidencia_extendida, red)
            
        return total_sum # Devolver la suma total de esta rama

def enumeration_ask(variable_X, evidencia_e, red):
    """
    Función principal que responde la consulta P(X|e).
    """
    
    # 1. Obtener la lista de todas las variables de la red,
    #    en orden topológico (padres antes que hijos).
    #    (Para esta red, el orden de las claves del dict ya funciona)
    variables_red = list(red.keys()) # ['Robo', 'Terremoto', ...]
    
    # 2. Crear el diccionario de puntuaciones (no normalizadas)
    Q = {}
    
    # 3. Iterar sobre los valores de la variable de consulta X
    for valor_X in [True, False]:
        
        # 4. Crear la evidencia extendida
        evidencia_extendida = evidencia_e.copy()
        evidencia_extendida[variable_X] = valor_X # Añadir X=valor_X
        
        # 5. Llamar a la recursión para calcular P(X=valor_X, e)
        #    Pasamos *todas* las variables de la red
        Q[valor_X] = enumerate_all(variables_red, evidencia_extendida, red)
        
    # 6. Normalizar el resultado y devolverlo
    return normalizar(Q)

# --- P3: Ejecutar el cálculo ---
print("--- 4. Inferencia por Enumeración ---") # Título

# La consulta: P(Robo | JuanLlama=True, MariaLlama=True)
# ¿Cuál es la probabilidad de un Robo, dado que ambos vecinos llamaron?
consulta_X = 'Robo'
evidencia = {
    'JuanLlama': True,
    'MariaLlama': True
}
print(f"\nConsulta: P({consulta_X} | JuanLlama=True, MariaLlama=True)")

# Llamar a la función principal de inferencia
distribucion_posterior = enumeration_ask(consulta_X, evidencia, red_alarma)

print("\n--- Resultado (Distribución Posterior) ---")
print(f"{distribucion_posterior}")


print("\nConclusión:")
print("Incluso si Juan y Maria llaman (fuerte evidencia), la probabilidad de")
print(f"un Robo es solo ~{distribucion_posterior[True]*100:.2f}%.")
print("Es más probable (71.53%) que haya sido un Terremoto (0.000365...) o")
print("que la alarma fallara (0.000628...).")