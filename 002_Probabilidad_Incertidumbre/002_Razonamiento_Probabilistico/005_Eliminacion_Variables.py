# Algoritmo de ELIMINACIÓN DE VARIABLES

# Este es un algoritmo de inferencia exacta, como la Enumeración, pero es mucho más eficiente.

# Definición:
# Es un algoritmo que responde consultas P(X|e) (igual que la Enumeración) pero evita los cálculos redundantes.

# El Problema de la Enumeración:
# Para calcular P(R|J,M), la Enumeración calcula la suma sobre 'T' y 'A'.
# Al hacerlo, calcula P(J|A)*P(M|A) *múltiples veces* para cada combinación de 'R' y 'T'.

# ¿Cómo funciona (La Solución - "Factores")?:
# 1. Representa la Red Bayesiana no como un grafo, sino como una lista de "Factores". Un Factor es una tabla de probabilidades (una CPT).
# 2. La consulta P(R | J=T, M=T) se reescribe como:
#    P(R, J=T, M=T) = Suma_sobre_A [ Suma_sobre_T [ P(R) * P(T) * P(A|R,T) * P(J=T|A) * P(M=T|A) ] ]
# 3. La clave es reordenar la suma:
#    = P(R) * [ Suma_sobre_T [ P(T) * [ Suma_sobre_A [ P(A|R,T) * P(J=T|A) * P(M=T|A) ] ] ] ]
#
# El Algoritmo:
# Trabaja desde "adentro hacia afuera", eliminando una variable oculta a la vez.
# 1. (Eliminar 'A'):
#    a. Toma todos los factores que mencionan 'A': f(A|R,T), f(J=T|A), f(M=T|A).
#    b. Los "une" (join) multiplicándolos para crear un factor gigante: f_1(A, R, T).
#    c. "Suma" (sum out) la variable 'A' de este factor para crear uno nuevo: f_2(R, T).
# 
# 2. (Eliminar 'T'):
#    a. Toma todos los factores que mencionan 'T': f(T) y el nuevo f_2(R, T).
#    b. Los "une": f_3(R, T) = f(T) * f_2(R, T).
#    c. "Suma" 'T': f_4(R).
# 3. (Final):
#    a. Toma los factores restantes: f(R) y f_4(R).
#    b. Los "une": f_5(R) = f(R) * f_4(R).
#    c. Normaliza f_5(R) para obtener la respuesta P(R | J=T, M=T).
#
# Componentes:
# 1. Factores: Una clase que almacena una CPT y las variables de las que depende.
# 2. Join (Unir): Una operación para multiplicar dos factores.
# 3. Sum Out (Sumar/Marginalizar): Una operación para eliminar una variable de un factor.
#
# Ventajas:
# - Mucho más eficiente que la Enumeración. Su velocidad depende del "treewidth" (ancho de árbol) del grafo, no del número total de variables.
# - Sigue siendo un algoritmo exacto.
#
# Desventajas:
# - El código es *mucho* más complejo de implementar.
# - La creación de factores intermedios (como f_1(A, R, T)) puede consumir muchísima memoria si una variable tiene muchos padres.
# - El "orden" en que se eliminan las variables importa mucho para la eficiencia.
#
# Ejemplo de uso:
# El programa calculará P(Robo | JuanLlama=True, MariaLlama=True)
# de forma eficiente, creando y uniendo factores.

import copy # Para copiar factores
from itertools import product # Para generar combinaciones de valores

# --- P0: FUNCIÓN AUXILIAR REQUERIDA (LA CORRECCIÓN) ---

def normalizar(puntuaciones):
    """
    Toma un diccionario de {etiqueta: puntuacion} y lo normaliza
    para que todos los valores sumen 1.0.
    """
    
    # 1. Calcular la suma total de todas las puntuaciones
    total = sum(puntuaciones.values()) # Suma las puntuaciones
    
    # 2. Manejar el caso de división por cero
    if total == 0:
        num_items = len(puntuaciones)
        if num_items > 0:
            # Si todas las puntuaciones son 0, devolver una distribución uniforme
            return {etiqueta: 1.0 / num_items for etiqueta in puntuaciones}
        else:
            return {} # Devolver un diccionario vacío si no hay ítems
            
    # 3. Devolver el diccionario normalizado (puntuacion / total)
    return {etiqueta: puntuacion / total for etiqueta, puntuacion in puntuaciones.items()}

# --- P1: Definición de la Clase Factor ---
# (Esta clase es idéntica a la anterior)

class Factor:
    """
    Representa una CPT o un factor intermedio.
    - variables: Lista de nombres de variables (ej. ['Robo', 'Alarma'])
    - cpt: Diccionario que mapea { (val1, val2, ...): prob }
    """
    def __init__(self, variables, cpt):
        self.variables = variables # Lista de nombres de variables
        self.cpt = cpt             # Diccionario de probabilidades
        
    def __str__(self): # Función para imprimir el factor de forma legible
        return f"Factor(Vars: {self.variables}, CPT: {self.cpt})"

# --- P2: Funciones Auxiliares para Factores ---
# (Estas funciones son idénticas a las anteriores)

def join_factors(f1, f2):
    """ Multiplica dos factores (f1 * f2) para crear uno nuevo """
    vars_f1 = set(f1.variables)
    vars_f2 = set(f2.variables)
    common_vars = list(vars_f1.intersection(vars_f2))
    new_vars = list(vars_f1.union(vars_f2))
    new_cpt = {}
    
    for new_vals in product([True, False], repeat=len(new_vars)):
        assignment = dict(zip(new_vars, new_vals))
        key_f1 = tuple(assignment[var] for var in f1.variables)
        key_f2 = tuple(assignment[var] for var in f2.variables)
        prob_f1 = f1.cpt.get(key_f1, 0.0)
        prob_f2 = f2.cpt.get(key_f2, 0.0)
        new_cpt[new_vals] = prob_f1 * prob_f2
        
    return Factor(new_vars, new_cpt)

def sum_out(factor, var):
    """ Elimina una variable 'var' de un factor sumando sobre ella """
    try:
        var_index = factor.variables.index(var)
    except ValueError:
        return factor
        
    new_vars = [v for v in factor.variables if v != var]
    new_cpt = {}
    
    for old_key, prob in factor.cpt.items():
        new_key = tuple(val for i, val in enumerate(old_key) if i != var_index)
        new_cpt[new_key] = new_cpt.get(new_key, 0.0) + prob
        
    return Factor(new_vars, new_cpt)

def set_evidence(factor, evidence):
    """ "Fija" el valor de una variable de evidencia en un factor """
    new_cpt = {}
    for key, prob in factor.cpt.items():
        assignment = dict(zip(factor.variables, key))
        is_consistent = True
        for var_e, val_e in evidence.items():
            if var_e in assignment and assignment[var_e] != val_e:
                is_consistent = False
                break
        if is_consistent:
            new_cpt[key] = prob
        else:
            new_cpt[key] = 0.0
            
    return Factor(factor.variables, new_cpt)

# --- P3: Algoritmo Principal de Eliminación de Variables ---
# (Esta función es idéntica, pero ahora su llamada a 'normalizar' funcionará)

def variable_elimination_ask(query_X, evidence_e, red, orden_elim):
    """
    Responde la consulta P(X|e) usando Eliminación de Variables.
    """
    factors = []
    for var, info in red.items():
        f = Factor(info['parents'] + [var], info['cpt'])
        f = set_evidence(f, evidence_e)
        factors.append(f)
        
    print(f"Factores iniciales (con evidencia aplicada): {len(factors)}")

    for var_to_elim in orden_elim:
        print(f"  Eliminando variable: '{var_to_elim}'...")
        factors_with_var = [f for f in factors if var_to_elim in f.variables]
        factors_without_var = [f for f in factors if var_to_elim not in f.variables]
        
        joined_factor = Factor([], {(): 1.0})
        for f in factors_with_var:
            joined_factor = join_factors(joined_factor, f)
            
        new_factor = sum_out(joined_factor, var_to_elim)
        factors = factors_without_var + [new_factor]
        
    print(f"Factores finales (antes del join): {len(factors)}")
    
    final_product = Factor([], {(): 1.0})
    for f in factors:
        final_product = join_factors(final_product, f)
        
    # Convertir la CPT final en un diccionario simple
    result_dict = {key[0]: prob for key, prob in final_product.cpt.items()}
    
    # ¡Esta línea ahora funcionará!
    return normalizar(result_dict)

# --- P4: Definición de la Red ---
# (Idéntica a la anterior)
red_alarma = {
    'Robo': {'parents': [], 'cpt': {(): 0.001}},
    'Terremoto': {'parents': [], 'cpt': {(): 0.002}},
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

# --- P5: Ejecutar el cálculo ---
# (Idéntico a la anterior)
print("Eliminación de Variables")

consulta_X = 'Robo'
evidencia = {
    'JuanLlama': True,
    'MariaLlama': True
}
orden = ['Alarma', 'Terremoto']

print(f"\nConsulta: P({consulta_X} | {evidencia})")
print(f"Orden de eliminación: {orden}")

distribucion_posterior = variable_elimination_ask(consulta_X, evidencia, red_alarma, orden)

print("\n--- Resultado (Distribución Posterior) ---")
print(f"{distribucion_posterior}")