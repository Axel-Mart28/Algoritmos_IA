# Algoritmo de RED BAYESIANA

# Este algoritmo es una estructura de datos gráfica para representar el conocimiento y la incertidumbre de forma compacta.

# Definición:
# Una Red Bayesiana (RB) es un Grafo Acíclico Dirigido (DAG) donde:
# 1. Los Nodos: Son las variables aleatorias del problema (ej. 'Lluvia', 'Robo').
# 2. Las Flechas (Arcos): Representan las dependencias condicionales directas.
#    Si una flecha va de A -> B, significa que "A es una causa directa de B".
#
# El componente clave de cada nodo es su CPT (Tabla de Probabilidad Condicionada).
# - Nodos raíz (sin padres): Tienen una CPT con su probabilidad a priori P(Nodo).
# - Nodos hijos (con padres): Tienen una CPT que define P(Nodo | Padres).
#
# ¿Cómo funciona este programa?
# Vamos a definir y representar una Red Bayesiana clásica en Python.
# Usaremos diccionarios anidados para almacenar los nodos y sus CPTs.

# Ejemplo de uso (El clásico ejemplo de la "Alarma"):
# - Un Robo puede activar una Alarma.
# - Un Terremoto  puede activar una Alarma.
# - La Alarma puede causar que en este caso, Juan Llame (JohnCalls).
# - La Alarma puede causar que en este caso,  Maria Llame (MaryCalls).
#
# 
#
# Aplicaciones:
# - Diagnóstico médico (Síntomas -> Enfermedades).
# - Filtros de spam (Palabras -> Probabilidad de Spam).
# - Reconocimiento de voz y visión artificial.
#
# Ventajas:
# - Representación visual e intuitiva de un problema complejo.
# - Es "compacta": gracias a la independencia condicional (que veremos),
#   no necesitamos almacenar la probabilidad de *todas* las combinaciones.
#
# Desventajas:
# - Diseñar la red requiere conocimiento de expertos.
# - Definir las CPTs es difícil (requiere muchos datos).
# - La inferencia (obtener respuestas) es computacionalmente costosa (NP-duro).

import math
from collections import defaultdict # (Útil para construir redes)

# --- P1: Definición de la Red Bayesiana 'Alarma' ---

# Usaremos True/False para los valores de las variables (ej. Robo=True)

# La estructura de la red será un diccionario.
# Cada clave es el nombre de la variable (un nodo).
# El valor es otro diccionario que define sus padres y su CPT.

red_alarma = {
    
    # 1. Nodo 'Robo' (Burglary)
    # - No tiene padres.
    # - Su CPT es solo su probabilidad a priori P(Robo).
    'Robo': {
        'parents': [],
        'cpt': {
            # P(Robo=True) = 0.001
            # (Usamos una tupla vacía () como clave para CPTs sin padres)
            (): 0.001 
        }
    },
    
    # 2. Nodo 'Terremoto' (Earthquake)
    # - No tiene padres.
    # - Su CPT es P(Terremoto).
    'Terremoto': {
        'parents': [],
        'cpt': {
            # P(Terremoto=True) = 0.002
            (): 0.002
        }
    },
    
    # 3. Nodo 'Alarma'
    # - Tiene dos padres: 'Robo' y 'Terremoto'.
    # - Su CPT define P(Alarma | Robo, Terremoto).
    'Alarma': {
        'parents': ['Robo', 'Terremoto'],
        'cpt': {
            # La clave de la CPT es una tupla con los valores de los padres,
            # en el orden listado en 'parents': (Robo, Terremoto)
            
            # P(Alarma=True | Robo=True, Terremoto=True)
            (True, True): 0.95,
            
            # P(Alarma=True | Robo=True, Terremoto=False)
            (True, False): 0.94,
            
            # P(Alarma=True | Robo=False, Terremoto=True)
            (False, True): 0.29,
            
            # P(Alarma=True | Robo=False, Terremoto=False)
            (False, False): 0.001
        }
    },
    
    # 4. Nodo 'JuanLlama' (JohnCalls)
    # - Tiene un padre: 'Alarma'.
    # - Su CPT define P(JuanLlama | Alarma).
    'JuanLlama': {
        'parents': ['Alarma'],
        'cpt': {
            # La clave es el valor de 'Alarma'
            
            # P(JuanLlama=True | Alarma=True)
            (True,): 0.90, # (Usamos una tupla (True,) para consistencia)
            
            # P(JuanLlama=True | Alarma=False)
            (False,): 0.05
        }
    },
    
    # 5. Nodo 'MariaLlama' (MaryCalls)
    # - Tiene un padre: 'Alarma'.
    # - Su CPT define P(MariaLlama | Alarma).
    'MariaLlama': {
        'parents': ['Alarma'],
        'cpt': {
            # P(MariaLlama=True | Alarma=True)
            (True,): 0.70,
            
            # P(MariaLlama=True | Alarma=False)
            (False,): 0.01
        }
    }
}

# --- P2: Función Auxiliar para Consultar la CPT ---
# (Este es el "algoritmo" más básico: cómo leer la red)

def get_prob_cpt(red, variable, valor, evidencia={}):
    """
    Obtiene la probabilidad P(variable=valor | evidencia)
    directamente de la CPT de la red.
    
    'evidencia' es un diccionario {variable_padre: valor_padre}
    """
    
    # 1. Obtener el nodo de la red
    nodo = red[variable] # Ej: red['Alarma']
    
    # 2. Obtener los padres de la variable
    padres = nodo['parents'] # Ej: ['Robo', 'Terremoto']
    
    # 3. Construir la clave para la CPT
    if not padres:
        # Si no tiene padres, la clave es una tupla vacía
        clave_cpt = ()
    else:
        # Si tiene padres, obtener sus valores desde la 'evidencia'
        # en el orden correcto
        valores_padres = [] # Lista para guardar los valores
        for padre in padres: # Iterar sobre ['Robo', 'Terremoto']
            valores_padres.append(evidencia[padre]) # Añadir el valor (ej. True)
        
        # Convertir la lista en una tupla (porque las tuplas
        # pueden ser claves de diccionario, las listas no)
        clave_cpt = tuple(valores_padres) # Ej: (True, False)
        
    # 4. Obtener la probabilidad P(variable=True) de la CPT
    prob_true = nodo['cpt'][clave_cpt] # Ej: 0.94
    
    # 5. Devolver la probabilidad correcta basada en el 'valor' solicitado
    if valor == True:
        # Si se pidió P(V=True), devolver el valor de la tabla
        return prob_true
    else:
        # Si se pidió P(V=False), devolver 1 - P(V=True)
        return 1.0 - prob_true

# --- P3: Demostración de cómo leer la Red ---
print("--- 1. Definición de Red Bayesiana 'Alarma' ---") # Título
print(f"Red definida con {len(red_alarma)} nodos.")

print("\n--- Consultas de ejemplo (Lectura de CPTs) ---")

# Ejemplo 1: Probabilidad a priori (nodo raíz)
# P(Robo=True)
prob_r = get_prob_cpt(red_alarma, 'Robo', True)
print(f"P(Robo=True) = {prob_r}") # Imprime 0.001

# Ejemplo 2: Probabilidad condicional (nodo hijo)
# P(Alarma=True | Robo=True, Terremoto=False)
evidencia_a = {'Robo': True, 'Terremoto': False}
prob_a = get_prob_cpt(red_alarma, 'Alarma', True, evidencia_a)
print(f"P(Alarma=True | Robo=True, Terremoto=False) = {prob_a}") # Imprime 0.94

# Ejemplo 3: Probabilidad condicional (nodo nieto)
# P(JuanLlama=False | Alarma=True)
evidencia_j = {'Alarma': True}
prob_j = get_prob_cpt(red_alarma, 'JuanLlama', False, evidencia_j)
print(f"P(JuanLlama=False | Alarma=True) = {prob_j:.2f}") # Imprime 0.10

# Ejemplo 4: Otra CPT
# P(MariaLlama=True | Alarma=False)
evidencia_m = {'Alarma': False}
prob_m = get_prob_cpt(red_alarma, 'MariaLlama', True, evidencia_m)
print(f"P(MariaLlama=True | Alarma=False) = {prob_m}") # Imprime 0.01

print("\n¡Red Bayesiana definida exitosamente!")