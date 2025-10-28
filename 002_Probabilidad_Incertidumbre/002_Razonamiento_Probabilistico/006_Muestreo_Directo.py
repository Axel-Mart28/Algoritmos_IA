# Algoritmo de MUESTREO DIRECTO

# Este es el algoritmo de inferencia aproximada más simple.

# Definición:
# Es un algoritmo que genera "muestras" (eventos completos) simulando la Red Bayesiana desde arriba (padres) hacia abajo (hijos).

# ¿Cómo funciona?
# 1. Se necesita una "ordenación topológica" de la red (asegurarse de que procesamos a los padres antes que a sus hijos).
# 2. Para cada nodo 'X' en ese orden:
# 3.   Mira los valores que *ya* simulamos para sus Padres.
# 4.   Busca la P(X=True | Padres) en su CPT.
# 5.   "Lanza una moneda" con esa probabilidad (ej. si P=0.7, genera un 70% de True y 30% de False).
# 6.   Añade el resultado (ej. X=True) a la muestra actual.
# 7. Repite para todos los nodos. El resultado es 1 muestra (un "evento completo").

# Componentes:
# 1. La Red Bayesiana (la estructura de la red_alarma).
# 2. Un orden topológico (en nuestro caso, el orden del diccionario ya lo es).
# 3. Un generador de números aleatorios (random.random()).
#
# Aplicaciones:
# - Estimar probabilidades a priori (ej. P(JuanLlama=True)).
# - Es el motor que usa el "Muestreo por Rechazo".
#
# Ventajas:
# - Muy simple y rápido de implementar.
# - Cada muestra se genera en tiempo lineal (muy rápido).
#
# Desventajas:
# - Por sí solo, *no puede* calcular probabilidades condicionales P(X|e).
# - Si un evento es muy raro (ej. P(Robo)=0.001), necesitarás
#   millones de muestras para verlo siquiera una vez.
#
# Ejemplo de uso:
# Generar 10,000 muestras de nuestra red 'Alarma' y luego
# contar cuántas de ellas tienen 'JuanLlama' = True.
# P(JuanLlama=True) ~= (Conteo de JuanLlama=True) / 10000

import random # Esencial para todos los algoritmos de muestreo

# --- P1: Definición de la Red y Funciones Auxiliares ---
# (Necesitamos la red  de los temas anteriores)

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

# (Función del tema #1)
def get_prob_cpt(red, variable, valor, evidencia={}):
    """ Obtiene P(variable=valor | evidencia) de la CPT """
    nodo = red[variable]
    padres = nodo['parents']
    if not padres:
        clave_cpt = ()
    else:
        # Crea la tupla de clave (ej. (True, False))
        clave_cpt = tuple([evidencia[padre] for padre in padres])
    prob_true = nodo['cpt'][clave_cpt]
    return prob_true if valor == True else (1.0 - prob_true)

# --- P2: Algoritmo de Muestreo Directo (para 1 muestra) ---

def muestreo_directo_una_muestra(red):
    """
    Genera UN evento completo (una muestra) de la red.
    """
    
    # 1. El orden de las claves en el dict es un orden topológico válido
    variables = red.keys() # ['Robo', 'Terremoto', 'Alarma', ...]
    
    # 2. Inicializar la muestra (el evento completo)
    muestra = {} # ej: {'Robo': True, 'Terremoto': False, ...}
    
    # 3. Iterar sobre las variables en orden
    for var in variables:
        
        # 4. Obtener la P(var=True | padres)
        #    'get_prob_cpt' usa la 'muestra' como evidencia,
        #    ya que la 'muestra' *ya contiene* los valores
        #    de los padres (que vinieron antes en la lista).
        prob_true = get_prob_cpt(red, var, True, muestra)
        
        # 5. "Lanzar la moneda"
        if random.random() < prob_true: # Si (ej) 0.45 < 0.94
            muestra[var] = True # Asignar True
        else:
            muestra[var] = False # Asignar False
            
    # 6. Devolver la muestra completa
    return muestra

# --- P3: Ejecutar Muestreo Directo para estimar P(JuanLlama) ---
print("Muestreo Directo") # Título
N_MUESTRAS = 10000 # Número de simulaciones

print(f"Generando {N_MUESTRAS} muestras directas...")
muestras_generadas = [] # Lista para guardar todas las muestras
for i in range(N_MUESTRAS):
    muestras_generadas.append(muestreo_directo_una_muestra(red_alarma))

# Ahora, contamos cuántas veces 'JuanLlama' fue True
conteo_juan_llama = 0
for muestra in muestras_generadas:
    if muestra['JuanLlama'] == True:
        conteo_juan_llama += 1

# Calcular la probabilidad estimada
P_juan_llama_estimada = conteo_juan_llama / N_MUESTRAS

print(f"\nMuestras generadas: {N_MUESTRAS}")
print(f"Veces que Juan llamó (conteo): {conteo_juan_llama}")
print(f"Probabilidad estimada P(JuanLlama=True): {P_juan_llama_estimada:.4f}")
# (La respuesta exacta es ~0.052, la estimación debería estar cerca)