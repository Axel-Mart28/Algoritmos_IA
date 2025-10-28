# Algoritmo de MUESTREO POR RECHAZO

# Este algoritmo usa el Muestreo Directo para responder consultas condicionales (P(X|e)).

# Definición:
# Es un algoritmo que estima P(X|e) generando muestras del mundo y rechazando (tirando a la basura) todas las que no coinciden con la evidencia 'e'.

# ¿Cómo funciona?
# 1. Decide cuántas muestras (N) quieres generar.
# 2. Inicializa contadores para la variable de consulta X (ej. {'True': 0, 'False': 0}).
# 3. Bucle N veces:
# 4.   Genera una muestra completa usando Muestreo Directo (del algoritmo 6a).
# 5.   Comprueba si esta muestra es *consistente* con la evidencia 'e'. (ej. si e = {JuanLlama:True} y la muestra tiene {JuanLlama:False}, RECHAZAR)
# 6.   Si la muestra es Rechazada: Tírala y continúa el bucle.
# 7.   Si la muestra es Aceptada:
# 8.     Mira el valor de la variable de consulta X (ej. 'Robo').
# 9.     Incrementa el contador para ese valor (ej. contador['Robo'=False] += 1).
# 10. Al final, normaliza los contadores (divide por el número total de muestras *aceptadas).

# Componentes:
# 1. Una función de Muestreo Directo (la que acabamos de hacer).
# 2. Una consulta X (ej. 'Robo').
# 3. Una evidencia 'e' (ej. {'JuanLlama': True}).

# Aplicaciones:
# - La forma más simple de responder P(Causa | Efecto) por aproximación.

# Ventajas:
# - Muy, muy simple de implementar (es solo un filtro sobre Muestreo Directo).
# - Funciona para cualquier consulta.

# Desventajas:
# - Terriblemente ineficiente Si la evidencia 'e' es rara (ej. P(e) = 0.01), rechazará el 99% de las muestras. Para obtener 1000 muestras útiles, tendrás que generar 100,000. El rendimiento colapsa.

# Ejemplo de uso:
# Estimar P(Robo | JuanLlama=True, MariaLlama=True)
# (La misma consulta que hicimos con Enumeración y Eliminación)

# Funciones Auxiliares para Muestreo por Rechazo

import random # ¡Esencial para todos los algoritmos de muestreo!

# --- P1: Definición de la Red y Funciones Auxiliares ---
# (Estas son las dependencias que faltaban)

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

def get_prob_cpt(red, variable, valor, evidencia={}):
    """
    Obtiene P(variable=valor | evidencia) de la CPT
    (Función del tema #1)
    """
    nodo = red[variable]
    padres = nodo['parents']
    if not padres:
        clave_cpt = ()
    else:
        # Crea la tupla de clave (ej. (True, False))
        clave_cpt = tuple([evidencia[padre] for padre in padres])
    prob_true = nodo['cpt'][clave_cpt]
    return prob_true if valor == True else (1.0 - prob_true)

# (Función de normalización del tema #3b, necesaria para el final)
def normalizar(puntuaciones):
    """ Normaliza un diccionario de {etiqueta: puntuacion} """
    total = sum(puntuaciones.values())
    if total == 0:
        num_items = len(puntuaciones)
        if num_items > 0:
            return {e: 1.0 / num_items for e in puntuaciones}
        else:
            return {}
    return {e: p / total for e, p in puntuaciones.items()}

# --- P2: Algoritmo de Muestreo Directo (6a) ---
# (Necesario para 6b)

def muestreo_directo_una_muestra(red):
    """
    Genera UN evento completo (una muestra) de la red.
    """
    variables = red.keys() # Orden topológico ['Robo', 'Terremoto', ...]
    muestra = {} # El evento completo que estamos construyendo
    
    for var in variables:
        # Obtener la P(var=True | padres ya muestreados)
        # 'muestra' contiene los valores de los padres
        prob_true = get_prob_cpt(red, var, True, muestra)
        
        # "Lanzar la moneda"
        if random.random() < prob_true:
            muestra[var] = True
        else:
            muestra[var] = False
            
    return muestra # Devuelve la muestra completa (ej. {'Robo': F, ...})

# --- P3: Funciones Auxiliares para Muestreo por Rechazo (6b) ---

def es_consistente(muestra, evidencia):
    """
    Devuelve True si la muestra coincide con *toda* la evidencia.
    """
    for var_e, val_e in evidencia.items():
        if muestra[var_e] != val_e:
            return False # No coincide, RECHAZAR
    return True # Coincide, ACEPTAR

# --- P4: Algoritmo de Muestreo por Rechazo (6b) ---

def muestreo_por_rechazo(query_X, evidencia_e, red, N):
    """
    Estima P(X|e) usando Muestreo por Rechazo con N muestras.
    """
    
    # 1. Inicializar contadores para la variable de consulta X
    conteo_X = {True: 0, False: 0}
    
    # 2. Bucle N veces (generar N muestras totales)
    for i in range(N):
        
        # 3. Generar una muestra del mundo (usando 6a)
        #    ¡Esta llamada ahora funciona!
        muestra = muestreo_directo_una_muestra(red)
        
        # 4. Comprobar si la muestra es consistente con la evidencia
        if es_consistente(muestra, evidencia_e):
            # --- MUESTRA ACEPTADA ---
            
            # 5. Obtener el valor de la variable de consulta X
            valor_X = muestra[query_X]
            
            # 6. Incrementar el contador para ese valor
            conteo_X[valor_X] += 1
        # else:
            # --- MUESTRA RECHAZADA ---
            # (No hacemos nada)
            pass
            
    # 7. Normalizar los conteos para obtener la distribución
    #    ¡Esta llamada ahora funciona!
    return normalizar(conteo_X)

# --- P5: Ejecutar el cálculo ---
print("Muestreo Directo y Por Rechazo")
N_RECHAZO = 500000 # Necesitamos *muchas* muestras

# La consulta: P(Robo | JuanLlama=True, MariaLlama=True)
consulta_X = 'Robo'
evidencia = {
    'JuanLlama': True,
    'MariaLlama': True
}

print(f"\nConsulta: P({consulta_X} | {evidencia})")
print(f"Generando {N_RECHAZO} muestras totales...")

# Llamar a la función de inferencia
# ¡Esta llamada ahora funcionará porque 'red_alarma' está definida!
dist_rechazo = muestreo_por_rechazo(consulta_X, evidencia, red_alarma, N_RECHAZO)

print("\n--- Resultado (Distribución Aproximada) ---")
print(f"{dist_rechazo}")

# Imprimir el total de muestras aceptadas (para ver la ineficiencia)
total_aceptadas = sum(dist_rechazo.values())
if N_RECHAZO > 0 and total_aceptadas > 0: # Evitar división por cero
    tasa_aceptacion = (total_aceptadas / N_RECHAZO) * 100
    # NOTA: dist_rechazo ya está normalizado, así que sus valores no son los conteos.
    # Necesitaríamos modificar 'muestreo_por_rechazo' para que devuelva
    # el conteo *antes* de normalizar si quisiéramos imprimir esto.
    # Por ahora, nos enfocamos en la probabilidad final.

print(f"\nConclusión:")
print(f"La estimación es P(Robo=True) ~= {dist_rechazo[True]:.4f}")
print(f"El resultado *exacto* (de Eliminación de Variables) era ~0.284.")