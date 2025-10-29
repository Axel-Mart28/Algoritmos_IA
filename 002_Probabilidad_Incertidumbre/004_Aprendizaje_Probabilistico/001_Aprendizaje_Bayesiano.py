# Algoritmo de APRENDIZAJE BAYESIANO (Algoritmo de Demostración)

# Este algoritmo demuestra el *concepto* de Aprendizaje Bayesiano.

# Definición:
# Es el proceso de aplicar la Regla de Bayes para actualizar nuestra *creencia* sobre un modelo (o sus parámetros) a medida que observamos nuevos datos.
#
# Fórmula Clave (La Regla de Bayes para el aprendizaje):
#
# P(Modelo | Datos) = [ P(Datos | Modelo) * P(Modelo) ] / P(Datos)
#
# O, de forma más simple (usando normalización):
# Posterior ∝ Likelihood * Prior
#
# ¿Cómo funciona este programa?
# 1. El "Modelo" (Hipótesis, H): No sabemos si una moneda es justa.
#    Crearemos un conjunto de hipótesis sobre cuál podría ser
#    la probabilidad de 'Cara' (p).
#    H = {p=0.1, p=0.2, p=0.3, p=0.4, p=0.5, ... p=0.9}
#
# 2. El "Prior" P(H) (Creencia A Priori):
#    Nuestra creencia *inicial*. Empezaremos con un "Prior Uniforme",
#    lo que significa que creemos que todas las hipótesis (0.1, 0.2, etc.)
#    son igualmente probables al principio.
#
# 3. Los "Datos" (D): Una secuencia de lanzamientos,
#    ej. ['Cara', 'Cara', 'Cara', 'Cruz']
#
# 4. El "Likelihood" P(D|H) (Verosimilitud):
#    ¿Qué tan probable es ver 3 'Caras' y 1 'Cruz' (Datos),
#    *asumiendo que* la moneda tiene un sesgo 'p' (Modelo)?
#    La fórmula es: p^(# de Caras) * (1-p)^(# de Cruces)
#
# 5. El "Posterior" P(H|D) (Creencia A Posteriori):
#    Nuestra creencia *actualizada*. El programa recalculará
#    la probabilidad de *cada* hipótesis (0.1, 0.2, ...) después
#    de *cada* lanzamiento de moneda.
#
# Veremos cómo la creencia "se mueve" de un prior uniforme (ignorancia)
# hacia un pico alrededor de la verdadera probabilidad.
#
# Aplicaciones:
# - Base conceptual de Naïve-Bayes.
# - Aprender los parámetros (las CPTs) de una Red Bayesiana.
#
# Ventajas:
# - Muestra explícitamente cómo la creencia se actualiza.
# - Funciona bien con pocos datos.
#
# Desventajas:
# - Este enfoque (discreto) es simple, pero en el mundo real,
#   los parámetros son continuos, lo que requiere matemáticas más
#   complejas (ej. "Distribuciones Beta").

import math # Para la función math.pow()
import copy # (No es estrictamente necesario, pero es buena práctica)

# --- P1: Funciones Auxiliares (Normalización y Likelihood) ---

def normalizar(puntuaciones): # (Función del tema #3b de Probabilidad)
    """ Normaliza un diccionario de {etiqueta: puntuacion} """
    total = sum(puntuaciones.values()) # Suma todas las puntuaciones
    if total == 0: # Evitar división por cero
        return {e: 0.0 for e in puntuaciones}
    return {e: p / total for e, p in puntuaciones.items()} # Devuelve {etiqueta: prob}

def calcular_likelihood(datos, hipotesis_p):
    """
    Calcula P(Datos | Hipótesis).
    'datos' es una tupla (num_caras, num_cruces).
    'hipotesis_p' es la prob. de 'Cara' (ej. 0.7).
    """
    num_caras, num_cruces = datos # Desempaquetar la tupla
    
    # p^(#caras) * (1-p)^(#cruces)
    # math.pow(base, exponente)
    prob_datos_dado_p = math.pow(hipotesis_p, num_caras) * \
                        math.pow(1 - hipotesis_p, num_cruces)
                        
    return prob_datos_dado_p # Devuelve la verosimilitud

def imprimir_distribucion(dist):
    """ Función auxiliar para imprimir la creencia """
    print("Creencia actual sobre P(Cara):") # Título
    for p, prob in dist.items(): # Iterar sobre {0.1: 0.05, 0.2: 0.1, ...}
        # Imprimir una barra de texto para visualizar
        barra = "#" * int(prob * 100) # (100 caracteres de ancho)
        print(f"  P(p={p:.1f}) = {prob:6.4f} | {barra}") # Imprime P(p=0.1) = 0.0500 | #####

# --- P2: Algoritmo de Aprendizaje Bayesiano Secuencial ---

def aprendizaje_bayesiano_secuencial(secuencia_datos, hipotesis):
    """
    Actualiza la creencia sobre 'p' después de *cada* dato observado.
    """
    
    # 1. Inicializar el PRIOR P(H)
    #    Empezamos con un Prior Uniforme (todas las hipótesis
    #    son igualmente probables).
    num_hipotesis = len(hipotesis) # Contar cuántas hipótesis hay
    prior = {h: (1.0 / num_hipotesis) for h in hipotesis} # {0.1: 0.1, 0.2: 0.1, ...}
    
    print("--- INICIO DEL APRENDIZAJE ---") # Mensaje
    print(f"Hipótesis posibles P(Cara) = {hipotesis}") # Mensaje
    imprimir_distribucion(prior) # Imprimir el prior uniforme
    
    # Variables para contar caras y cruces
    num_caras = 0
    num_cruces = 0
    
    # 2. Bucle de Aprendizaje (actualizar con cada dato)
    for i, dato in enumerate(secuencia_datos): # Iterar sobre ['Cara', 'Cara', 'Cruz', ...]
        
        print(f"\n--- Paso {i+1}: Observado '{dato}' ---") # Mensaje
        
        # 3. Actualizar conteos
        if dato == 'Cara':
            num_caras += 1
        else:
            num_cruces += 1
            
        datos_actuales = (num_caras, num_cruces) # Tupla (ej. (3, 1))
        
        # 4. Calcular el POSTERIOR
        #    Posterior ∝ Likelihood * Prior
        posterior_no_normalizado = {} # Diccionario para las puntuaciones
        
        for h in hipotesis: # Para cada p (0.1, 0.2, ...)
            
            # Obtener P(Datos | H)
            likelihood = calcular_likelihood(datos_actuales, h)
            
            # Obtener P(H) (el prior de este paso)
            prior_prob = prior[h]
            
            # Calcular Puntuación = Likelihood * Prior
            posterior_no_normalizado[h] = likelihood * prior_prob
            
        # 5. Normalizar el Posterior
        #    (Esto calcula P(D) por nosotros y divide)
        posterior = normalizar(posterior_no_normalizado)
        
        # 6. ¡El Posterior se convierte en el Prior para el siguiente paso!
        #    Esta es la esencia del aprendizaje.
        prior = posterior
        
        # 7. Imprimir la nueva creencia
        imprimir_distribucion(posterior)
        
    # Devolver la creencia final
    return prior

# --- P3: Ejecutar la Simulación ---

# 1. Definir las hipótesis (el "modelo")
#    (Usamos 11 hipótesis para incluir 0.5)
hipotesis_posibles = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

# 2. Definir los Datos (una moneda *muy* sesgada)
#    (8 Caras, 2 Cruces)
datos_observados = [
    'Cara', 'Cara', 'Cara', 'Cruz', 'Cara', 
    'Cara', 'Cara', 'Cruz', 'Cara', 'Cara'
]

# 3. Ejecutar el algoritmo de aprendizaje
creencia_final = aprendizaje_bayesiano_secuencial(datos_observados, hipotesis_posibles)

print("\n--- APRENDIZAJE FINALIZADO ---") # Mensaje
print(f"Datos observados: {datos_observados.count('Cara')} Caras, {datos_observados.count('Cruz')} Cruces") # Resumen
print("Creencia final (Posterior):") # Título final
imprimir_distribucion(creencia_final) # Imprimir el resultado final

print("\nConclusión:")
print("Observa cómo la creencia (las barras '#') empezó 'plana' (Uniforme)")
print("y, con cada observación, se fue 'moviendo' hasta centrarse")
print(f"alrededor de p=0.8, que es la frecuencia real de los datos (8/10).")