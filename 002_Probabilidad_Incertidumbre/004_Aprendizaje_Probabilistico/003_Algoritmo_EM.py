# ALGORITMO EM (EXPECTATION-MAXIMIZATION) 

# Este es un algoritmo de *Aprendizaje No Supervisado*.
# Definición:
# El algoritmo EM es un método iterativo para encontrar los
# *parámetros* de un modelo probabilístico (ej. la media y
# varianza de un cluster) cuando el modelo depende de
# *variables latentes* (información oculta).
#
# El Problema del Huevo y la Gallina:
# 1. Si *supiéramos* a qué cluster (A o B) pertenece cada punto de datos (la variable latente), sería *fácil* calcular la media y varianza de A y B (los parámetros).
# 2. Si *supiéramos* la media y varianza de A y B (los parámetros), sería *fácil* calcular la probabilidad de que un punto pertenezca a A o B (la variable latente).
#
# ¿Cómo funciona? (La Solución EM):
# El algoritmo "rompe" el círculo vicioso alternando dos pasos:
#
# 1. PASO E (Expectation - Esperanza):
#    - "Adivina" los valores de las variables latentes.
#    - Usa los *parámetros actuales* (ej. medias iniciales)
#      para calcular la *probabilidad* (o "responsabilidad")
#      de que cada punto de datos pertenezca a cada cluster.
#    - Ej: "Dado mi PÉSIMO guess inicial, el punto 1
#      pertenece 60% al Cluster A y 40% al Cluster B".
#
# 2. PASO M (Maximization - Maximización):
#    - "Actualiza" los parámetros del modelo.
#    - Usa las "responsabilidades" (el guess del Paso E)
#      como *pesos* para recalcular los parámetros.
#    - Ej: "La nueva media del Cluster A es el *promedio
#      ponderado* de todos los puntos, usando sus
#      responsabilidades de 60%/40% como pesos".
#
# 3. REPETIR: Se repiten los Pasos E y M hasta que los
#    parámetros dejen de cambiar (convergencia).
#
# Componentes:
# 1. Parámetros (theta, $\theta$): Lo que queremos aprender
# 2. Variables Latentes (Z): La información oculta (ej. a qué cluster pertenece cada punto).
# 3. Función de Verosimilitud (Likelihood): P(Datos | $\theta$).
#
# Aplicaciones:
# - Agrupamiento (Clustering) con Modelos de Mezcla Gaussiana (GMMs).
#   (Este es el ejemplo clásico que implementaremos).
# - Aprender los parámetros de los HMMs (se llama "Algoritmo Baum-Welch").
# - Visión por computadora, procesamiento de lenguaje natural.
#
# Ventajas:
# - Muy poderoso para problemas con datos faltantes.
# - Es la base de muchos otros algoritmos de IA.
#
# Desventajas:
# - No hay garantía de encontrar el *mejor* óptimo global
#   (es sensible a la inicialización y puede atascarse
#   en óptimos locales).
# - Puede ser lento para converger.
#
# Ejemplo de uso:
# Tenemos un conjunto de datos 1D que provienen de *dos*
# curvas de campana (Gaussianas), pero no sabemos cuáles son.
# EM descubrirá la media y varianza de ambas curvas.

import math
import random
from collections import namedtuple # Para una estructura de datos simple

# --- P1: Funciones Auxiliares (Gaussianas) ---

def gaussian_pdf(x, media, varianza):
    """
    Calcula la "altura" de la curva de campana (PDF) en el punto 'x'.
    Esto nos da P(x | media, varianza).
    """
    # Manejar varianza cero para evitar división por cero
    if varianza == 0:
        return 1.0 if x == media else 0.0
        
    # Fórmula de la Distribución Normal (Gaussiana)
    termino1 = 1.0 / math.sqrt(2 * math.pi * varianza) # Constante de normalización
    termino2_exponente = -((x - media)**2) / (2 * varianza) # Exponente
    termino2 = math.exp(termino2_exponente) # e^(exponente)
    
    return termino1 * termino2 # Devuelve la probabilidad/densidad

# --- P2: Implementación del Algoritmo EM ---

# Usaremos una tupla con nombre para guardar los parámetros de un cluster
ClusterParams = namedtuple('ClusterParams', ['media', 'varianza', 'peso'])

def algoritmo_em(datos, k_clusters, n_iteraciones):
    """
    Implementa el algoritmo EM para un Modelo de Mezcla Gaussiana (GMM) 1D.
    
    'datos': una lista de números (ej. [1.1, 5.2, 0.9, 4.8])
    'k_clusters': cuántos clusters (curvas) queremos encontrar (ej. 2)
    'n_iteraciones': cuántas veces repetir E-M
    """
    
    # --- 1. Inicialización (Adivinanza Inicial) ---
    # Adivinamos parámetros iniciales para nuestros 'k' clusters.
    # (Esta es una forma simple de inicializar, hay métodos mejores)
    parametros = [] # Lista para guardar los (media, var, peso) de cada cluster
    min_dato, max_dato = min(datos), max(datos) # Encontrar el rango
    
    for i in range(k_clusters):
        # Adivinar una media aleatoria dentro del rango de datos
        media_inicial = random.uniform(min_dato, max_dato)
        # Adivinar una varianza (ej. 1.0)
        var_inicial = 1.0
        # Adivinar un peso (todos empiezan iguales)
        peso_inicial = 1.0 / k_clusters
        
        parametros.append(ClusterParams(media_inicial, var_inicial, peso_inicial))
        
    print(f"Parámetros Iniciales (Guess): {parametros}")

    # Lista para guardar las "responsabilidades" (del Paso E)
    # responsabilidades[i][j] = P(cluster_j | dato_i)
    responsabilidades = [[0.0] * k_clusters for _ in datos]

    # --- 2. Bucle Iterativo (E-M) ---
    for iter in range(n_iteraciones):
        
        # --- PASO E (Expectation) ---
        # Calcular las "responsabilidades" P(Cluster | Dato)
        # para *cada* punto de datos.
        
        for i, dato in enumerate(datos): # Iterar sobre cada punto de dato
            
            puntuaciones = [] # Puntuaciones no normalizadas
            for j in range(k_clusters): # Iterar sobre cada cluster
                # P(Dato | Cluster_j) * P(Cluster_j)
                params_j = parametros[j] # Obtener params del cluster j
                likelihood = gaussian_pdf(dato, params_j.media, params_j.varianza) # P(Dato|C_j)
                prior = params_j.peso # P(C_j)
                puntuacion = likelihood * prior
                puntuaciones.append(puntuacion)
                
            # Normalizar las puntuaciones para obtener probabilidades
            suma_puntuaciones = sum(puntuaciones)
            if suma_puntuaciones == 0: # Evitar división por cero
                for j in range(k_clusters):
                    responsabilidades[i][j] = 1.0 / k_clusters # Asignar uniformemente
            else:
                for j in range(k_clusters):
                    responsabilidades[i][j] = puntuaciones[j] / suma_puntuaciones
        
        # --- PASO M (Maximization) ---
        # Recalcular los parámetros (media, var, peso)
        # usando las 'responsabilidades' como pesos.
        
        nuevos_parametros = [] # Lista para los nuevos params
        
        for j in range(k_clusters): # Iterar sobre cada cluster
            
            # 1. Calcular el "peso" total de este cluster
            suma_responsabilidad_j = sum(responsabilidades[i][j] for i in range(len(datos)))
            
            # 2. Calcular nueva Media (promedio ponderado)
            # Suma[ resp(i, j) * dato_i ] / Suma[ resp(i, j) ]
            suma_ponderada = sum(responsabilidades[i][j] * datos[i] for i in range(len(datos)))
            if suma_responsabilidad_j == 0: # Evitar división por cero
                nueva_media = parametros[j].media # Mantener la media anterior
            else:
                nueva_media = suma_ponderada / suma_responsabilidad_j
            
            # 3. Calcular nueva Varianza (varianza ponderada)
            suma_var_ponderada = sum(
                responsabilidades[i][j] * ((datos[i] - nueva_media)**2) 
                for i in range(len(datos))
            )
            if suma_responsabilidad_j == 0:
                nueva_varianza = parametros[j].varianza
            else:
                nueva_varianza = suma_var_ponderada / suma_responsabilidad_j
                # Evitar que la varianza colapse a cero
                if nueva_varianza < 0.01: nueva_varianza = 0.01 
            
            # 4. Calcular nuevo Peso (fracción de responsabilidad total)
            nuevo_peso = suma_responsabilidad_j / len(datos)
            
            # 5. Guardar los nuevos parámetros
            nuevos_parametros.append(ClusterParams(nueva_media, nueva_varianza, nuevo_peso))
            
        # 6. Actualizar los parámetros para la siguiente iteración
        parametros = nuevos_parametros
        
        # Imprimir progreso (opcional)
        if (iter + 1) % 10 == 0:
            print(f"Iteración {iter+1}/{n_iteraciones} completada.")

    # Devolver los parámetros finales aprendidos
    return parametros

# --- P3: Ejecutar el Algoritmo ---
print("Algoritmo EM") # Título

# 1. Crear datos de ejemplo
#    (Generaremos datos de *dos* distribuciones conocidas)
# [Image of a Gaussian Mixture Model (GMM) showing two overlapping distributions]
datos_reales = []
# Cluster A (media=0, var=1)
for _ in range(50):
    datos_reales.append(random.gauss(0, 1))
# Cluster B (media=5, var=1)
for _ in range(50):
    datos_reales.append(random.gauss(5, 1))

random.shuffle(datos_reales) # ¡Mezclar los datos!
print(f"Datos generados: {len(datos_reales)} puntos (de 2 clusters ocultos)")

# 2. Llamar al algoritmo EM para *encontrar* esos dos clusters
k = 2 # Queremos encontrar 2 clusters
iteraciones = 50 # Repetir 50 veces

parametros_finales = algoritmo_em(datos_reales, k, iteraciones)

print("\n--- APRENDIZAJE FINALIZADO ---") # Mensaje
print("Parámetros finales (aprendidos por EM):") # Título
for i, params in enumerate(parametros_finales): # Iterar sobre los params finales
    print(f"  Cluster {i+1}:")
    print(f"    Media    = {params.media:.4f} (Real era ~0.0 o ~5.0)")
    print(f"    Varianza = {params.varianza:.4f} (Real era ~1.0)")
    print(f"    Peso     = {params.peso:.4f} (Real era ~0.5)")

print("\nConclusión:")
print("EM empezó con adivinanzas aleatorias y, alternando E y M,")
print("convergió a los parámetros reales de los clusters ocultos.")