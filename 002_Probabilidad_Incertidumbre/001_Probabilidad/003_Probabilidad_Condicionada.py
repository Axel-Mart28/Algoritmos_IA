# Algoritmo de PROBABILIDAD CONDICIONADA

# Este algoritmo calcula la probabilidad de un evento A, *dado que* (o "a condición de") un evento B ya ha ocurrido. Se escribe P(A|B).

# Definición:
# Es la actualización de nuestra creencia sobre A, después de observar la evidencia B.
# Responde a la pregunta: "¿Qué tan probable es A, *sabiendo que B es verdad*?"

# ¿Cómo funciona (Fórmula)?:
# P(A|B) = P(A y B) / P(B)
# P(A y B) = Probabilidad conjunta de que A y B ocurran juntos.
# P(B) = Probabilidad a priori de que B ocurra (la vimos en el tema #2).
#
# ¿Cómo funciona (Intuitivamente)?:
# El algoritmo "encoge" el universo de posibilidades.
# 1. Ignora todos los datos donde la evidencia B *no* ocurrió.
# 2. Nuestro "nuevo universo" o "nuevo denominador" es ahora P(B).
# 3. Dentro de ese nuevo universo, calcula la probabilidad de A.
#
# Componentes:
# 1. Hipótesis (A): El evento sobre el que queremos saber (ej. 'Lluvia').
# 2. Evidencia (B): La información nueva que acabamos de recibir (ej. 'Tráfico').

# Aplicaciones:
# - El núcleo del diagnóstico médico: P(Enfermedad | Síntoma)
# - Filtros de Spam: P(Spam | Palabra "Viagra")
# - Todo el razonamiento en IA (visión por computadora, predicciones).

# Ventajas:
# - Es la forma matemática de "aprender de la experiencia".

# Desventajas:
# - A veces es difícil calcular P(A y B) directamente (esto es lo que la Regla de Bayes resuelve).

# Ejemplo de uso:
# Tenemos datos de Clima y Tráfico.
# P(Lluvioso) (a priori) puede ser baja.
# Pero P(Lluvioso | Tráfico='Sí') (condicionada) será mucho más alta.

# --- Datos de ejemplo (Observaciones de Clima y Tráfico) ---
# Cada tupla es una observación (Clima, Tráfico)
historial_completo = [
    ('Soleado', 'No'),    # Día 1
    ('Lluvioso', 'Sí'),   # Día 2
    ('Nublado', 'Sí'),    # Día 3
    ('Soleado', 'No'),    # Día 4
    ('Lluvioso', 'Sí'),   # Día 5
    ('Lluvioso', 'No'),   # Día 6
    ('Nublado', 'No'),    # Día 7
    ('Soleado', 'Sí')     # Día 8
]

# --- Algoritmo de Probabilidad Condicionada (Método de conteo intuitivo) ---

def calcular_probabilidad_condicionada(datos, hipotesis_A, evidencia_B):
    """
    Calcula P(A|B) directamente desde los datos.
    P(A|B) = (Conteo de A y B) / (Conteo de B)
    """
    
    # 1. Inicializar contadores
    conteo_evidencia_B = 0     # Contador para cuántas veces B es verdad
    conteo_hipotesis_A_y_B = 0 # Contador para A y B son verdad *juntas*
    
    # 2. Iterar sobre todos los datos
    for observacion in datos:
        evento_clima, evento_trafico = observacion # Desempaquetar la tupla
        
        # 3. Comprobar si la evidencia B está presente en esta observación
        #    (B = 'Tráfico es Sí')
        if evento_trafico == evidencia_B:
            # Si B es verdad, nuestro "nuevo universo" crece
            conteo_evidencia_B += 1
            
            # 4. *Dentro* de este nuevo universo, ver si A también es verdad
            #    (A = 'Clima es Lluvioso')
            if evento_clima == hipotesis_A:
                conteo_hipotesis_A_y_B += 1
                
    # 5. Calcular la probabilidad
    if conteo_evidencia_B == 0:
        # Si la evidencia B nunca ocurrió, no podemos calcular la probabilidad
        return 0.0
        
    # P(A|B) = (Veces que A y B pasaron) / (Veces que B pasó)
    probabilidad = conteo_hipotesis_A_y_B / conteo_evidencia_B
    
    return probabilidad

# --- Ejecutar el cálculo ---
print("--- 3a. Probabilidad Condicionada P(A|B) ---") # Título
print(f"Historial de datos: {historial_completo}") # Muestra los datos

# Queremos calcular P(Lluvioso | Tráfico='Sí')
hipotesis = 'Lluvioso'
evidencia = 'Sí'

# Búsqueda manual para comprobación:
# Evidencia (Tráfico='Sí'): [('Lluvioso', 'Sí'), ('Nublado', 'Sí'), ('Lluvioso', 'Sí'), ('Soleado', 'Sí')] -> conteo_evidencia_B = 4
# Hipótesis (Lluvioso) dentro de esa evidencia: [('Lluvioso', 'Sí'), ('Lluvioso', 'Sí')] -> conteo_hipotesis_A_y_B = 2
# Resultado esperado: 2 / 4 = 0.5

# Llamar a la función del algoritmo
prob_cond = calcular_probabilidad_condicionada(historial_completo, hipotesis, evidencia)

print(f"\nHipótesis (A): {hipotesis}")
print(f"Evidencia (B): Tráfico = '{evidencia}'")
print(f"\nProbabilidad Condicionada P(A|B): {prob_cond}") # Imprime 0.5

# Comparar con la probabilidad a priori (del tema #2)
# P(Lluvioso) = 3 / 8 = 0.375
# P(Lluvioso | Tráfico='Sí') = 0.5
# ¡Nuestra creencia en "Lluvia" aumentó al observar "Tráfico"!