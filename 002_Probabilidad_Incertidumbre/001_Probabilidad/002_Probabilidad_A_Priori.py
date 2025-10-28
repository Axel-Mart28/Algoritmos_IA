# Algoritmo de PROBABILIDAD A PRIORI

# Definición:
# Es la probabilidad de un evento *antes* de que tengamos cualquier evidencia o información nueva. Es nuestra "creencia inicial" o "creencia por defecto".
# Se denota como P(A).
# Ej: P(Lluvia) = 0.3 (Nuestra creencia inicial es que hay un 30% de prob. de lluvia hoy, sin haber mirado el cielo o el pronóstico).

# ¿Cómo se calcula?
# Usualmente se basa en datos históricos o frecuencias relativas.
# P(A) = (Número de veces que A ocurrió en el pasado) / (Número total de observaciones)
#
# Componentes:
# 1. Un "espacio muestral" (el total de observaciones o datos).
# 2. Un "evento" (el suceso específico que queremos medir, ej. 'Lluvia').
#
# Aplicaciones:
# - Es el punto de partida fundamental para la Regla de Bayes (que veremos después).
# - Establecer una línea base para un diagnóstico (ej. "La probabilidad de que un
#   paciente *cualquiera* tenga esta enfermedad es P(Enfermedad) = 0.01").
#
# Ventajas:
# - Muy simple de calcular y entender.
#
# Desventajas:
# - Es una creencia "ingenua", ya que no usa ninguna evidencia contextual.
#   (Ej. P(Lluvia)=0.3 ignora si el cielo está negro y lleno de nubes).
#
# Ejemplo de uso:
# Tenemos una base de datos de 10 días de clima. Queremos saber la probabilidad
# a priori de "Lluvioso".
# P(Lluvioso) = (Veces que llovió) / (Total de días)

# --- Datos de ejemplo (nuestro "historial de observaciones") ---
# Imaginemos que observamos el clima durante 10 días
historial_clima = ['Soleado', 'Lluvioso', 'Nublado', 'Soleado', 'Soleado', 
                   'Lluvioso', 'Soleado', 'Nublado', 'Soleado', 'Soleado']

# El evento que nos interesa es 'Lluvioso'
evento_A = 'Lluvioso'

# El total de observaciones (nuestro espacio muestral)
total_observaciones = len(historial_clima) # El valor es 10

# --- Algoritmo de Probabilidad a Priori ---

def calcular_probabilidad_a_priori(datos, evento): # Función para calcular P(A)
    """
    Calcula la probabilidad a priori de un evento basado en un historial de datos.
    P(A) = (Conteo de A) / (Total de datos)
    """
    
    # 1. Contar cuántas veces ocurrió el evento
    # (Usamos el método .count() de las listas de Python)
    conteo_evento = datos.count(evento) # Contará cuántas veces aparece 'evento'
    
    # 2. Obtener el total de observaciones
    total_datos = len(datos) # Longitud de la lista
    
    # 3. Calcular la probabilidad
    if total_datos == 0: # Evitar división por cero
        return 0.0 # Si no hay datos, la probabilidad es 0
    
    # Esta es la fórmula P(A) = Conteo(A) / Total
    probabilidad = conteo_evento / total_datos # Ej: 2 / 10 = 0.2
    
    return probabilidad # Devuelve el resultado

# --- Ejecutar el cálculo ---
print("--- 2. Probabilidad a Priori (P(A)) ---") # Título
print(f"Historial de datos: {historial_clima}") # Muestra los datos
print(f"Total de observaciones: {total_observaciones}") # Muestra el total

# Calcular P(Lluvioso)
# El historial tiene 2 'Lluvioso' de un total de 10.
prob_lluvia = calcular_probabilidad_a_priori(historial_clima, 'Lluvioso') # Llamada a la función
print(f"\nProbabilidad a Priori de 'Lluvioso' (P(Lluvioso)): {prob_lluvia}") # Imprime 0.2

# Calcular P(Soleado)
# El historial tiene 6 'Soleado' de un total de 10.
prob_soleado = calcular_probabilidad_a_priori(historial_clima, 'Soleado') # Llamada a la función
print(f"Probabilidad a Priori de 'Soleado' (P(Soleado)): {prob_soleado}") # Imprime 0.6