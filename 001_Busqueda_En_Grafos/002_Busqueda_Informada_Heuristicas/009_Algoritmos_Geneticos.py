# Los algoritmos genéticos son técnicas de búsqueda inspiradas en el proceso de evolución natural (selección natural, cruza,  mutación y supervivencia del más apto).
# Su objetivo es buscar soluciones óptimas (o cercanas al óptimo) en espacios grandes y complejos donde los métodos tradicionales fallan o serían demasiado lentos.
# Entre sus ventajas se encuentran:
# - Capacidad para escapar de óptimos locales: Al aceptar soluciones peores temporalmente, el algoritmo puede explorar más ampliamente el espacio de soluciones.
# - Flexibilidad: Puede adaptarse a una amplia variedad de problemas de optimización.
# Entre sus desventajas se encuentran:
# - Parámetros sensibles: La elección de la temperatura inicial y la tasa de enfriamiento puede afectar significativamente el rendimiento del algoritmo.
# En este programa se muestra un ejemplo de su funcionamiento con la funcion f(x) = x^2 para x en el rango de 0 a 31.

import random #Libreria para operaciones aleatorias

# === Función objetivo ===
def fitness(x):
    return x ** 2  # Queremos maximizar f(x) = x^2

# === Convertir entre binario y decimal ===
def binario_a_decimal(cromosoma):
    return int("".join(str(bit) for bit in cromosoma), 2) # Convertir lista de bits a entero decimal

def decimal_a_binario(x, longitud=5): # Convertir entero decimal a lista de bits
    return [int(bit) for bit in format(x, f"0{longitud}b")] # Formatear como binario con longitud fija

# === Crear población inicial ===
def crear_poblacion(tamaño, longitud):
    return [[random.randint(0, 1) for _ in range(longitud)] for _ in range(tamaño)] # Lista de listas de bits

# === Selección por ruleta ===
def seleccion_ruleta(poblacion): # Selección basada en fitness proporcional
    fitnesses = [fitness(binario_a_decimal(ind)) for ind in poblacion] # Calcular fitness de cada individuo
    total_fit = sum(fitnesses) # Suma total de fitness
    probabilidades = [f / total_fit for f in fitnesses] # Probabilidades proporcionales al fitness
    elegido = random.choices(poblacion, weights=probabilidades, k=1)[0] # Selección aleatoria ponderada
    return elegido # Devolver el individuo seleccionado

# === Cruzamiento (crossover) ===
def cruzar(padre1, padre2): # Cruce de un punto
    punto = random.randint(1, len(padre1) - 2) # Elegir punto de cruce
    hijo1 = padre1[:punto] + padre2[punto:] # Crear hijos combinando padres
    hijo2 = padre2[:punto] + padre1[punto:] # Crear segundo hijo
    return hijo1, hijo2 # Devolver los hijos

# === Mutación ===
def mutar(cromosoma, prob_mut=0.1): # Mutar bits con cierta probabilidad
    return [bit if random.random() > prob_mut else 1 - bit for bit in cromosoma] # Invertir bit con probabilidad prob_mut

# === Algoritmo Genético ===
def algoritmo_genetico(tamaño_poblacion=6, generaciones=10, prob_mut=0.1): # Parámetros del algoritmo
    longitud = 5  # Longitud del cromosoma (5 bits)
    poblacion = crear_poblacion(tamaño_poblacion, longitud) # Crear población inicial

    print("=== Algoritmo Genético ===\n")
    print("Población inicial:")
    for ind in poblacion: # Mostrar población inicial
        x = binario_a_decimal(ind) # Convertir a decimal
        print(ind, f"x={x}, f(x)={fitness(x)}")
    print()

    for gen in range(generaciones): # Bucle sobre generaciones
        nueva_poblacion = [] # Lista para nueva población
        print(f"--- Generación {gen + 1} ---")

        while len(nueva_poblacion) < tamaño_poblacion: # Crear nueva población
            padre1 = seleccion_ruleta(poblacion) # Seleccionar padres
            padre2 = seleccion_ruleta(poblacion) # Seleccionar segundo padre
            hijo1, hijo2 = cruzar(padre1, padre2) # Cruzar padres para crear hijos

            hijo1 = mutar(hijo1, prob_mut) # Mutar hijos
            hijo2 = mutar(hijo2, prob_mut) # Mutar segundo hijo

            nueva_poblacion.extend([hijo1, hijo2]) # Agregar hijos a la nueva población

        poblacion = nueva_poblacion[:tamaño_poblacion] # Asegurar tamaño de población

        # Mostrar información de la generación
        mejores = sorted(poblacion, key=lambda ind: fitness(binario_a_decimal(ind)), reverse=True) # Ordenar por fitness
        mejor = mejores[0] # Mejor individuo
        x_mejor = binario_a_decimal(mejor) # Convertir a decimal
        print(f"Mejor individuo: {mejor} → x={x_mejor}, f(x)={fitness(x_mejor)}\n")

    print("Proceso completado.\n")
    mejor = max(poblacion, key=lambda ind: fitness(binario_a_decimal(ind))) # Mejor individuo final
    x_mejor = binario_a_decimal(mejor) # Convertir a decimal
    print(f"Mejor solución encontrada: x={x_mejor}, f(x)={fitness(x_mejor)}")

# Ejecutar el algoritmo
algoritmo_genetico()
