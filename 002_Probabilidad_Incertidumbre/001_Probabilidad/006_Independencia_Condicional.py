# Algoritmo de INDEPENDENCIA CONDICIONAL

# Este es un CONCEPTO fundamental, no un algoritmo en sí.
# Es la idea que hace que las Redes Bayesianas (un tema posterior) sean posibles.

# Definición:
# Dos variables (A y B) son *condicionalmente independientes* dada una tercera variable (C), si saber el valor de C hace que A y B dejen de estar relacionadas.
#
# En otras palabras: si ya conozco C, aprender sobre B no me da *ninguna
# información nueva* sobre A.
#
# Analogía Clásica (Causa Común):
# - Variable A = Ventas de helados
# - Variable B = Ahogamientos en la playa
# - Variable C = Temperatura (el clima)
#
# 1. ¿A y B están correlacionadas? SÍ. En un día con muchos ahogamientos (B), probablemente también haya muchas ventas de helados (A). P(A|B) != P(A).
# 2. ¿Son independientes *dado C*? SÍ. Si decimos "Hoy la temperatura es de 40°C" (ya conozco C), y luego digo "hubo 5 ahogamientos" (B), ¿Sorprende que las ventas de helados (A) sean altas? No. La temperatura (C) ya explicaba ambas cosas. C es la "causa común" que bloquea la conexión entre A y B.
#
# Notación Matemática:
# P(A | B, C) = P(A | C)
# "La probabilidad de A, sabiendo B y C, es la misma que la probabilidad de A, sabiendo solo C."
#
# ¿Cómo funciona este programa?:
# Vamos a *demostrar* esta igualdad usando un conjunto de datos.
# 1. Demostraremos que P(Helados | Ahogamientos) es *diferente* de P(Helados).
#    (Esto prueba que NO son independientes).
# 2. Demostraremos que P(Helados | Ahogamientos, Temp='Calor') es *igual* a
#    P(Helados | Temp='Calor').
#    (Esto prueba que SÍ son condicionalmente independientes).
#
# Aplicaciones:
# - Permite simplificar modelos probabilísticos enormemente.
# - Es la suposición central de las Redes Bayesianas y el clasificador Naive Bayes.
#
# Ventajas:
# - Reduce el número de probabilidades que necesitamos calcular.
#   (Es más fácil estimar P(A|C) que P(A|B,C,D,E...)).
#
# Desventajas:
# - Es una "suposición". Si suponemos que dos variables son condicionalmente
#   independientes pero en realidad no lo son, nuestro modelo será incorrecto.

import math # Para la comparación de decimales

# --- P1: Datos de Ejemplo (Causa Común) ---
# Columnas: (C: Temperatura, A: Ventas Helados, B: Ahogamientos)
historial = [
    # Días de Calor (5)
    ('Calor', 'Altas', 'Sí'), # Causa común lleva a A y B
    ('Calor', 'Altas', 'Sí'), # Causa común lleva a A y B
    ('Calor', 'Altas', 'Sí'), # Causa común lleva a A y B
    ('Calor', 'Altas', 'Sí'), # Causa común lleva a A y B
    ('Calor', 'Altas', 'No'), # Causa común lleva a A, pero B no ocurrió (suerte)
    
    # Días de Frío (5)
    ('Frío', 'Bajas', 'No'), # Causa común (frío) lleva a no-A y no-B
    ('Frío', 'Bajas', 'No'), 
    ('Frío', 'Bajas', 'No'), 
    ('Frío', 'Bajas', 'No'), 
    ('Frío', 'Bajas', 'No')
]
total_datos = len(historial) # Total 10 observaciones

# --- P2: Funciones Auxiliares (Contadores) ---
# (Estas son versiones más específicas de nuestro algoritmo P(A|B) anterior)

def p_a_dado_b(datos, A, B):
    """ Calcula P(A|B) = (Conteo de A y B) / (Conteo de B) """
    conteo_B = 0
    conteo_A_y_B = 0
    for C_i, A_i, B_i in datos: # Iterar sobre (Temp, Helados, Ahog.)
        if B_i == B: # Si la evidencia B es verdad...
            conteo_B += 1
            if A_i == A: # ...y la hipótesis A también es verdad
                conteo_A_y_B += 1
    if conteo_B == 0: return 0.0 # Evitar división por cero
    return conteo_A_y_B / conteo_B # Devuelve P(A|B)

def p_a_dado_b_y_c(datos, A, B, C):
    """ Calcula P(A | B, C) = (Conteo A,B,C) / (Conteo B,C) """
    conteo_B_y_C = 0
    conteo_A_y_B_y_C = 0
    for C_i, A_i, B_i in datos: # Iterar sobre (Temp, Helados, Ahog.)
        if B_i == B and C_i == C: # Si ambas evidencias B y C son verdad...
            conteo_B_y_C += 1
            if A_i == A: # ...y la hipótesis A también es verdad
                conteo_A_y_B_y_C += 1
    if conteo_B_y_C == 0: return 0.0 # Evitar división por cero
    return conteo_A_y_B_y_C / conteo_B_y_C # Devuelve P(A|B,C)

def p_a_dado_c(datos, A, C):
    """ Calcula P(A | C) = (Conteo A,C) / (Conteo C) """
    # (Esta es la misma función que p_a_dado_b, pero con C)
    conteo_C = 0
    conteo_A_y_C = 0
    for C_i, A_i, B_i in datos: # Iterar sobre (Temp, Helados, Ahog.)
        if C_i == C: # Si la evidencia C es verdad...
            conteo_C += 1
            if A_i == A: # ...y la hipótesis A también es verdad
                conteo_A_y_C += 1
    if conteo_C == 0: return 0.0 # Evitar división por cero
    return conteo_A_y_C / conteo_C # Devuelve P(A|C)

# --- P3: Demostración ---
print("--- 5. Demostración de Independencia Condicional ---") # Título
print(f"Datos: {total_datos} días de (Temp, Helados, Ahogamientos)")
print("A = Ventas Helados ('Altas')")
print("B = Ahogamientos ('Sí')")
print("C = Temperatura ('Calor')")

# --- Paso 1: Demostrar que A y B NO son independientes ---
print("\n--- Paso 1: ¿Son A y B independientes? (P(A|B) == P(A)) ---")

# P(A) = P(Helados='Altas')
# Conteo de 'Altas' = 5. Total = 10.
conteo_A = sum(1 for C,A,B in historial if A == 'Altas') # Contar 'Altas'
p_A = conteo_A / total_datos # 5 / 10 = 0.5
print(f"P(A) = P(Helados='Altas') = {p_A:.4f}") # Imprime 0.5000

# P(A|B) = P(Helados='Altas' | Ahogamientos='Sí')
# Conteo de 'Sí' = 4.
# Conteo de 'Altas' y 'Sí' = 4.
p_A_dado_B = p_a_dado_b(historial, A='Altas', B='Sí') # 4 / 4 = 1.0
print(f"P(A|B) = P(Helados='Altas' | Ahogamientos='Sí') = {p_A_dado_B:.4f}") # Imprime 1.0000

# Comprobar la independencia
print(f"\nResultado: {p_A_dado_B:.4f} != {p_A:.4f}") # 1.0 != 0.5
print("Conclusión: A y B NO son independientes. Saber de Ahogamientos (B) cambia")
print("   drásticamente nuestra creencia sobre las Ventas de Helados (A).")

# --- Paso 2: Demostrar que A y B SÍ son condicionalmente independientes dado C ---
print("\n--- Paso 2: ¿Son A y B independientes *dado C*? (P(A|B,C) == P(A|C)) ---")

# La condición es: C = Temperatura ('Calor')

# P(A|C) = P(Helados='Altas' | Temp='Calor')
# Conteo de 'Calor' = 5.
# Conteo de 'Altas' y 'Calor' = 5.
p_A_dado_C = p_a_dado_c(historial, A='Altas', C='Calor') # 5 / 5 = 1.0
print(f"P(A|C) = P(Helados='Altas' | Temp='Calor') = {p_A_dado_C:.4f}") # Imprime 1.0000

# P(A|B,C) = P(Helados='Altas' | Ahogamientos='Sí' Y Temp='Calor')
# Conteo de 'Sí' y 'Calor' = 4.
# Conteo de 'Altas', 'Sí' y 'Calor' = 4.
p_A_dado_B_y_C = p_a_dado_b_y_c(historial, A='Altas', B='Sí', C='Calor') # 4 / 4 = 1.0
print(f"P(A|B,C) = P(Helados='Altas' | Ahogamientos='Sí', Temp='Calor') = {p_A_dado_B_y_C:.4f}") # Imprime 1.0000

# Comprobar la independencia condicional
print(f"\nResultado: {p_A_dado_B_y_C:.4f} == {p_A_dado_C:.4f}") # 1.0 == 1.0
print(" Conclusión: A y B SÍ son condicionalmente independientes dado C.")
print("   Una vez que sabemos que la Temp es 'Calor' (C), saber sobre")
print("   Ahogamientos (B) no nos da NINGUNA información nueva sobre Helados (A).")
print("   ¡La Temperatura (C) 'bloquea' la conexión!")