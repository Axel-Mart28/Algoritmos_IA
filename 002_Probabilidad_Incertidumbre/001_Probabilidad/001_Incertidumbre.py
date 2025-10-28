# Algoritmo de INCERTIDUMBRE (demostracion del concepto)

# Este programa no "calcula" la incertidumbre, sino que la "demuestra".
# Es un concepto, no un cálculo.
# Definición:
# Incertidumbre es la falta de certeza absoluta sobre el resultado de un evento.
# Es el "ruido", la "aleatoriedad" o la "información faltante" del mundo real.
# ¿Cómo funciona este programa?
# Vamos a crear dos "mundos" para un robot que intenta avanzar:
# 1. Mundo Cierto: Si el robot intenta "Avanzar", *siempre* avanza 1 paso. (Resultado 100% predecible).
# 2. Mundo Incierto: Si el robot intenta "Avanzar", el resultado es estocástico (aleatorio).
#    - 80% de prob: Avanza 1 paso (éxito).
#    - 10% de prob: Se queda quieto (el motor falla).
#    - 10% de prob: Retrocede 1 paso (se resbala).
# ¿Qué demuestra esto?
# Que en el mundo real (incierto), la *intención* del agente ("Avanzar") no
# siempre es igual al *resultado*. La probabilidad nos ayuda a modelar y
# razonar sobre esta diferencia.
# Aplicaciones del concepto:
# - Robótica (un robot nunca se mueve perfectamente).
# - Finanzas (el precio de una acción es incierto).
# - Medicina (un diagnóstico nunca es 100% seguro).
# Ventajas de *modelar* la incertidumbre (con probabilidad):
# - Nos permite crear sistemas robustos que pueden manejar fallos.
# - Nos permite tomar decisiones óptimas *dado* el riesgo.
# Desventajas de *ignorar* la incertidumbre:
# - Se crean sistemas frágiles que fallan catastróficamente cuando el mundo no se comporta como se esperaba.

import random # Necesitamos la biblioteca 'random' para simular la aleatoriedad

# --- Definición de nuestros dos "Mundos" ---

def mundo_cierto(posicion_actual, accion):
    """
    Simula un mundo determinista (sin incertidumbre).
    Si la acción es 'Avanzar', *siempre* avanza 1.
    """
    if accion == 'Avanzar':
        return posicion_actual + 1 # El resultado es 100% predecible
    else:
        return posicion_actual # Otra acción

def mundo_incierto(posicion_actual, accion):
    """
    Simula un mundo estocástico (CON incertidumbre).
    El resultado de 'Avanzar' es aleatorio.
    """
    if accion == 'Avanzar':
        # 1. Generar un número aleatorio entre 0.0 y 1.0
        prob = random.random() # Ej: 0.735...
        
        # 2. Comprobar el resultado basado en el número aleatorio
        
        if prob < 0.8: # Pasa el 80% de las veces (ej. 0.0 a 0.799...)
            # Éxito: Avanza 1 paso
            return posicion_actual + 1
        
        elif prob < 0.9: # Pasa el 10% de las veces (ej. 0.8 a 0.899...)
            # Fallo: Se queda quieto
            return posicion_actual
        
        else: # Pasa el 10% restante de las veces (ej. 0.9 a 1.0)
            # Fallo catastrófico: Retrocede 1 paso
            return posicion_actual - 1
            
    else:
        return posicion_actual # Otra acción

# --- P1: Ejecución en el MUNDO CIERTO ---

print("--- 1. Demostración de Incertidumbre ---")
print("\n=== MUNDO CIERTO (Sin Incertidumbre) ===")
print("El robot intentará 'Avanzar' 5 veces.")

posicion = 0 # El robot empieza en 0
accion = 'Avanzar' # La intención del robot

for i in range(5): # Bucle de 5 pasos
    posicion = mundo_cierto(posicion, accion) # Llama al mundo predecible
    print(f"  Paso {i+1}: El robot intentó '{accion}', posición final: {posicion}")

print("-> Resultado del Mundo Cierto: Siempre termina en 5. 100% predecible.")

# --- P2: Ejecución en el MUNDO INCIERTO ---

print("\n=== MUNDO INCIERTO (Estocástico) ===")
print("El robot intentará 'Avanzar' 5 veces.")

posicion = 0 # El robot empieza en 0 (reiniciamos)
accion = 'Avanzar' # La misma intención

for i in range(5): # Bucle de 5 pasos
    posicion = mundo_incierto(posicion, accion) # Llama al mundo impredecible
    print(f"  Paso {i+1}: El robot intentó '{accion}', posición final: {posicion}")

print(f"-> Resultado del Mundo Incierto: Terminó en {posicion}.")
print("(Ejecuta este código de nuevo y el resultado será diferente)")

# --- P3: Múltiples simulaciones del MUNDO INCIERTO ---

print("\n--- Simulando el MUNDO INCIERTO 10 veces (10 'vidas' del robot) ---")

resultados_finales = [] # Lista para guardar dónde termina el robot cada vez
for vida in range(10): # Simular 10 "vidas" o "futuros"
    posicion_sim = 0 # Empezar en 0 cada vez
    for i in range(5): # Dar 5 pasos
        posicion_sim = mundo_incierto(posicion_sim, accion)
    resultados_finales.append(posicion_sim) # Guardar la posición final

print(f"Posiciones finales después de 10 vidas: {resultados_finales}")
print("\nConclusión:")
print("La Incertidumbre es el hecho de que no podemos predecir cuál será")
print("el resultado exacto. Solo podemos saber que el resultado más")
print("probable es 5, pero a veces será 4, 3, 2, etc.")