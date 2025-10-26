# Este algoritmo se basa en la técnica de temple simulado, que es una variante de la búsqueda local.
# La idea viene de la metalurgia, donde un metal caliente se enfría lentamente para evitar defectos y alcanzar una estructura más estable.
# En este algoritmo, el “calor” permite aceptar movimientos hacia soluciones peores temporalmente, para escapar de óptimos locales y aumentar la probabilidad de encontrar el óptimo global.

#Entre sus características principales se encuentran:
# - Control de temperatura: La probabilidad de aceptar soluciones peores disminuye a medida que la temperatura baja.
# - Enfriamiento gradual: La temperatura se reduce lentamente para permitir una exploración adecuada del espacio de soluciones.
# - Eficiencia en memoria: No necesita una lista completa de vecinos, lo que lo hace eficiente en términos de memoria.

import math
import random

grafo = {
    'A': ['B', 'C'],
    'B': ['D', 'E'],
    'C': ['F'],
    'D': [], 'E': ['G'], 'F': [], 'G': []
}

# Heurística (distancia estimada a la meta, menor es mejor)
heuristica = {
    'A': 7,
    'B': 6,
    'C': 4,
    'D': 5,
    'E': 3,
    'F': 2,
    'G': 0  # Meta
}

# Función de búsqueda por temple simulado
def temple_simulado(grafo, heuristica, inicio, objetivo, temperatura_inicial=100, enfriamiento=0.9): # Función de búsqueda por temple simulado
    actual = inicio # Nodo actual comienza en el nodo de inicio
    mejor = actual # Mejor nodo encontrado
    camino = [actual] # Lista para registrar el camino seguido
    temperatura = temperatura_inicial # Temperatura inicial

    print("=== Búsqueda por Temple Simulado ===\n")
    print(f"Inicio en nodo: {actual} (h={heuristica[actual]})\n")

    while temperatura > 1: # Mientras la temperatura sea mayor que 1
        vecinos = grafo[actual] # Obtener los vecinos del nodo actual
         # Si no hay vecinos, se detiene la búsqueda
        if not vecinos:
            print("No hay vecinos, se detiene la búsqueda.\n")
            break

        vecino = random.choice(vecinos) # Elegir un vecino al azar
        delta = heuristica[vecino] - heuristica[actual] # Diferencia de heurística

        # Aceptar si mejora o con cierta probabilidad si empeora
        if delta < 0 or random.random() < math.exp(-delta / temperatura): # Si mejora o se acepta con probabilidad
            actual = vecino # Moverse al vecino
            camino.append(actual) # Registrar el nodo en el camino
            print(f"→ Movimiento a {actual} (h={heuristica[actual]}) [T={temperatura:.2f}]") # Mostrar movimiento

            if heuristica[actual] < heuristica[mejor]: # Si se encuentra un mejor nodo
                mejor = actual # Actualizar mejor nodo

            if actual == objetivo: # Si se alcanza el objetivo
                print("\nObjetivo alcanzado!\n") 
                break

        temperatura *= enfriamiento  # Disminuye la temperatura

    return camino, mejor # Devolver el camino seguido y el mejor nodo encontrado

# Ejemplo de ejecución
camino, mejor = temple_simulado(grafo, heuristica, 'A', 'G')
print("Camino recorrido:", " → ".join(camino))
print("Mejor nodo encontrado:", mejor)
