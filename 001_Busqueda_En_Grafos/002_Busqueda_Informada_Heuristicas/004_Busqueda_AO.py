# Algoritmo de Búsqueda AO* (And-Or Star)
#La característica principal de AO* es su capacidad para manejar problemas que pueden descomponerse en subproblemas independientes (AND) y alternativas (OR).
#La diferencia entre A* y AO* radica en que AO* está diseñado para trabajar con grafos que representan problemas con estructuras más complejas, donde un nodo puede tener múltiples hijos que deben ser explorados conjuntamente (AND) o alternativamente (OR).

#Grafo representado como una estructura AND-OR
#         A
#       / | \
#   (AND) | (OR)
#   B   C    D
#  / \       |
# E   F      G

#Este algoritmo evalúa nodos AND y OR de manera diferente:
# - Para nodos OR, selecciona el hijo con el costo mínimo.
# - Para nodos AND, suma los costos de todos los hijos.

# Cada nodo tiene una lista de posibles conexiones.
# Cada conexión puede ser un conjunto de nodos (AND) o un solo nodo (OR)
grafo = {
    'A': [[('B', 1), ('C', 1)], [('D', 3)]],  # A -> (B y C) o D
    'B': [[('E', 1), ('F', 2)]],              # B -> (E y F)
    'C': [[('G', 2)]],                        # C -> G
    'D': [],                                  # D es terminal
    'E': [], 'F': [], 'G': []                 # Terminales
}

# Heurísticas iniciales (estimación de costo restante)
h = {'A': 5, 'B': 4, 'C': 3, 'D': 2, 'E': 0, 'F': 0, 'G': 0}

# Función principal AO*
def ao_star(nodo):
    print(f"\nEvaluando nodo: {nodo}")

    # Si es un nodo terminal, regresamos su heurística
    if not grafo[nodo]:
        print(f"  Nodo {nodo} es terminal (h={h[nodo]})")
        return h[nodo]

    costos_ramas = [] # Lista para almacenar costos de cada rama

    # Evaluar cada conexión (puede ser AND o OR)
    for conexion in grafo[nodo]: # Cada conexión puede ser un conjunto de nodos (AND) o un solo nodo (OR)
        costo_total = 0 # Costo total para esta conexión
        nombres_hijos = [] # Nombres de los hijos en esta conexión

        # Cada conexión puede tener varios hijos (AND)
        for (hijo, costo) in conexion: # Iterar sobre los hijos de la conexión
            costo_total += costo + ao_star(hijo) # Sumar costo del hijo y su heurística
            nombres_hijos.append(hijo) # Agregar el nombre del hijo

        costos_ramas.append((costo_total, nombres_hijos)) # Almacenar el costo total y los nombres de los hijos

    # Seleccionar la rama con menor costo total
    mejor_costo, mejor_rama = min(costos_ramas, key=lambda x: x[0]) # Elegir la rama con el costo mínimo
    h[nodo] = mejor_costo  # Actualizar heurística del nodo

    print(f"  Nodo {nodo} actualiza h = {mejor_costo}, usando hijos {mejor_rama}") # Mostrar actualización de heurística
    return mejor_costo # Devolver el mejor costo encontrado


# Ejecución del algoritmo AO*
print("=== Búsqueda AO* (AND-OR A estrella) ===")
costo_total = ao_star('A')
print("\n Costo total estimado para resolver A:", costo_total)
