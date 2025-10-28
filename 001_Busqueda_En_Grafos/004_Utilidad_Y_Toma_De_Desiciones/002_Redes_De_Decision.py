#Implementacion del algoritmo de redes de decisión
# Este algoritmo permite modelar decisiones bajo incertidumbre utilizando una red de nodos y arcos.
# Cada nodo representa una decisión y cada arco representa una posible consecuencia con su probabilidad y utilidad asociada.
#Entre sus principales aplicaciones se encuentran la planificación, la gestión de riesgos y la toma de decisiones estratégicas.
# Entre sus ventajas se incluye la capacidad de visualizar y analizar decisiones complejas de manera más clara.
#Entre sus desventajas se encuentran la necesidad de datos precisos y la complejidad en la construcción de la red.
#En este caso, se esta haciendo un ejemplo simple de una red de decisión para llevar o no un paraguas, dependiendo el valor de la utilidad.

# Representamos la red como un diccionario:
red_decision = { # Diccionario que representa la red de decisión
    'Llevar paraguas': [ # Nodo de decisión
        {'estado': 'Llueve', 'prob': 0.3, 'utilidad': 5},
        {'estado': 'No llueve', 'prob': 0.7, 'utilidad': 1}
    ],
    'No llevar paraguas': [ # Nodo de decisión
        {'estado': 'Llueve', 'prob': 0.3, 'utilidad': -5},
        {'estado': 'No llueve', 'prob': 0.7, 'utilidad': 4}
    ]
}

def utilidad_esperada(accion): # Función para calcular la utilidad esperada
    """Calcula la utilidad esperada de una acción."""
    return sum(e['prob'] * e['utilidad'] for e in red_decision[accion]) # Función para calcular la utilidad esperada

def mejor_decision(red_decision): # Función para encontrar la mejor decisión
    """Devuelve la decisión con mayor utilidad esperada."""
    utilidades = {a: utilidad_esperada(a) for a in red_decision} # Función para calcular la utilidad esperada
    mejor = max(utilidades, key=utilidades.get) # Función para encontrar la mejor decisión
    return mejor, utilidades # Función para encontrar la mejor decisión 

# Calcular la mejor acción
mejor_accion, utilidades = mejor_decision(red_decision) # Función para encontrar la mejor decisión

print("=== RED DE DECISIÓN ===\n")
for a, u in utilidades.items(): # Función para calcular la utilidad esperada
    print(f"Utilidad esperada de '{a}': {u:.2f}")

print(f"\nMejor decisión: {mejor_accion}")
