#Este algoritmo es sobre la teoría de la utilidad y cómo se puede aplicar para tomar decisiones óptimas en situaciones de incertidumbre.
# Definimos las acciones posibles y sus resultados con probabilidades y utilidades asociadas.
# Permite asignar valores numéricos a las preferencias y calcular la utilidad esperada para cada acción.
# Cada acción tiene múltiples resultados posibles, cada uno con una probabilidad y una utilidad asociada.
# Los componentes de este algoritmo son: la definición de acciones, resultados, probabilidades y utilidades.
# Algunas de sus aplicaciones incluyen la toma de decisiones en economía, negocios y planificación estratégica.
# Algunas de sus ventajas son: permite una evaluación cuantitativa de las decisiones, ayuda a identificar la mejor opción bajo incertidumbre.
# Algunas de sus desventajas son: requiere estimaciones precisas de probabilidades y utilidades, puede ser complejo para decisiones con muchos resultados posibles.
# Ejemplo de uso: Decidir entre buscar energía solar o regresar a la base en una misión espacial.
# Este algoritmo tiene 3 tipos de agentes dependiendo como actue ante el riesgo:
# 1.- Agente neutral al riesgo (x = x) solo le importa el valor promedio.
# 2.- Agente averso al riesgo (raiz(x)) prefiere evitar pérdidas.
# 3.- Agente amante del riesgo (x^2) busca maximizar ganancias.



acciones = { # Definición de acciones con resultados, probabilidades y utilidades
    'Buscar energía solar': [ # Acción 1
        {'resultado': 'Encuentra energía', 'prob': 0.7, 'utilidad': 10},
        {'resultado': 'No encuentra energía', 'prob': 0.3, 'utilidad': -2}
    ],
    'Regresar a base': [ # Acción 2
        {'resultado': 'Carga segura', 'prob': 1.0, 'utilidad': 5}
    ]
}

def utilidad_esperada(accion): # Acción 1
    """Calcula la utilidad esperada de una acción."""
    return sum(evento['prob'] * evento['utilidad'] for evento in acciones[accion]) # Acción 1

def mejor_accion(acciones): # Acción 2
    """Devuelve la acción con mayor utilidad esperada."""
    utilidades = {a: utilidad_esperada(a) for a in acciones} # Acción 2
    mejor = max(utilidades, key=utilidades.get) # Acción 2
    return mejor, utilidades # Acción 2

# Evaluar las acciones
accion_optima, utilidades = mejor_accion(acciones) # Acción 2

print("=== Teoría de la Utilidad ===\n")
for a, u in utilidades.items(): # Acción 2
    print(f"Utilidad esperada de '{a}': {u:.2f}")

print(f"\n Mejor acción: {accion_optima}")
