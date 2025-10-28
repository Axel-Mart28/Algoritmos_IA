# Este es un algoritmo de Valor de la Información
#Este algoritmo tiene como caracteristica calcular el valor que tiene obtener información adicional antes de tomar una decisión.
#Entre sus aplicaciones se encuentran la planificación, la gestión de riesgos y la toma de decisiones estratégicas.
#Su objetivo es ayudar a los tomadores de decisiones a evaluar si vale la pena invertir en información adicional.
#Entre sus ventajas se incluye la capacidad de mejorar la calidad de las decisiones y reducir la incertidumbre.
#Entre sus desventajas se encuentran los costos asociados a la obtención de información y la posibilidad de que esta sea incorrecta.
#En este caso, se evaluará el valor de la información en el contexto de llevar o no un paraguas.
#Valor de la informacion = Utilidad Esperada con Información - Utilidad Esperada sin Información.

# Probabilidades y utilidades base
red_decision = {
    'Llevar paraguas': [ # Nodo de decisión
        {'estado': 'Llueve', 'prob': 0.3, 'utilidad': 5}, # Si llueve entonces elegir "Llevar paraguas" (5)
        {'estado': 'No llueve', 'prob': 0.7, 'utilidad': 1} # Si no llueve entonces elegir "Llevar paraguas" (1)
    ],
    'No llevar paraguas': [ # Nodo de decisión
        {'estado': 'Llueve', 'prob': 0.3, 'utilidad': -5}, # Si llueve entonces elegir "No llevar paraguas" (-5)
        {'estado': 'No llueve', 'prob': 0.7, 'utilidad': 4} # Si no llueve entonces elegir "No llevar paraguas" (4)
    ]
}

def utilidad_esperada(accion): # Función para calcular la utilidad esperada
    """Calcula la utilidad esperada de una acción."""
    return sum(e['prob'] * e['utilidad'] for e in red_decision[accion]) # Función para calcular la utilidad esperada

# --- 1.- Sin información (decisión directa) ---
UE_sin_info = {a: utilidad_esperada(a) for a in red_decision}
mejor_sin_info = max(UE_sin_info, key=UE_sin_info.get)
UE0 = UE_sin_info[mejor_sin_info]

# --- 2.- Con información (decisión perfecta) ---
# Si sabemos el clima, elegimos la mejor acción en cada caso
UE_lluvia = max(5, -5)   # Si llueve entonces elegir "Llevar paraguas" (5)
UE_sol = max(1, 4)       # Si no llueve entonces elegir "No llevar paraguas" (4)

UE1 = (0.3 * UE_lluvia) + (0.7 * UE_sol)

# --- 3.- Valor de la Información ---
VOI = UE1 - UE0 # Valor de la Información

print("=== VALOR DE LA INFORMACIÓN ===\n")
print(f"Utilidad esperada sin información: {UE0:.2f}")
print(f"Utilidad esperada con información perfecta: {UE1:.2f}")
print(f"\n➡️  Valor esperado de la información: {VOI:.2f}")
