# Algoritmo de TEORÍA DE JUEGOS:
# Este algoritmo implementa el famoso Dilema del Prisionero, un juego de teoría de juegos
# que muestra por qué dos individuos racionales podrían no cooperar, incluso si parece
# que es en su mejor interés hacerlo. Es un ejemplo clásico de conflicto entre interés
# individual y beneficio colectivo.
# Entre sus aplicaciones están: economía, sociología, biología evolutiva y ciencias políticas.
# Ventajas: ilustra conceptos fundamentales de cooperación y competencia en sistemas interactivos.
# Desventajas: asume racionalidad perfecta y puede no capturar completamente el comportamiento humano real.
#En este codigo se hace el ejemplo de un juego de 2 prisioneros, el cual es de dos jugadores donde cada uno puede Cooperar (C) o Traicionar (T).

# Estrategias posibles para cada jugador
estrategias = ['C', 'T']  # 'C' = Cooperar (Confesar), 'T' = Traicionar (Mantener silencio)

# Matriz de pagos: (pago_jugador_A, pago_jugador_B)
# Los números representan años de prisión (valores negativos = castigo)
pagos = {
    ('C', 'C'): (-1, -1),  # Ambos confiesan: 1 año cada uno
    ('C', 'T'): (-3, 0),   # A confiesa, B calla: A 3 años, B libre
    ('T', 'C'): (0, -3),   # A calla, B confiesa: A libre, B 3 años  
    ('T', 'T'): (-2, -2)   # Ambos callan: 2 años cada uno
}

# --- Función para encontrar equilibrios de Nash ---
def encontrar_equilibrio_nash():
    """Encuentra todos los Equilibrios de Nash en el Dilema del Prisionero."""
    equilibrios = []  # Lista para almacenar los equilibrios encontrados
    
    # Itera sobre todas las combinaciones posibles de estrategias
    for sA in estrategias:  # Para cada estrategia del jugador A
        for sB in estrategias:  # Para cada estrategia del jugador B
            # Obtiene los pagos actuales para esta combinación de estrategias
            pagoA, pagoB = pagos[(sA, sB)]

            # --- Verifica si el jugador A está en su mejor respuesta ---
            # ¿Puede A mejorar cambiando unilateralmente su estrategia?
            mejorA = True  # Asume inicialmente que A no puede mejorar
            for sA_alt in estrategias:  # Prueba todas las estrategias alternativas de A
                if pagos[(sA_alt, sB)][0] > pagoA:  # Si alguna alternativa da mejor pago
                    mejorA = False  # A puede mejorar cambiando
                    break

            # --- Verifica si el jugador B está en su mejor respuesta ---
            # ¿Puede B mejorar cambiando unilateralmente su estrategia?
            mejorB = True  # Asume inicialmente que B no puede mejorar
            for sB_alt in estrategias:  # Prueba todas las estrategias alternativas de B
                if pagos[(sA, sB_alt)][1] > pagoB:  # Si alguna alternativa da mejor pago
                    mejorB = False  # B puede mejorar cambiando
                    break

            # Si ambos jugadores están en su mejor respuesta, es un Equilibrio de Nash
            if mejorA and mejorB:
                equilibrios.append((sA, sB))

    return equilibrios

# --- Ejecutar búsqueda de equilibrios ---
equilibrios = encontrar_equilibrio_nash()  # Encuentra todos los equilibrios de Nash

print("=== TEORÍA DE JUEGOS: DILEMA DEL PRISIONERO ===\n")
# Imprime cada equilibrio encontrado
for eq in equilibrios:
    print(f"Equilibrio de Nash encontrado: Jugador A -> {eq[0]}, Jugador B -> {eq[1]}")