# Implementación del algoritmo de iteración de valores
# Este algoritmo se utiliza para resolver procesos de decisión de Markov (MDP) 
# Calcula la utilidad óptima de cada estado mediante iteraciones sucesivas.
# Entre sus principales aplicaciones se encuentran la robótica, sistemas de recomendación y planificación automática en inteligencia artificial.
# Entre sus ventajas se incluye la garantía de convergencia a los valores óptimos.
# Entre sus desventajas se encuentra la necesidad de conocer el modelo completo del entorno y puede ser computacionalmente costoso para espacios de estado grandes.
#Su objetivo principal es determinar cual accion debera tomarse en cada estado para maximizar la utilidad esperada.


# Definición de los componentes del MDP:
estados = ['A', 'B', 'C']  # Lista de estados posibles del sistema
recompensa = {'A': 0, 'B': 0, 'C': 1}  # Utilidades inmediatas por estado

# Transiciones P(s'|s,a) - probabilidades de transición entre estados
# En este caso hay una sola acción por estado (proceso de decisión simplificado)
transiciones = {  # Diccionario que representa las transiciones entre estados
    'A': {'B': 1.0},  # Desde A siempre se va a B
    'B': {'A': 0.5, 'C': 0.5},  # Desde B hay 50% de probabilidad de ir a A o C
    'C': {'C': 1.0}  # C es estado terminal (se queda en C con probabilidad 1)
}

# Parámetros del algoritmo
gamma = 0.9  # Factor de descuento para recompensas futuras
epsilon = 0.001  # Criterio de convergencia (margen de error aceptable)
U = {s: 0 for s in estados}  # Inicialización de utilidades en 0 para todos los estados

# --- Implementación del algoritmo de iteración de valores ---
def iteracion_valores():
    """Ejecuta el algoritmo de iteración de valores hasta alcanzar convergencia."""
    global U
    while True:
        delta = 0  # Inicializa la diferencia máxima entre iteraciones
        U_nuevo = U.copy()  # Crea una copia de las utilidades actuales
        
        # Actualiza la utilidad para cada estado
        for s in estados:
            # Calcula la nueva utilidad usando la ecuación de Bellman
            U_nuevo[s] = recompensa[s] + gamma * sum(
                transiciones[s][s2] * U[s2] for s2 in transiciones[s]
            )
            # Actualiza el valor delta (máximo cambio en esta iteración)
            delta = max(delta, abs(U_nuevo[s] - U[s]))
        
        U = U_nuevo  # Actualiza las utilidades con los nuevos valores
        
        # Verifica criterio de convergencia
        if delta < epsilon:
            break
    
    return U

# Ejecutar el algoritmo
U_final = iteracion_valores()  # Obtiene las utilidades finales convergidas

print("=== ITERACIÓN DE VALORES ===\n")
# Imprime los resultados formateados
for s in estados:
    print(f"U({s}) = {U_final[s]:.4f}")