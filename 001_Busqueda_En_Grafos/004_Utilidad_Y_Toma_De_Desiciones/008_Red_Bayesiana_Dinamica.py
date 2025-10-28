# Algoritmo de Red Bayesiana Dinámica simple (DBN) 
# Este algoritmo implementa una Red Bayesiana Dinámica (DBN) para modelar sistemas que evolucionan en el tiempo.
# Las DBNs extienden las redes bayesianas para representar procesos temporales con estados ocultos y observaciones.
# Entre sus aplicaciones están: filtrado de señales, seguimiento de objetos, diagnóstico de fallos en tiempo real, y sistemas de monitorización donde el estado real no es directamente observable.
# Ventajas: modela eficientemente sistemas dinámicos con incertidumbre, permite razonamiento temporal.
# Desventajas: complejidad computacional que crece con el tiempo, requiere conocimiento del modelo.
#En el algoritmo se usa el ejemplo de un modelo de clima con uso de paraguas o no.


# Estados ocultos - variables no observables que representan el estado real del sistema
estados = ['Lluvia', 'Sol']  # Estados posibles del clima (no directamente observables)

# Observaciones - evidencias que podemos percibir directamente
observaciones = ['Paraguas', 'Sin paraguas']  # Lo que podemos ver (si la gente lleva paraguas o no)

# Probabilidades iniciales P(X0) - distribución de probabilidad en el tiempo inicial
P_inicial = {'Lluvia': 0.5, 'Sol': 0.5}  # Creencia inicial sobre el estado (50% para cada uno)

# Probabilidades de transición P(X_t | X_{t-1}) - cómo evoluciona el estado en el tiempo
P_trans = {
    ('Lluvia','Lluvia'): 0.7,  # Probabilidad de que siga lloviendo si está lloviendo
    ('Lluvia','Sol'): 0.3,     # Probabilidad de que salga el sol si está lloviendo
    ('Sol','Lluvia'): 0.2,     # Probabilidad de que llueva si hace sol
    ('Sol','Sol'): 0.8         # Probabilidad de que siga haciendo sol si hace sol
}

# Probabilidades de observación P(O_t | X_t) - relación entre estado real y observaciones
P_obs = {
    ('Lluvia','Paraguas'): 0.9,      # Alta probabilidad de ver paraguas cuando llueve
    ('Lluvia','Sin paraguas'): 0.1,  # Baja probabilidad de no ver paraguas cuando llueve
    ('Sol','Paraguas'): 0.2,         # Baja probabilidad de ver paraguas cuando hace sol
    ('Sol','Sin paraguas'): 0.8      # Alta probabilidad de no ver paraguas cuando hace sol
}

# Secuencia de observaciones - evidencias recogidas a lo largo del tiempo
evidencia = ['Paraguas', 'Paraguas', 'Sin paraguas']  # Observaciones en tiempos t=1,2,3

# --- Filtrado: calcular belief state en cada tiempo ---
def filtrado(evidencia, P_inicial, P_trans, P_obs):
    """
    Implementa el algoritmo de filtrado para DBNs (algoritmo de avance).
    Calcula la distribución de probabilidad del estado actual dada toda la evidencia hasta el momento.
    """
    belief = P_inicial.copy()  # Inicializa el belief con las probabilidades iniciales
    historial = [belief.copy()]  # Almacena el historial de beliefs para análisis
    
    # Procesa cada observación en la secuencia temporal
    for o in evidencia:
        # --- Paso de PREDICCIÓN ---
        # Calcula P(X_t | e_{1:t-1}) = sum_{x_{t-1}} P(X_t | x_{t-1}) * P(x_{t-1} | e_{1:t-1})
        pred = {}
        for s2 in estados:  # Para cada estado posible en tiempo t
            pred[s2] = sum(P_trans[(s1,s2)] * belief[s1] for s1 in estados)
        
        # --- Paso de ACTUALIZACIÓN ---
        # Calcula P(X_t | e_{1:t}) ∝ P(e_t | X_t) * P(X_t | e_{1:t-1})
        belief_new = {}
        for s in estados:  # Para cada estado posible
            belief_new[s] = P_obs[(s,o)] * pred[s]  # P(observación|estado) * P(estado|evidencia_anterior)
        
        # Normalizar para obtener una distribución de probabilidad válida
        total = sum(belief_new.values())
        for s in belief_new:
            belief_new[s] /= total  # Normaliza dividiendo por la suma total
        
        belief = belief_new  # Actualiza el belief para el siguiente paso
        historial.append(belief.copy())  # Guarda el belief actual en el historial
    
    return historial  # Retorna la evolución completa del belief

# --- Ejecutar filtrado ---
belief_history = filtrado(evidencia, P_inicial, P_trans, P_obs)  # Ejecuta el algoritmo de filtrado

print("=== RED BAYESIANA DINÁMICA (DBN) ===\n")
# Imprime la evolución del belief a lo largo del tiempo
for t, b in enumerate(belief_history):
    print(f"Tiempo {t}: {b}")  # Muestra el belief en cada paso temporal