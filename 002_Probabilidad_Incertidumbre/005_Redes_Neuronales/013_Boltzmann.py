# --- ALGORITMO DE  MÁQUINA DE BOLTZMANN (BOLTZMANN MACHINE) ---

# Este es un tipo de *Red Neuronal Recurrente* y *Estocástica* (aleatoria).
# Es conceptualmente importante como conexión entre redes neuronales y
# la física estadística, y es el precursor de las RBMs usadas en Deep Learning.
#
# Definición:
# Una Máquina de Boltzmann es una red de neuronas binarias (0 o 1)
# conectadas simétricamente (w_ij = w_ji), donde las neuronas
# actualizan sus estados de forma *probabilística* basándose en
# una "temperatura" (T) y una "función de energía" global.
#
# ¿Cómo aprende? 
# - El entrenamiento busca ajustar los pesos (w) y sesgos (theta)
#   para que la *distribución de probabilidad* de los estados de la red
#   (cuando está en "equilibrio térmico") coincida con la
#   *distribución de los datos* de entrenamiento.
# - Requiere muestreo extensivo (como Gibbs/MCMC) en dos fases
#   ("positiva" con datos, "negativa" sin datos) y es *muy* lento.
#   (Por eso se inventaron las RBMs - Restricted Boltzmann Machines).
#
# ¿Cómo funciona este programa?
# NO implementaremos el algoritmo de aprendizaje.
# Implementaremos la *dinámica* de la red:
# 1. Definiremos una red pequeña con pesos fijos.
# 2. Implementaremos la función de energía.
# 3. Implementaremos la regla de actualización estocástica.
# 4. Simularemos cómo el estado de la red "pasea" aleatoriamente.

import numpy as np # Para operaciones matriciales y exp()
import random # Para elegir neuronas y aceptar cambios

class BoltzmannMachine: # Clase para la Máquina de Boltzmann
    def __init__(self, num_neuronas): # Constructor
        self.num_neuronas = num_neuronas # Número de neuronas
        # Inicializar pesos simétricos (w_ij = w_ji) y diagonal cero
        self.pesos = np.random.randn(num_neuronas, num_neuronas) * 0.1
        self.pesos = (self.pesos + self.pesos.T) / 2 # Hacer simétrico
        np.fill_diagonal(self.pesos, 0) # Diagonal cero
        # Inicializar sesgos (theta) aleatorios
        self.sesgos = np.random.randn(num_neuronas) * 0.1
        # Inicializar estados binarios (0 o 1) aleatorios
        self.estados = np.random.randint(0, 2, size=num_neuronas)
        print("Máquina de Boltzmann inicializada.")
        #print(f" Pesos iniciales (W):\n{self.pesos}")
        #print(f" Sesgos iniciales (theta): {self.sesgos}")
        print(f" Estado inicial (s): {self.estados}")

    def calcular_energia(self, estado_s): # Calcula la Energía (E) de un estado
        """ Calcula la energía de una configuración de estados dada """
        # E = - Sum_{i<j} (w_ij * s_i * s_j) - Sum_i (theta_i * s_i)
        # Forma matricial eficiente: E = -0.5 * s @ W @ s.T - theta @ s.T
        energia_interaccion = -0.5 * np.dot(estado_s.T, np.dot(self.pesos, estado_s))
        energia_sesgo = -np.dot(self.sesgos, estado_s.T)
        return energia_interaccion + energia_sesgo # Devuelve E

    def run_step(self, temperatura=1.0): # Ejecuta un paso de actualización
        """ Elige una neurona y decide si cambia su estado probabilísticamente """
        
        # 1. Elegir una neurona 'k' al azar
        k = random.randint(0, self.num_neuronas - 1) # Índice de la neurona
        
        # 2. Calcular Delta E si 'k' cambiara de estado
        estado_actual_k = self.estados[k] # 0 o 1
        estado_nuevo_k = 1 - estado_actual_k # 1 o 0
        
        # Calcular el cambio en la energía (forma eficiente):
        # Delta_E_k = (1 - 2*s_k_actual) * (Sum_j(w_kj * s_j) + theta_k)
        suma_ponderada_k = np.dot(self.pesos[k, :], self.estados) + self.sesgos[k]
        delta_E = (estado_nuevo_k - estado_actual_k) * suma_ponderada_k # Cambio de energía
        
        # 3. Calcular la probabilidad de aceptar el cambio
        # P(cambiar) = 1 / (1 + exp(Delta_E / T))
        if temperatura <= 0: # Evitar división por cero
             prob_cambio = 1.0 if delta_E < 0 else 0.0 # Determinista si T=0
        else:
             # Usar clip para evitar overflow en exp()
             exp_arg = np.clip(delta_E / temperatura, -500, 500)
             prob_cambio = 1.0 / (1.0 + np.exp(exp_arg))
             
        # 4. Decidir si cambiar o no
        if random.random() < prob_cambio: # Si el número aleatorio es menor
            self.estados[k] = estado_nuevo_k # ¡Cambiar estado!
            #print(f"    Neurona {k} cambió a {estado_nuevo_k} (DeltaE={delta_E:.2f}, P={prob_cambio:.2f})")
            return True # Indicar que hubo cambio
        else:
            #print(f"    Neurona {k} NO cambió (DeltaE={delta_E:.2f}, P={prob_cambio:.2f})")
            return False # Indicar que no hubo cambio

# --- Ejecutar la Simulación ---
print("\n--- 8d. Máquina de Boltzmann (Boltmann ---") # Título

# Crear una máquina pequeña
bm = BoltzmannMachine(num_neuronas=5) # 5 neuronas

# Simular varios pasos
N_PASOS_BM = 50
TEMPERATURA = 1.0 # Temperatura constante (en Simulated Annealing, T bajaría)
print(f"\nSimulando {N_PASOS_BM} pasos con Temperatura T={TEMPERATURA}...")

num_cambios = 0
for i in range(N_PASOS_BM):
    cambio = bm.run_step(temperatura=TEMPERATURA) # Ejecutar un paso
    if cambio: num_cambios += 1
    energia_actual = bm.calcular_energia(bm.estados) # Calcular energía
    print(f"  Paso {i+1}: Estado = {bm.estados}, Energía = {energia_actual:.3f}") # Imprimir estado

print(f"\nSimulación completada. Hubo {num_cambios} cambios de estado.")
print(f"Estado final: {bm.estados}") # Estado final
print(f"Energía final: {energia_actual:.3f}") # Energía final

print("\nConclusión:")
print("La Máquina de Boltzmann actualizó sus estados de forma estocástica.")
print("La probabilidad de cambiar dependía de la diferencia de energía")
print("y la temperatura. La red tiende a explorar estados de baja energía.")
print("El *entrenamiento* de esta red (ajustar W y theta) es computacionalmente")
print("muy intensivo y requiere técnicas más avanzadas (como RBMs).")