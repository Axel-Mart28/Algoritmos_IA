# --- ALGORITMO DE  RED DE HOPFIELD (HOPFIELD NETWORK) ---

# Este es un tipo de *Red Neuronal Recurrente* (las conexiones forman ciclos)
# utilizada como *Memoria Asociativa* (o Content-Addressable Memory).
#
# Definición:
# Una Red de Hopfield almacena un conjunto de patrones (generalmente binarios bipolares).
# Cuando se le presenta un patrón de entrada (posiblemente ruidoso o incompleto),
# la red *evoluciona* (sus neuronas cambian de estado iterativamente) hasta
# *converger* a uno de los patrones almacenados al que más se parece la entrada.
# "Recuerda" el patrón completo a partir de una pista.
#
# ¿Cómo funciona?
# 1. ALMACENAMIENTO (Regla de Hebb Modificada):
#    - Los patrones a memorizar (P1, P2...) deben ser vectores bipolares (+1, -1).
#    - Se crea una *matriz de pesos* (W) donde las conexiones son *simétricas* (w_ij = w_ji)
#      y *sin auto-conexiones* (w_ii = 0).
#    - La regla de almacenamiento es (simplificada):
#      W = Suma sobre todos los patrones 'p' [ (p^T * p) - Identidad ]
#      (Esencialmente, suma los productos externos de cada patrón consigo mismo).
#
# 2. RECUPERACIÓN (Iteración Asíncrona o Síncrona):
#    - Se inicializa el estado de la red con el patrón de entrada (ruidoso).
#    - Se repite hasta la convergencia:
#      - Se elige una neurona 'i' (al azar si es asíncrona, todas si es síncrona).
#      - Se calcula la suma ponderada de sus entradas: h_i = Sum_j (w_ij * s_j)
#        (donde s_j es el estado *actual* de la neurona 'j').
#      - Se actualiza el estado de la neurona 'i' con una función escalón (signo):
#        s_i (nuevo) = +1 si h_i >= 0
#        s_i (nuevo) = -1 si h_i < 0
#    - La red converge cuando los estados de las neuronas dejan de cambiar. El estado
#      final es (idealmente) uno de los patrones almacenados.
#
# 
#
# Componentes:
# 1. Neuronas binarias bipolares (+1, -1).
# 2. Matriz de Pesos (W) simétrica y con diagonal cero.
# 3. Regla de Actualización (basada en suma ponderada y función signo).
#
# Aplicaciones:
# - Memoria asociativa (recuperar recuerdos de pistas).
# - Optimización (puede encontrar soluciones a problemas como el Viajante de Comercio,
#   definiendo una "función de energía" que la red minimiza).
# - Reconocimiento de patrones.
#
# Ventajas:
# - Modelo simple de memoria asociativa.
# - Garantiza converger a un estado estable (un "mínimo local" de energía).
#
# Desventajas:
# - Capacidad de almacenamiento *limitada* (aprox. 0.14 * N patrones, donde N es #neuronas).
#   Si almacenas demasiados patrones, ocurren "estados espurios" (recuerdos falsos).
# - Puede converger al patrón "equivocado" si la entrada está a medio camino.
# - Sensible a patrones muy similares entre sí.

import numpy as np # Para operaciones matriciales

class HopfieldNetwork: # Clase para la Red de Hopfield
    def __init__(self, num_neuronas): # Constructor
        self.num_neuronas = num_neuronas # Número de neuronas (longitud del patrón)
        # Inicializar la matriz de pesos con ceros
        self.pesos = np.zeros((num_neuronas, num_neuronas))

    def store_patterns(self, patrones): # Método para "memorizar" patrones
        """ Almacena una lista de patrones usando la regla de Hebb modificada """
        print(f"Almacenando {len(patrones)} patrones...") # Mensaje
        P = np.array(patrones) # Convertir lista de patrones a matriz NumPy [num_patrones, num_neuronas]
        
        # Calcular W = (1/N) * Suma(p^T * p) - (M/N)*I  (Forma más estable)
        # O la forma simple W = Suma(p^T * p)
        
        # Forma simple: W = Suma(p^T * p)
        for p in P: # Iterar sobre cada patrón
            # p[:, np.newaxis] convierte el vector fila [1, -1] en columna [[1], [-1]]
            # El producto externo p^T * p crea una matriz N x N
            self.pesos += np.outer(p, p)
            
        # Asegurar diagonal cero (sin auto-conexiones)
        np.fill_diagonal(self.pesos, 0)
        
        # (Opcional) Normalizar pesos (a veces ayuda)
        # self.pesos /= len(patrones)
        print("Pesos calculados.") # Mensaje

    def retrieve(self, patron_entrada, max_iter=20): # Método para "recordar"
        """ Intenta recuperar un patrón almacenado a partir de una entrada """
        print(f"\nIntentando recuperar desde: {patron_entrada}") # Mensaje
        estado_actual = np.copy(patron_entrada) # Copiar la entrada al estado inicial
        
        for i in range(max_iter): # Bucle de iteraciones (actualización síncrona)
            print(f"  Iteración {i+1}: Estado = {estado_actual}") # Imprimir estado
            
            # Calcular la "energía" (opcional, para ver convergencia)
            # energia = -0.5 * np.dot(estado_actual.T, np.dot(self.pesos, estado_actual))
            # print(f"    Energía: {energia:.2f}")

            # Calcular las activaciones (sumas ponderadas) para TODAS las neuronas
            # h = W * s
            activaciones = np.dot(self.pesos, estado_actual)
            
            # Aplicar la función signo para obtener el nuevo estado
            nuevo_estado = np.sign(activaciones)
            # np.sign devuelve 0 si la activación es 0, lo cambiamos a +1 o -1 (convención)
            nuevo_estado[nuevo_estado == 0] = 1 
            
            # Comprobar convergencia
            if np.array_equal(nuevo_estado, estado_actual): # Si no hubo cambios
                print(f"  Convergencia alcanzada en la iteración {i+1}.") # Mensaje
                return nuevo_estado # Devolver el estado estable
                
            estado_actual = nuevo_estado # Actualizar el estado para la siguiente iteración
            
        print("  Máximo de iteraciones alcanzado.") # Mensaje si no converge
        return estado_actual # Devolver el último estado

# --- Ejecutar la Red de Hopfield ---
print("\n--- HOPFIELD---") # Título

# 1. Definir los patrones a almacenar
patron1 = np.array([+1, +1, -1, -1])
patron2 = np.array([-1, -1, +1, +1])
patrones_memoria = [patron1, patron2]

# 2. Crear y entrenar (almacenar) la red
hopfield_net = HopfieldNetwork(num_neuronas=4) # Crear red de 4 neuronas
hopfield_net.store_patterns(patrones_memoria) # Almacenar patrones

# 3. Crear una entrada ruidosa (Patrón 1 con un error)
entrada_ruidosa = np.array([+1, -1, -1, -1]) # El segundo bit está mal

# 4. Intentar recuperar el patrón original
patron_recuperado = hopfield_net.retrieve(entrada_ruidosa) # Ejecutar recuperación

print("\n--- Resultado ---") # Título resultado
print(f"Entrada Ruidosa:   {entrada_ruidosa}") # Imprimir entrada
print(f"Patrón Recuperado: {patron_recuperado}") # Imprimir salida
print(f"Patrón Original 1: {patron1}") # Imprimir original para comparar

# Comprobar si la recuperación fue exitosa
if np.array_equal(patron_recuperado, patron1): # Comparar
    print("¡Recuperación exitosa! La red recordó el Patrón 1.") # Éxito
else:
    print("La recuperación falló o convergió a otro estado.") # Fallo

print("\nConclusión:")
print("La Red de Hopfield utilizó sus conexiones recurrentes y simétricas")
print("para evolucionar desde un estado inicial ruidoso hasta uno de los")
print("patrones estables (memorizados), actuando como una memoria asociativa.")