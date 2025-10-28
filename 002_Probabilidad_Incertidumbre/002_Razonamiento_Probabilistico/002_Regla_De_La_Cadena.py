# Algoritmo de  REGLA DE LA CADENA

# Este es el principio matemático que justifica por qué las Redes Bayesianas son tan poderosas y compactas.

# Definición:
# La Regla de la Cadena de la probabilidad nos dice cómo calcular la probabilidad conjunta* (la probabilidad de que un conjunto completo de variables ocurran) multiplicando probabilidades condicionales.

# Fórmula General: P(A, B, C, D) = P(A) * P(B|A) * P(C|A, B) * P(D|A, B, C)
# Necesitaríamos una CPT gigante para P(D|A,B,C).
#
# ¿Cómo funciona (LA CLAVE de las Redes Bayesianas)?:
# Una Red Bayesiana es una estructura de datos que hace una suposición de  Independencia Condicional.
# Suposición: Un nodo es condicionalmente independiente de todos sus no-descendientes, dado el valor de sus padres.
#
# Esto simplifica drásticamente la Regla de la Cadena a:
#
# Fórmula de la Red Bayesiana: P(X1, X2, ..., Xn) = Producto de [ P(Xi | Padres(Xi)) ]
#
# ¿Cómo funciona este programa?
# Vamos a implementar esta formula. Escribiremos un algoritmo que calcula la probabilidad conjunta de un "evento completo" (un valor para *cada* variable en la red).

# Lo hará iterando sobre cada nodo y multiplicando la probabilidad condicional de su valor, dados los valores de sus padres (los cuales también están en el "evento completo").

# Componentes:
# 1. La Red Bayesiana que se uso en el algoritmo anterior.
# 2. Un "evento completo" (un robo, etc)
#
# Aplicaciones:
# - Este algoritmo es el bloque de construcción fundamental para la "Inferencia por Enumeración".
# - Permite calcular la probabilidad de *cualquier* estado específico del mundo.
#
# Ventajas:
# - Muestra la "compacidad" de la Red Bayesiana.
# - Es exponencialmente más rápido que la Regla de la Cadena general.
#
# Desventajas:
# - El resultado solo es correcto si las suposiciones de independencia (la estructura del grafo) son correctas.
#
# Ejemplo de uso (Usando nuestra red):
# Calcular P(R=F, T=T, A=T, J=T, M=F)
# Nuestra fórmula será:
# P(R=F) * P(T=T) * P(A=T|R=F, T=T) * P(J=T|A=T) * P(M=F|A=T)
#
# 

# P1: Definición de la Red y Funciones Auxiliares
red_alarma = {
    'Robo': {
        'parents': [],
        'cpt': {(): 0.001}
    },
    'Terremoto': {
        'parents': [],
        'cpt': {(): 0.002}
    },
    'Alarma': {
        'parents': ['Robo', 'Terremoto'],
        'cpt': {
            (True, True): 0.95, (True, False): 0.94,
            (False, True): 0.29, (False, False): 0.001
        }
    },
    'JuanLlama': {
        'parents': ['Alarma'],
        'cpt': {(True,): 0.90, (False,): 0.05}
    },
    'MariaLlama': {
        'parents': ['Alarma'],
        'cpt': {(True,): 0.70, (False,): 0.01}
    }
}

def get_prob_cpt(red, variable, valor, evidencia={}):
    """
    Obtiene la probabilidad P(variable=valor | evidencia)
    directamente de la CPT de la red.
    """
    nodo = red[variable] # Obtener el nodo
    padres = nodo['parents'] # Obtener la lista de padres
    
    # Construir la clave de la CPT
    if not padres:
        clave_cpt = () # Clave vacía para nodos raíz
    else:
        # Construir la tupla de valores de los padres
        clave_cpt = tuple([evidencia[padre] for padre in padres])
        
    # Obtener la probabilidad de la tabla
    prob_true = nodo['cpt'][clave_cpt]
    
    # Devolver la prob correcta (True o False)
    return prob_true if valor == True else (1.0 - prob_true)

# --- P2: Algoritmo de la Regla de la Cadena ---

def calcular_probabilidad_conjunta(red, evento_completo):
    """
    Calcula la probabilidad conjunta de un evento completo
    usando la Regla de la Cadena simplificada por la Red Bayesiana.
    
    P(X1...Xn) = Producto de [ P(Xi | Padres(Xi)) ]
    
    'evento_completo' es un dict: {var: valor, var: valor, ...}
    """
    
    # 1. Inicializar la probabilidad total en 1.0
    #    (Usamos 1.0 porque es la identidad para la multiplicación)
    prob_conjunta_total = 1.0
    
    # 2. Iterar sobre *cada* variable en la red
    #    (El orden no importa gracias a la estructura de la red)
    for variable in red.keys(): # Ej: 'Robo', 'Terremoto', 'Alarma', ...
        
        # 3. Obtener el valor de esta variable del evento
        valor = evento_completo[variable] # Ej: evento_completo['Alarma'] -> True
        
        # 4. Obtener la probabilidad condicional de este nodo
        #    P(Xi | Padres(Xi))
        #    Usamos la función 'get_prob_cpt'
        #    Pasamos el 'evento_completo' como la 'evidencia'
        #    porque contiene los valores de los padres que 'get_prob_cpt' necesita.
        prob_condicional = get_prob_cpt(red, variable, valor, evento_completo)
        
        # 5. Multiplicar esta probabilidad a nuestro total
        prob_conjunta_total *= prob_condicional
        
    # 6. Devolver el producto final
    return prob_conjunta_total

# --- P3: Ejecutar el cálculo ---
print("--- 2. Regla de la Cadena (Cálculo de Probabilidad Conjunta) ---") # Título

# Vamos a calcular la probabilidad de un escenario específico:
# "No hubo Robo, SÍ hubo un Terremoto, la Alarma se activó,
#  Juan llamó, pero María NO llamó."
evento_consulta = {
    'Robo': False,
    'Terremoto': True,
    'Alarma': True,
    'JuanLlama': True,
    'MariaLlama': False
}

print(f"\nEvento a calcular: P(R=F, T=T, A=T, J=T, M=F)") # Muestra la consulta

# --- Desglose manual (lo que el algoritmo hará): ---
# P(Robo=False) = 1.0 - 0.001 = 0.999
# P(Terremoto=True) = 0.002
# P(Alarma=True | Robo=F, Terremoto=T) = 0.29
# P(JuanLlama=True | Alarma=T) = 0.90
# P(MariaLlama=False | Alarma=T) = 1.0 - 0.70 = 0.30
#
# Resultado esperado = 0.999 * 0.002 * 0.29 * 0.90 * 0.30
# Resultado esperado = 0.0001564446

# 1. Llamar a la función del algoritmo
prob_final = calcular_probabilidad_conjunta(red_alarma, evento_consulta)

print(f"\n--- Desglose del Cálculo ---")
print(f"  P(R=F)         = {get_prob_cpt(red_alarma, 'Robo', False, evento_consulta):.3f}")
print(f"x P(T=T)         = {get_prob_cpt(red_alarma, 'Terremoto', True, evento_consulta):.3f}")
print(f"x P(A=T | R=F,T=T) = {get_prob_cpt(red_alarma, 'Alarma', True, evento_consulta):.2f}")
print(f"x P(J=T | A=T)   = {get_prob_cpt(red_alarma, 'JuanLlama', True, evento_consulta):.2f}")
print(f"x P(M=F | A=T)   = {get_prob_cpt(red_alarma, 'MariaLlama', False, evento_consulta):.2f}")
print("--------------------------")
print(f"Probabilidad Conjunta Total: {prob_final}")
print(f"(Valor científico: {prob_final:e})")