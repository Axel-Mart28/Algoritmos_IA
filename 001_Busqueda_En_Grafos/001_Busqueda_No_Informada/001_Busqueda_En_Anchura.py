from collections import deque #Importamos deque para usarlo como cola

#Este metodo comienza desde lo amplio, es decir, le da prioridad a lo amplio antes que a lo profundo.
#Ejemplo de la lista de un grafo:

grafo = {
    'A': ['B', 'C', 'D'], #A tiene tres vecinos: B, C y D
    'B': [], #B no tiene vecinos
    'C': ['E', 'F'], #C tiene dos vecinos: E y F
    'D': [], #D no tiene vecinos
    'E': [], #E no tiene vecinos
    'F': ['G'], #F tiene un vecino: G
    'G': [] #G no tiene vecinos
}

def busqueda_en_anchura(grafo, nodo_inicial): #Funcion de busqueda en anchura
    queue = deque([nodo_inicial])  # Cola para los nodos por visitar

    visitados = []  # Conjunto para los nodos ya visitados

    while queue: #Mientras que la cola no este vacia
        nodo_actual = queue.popleft()  # Sacar el primer nodo de la cola

        if nodo_actual not in visitados:  # Si el nodo no ha sido visitado
            visitados.append(nodo_actual)  # Marcar como visitado y lo agrega

            vecinos = grafo[nodo_actual]  # Obtener los vecinos del nodo actual
            for vecino in vecinos:  # Para cada vecino
                if vecino not in visitados:  # Si no ha sido visitado
                    queue.append(vecino)  # Agregar a la cola

    return visitados

#Comenzando la busqueda en anchura desde el nodo 'A':
result = busqueda_en_anchura(grafo, 'A')
print("Recorrido en anchura:", result)