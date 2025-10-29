# --- ALGORITMO DE EXTRACCIÓN DE INFORMACIÓN  ---

# Concepto: Extraer información estructurada (entidades, relaciones)
#           de texto no estructurado.
# Objetivo: Identificar Nombres (PER), Lugares (LOC) y Fechas (DATE)
#           en un texto usando reglas simples (regex).
#           (Simula el *resultado* de un modelo probabilístico).

import re # Biblioteca para Expresiones Regulares
from collections import defaultdict

# --- P1: Texto de Ejemplo ---
texto = """
El Dr. Juan Pérez visitó Guadalajara el 15 de Octubre de 2024.
Trabaja en Google Inc. desde Enero del 2020. María López también
viajó a Jalisco, pero llegó el 1 de Noviembre. Google planea
abrir oficinas en Tlaquepaque.
"""

print("---  Extracción de Información ---")
print(f"Texto de Entrada:\n{texto}")

# --- P2: Definir Patrones (Reglas Simples con Regex) ---

# Expresión Regular (muy simplificada) para nombres propios
# (Busca palabras capitalizadas consecutivas, posiblemente con 'Dr.', 'Sra.', etc.)
# \b -> Límite de palabra
# (?:Dr\.|Sr\.|Sra\.\s)? -> Título opcional (grupo no capturado)
# ([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*) -> Una o más palabras capitalizadas (grupo capturado)
regex_persona = r'\b(?:Dr\.|Sr\.|Sra\.\s)?([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)\b'
# (Nota: Esto es muy básico y capturará falsos positivos como inicios de oración)

# Expresión Regular para lugares comunes (simplificado)
# (Busca nombres de lugares conocidos o palabras capitalizadas cerca de preposiciones)
# (Esta es difícil de generalizar con regex simples)
regex_lugar = r'\b(Guadalajara|Jalisco|Tlaquepaque|México)\b' # Lista fija
# Podríamos añadir algo como: (?:en|a|de)\s+([A-Z][a-z]+) pero sería muy ruidoso

# Expresión Regular para fechas (simplificado)
# (Busca patrones como DD de Mes de AAAA o Mes del AAAA)
regex_fecha = r'\b(\d{1,2}\s+de\s+[A-Za-z]+\s+de\s+\d{4}|\b[A-Za-z]+\s+del\s+\d{4})\b'

# --- P3: Aplicar los Patrones para Extraer Entidades ---

def extraer_entidades(texto, patrones_etiquetas):
    """
    Busca patrones regex en el texto y devuelve un diccionario de entidades.
    patrones_etiquetas: {'PER': regex_per, 'LOC': regex_loc, ...}
    """
    entidades_encontradas = defaultdict(list) # {'PER': ['Juan Pérez', ...], 'LOC': [...]}

    for etiqueta, regex in patrones_etiquetas.items(): # Iterar sobre PER, LOC, DATE
        # re.findall() encuentra *todas* las ocurrencias que coinciden
        matches = re.findall(regex, texto)
        if matches:
            # Añadir las coincidencias (limpiando espacios extra si es necesario)
            entidades_encontradas[etiqueta].extend([match.strip() for match in matches if isinstance(match, str)])
            # Si el regex tiene grupos, findall puede devolver tuplas, manejamos eso
            for match in matches:
                 if isinstance(match, tuple):
                      entidades_encontradas[etiqueta].extend([m.strip() for m in match if m])


    # Post-procesamiento simple para eliminar duplicados manteniendo orden
    for etiqueta in entidades_encontradas:
         items_unicos = []
         [items_unicos.append(item) for item in entidades_encontradas[etiqueta] if item not in items_unicos]
         entidades_encontradas[etiqueta] = items_unicos


    return dict(entidades_encontradas) # Convertir de nuevo a dict normal

# Definir los patrones a buscar
patrones = {
    'PER': regex_persona,
    'LOC': regex_lugar,
    'DATE': regex_fecha
}

# Ejecutar la extracción
entidades = extraer_entidades(texto, patrones)

# --- P4: Imprimir Resultados ---
print("\n--- Entidades Extraídas (Resultado Simulado) ---")
if entidades:
    for etiqueta, lista_entidades in entidades.items(): # Iterar sobre PER, LOC, DATE
        print(f"  {etiqueta}:") # Imprimir etiqueta
        for entidad in lista_entidades: # Iterar sobre las encontradas
            print(f"    - {entidad}") # Imprimir entidad
else:
    print("  No se encontraron entidades con los patrones definidos.")

print("\nConclusión:")
print("Este ejemplo usó regex simples para simular la tarea de NER.")
print("Un sistema real usaría modelos probabilísticos (HMM, CRF, Redes Neuronales)")
print("entrenados con datos etiquetados para lograr mayor precisión y robustez,")
print("aprendiendo patrones contextuales que los regex no capturan fácilmente.")
print("(Por ejemplo, distinguir 'Apple' (ORG) de 'apple' (fruta)).")