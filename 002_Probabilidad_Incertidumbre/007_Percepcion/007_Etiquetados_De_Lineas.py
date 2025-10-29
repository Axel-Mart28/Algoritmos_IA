import cv2
import numpy as np
import matplotlib.pyplot as plt

print("--- 7. Etiquetado de Líneas (imagen generada automáticamente) ---")

# Crear imagen negra
imagen = np.zeros((300, 400), dtype=np.uint8)

# Dibujar figuras blancas
cv2.circle(imagen, (100, 150), 40, 255, -1)
cv2.rectangle(imagen, (200, 100), (300, 200), 255, -1)
cv2.putText(imagen, "AI", (320, 180), cv2.FONT_HERSHEY_SIMPLEX, 1, 255, 3)

# Etiquetar componentes conectados
_, binaria = cv2.threshold(imagen, 127, 255, cv2.THRESH_BINARY)
num_labels, labels = cv2.connectedComponents(binaria)

print(f"Se detectaron {num_labels - 1} objetos (sin contar el fondo).")

# Asignar colores
etiquetas_color = np.zeros((labels.shape[0], labels.shape[1], 3), dtype=np.uint8)
for i in range(1, num_labels):
    etiquetas_color[labels == i] = np.random.randint(0, 255, 3)

# Mostrar
plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.title("Imagen binaria")
plt.imshow(binaria, cmap='gray')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.title("Objetos etiquetados")
plt.imshow(etiquetas_color)
plt.axis('off')

plt.tight_layout()
plt.show()
