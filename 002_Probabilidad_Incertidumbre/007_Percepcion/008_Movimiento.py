# ALGORITMO DE DETECTOR DE MOVIMIENTO CON OPENCV Y GRADIO 
# Concepto: Detectar movimiento en un video analizando los cambios entre cuadros consecutivos.
# Técnica: Sustracción de fondo (Background Subtraction) usando MOG2.
# Librerías: OpenCV (procesamiento de video), Gradio (interfaz gráfica web).

import cv2
import gradio as gr

def vid_inf(vid_path):
    """
    Procesa un video cuadro por cuadro y detecta movimiento.
    Resalta las regiones donde se detecta movimiento y genera un video de salida.
    """

    # Abrir el video desde la ruta proporcionada
    cap = cv2.VideoCapture(vid_path)

    # Verificar si el archivo se abrió correctamente
    if not cap.isOpened():
        print("❌ Error al abrir el video. Verifica el nombre o la ruta del archivo.")
        return

    # Obtener propiedades del video: tamaño y FPS (frames por segundo)
    frame_size = (
        int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    )
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    # Nombre del archivo de salida
    output_video = "output.mp4"

    # Crear un objeto para guardar el video procesado
    # 'fourcc' define el códec de video; 'mp4v' sirve para .mp4
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video, fourcc, fps, frame_size)

    # Crear un objeto de sustracción de fondo MOG2
    # Este algoritmo aprende el fondo y detecta los píxeles que cambian (movimiento)
    back_sub = cv2.createBackgroundSubtractorMOG2()

    frame_count = 0  # Contador de cuadros procesados

    # Bucle principal: procesar cada frame del video
    while cap.isOpened():
        ret, frame = cap.read()  # Leer cuadro
        if not ret:  # Si no hay más cuadros, salir del bucle
            break

        # --- 1. Aplicar sustracción de fondo ---
        # Genera una máscara donde los píxeles en movimiento aparecen en blanco
        fg_mask = back_sub.apply(frame)

        # --- 2. Umbral para binarizar la máscara (solo conservar movimiento fuerte) ---
        _, mask_thresh = cv2.threshold(fg_mask, 150, 255, cv2.THRESH_BINARY)

        # --- 3. Operación morfológica para eliminar ruido (pequeños puntos blancos) ---
        mask_open = cv2.morphologyEx(
            mask_thresh, cv2.MORPH_OPEN,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
        )

        # --- 4. Encontrar contornos de las áreas en movimiento ---
        contours, _ = cv2.findContours(mask_open, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # --- 5. Filtrar los contornos demasiado pequeños ---
        min_contour_area = 4000  # área mínima para considerar un "movimiento"
        large_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > min_contour_area]

        # --- 6. Dibujar rectángulos alrededor de los objetos en movimiento ---
        frame_out = frame.copy()
        for cnt in large_contours:
            x, y, w, h = cv2.boundingRect(cnt)
            cv2.rectangle(frame_out, (x, y), (x + w, y + h), (0, 255, 0), 3)  # Rectángulo verde

        # --- 7. Guardar el cuadro procesado en el video de salida ---
        out.write(frame_out)

        # --- 8. Convertir el cuadro a RGB (para que Gradio lo muestre correctamente) ---
        frame_out_RGB = cv2.cvtColor(frame_out, cv2.COLOR_BGR2RGB)

        # --- 9. Mostrar un frame cada cierto tiempo para no saturar la interfaz ---
        if frame_count % 24 == 0:  # Mostrar 1 de cada 24 cuadros (~1 por segundo)
            yield frame_out_RGB

        frame_count += 1  # Avanzar contador

    # --- 10. Liberar recursos ---
    cap.release()
    out.release()

# --- INTERFAZ DE GRADIO ---
# Permite subir un video y ver el resultado directamente desde el navegador.

app = gr.Interface(
    fn=vid_inf,  # Función principal que procesa el video
    inputs=gr.Video(label="🎥 Video de entrada"),
    outputs=gr.Image(label="🟢 Detección de movimiento"),
    examples=[["thief.mp4"]]  # Ejemplo de video precargado (opcional)
)

# Ejecutar la app en modo cola (permite procesar por frames)
app.queue().launch()
