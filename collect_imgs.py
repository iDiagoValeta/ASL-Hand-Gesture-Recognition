import os
import cv2

# --- Configuración del Directorio de Datos ---
DATA_DIR = './data'
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR) # Crea el directorio 'data' si no existe.

# --- Definición de Clases y Tamaño del Dataset ---
labels_to_collect = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', 'BLANK']
number_of_classes = len(labels_to_collect)
dataset_size_per_class = 500 # Número de imágenes a recolectar por cada clase.

# --- Inicialización de la Cámara ---
cap = cv2.VideoCapture(0) # Abre la cámara web (0 es la cámara por defecto).
if not cap.isOpened():
    print("Error: No se pudo abrir la cámara.")
    exit()

# --- Bucle de Recolección de Datos por Clase ---
for i, class_name in enumerate(labels_to_collect):
    class_dir = os.path.join(DATA_DIR, class_name)
    if not os.path.exists(class_dir):
        os.makedirs(class_dir) # Crea un subdirectorio para cada clase (ej., 'data/A').

    print(f'Collecting data for class: {class_name} ({i + 1}/{number_of_classes})')

    # Espera que el usuario presione 'q' para iniciar la captura de la clase actual.
    while True:
        ret, frame = cap.read()
        cv2.putText(frame, f'Ready for {class_name}? Press "Q" ! :)', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.imshow('frame', frame)
        if cv2.waitKey(25) == ord('q'):
            break

    # Captura y guarda las imágenes para la clase actual.
    counter = 0
    while counter < dataset_size_per_class:
        ret, frame = cap.read()
        cv2.imshow('frame', frame)
        cv2.waitKey(25)
        cv2.imwrite(os.path.join(class_dir, f'{counter}.jpg'), frame) # Guarda cada frame como una imagen.
        counter += 1

# --- Liberación de Recursos ---
cap.release() # Libera la cámara.
cv2.destroyAllWindows() # Cierra todas las ventanas de OpenCV.