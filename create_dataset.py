import os
import pickle
import mediapipe as mp
import cv2
import numpy as np

# Configuración de MediaPipe Hands
mp_hands = mp.solutions.hands
# Opciones para la detección de manos: modo de imagen estática y confianza mínima.
hands_options = dict(static_image_mode=True, min_detection_confidence=0.5, min_tracking_confidence=0.5)

DATA_DIR = './data' # Directorio de las imágenes del dataset.

data = [] # Lista para almacenar los datos de los puntos clave normalizados.
labels = [] # Lista para almacenar las etiquetas de las clases.

# --- Procesamiento de Imágenes y Extracción de Puntos Clave ---
with mp_hands.Hands(**hands_options) as hands: # Inicializa el detector de manos de MediaPipe.
    for dir_name in os.listdir(DATA_DIR): # Itera por cada subdirectorio (clase).
        dir_path = os.path.join(DATA_DIR, dir_name)
        if not os.path.isdir(dir_path):
            continue

        print(f"Processing class: {dir_name}")
        for img_name in os.listdir(dir_path): # Itera por cada imagen en la clase.
            img_file = os.path.join(dir_path, img_name)

            img = cv2.imread(img_file)
            if img is None:
                print(f"Could not read image {img_file}, skipping...")
                continue

            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # Convierte a RGB, formato esperado por MediaPipe.
            results = hands.process(img_rgb) # Detecta manos y extrae puntos clave.

            if results.multi_hand_landmarks: # Si se detectan manos:
                for hand_landmarks in results.multi_hand_landmarks:
                    # Normaliza las coordenadas de los puntos clave para hacerlas independientes del tamaño/posición.
                    x_coords = [lm.x for lm in hand_landmarks.landmark]
                    y_coords = [lm.y for lm in hand_landmarks.landmark]

                    x_min, y_min = min(x_coords), min(y_coords)
                    x_max, y_max = max(x_coords), max(y_coords)

                    range_x = x_max - x_min if (x_max - x_min) > 0 else 1.0
                    range_y = y_max - y_min if (y_max - y_min) > 0 else 1.0

                    data_aux = []
                    for lm in hand_landmarks.landmark:
                        data_aux.append((lm.x - x_min) / range_x)
                        data_aux.append((lm.y - y_min) / range_y)

                    data.append(data_aux) # Almacena los puntos clave normalizados.
                    labels.append(dir_name) # Almacena la etiqueta de la clase.
            else:
                # Si no se detectan manos en una imagen 'BLANK', se añaden ceros como datos.
                if dir_name == 'BLANK':
                    data_aux_blank = [0.0] * 42 # 21 puntos * 2 coordenadas (x,y) = 42 características.
                    data.append(data_aux_blank)
                    labels.append(dir_name)
                else:
                    print(f"No hands detected in image {img_file} for class {dir_name}. This sample might be noisy.")

# --- Relleno de Datos (Padding) ---
max_len = 0
for d_item in data: # Encuentra la longitud máxima de los datos de puntos clave.
    if len(d_item) > max_len:
        max_len = len(d_item)

for i in range(len(data)): # Rellena todos los arrays de datos con ceros para tener la misma longitud.
    while len(data[i]) < max_len:
        data[i].append(0.0)

# --- Guardar Datos Procesados ---
output_file = 'data.pickle'
try:
    # Guarda los datos procesados y las etiquetas en un archivo pickle para su uso posterior en el entrenamiento.
    with open(output_file, 'wb') as f:
        pickle.dump({'data': data, 'labels': labels}, f)
    print(f"File '{output_file}' created successfully. Total samples: {len(data)}")
except Exception as e:
    print(f"Error saving pickle file: {e}")