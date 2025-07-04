import pickle
import cv2
import mediapipe as mp
import numpy as np
from sklearn.preprocessing import LabelEncoder
import time
import tensorflow as tf # Importar tensorflow
from tensorflow import keras # Importar keras

# --- Cargar Modelo Pre-entrenado y Codificador de Etiquetas ---
try:
    # Cargar el modelo de Keras
    model = keras.models.load_model('./deep_model.h5')
    print("Modelo de Keras 'deep_model.h5' cargado correctamente.")

    # Cargar el LabelEncoder
    with open('./label_encoder.pkl', 'rb') as f:
        label_encoder = pickle.load(f)
    print("LabelEncoder 'label_encoder.pkl' cargado correctamente.")

    # Crear el diccionario de etiquetas
    labels_dict = {i: label for i, label in enumerate(label_encoder.classes_)}
    print(f"Loaded labels_dict: {labels_dict}")
except FileNotFoundError:
    print("Error: Los archivos del modelo (deep_model.h5 o label_encoder.pkl) no se encontraron. Por favor, entrena el clasificador primero usando train_deep_classifier.py.")
    exit()
except Exception as e:
    print(f"Error al cargar el modelo o LabelEncoder: {e}")
    exit()

# --- Inicializar Cámara y MediaPipe ---
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: No se pudo abrir la cámara.")
    exit()

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(static_image_mode=False, min_detection_confidence=0.5, min_tracking_confidence=0.5)

# --- Variables para la Construcción de Palabras y Temporizadores ---
current_word = ""
last_prediction = ""
prediction_start_time = 0

blank_start_time = 0
blank_threshold_clear = 5

# --- Bucle Principal de Detección y Predicción en Tiempo Real ---
while True:
    data_aux = []
    x_coords = []
    y_coords = []

    ret, frame = cap.read()
    if not ret:
        print("Error: No se pudo leer el frame de la cámara.")
        break

    H, W, _ = frame.shape
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)

    predicted_character = "BLANK"

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style())

            x_coords = [lm.x for lm in hand_landmarks.landmark]
            y_coords = [lm.y for lm in hand_landmarks.landmark]

            x_min, y_min = min(x_coords), min(y_coords)
            x_max, y_max = max(x_coords), max(y_coords)

            range_x = x_max - x_min if (x_max - x_min) > 0 else 1.0
            range_y = y_max - y_min if (y_max - y_min) > 0 else 1.0

            for lm in hand_landmarks.landmark:
                data_aux.append((lm.x - x_min) / range_x)
                data_aux.append((lm.y - y_min) / range_y)

        expected_feature_length = 42
        if len(data_aux) < expected_feature_length:
            data_aux.extend([0.0] * (expected_feature_length - len(data_aux)))
        elif len(data_aux) > expected_feature_length:
            data_aux = data_aux[:expected_feature_length]

        # Realiza la predicción usando el modelo de Keras.
        # Asegúrate de que los datos de entrada tengan la forma correcta (batch_size, num_features)
        prediction_probs = model.predict(np.asarray([data_aux]), verbose=0)[0]
        predicted_numeric_label = np.argmax(prediction_probs) # Obtiene el índice de la clase con mayor probabilidad
        predicted_character = labels_dict[predicted_numeric_label] # Mapea a la etiqueta de texto

        x1 = int(min(x_coords) * W) - 10
        y1 = int(min(y_coords) * H) - 10
        x2 = int(max(x_coords) * W) - 10
        y2 = int(max(y_coords) * H) - 10

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), 4)
        cv2.putText(frame, predicted_character, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 0), 3, cv2.LINE_AA)

    # --- Lógica de Construcción de Palabra con tiempo mínimo (para añadir letras) ---
    current_time = time.time()

    if predicted_character != last_prediction:
        prediction_start_time = current_time
        last_prediction = predicted_character
        if predicted_character != "BLANK":
            blank_start_time = 0
    elif predicted_character != "BLANK" and (current_time - prediction_start_time >= 3):
        current_word += predicted_character
        last_prediction = "BLANK"
        prediction_start_time = current_time
        blank_start_time = 0

    # --- Lógica para Borrar la Palabra con "BLANK" Sostenido ---
    if predicted_character == "BLANK":
        if blank_start_time == 0:
            blank_start_time = current_time
        elif current_time - blank_start_time >= blank_threshold_clear:
            current_word = ""
            blank_start_time = 0
            print("Palabra borrada por BLANK sostenido.")
    else:
        blank_start_time = 0

    # --- Mostrar Información en Pantalla ---
    cv2.putText(frame, f"Sign: {predicted_character}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(frame, f"Word: {current_word}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

    cv2.imshow('frame', frame)
    key = cv2.waitKey(1) & 0xFF

    # --- Comandos del Teclado ---
    if key == ord('q'):
        break
    elif key == ord('c'):
        current_word = current_word[:-1]
        blank_start_time = 0
    elif key == ord(' '):
        current_word += ' '
        blank_start_time = 0

# --- Liberar Recursos ---
cap.release()
cv2.destroyAllWindows()