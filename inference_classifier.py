import pickle
import cv2
import mediapipe as mp
import numpy as np
from sklearn.preprocessing import LabelEncoder
import time # Importa el módulo time para trabajar con temporizadores

# --- Cargar Modelo Pre-entrenado y Codificador de Etiquetas ---
# Esta sección es crucial para que el script pueda usar el modelo de IA que has entrenado.
# Carga el archivo 'model.p' que contiene el modelo de clasificación de gestos y el codificador
# de etiquetas que mapea los números a las letras (ej. 0 a 'A', 1 a 'B').
try:
    model_data = pickle.load(open('./model.p', 'rb')) # Intenta cargar el archivo del modelo
    model = model_data['model'] # Extrae el modelo de clasificación
    label_encoder = model_data['label_encoder'] # Extrae el codificador de etiquetas
    # Crea un diccionario para mapear los IDs numéricos del modelo a las etiquetas de texto originales.
    labels_dict = {i: label for i, label in enumerate(label_encoder.classes_)}
    print(f"Loaded labels_dict: {labels_dict}")
except FileNotFoundError:
    print("Error: model.p not found. Please train the classifier first.")
    exit() # Sale del programa si el archivo del modelo no se encuentra
except KeyError:
    print("Error: label_encoder not found in model.p. Please retrain your model with the updated train_classifier.py.")
    exit() # Sale si falta el codificador de etiquetas
except Exception as e:
    print(f"Error loading model.p or label_encoder: {e}")
    exit() # Captura cualquier otro error al cargar

# --- Inicializar Cámara y MediaPipe ---
# Prepara la cámara web y configura las herramientas de MediaPipe para la detección de manos.
# MediaPipe es una biblioteca de Google para detectar puntos clave en el cuerpo, manos y rostro.
cap = cv2.VideoCapture(0) # Abre la cámara web por defecto (0 es usualmente la cámara integrada)
if not cap.isOpened():
    print("Error: No se pudo abrir la cámara.")
    exit() # Sale si la cámara no se puede abrir

mp_hands = mp.solutions.hands # Herramientas específicas de MediaPipe para detección de manos
mp_drawing = mp.solutions.drawing_utils # Utilidades para dibujar los puntos clave (landmarks) en el frame
mp_drawing_styles = mp.solutions.drawing_styles # Estilos visuales para los dibujos de MediaPipe

# Configura el detector de manos de MediaPipe.
# static_image_mode=False es para procesamiento de video (detecta y sigue manos a lo largo de los frames).
# min_detection_confidence: umbral de confianza para la primera detección de una mano.
# min_tracking_confidence: umbral de confianza para seguir una mano ya detectada.
hands = mp_hands.Hands(static_image_mode=False, min_detection_confidence=0.5, min_tracking_confidence=0.5)

# --- Variables para la Construcción de Palabras y Temporizadores ---
# Estas variables controlan cómo se acumulan las letras para formar palabras
# y gestionan los tiempos de espera para añadir letras o borrar la palabra.
current_word = ""          # Almacena la palabra que se está formando
last_prediction = ""       # Guarda la última letra (o "BLANK") predicha para evitar duplicados
prediction_start_time = 0  # Marca el tiempo en que la 'last_prediction' actual comenzó a ser detectada

# Nuevas variables para la lógica de borrado por "BLANK" sostenido
blank_start_time = 0       # Marca el tiempo en que el estado "BLANK" comenzó a ser detectado continuamente
blank_threshold_clear = 5  # Tiempo en segundos que "BLANK" debe ser sostenido para borrar la palabra

# --- Bucle Principal de Detección y Predicción en Tiempo Real ---
# Este es el corazón del programa. Se ejecuta continuamente, procesando cada frame de la cámara.
while True:
    data_aux = [] # Lista temporal para almacenar los datos normalizados de los puntos clave de la mano
    x_coords = [] # Almacena las coordenadas X de los puntos clave
    y_coords = [] # Almacena las coordenadas Y de los puntos clave

    ret, frame = cap.read() # Lee un frame de la cámara. 'ret' es True si se leyó correctamente.
    if not ret:
        print("Error: No se pudo leer el frame de la cámara.")
        break # Sale del bucle si no se puede leer el frame

    H, W, _ = frame.shape # Obtiene la altura (H) y el ancho (W) del frame
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) # Convierte el frame de BGR (OpenCV) a RGB (MediaPipe)
    results = hands.process(frame_rgb) # Procesa el frame con MediaPipe para detectar manos y sus landmarks

    predicted_character = "BLANK" # Valor por defecto si no se detecta ninguna mano o un signo claro

    if results.multi_hand_landmarks: # Si MediaPipe detectó una o más manos en el frame
        for hand_landmarks in results.multi_hand_landmarks: # Itera sobre cada mano detectada
            # Dibuja los puntos clave (landmarks) y las conexiones de la mano en el frame.
            mp_drawing.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style())

            # Extrae y normaliza las coordenadas de los puntos clave.
            # La normalización es crucial para que el modelo funcione independientemente del tamaño
            # o posición de la mano en la imagen, basándose solo en las proporciones relativas.
            x_coords = [lm.x for lm in hand_landmarks.landmark]
            y_coords = [lm.y for lm in hand_landmarks.landmark]

            x_min, y_min = min(x_coords), min(y_coords) # Encuentra las coordenadas mínimas (arriba-izquierda)
            x_max, y_max = max(x_coords), max(y_coords) # Encuentra las coordenadas máximas (abajo-derecha)

            # Calcula el rango de las coordenadas para la normalización.
            # Se añade un pequeño valor (1.0) para evitar división por cero si el rango es 0.
            range_x = x_max - x_min if (x_max - x_min) > 0 else 1.0
            range_y = y_max - y_min if (y_max - y_min) > 0 else 1.0

            for lm in hand_landmarks.landmark:
                # Normaliza las coordenadas de cada landmark y las añade a 'data_aux'.
                # Cada landmark tiene una coordenada X e Y, por lo que se añaden 2 valores por landmark.
                data_aux.append((lm.x - x_min) / range_x)
                data_aux.append((lm.y - y_min) / range_y)

        expected_feature_length = 42 # El modelo espera 42 características (21 puntos * 2 coordenadas)
        # Asegura que 'data_aux' tenga la longitud correcta, rellenando con ceros o truncando si es necesario.
        if len(data_aux) < expected_feature_length:
            data_aux.extend([0.0] * (expected_feature_length - len(data_aux)))
        elif len(data_aux) > expected_feature_length:
            data_aux = data_aux[:expected_feature_length]

        # Realiza la predicción usando el modelo de IA cargado.
        prediction = model.predict([np.asarray(data_aux)])
        # Convierte la predicción numérica (ej. 0) de vuelta a la etiqueta de texto (ej. 'A').
        predicted_character = labels_dict[int(prediction[0])]

        # --- Dibujar Cuadro Delimitador y Predicción ---
        # Calcula las coordenadas del cuadro delimitador alrededor de la mano detectada
        # y dibuja el carácter predicho encima.
        x1 = int(min(x_coords) * W) - 10 # Coordenada X superior izquierda (ajustada)
        y1 = int(min(y_coords) * H) - 10 # Coordenada Y superior izquierda (ajustada)
        x2 = int(max(x_coords) * W) - 10 # Coordenada X inferior derecha (ajustada)
        y2 = int(max(y_coords) * H) - 10 # Coordenada Y inferior derecha (ajustada)

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), 4) # Dibuja el rectángulo negro
        cv2.putText(frame, predicted_character, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 0), 3, cv2.LINE_AA) # Dibuja el texto

    # --- Lógica de Construcción de Palabra con tiempo mínimo (para añadir letras) ---
    # Esta sección controla cuándo una letra detectada se añade a la 'current_word'.
    # Requiere que la misma letra sea detectada continuamente por un mínimo de 3 segundos.
    current_time = time.time() # Obtiene el tiempo actual

    if predicted_character != last_prediction:
        # Si la predicción actual es diferente de la última, significa que hay un cambio (nueva letra o BLANK).
        # Reinicia el temporizador para la nueva predicción.
        prediction_start_time = current_time
        last_prediction = predicted_character
        # Si el nuevo carácter no es "BLANK", resetea el temporizador de "BLANK" para evitar borrados.
        if predicted_character != "BLANK":
            blank_start_time = 0
    elif predicted_character != "BLANK" and (current_time - prediction_start_time >= 3):
        # Si la predicción es la misma que la última Y no es "BLANK"
        # Y ha sido detectada continuamente durante al menos 3 segundos:
        current_word += predicted_character # Añade la letra a la palabra
        last_prediction = "BLANK" # Resetea last_prediction a "BLANK" para forzar una "pausa"
                                  # (ej. quitar la mano o cambiar de signo) antes de que la MISMA letra se añada de nuevo.
        prediction_start_time = current_time # Reinicia el temporizador para la próxima letra
        blank_start_time = 0 # Si se añadió una letra, se resetea el temporizador de BLANK

    # --- Lógica para Borrar la Palabra con "BLANK" Sostenido ---
    # Esta sección borra la 'current_word' si "BLANK" es detectado continuamente
    # durante un umbral de tiempo definido (blank_threshold_clear, ej. 5 segundos).
    if predicted_character == "BLANK":
        if blank_start_time == 0: # Si es la primera vez que entramos en un periodo "BLANK" continuo
            blank_start_time = current_time # Inicia el temporizador de "BLANK"
        elif current_time - blank_start_time >= blank_threshold_clear:
            # Si "BLANK" ha sido detectado por más del umbral definido
            current_word = "" # Borra toda la palabra
            blank_start_time = 0 # Reinicia el temporizador de "BLANK" para no borrar repetidamente
            print("Palabra borrada por BLANK sostenido.") # Mensaje de depuración en la consola
    else:
        # Si el carácter predicho NO es "BLANK", se resetea el temporizador de "BLANK".
        # Esto evita que se borre la palabra si el usuario está mostrando letras.
        blank_start_time = 0

    # --- Mostrar Información en Pantalla ---
    # Usa OpenCV para dibujar el carácter actual y la palabra en construcción en el frame.
    cv2.putText(frame, f"Sign: {predicted_character}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(frame, f"Word: {current_word}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

    cv2.imshow('frame', frame) # Muestra el frame procesado en una ventana
    key = cv2.waitKey(1) & 0xFF # Captura la tecla presionada (espera 1 ms)

    # --- Comandos del Teclado ---
    # Permite al usuario interactuar con el programa mediante el teclado.
    if key == ord('q'): # Si se presiona 'q', sale del bucle (cierra el programa)
        break
    elif key == ord('c'): # Si se presiona 'c', borra la última letra de la palabra
        current_word = current_word[:-1]
        blank_start_time = 0 # Resetea el temporizador de BLANK si el usuario borra manualmente
    elif key == ord(' '): # Si se presiona la barra espaciadora, añade un espacio a la palabra
        current_word += ' '
        blank_start_time = 0 # Resetea el temporizador de BLANK si el usuario añade un espacio

# --- Liberar Recursos ---
# Limpia los recursos utilizados antes de que el programa termine.
cap.release() # Libera la cámara web
cv2.destroyAllWindows() # Cierra todas las ventanas de OpenCV