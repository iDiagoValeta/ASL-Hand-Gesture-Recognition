import pickle
import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from keras.utils import to_categorical # Importar to_categorical desde keras.utils

# --- Cargar Datos ---
data_dict = pickle.load(open('./data.pickle', 'rb'))

data = np.asarray(data_dict['data'])
labels = np.asarray(data_dict['labels'])

# --- Codificación de Etiquetas ---
label_encoder = LabelEncoder()
numeric_labels = label_encoder.fit_transform(labels)
num_classes = len(label_encoder.classes_)

# Convertir etiquetas numéricas a one-hot encoding
# Ejemplo: si num_classes = 3 y una etiqueta es 0, se convierte en [1, 0, 0]
y_one_hot = to_categorical(numeric_labels, num_classes=num_classes)

# --- División Entrenamiento/Prueba ---
x_train, x_test, y_train_one_hot, y_test_one_hot = train_test_split(
    data, y_one_hot, test_size=0.2, shuffle=True, stratify=numeric_labels, random_state=42
)

# --- Construir el Modelo de Red Neuronal ---
model = keras.Sequential([
    keras.layers.Dense(128, activation='relu', input_shape=(x_train.shape[1],)),
    keras.layers.Dropout(0.3), # Previene el sobreajuste
    keras.layers.Dense(64, activation='relu'),
    keras.layers.Dropout(0.3),
    keras.layers.Dense(num_classes, activation='softmax') # La capa de salida con activación softmax para clasificación multiclase
])

# --- Compilar el Modelo ---
# optimizer='adam' es una buena opción para empezar
# loss='categorical_crossentropy' es para clasificación multiclase con one-hot encoded labels
# metrics=['accuracy'] para monitorear la precisión durante el entrenamiento
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# --- Entrenamiento del Modelo ---
# Puedes ajustar epochs y batch_size según tus necesidades y la potencia de tu GPU
print("Iniciando entrenamiento del modelo...")
history = model.fit(x_train, y_train_one_hot,
                    epochs=50, # Número de veces que el modelo verá todos los datos de entrenamiento
                    batch_size=32, # Número de muestras por actualización de gradiente
                    validation_split=0.1, # Usa una parte del set de entrenamiento para validación
                    verbose=1) # Muestra el progreso del entrenamiento

# --- Evaluación ---
print("\nEvaluando el modelo en el conjunto de prueba...")
loss, accuracy = model.evaluate(x_test, y_test_one_hot, verbose=0)
print(f'Precisión en el conjunto de prueba: {accuracy*100:.2f}%')

# --- Guardar Modelo y Codificador ---
# Guarda el modelo de Keras en formato HDF5
model_save_path = 'deep_model.h5'
model.save(model_save_path)
print(f"Modelo de Keras guardado en '{model_save_path}'")

# Guarda el LabelEncoder por separado para la inferencia
encoder_save_path = 'label_encoder.pkl'
with open(encoder_save_path, 'wb') as f:
    pickle.dump(label_encoder, f)
print(f"LabelEncoder guardado en '{encoder_save_path}'")