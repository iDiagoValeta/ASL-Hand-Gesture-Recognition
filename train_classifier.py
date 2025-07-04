import pickle
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder # <-- Importa para convertir etiquetas a números.

# --- Cargar Datos ---
data_dict = pickle.load(open('./data.pickle', 'rb')) # Carga los datos preprocesados de las manos.

data = np.asarray(data_dict['data'])
labels = np.asarray(data_dict['labels'])

# --- Codificación de Etiquetas ---
label_encoder = LabelEncoder()
numeric_labels = label_encoder.fit_transform(labels) # Convierte etiquetas de texto (A, B, BLANK) a números (0, 1, 2).

# --- División Entrenamiento/Prueba ---
# Divide los datos para entrenar (80%) y probar (20%) el modelo.
# `stratify` asegura una distribución equitativa de las clases.
x_train, x_test, y_train_numeric, y_test_numeric = train_test_split(
    data, numeric_labels, test_size=0.2, shuffle=True, stratify=numeric_labels, random_state=42
)

# --- Entrenamiento del Modelo ---
model = RandomForestClassifier(random_state=42) # Define el modelo de clasificación.
model.fit(x_train, y_train_numeric) # Entrena el modelo con los datos.

# --- Evaluación ---
y_predict_numeric = model.predict(x_test) # Realiza predicciones.
score = accuracy_score(y_predict_numeric, y_test_numeric) # Calcula la precisión.

print('{}% of samples were classified correctly !'.format(score * 100))

# --- Guardar Modelo y Codificador ---
f = open('model.p', 'wb')
# Guarda el modelo entrenado y el codificador de etiquetas para usarlos en el futuro.
pickle.dump({'model': model, 'label_encoder': label_encoder}, f)
f.close()

print("Model saved to model.p with label encoder.")