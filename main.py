import numpy as np
import pandas as pd
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow import keras
from sklearn.preprocessing import LabelEncoder

# Cargar datos de entrenamiento y prueba
train_data = pd.read_csv('training.csv')
test_data = pd.read_csv('testing.csv')

# Dividir los datos en características (X) y etiquetas (y)
X_train = train_data.drop('CLASE', axis=1)
y_train = train_data['CLASE']
X_test = test_data.drop('CLASE', axis=1)
y_test = test_data['CLASE']

# Entrenar un clasificador de Naive Bayes
naive_bayes = MultinomialNB()
naive_bayes.fit(X_train, y_train)

# Realizar predicciones y calcular la precisión del clasificador de Naive Bayes
y_pred_nb = naive_bayes.predict(X_test)
accuracy_nb = accuracy_score(y_test, y_pred_nb)
print("Precisión del clasificador de Naive Bayes:", accuracy_nb)



# Crear un objeto LabelEncoder para las etiquetas de clase
label_encoder = LabelEncoder()

# Codificar las etiquetas de clase en valores numéricos
y_train_encoded = label_encoder.fit_transform(y_train)
y_test_encoded = label_encoder.transform(y_test)

# Entrenar una RNA
model = keras.Sequential([
    keras.layers.Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
    keras.layers.Dense(32, activation='relu'),
    keras.layers.Dense(len(label_encoder.classes_), activation='softmax')
])
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(X_train, y_train_encoded, epochs=10, batch_size=32)

# Evaluar la RNA en los datos de prueba
_, accuracy_rnn = model.evaluate(X_test, y_test_encoded)
print("Precisión de la RNA:", accuracy_rnn)