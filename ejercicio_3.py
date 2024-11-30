import tensorflow as tf
from tensorflow.keras import layers, models

# Cargar dataset MNIST
(X_train, y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data()
X_train, X_test = X_train / 255.0, X_test / 255.0

# Configuraciones a probar
configurations = [
    [128],                  # Una capa densa con 128 neuronas
    [128, 64],              # Dos capas densas con 128 y 64 neuronas
    [256, 128, 64],         # Tres capas densas con 256, 128, y 64 neuronas
]

# Probar cada configuración
for config in configurations:
    print(f"Probando configuración: {config}")
    model = models.Sequential()
    model.add(layers.Flatten(input_shape=(28, 28)))
    for neurons in config:
        model.add(layers.Dense(neurons, activation='relu'))
    model.add(layers.Dense(10, activation='softmax'))

    # Compilar y entrenar
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    model.fit(X_train, y_train, epochs=5, validation_data=(X_test, y_test))

    # Evaluar
    test_loss, test_acc = model.evaluate(X_test, y_test)
    print(f"Precisión en prueba: {test_acc:.4f}")
