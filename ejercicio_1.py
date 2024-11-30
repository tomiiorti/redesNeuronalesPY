from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt


# Cargar dataset de dígitos
digits = load_digits()
X, y = digits.data, digits.target

# Dividir en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Crear un Perceptrón Multicapa (MLP)
mlp = MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=500, random_state=42)

# Entrenar la red
mlp.fit(X_train, y_train)

# Predicciones
y_pred = mlp.predict(X_test)

# Reporte de clasificación
print(classification_report(y_test, y_pred))

plt.gray()
plt.matshow(digits.images[0])
plt.show()
