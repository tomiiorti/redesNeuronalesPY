### **¿Qué son las redes neuronales y en qué se inspiran? (Concepto básico)**

Las redes neuronales son un tipo de modelo de aprendizaje automático diseñado para identificar patrones y realizar predicciones o clasificaciones. Se inspiran en el funcionamiento del cerebro humano, específicamente en cómo las neuronas biológicas procesan y transmiten información.

- **Estructura básica:** Una red neuronal consta de nodos (neuronas artificiales) organizados en capas. Cada nodo recibe entradas, realiza un cálculo (generalmente a través de una función de activación), y pasa el resultado como salida.
- **Inspiración biológica:** Las conexiones entre las neuronas en una red artificial intentan emular las sinapsis biológicas, donde cada conexión tiene un peso que determina su importancia.

---

### **Función de cada tipo de capa en una red neuronal**

1. **Capa de entrada:**
    - **Rol:** Recibe los datos iniciales del problema (como píxeles de una imagen o características de un dataset).
    - **Descripción:** Los nodos en esta capa son equivalentes al número de características de entrada.
    - **Ejemplo:** Para un modelo que clasifica imágenes de 28x28 píxeles, habrá 784 nodos en esta capa.
2. **Capas ocultas:**
    - **Rol:** Procesan la información mediante la combinación de pesos, funciones de activación y sesgos.
    - **Descripción:** Estas capas extraen patrones y características intermedias de los datos.
    - **Ejemplo:** En un modelo que clasifica imágenes, las capas ocultas podrían aprender bordes, texturas y formas más complejas.
3. **Capa de salida:**
    - **Rol:** Proporciona la predicción final o clasificación.
    - **Descripción:** El número de nodos depende del tipo de problema:
        - En un problema de clasificación binaria, hay un solo nodo con una salida probabilística (0 o 1).
        - En problemas de clasificación multiclase, habrá tantos nodos como clases.
    - **Ejemplo:** Para clasificar dígitos del 0 al 9, habrá 10 nodos.

---

### **¿Qué es la retropropagación y cuál es su propósito en el entrenamiento de una red?**

- **Definición:** La retropropagación (o *backpropagation*) es un algoritmo que ajusta los pesos de la red neuronal durante el entrenamiento. Es una forma eficiente de calcular el gradiente del error con respecto a los pesos y sesgos utilizando el método del descenso del gradiente.
- **Propósito:**
    1. **Minimizar el error:** Reduce la diferencia entre las predicciones de la red y los valores reales (error).
    2. **Optimizar la red:** Ajusta los pesos de cada conexión para mejorar la precisión en cada iteración.
- **Proceso:**
    1. **Propagación hacia adelante:** Se calcula la salida de la red con los pesos actuales.
    2. **Cálculo del error:** Se compara la salida predicha con el valor real usando una función de pérdida.
    3. **Propagación hacia atrás:** El error se retropropaga para calcular cómo deben ajustarse los pesos.
    4. **Actualización de pesos:** Se utiliza el descenso del gradiente para actualizar los pesos y reducir el error.

---

### **¿En qué casos se puede usar Scikit-learn para redes neuronales y cuándo es preferible usar TensorFlow o PyTorch?**

1. **Scikit-learn:**
    - **Cuándo usarlo:**
        - Cuando se necesitan modelos simples para redes neuronales básicas.
        - Para prototipar rápidamente o realizar pruebas con datasets pequeños.
        - En proyectos que no requieren alta personalización o redes profundas.
    - **Ventaja:** Fácil de usar, con una API clara para principiantes.
2. **TensorFlow o PyTorch:**
    - **Cuándo usarlo:**
        - Para construir redes neuronales profundas (Deep Learning) y modelos avanzados.
        - Cuando se requiere un alto rendimiento y entrenamiento en GPU.
        - En casos que exigen personalización completa del modelo (arquitectura de red compleja, ajuste de hiperparámetros avanzado, etc.).
        - En proyectos que trabajan con grandes cantidades de datos o modelos de producción a gran escala.
    - **Ventajas:**
        - **TensorFlow:** Ideal para producción, herramientas como TensorBoard para monitoreo, y compatibilidad con despliegue en dispositivos móviles.
        - **PyTorch:** Más flexible y popular en investigación académica debido a su enfoque dinámico y facilidad para depuración.


### Aclaraciónes para los ejercicios
**Dependencias a instalar:**
*sklearn:*
```
pip install sklearn
```
___
*matplotlib:*
```
pip install matplotlib

```
___
*tensorflow:*
```
pip install tensorflow
```