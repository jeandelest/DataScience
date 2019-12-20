import tensorflow as tf
import numpy as np

# Avec tf 2.0 par defaut on utilise le mode graph
# Add @tf.function
a = np.array([1., 2.])
b = np.array([2., 5.])

@tf.function #-> mode graph
def add_fc(a, b):
    return tf.add(a, b)

model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Dense(256, activation="relu"))
model.add(tf.keras.layers.Dense(128, activation="relu"))
model.add(tf.keras.layers.Dense(10, activation="softmax"))

# model(x) -> mode graph
# model.predict(x) -> mode earger

# mode graph plus rapid que le mode earger