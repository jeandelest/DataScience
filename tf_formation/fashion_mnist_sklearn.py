import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import pandas as pd
import sys

from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler

fashion_mnist = tf.keras.datasets.fashion_mnist
(images, targets), (_, _) = fashion_mnist.load_data()
images = images[:10000]
targets = targets[:10000]

targets_name = ["T-shirt/top", "Trouser", "Pullover", "Dress", 
"Coat", "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]

images = images.astype(np.float32).reshape(-1, 28*28)

scaler = MinMaxScaler()
images = scaler.fit_transform(images)

model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Dense(256, activation="relu"))
model.add(tf.keras.layers.Dense(128, activation="relu"))
# Softmax permet que la sommes du layer = 1 (Probabilité)
model.add(tf.keras.layers.Dense(10, activation="softmax"))

# Define error
model.compile(
    loss = "sparse_categorical_crossentropy",
    optimizer = "sgd",
    metrics = ["accuracy"]
)

history = model.fit(images, targets, epochs=10)

model_output = model.predict(images[0:1])
print(np.max(model_output), targets[0:1])
