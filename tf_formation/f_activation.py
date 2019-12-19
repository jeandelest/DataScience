import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter
import tensorflow as tf
import numpy as np
import pandas as pd
import sys

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

###################### ACTIVATION #########################
# Lineaire (Activation = preactivation => pas d'activation)
# Tanh scale les val entre (-1 et 1)
# Sigmoid (- inf to + inf ) => probality in result (scale entre 0 et 1)
# Softmax (preactivtation de toute les entrées) => sum (sortie) = 1
# Relu if x < 0 = y = 0 else y = x
##############################################################
fashion_mnist = tf.keras.datasets.fashion_mnist
(images, targets), (_, _) = fashion_mnist.load_data()
images = images[:10000]
targets = targets[:10000]

targets_name = ["T-shirt/top", "Trouser", "Pullover", "Dress", 
"Coat", "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]

images = images.astype(np.float32).reshape(-1, 28*28)
scaler = MinMaxScaler()
images = scaler.fit_transform(images)

im_train, im_test, tar_train, tar_test = train_test_split(images, targets, test_size=0.2, random_state=1)

model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Dense(256, activation="relu"))
model.add(tf.keras.layers.Dense(128, activation="relu"))
model.add(tf.keras.layers.Dense(10, activation="softmax"))

model.compile(
    loss = "sparse_categorical_crossentropy",
    optimizer = "sgd",
    metrics = ["accuracy"]
)

