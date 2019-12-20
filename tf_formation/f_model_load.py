
import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter
import tensorflow as tf
import numpy as np
import pandas as pd
import sys

from sklearn.preprocessing import MinMaxScaler

fashion_mnist = tf.keras.datasets.fashion_mnist
(images, targets), (_, _) = fashion_mnist.load_data()
images = images[10000:10100]
targets = targets[10000:10100]
images = images.astype(np.float32).reshape(-1, 28*28)
scaler = MinMaxScaler()
images = scaler.fit_transform(images)

model = tf.keras.models.load_model("f_mnist.h5")

loss, acc = model.evaluate(images, targets)
print(loss)
print(acc)