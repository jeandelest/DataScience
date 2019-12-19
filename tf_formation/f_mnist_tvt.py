import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter
import tensorflow as tf
import numpy as np
# import pandas as pd
import sys

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

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
# validation 20%
history = model.fit(im_train, tar_train, epochs=40, validation_split=0.2)

loss_c = history.history["loss"]
acc_c = history.history["accuracy"]

val_loss = history.history["val_loss"]
val_acc = history.history["val_accuracy"]

plt.figure(1)
plt.subplot(221)
plt.plot(loss_c, 'b')
plt.title('Loss train')
plt.grid(True)

plt.subplot(222)
plt.plot(acc_c, 'r')
plt.title('Accuracy train')
plt.grid(True)

plt.subplot(223)
plt.plot(val_loss, 'g')
plt.title('Loss validation')
plt.grid(True)

plt.subplot(224)
plt.plot(val_acc, 'y')
plt.title('Accuracy validation')
plt.grid(True)

# Format the minor tick labels of the y-axis into empty strings with
# `NullFormatter`, to avoid cumbering the axis with too many labels.
plt.gca().yaxis.set_minor_formatter(NullFormatter())
# Adjust the subplot layout, because the logit one may take more space
# than usual, due to y-tick labels like "1 - 10^{-3}"
plt.subplots_adjust(top=0.92, bottom=0.08, left=0.10, right=0.95, hspace=0.25,
                    wspace=0.35)
plt.show()

model.evaluate(im_test, tar_test)