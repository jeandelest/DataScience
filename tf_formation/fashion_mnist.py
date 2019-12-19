import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import pandas as pd
import sys

# Which class is in History
# print(dir(tf.keras.callbacks.History))
assert hasattr(tf, "function")

fashion_mnist = tf.keras.datasets.fashion_mnist
(images, targets), (_, _) = fashion_mnist.load_data()
images = images[:10000]

# print(type(images))
targets = targets[:10000]
print(images)
# print(images.shape)
# print(targets.shape)

targets_name = ["T-shirt/top", "Trouser", "Pullover", "Dress", 
"Coat", "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]

# plt.imshow(images[10], cmap="binary")
# plt.title(targets_name[targets[10]])
# plt.show()

# Model
model = tf.keras.models.Sequential()
# Need a vector not a matrix
# model.add(tf.keras.layers.Flatten(input_shape=[28, 28]))
images = images.astype(np.float32).reshape(-1, 28*28) / 255.0
model.add(tf.keras.layers.Dense(256, activation="relu"))
model.add(tf.keras.layers.Dense(128, activation="relu"))
# Softmax permet que la sommes du layer = 1 (Probabilité)
model.add(tf.keras.layers.Dense(10, activation="softmax"))

# model_output = model.predict(images[0:1])
# print(np.sum(model_output), targets[0:1])
# Neuron network summary
# model.summary()
# Define error
model.compile(
    loss = "sparse_categorical_crossentropy",
    optimizer = "sgd",
    metrics = ["accuracy"]
)

history = model.fit(images, targets, epochs=20)


"""loss_curve = history.history["loss"]
acc_curve = history.history["accuracy"]

plt.plot(loss_curve)
plt.title("Loss")
plt.show()

plt.plot(acc_curve)
plt.title("Acc")
plt.show() """