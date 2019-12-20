import tensorflow as tf
import numpy as np


model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Dense(256, activation="relu"))
model.add(tf.keras.layers.Dense(128, activation="relu"))
model.add(tf.keras.layers.Dense(2, activation="softmax"))

obj_loss = tf.keras.losses.SparseCategoricalCrossentropy()
optimizer = tf.keras.optimizers.Adam()
# Accumulateur -> track l'évolution de l'erreur
train_loss = tf.keras.metrics.Mean(name='train_loss')

@tf.function
def train_step(image, targets):
    with tf.GradientTape() as tape:
        prediction = model(image)
        # Error
        loss = obj_loss(targets, prediction)
    # Gradient
    gradients = tape.gradient(loss, model.trainable_variables)
    # Change weights
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    # Accumulate
    train_loss(loss)

for epoch in range(0, 10):
    for _ in range(0, 100):
        # Fake data
        inputs = np.zeros((2, 30))
        inputs[0] -= 1
        inputs[1] = 1
        # Fake targets
        targets = np.zeros((2, 1))
        targets[0] = 0
        targets[1] = 1
        #Training
        train_step(inputs, targets)
    print("Loss: %s", train_loss.result())
    train_loss.reset_states()