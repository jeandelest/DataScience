import tensorflow as tf
import matplotlib as mlp 
import numpy as np 

import sys
from DefaultModel import *
import json

vocab, text = get_vocab()

inputs, targets, vocab_to_int, int_to_vocab = get_inputs_targets(vocab, text)

vocab_size = len(vocab)

model = build_model(vocab_size)

loss_object = tf.keras.losses.SparseCategoricalCrossentropy()
optimizer = tf.keras.optimizers.Adam(lr=0.001)

# Loss
train_loss = tf.keras.metrics.Mean(name='train_loss')
# Accuracy
train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')

@tf.function
def train_step(inputs, targets):
    with tf.GradientTape() as tape:
        # Make a prediction on all the batch
        predictions = model(inputs)
        # Get the error/loss on these predictions
        loss = loss_object(targets, predictions)
    # Compute the gradient which respect to the loss
    gradients = tape.gradient(loss, model.trainable_variables)
    # Change the weights of the model
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    # The metrics are accumulate over time. You don't need to average it yourself.
    train_loss(loss)
    train_accuracy(targets, predictions)

# Start by resetting the cells of the RNN
model.reset_states()

for epoch in range(4000):
    for batch_inputs, batch_targets in gen_batch(inputs, targets, 100, 64, vocab_size, noise=13):
        train_step(batch_inputs, batch_targets)
    template = '\r Epoch {}, Train Loss: {}, Train Accuracy: {}'
    print(template.format(epoch, train_loss.result(), train_accuracy.result()*100), end="")
    model.reset_states()

model.save("model_rnn.h5")

with open("model_rnn_vocab_to_int", "w") as f:
    f.write(json.dumps(vocab_to_int))
with open("model_rnn_int_to_vocab", "w") as f:
    f.write(json.dumps(int_to_vocab))

