import random
import json
import tensorflow as tf 
from DefaultModel import *

vocab, text = get_vocab()
inputs, targets, _, _ = get_inputs_targets(vocab, text)

with open("model_rnn_vocab_to_int", "r") as f:
    vocab_to_int = json.loads(f.read())
with open("model_rnn_int_to_vocab", "r") as f:
    int_to_vocab = json.loads(f.read())
    int_to_vocab = {int(key):int_to_vocab[key] for key in int_to_vocab}

model = build_model(len(vocab))

model.load_weights("model_rnn.h5")

# model.reset_states()

size_poetries = 300

poetries = np.zeros((64, size_poetries, 1))
sequences = np.zeros((64, 100))
for b in range(64):
    rd = np.random.randint(0, len(inputs) - 100)
    sequences[b] = inputs[rd:rd+100]

for i in range(size_poetries+1):
    if i > 0:
        poetries[:,i-1,:] = sequences
    softmax = predict(sequences, model)
    # Set the next sequences
    sequences = np.zeros((64, 1))
    for b in range(64):
        argsort = np.argsort(softmax[b][0])
        argsort = argsort[::-1]
        # Select one of the strongest 4 proposals
        sequences[b] = argsort[0]

print("=============================")
for b in range(2):
    sentence = "".join([int_to_vocab[i[0]] for i in poetries[b]])
    print(sentence)
    print("\n=====================\n")