import tensorflow as tf
from RnnModel import *
import unidecode
import numpy as np

# Genere des batchs à partir de la dataset
def gen_batch(inputs, targets, seq_len, batch_size, vocab_size, noise=0):
    # Size of each chunk
    chuck_size = (len(inputs) -1)  // batch_size
    # Numbef of sequence per chunk
    sequences_per_chunk = chuck_size // seq_len

    for s in range(0, sequences_per_chunk):
        batch_inputs = np.zeros((batch_size, seq_len))
        batch_targets = np.zeros((batch_size, seq_len))
        for b in range(0, batch_size):
            fr = (b*chuck_size)+(s*seq_len)
            to = fr+seq_len
            batch_inputs[b] = inputs[fr:to]
            batch_targets[b] = inputs[fr+1:to+1]
            
            if noise > 0:
                noise_indices = np.random.choice(seq_len, noise)
                batch_inputs[b][noise_indices] = np.random.randint(0, vocab_size)
            
        yield batch_inputs, batch_targets


def build_model(vocab_size):
    ### Creat the layers

    # Set the input of the model
    tf_inputs = tf.keras.Input(shape=(None,), batch_size=64)
    # Convert each value of the  input into a one encoding vector
    one_hot = OneHot(vocab_size, autocast=False)(tf_inputs)
    # Stack LSTM cells
    rnn_layer1 = tf.keras.layers.LSTM(128, return_sequences=True, stateful=True, autocast=False)(one_hot)
    rnn_layer2 = tf.keras.layers.LSTM(128, return_sequences=True, stateful=True, autocast=False)(rnn_layer1)
    # Create the outputs of the model
    hidden_layer = tf.keras.layers.Dense(128, activation="relu")(rnn_layer2)
    outputs = tf.keras.layers.Dense(vocab_size, activation="softmax")(hidden_layer)

    ### Setup the model
    return tf.keras.Model(inputs=tf_inputs, outputs=outputs)

def get_vocab():
    with open("victorhugo.txt", "r") as f:
        text = f.read()

    # print(type(text))
    text = unidecode.unidecode(text)
    text = text.lower()

    text = text.replace("2", "")
    text = text.replace("1", "")
    text = text.replace("8", "")
    text = text.replace("5", "")
    text = text.replace(">", "")
    text = text.replace("<", "")
    text = text.replace("!", "")
    text = text.replace("?", "")
    text = text.replace("-", "")
    text = text.replace("$", "")
    # ----------------
    text = text.replace("|", "")
    text = text.replace("%", "")
    text = text.replace("/", "")
    text = text.replace("0", "")
    text = text.replace("\n", "")

    text = text.strip()

    return set(text), text

def get_inputs_targets(vocab, text):
    # Vocab to in and int to vocab
    vocab_to_int = {l:i for i,l in enumerate(vocab)}
    int_to_vocab = {i:l for i,l in enumerate(vocab)}

    #Convert Pharse to int and vice versa
    encoded = [vocab_to_int[l] for l in text]

    # Le target de l'input courant est la lettre suivante
    inputs, targets = encoded, encoded[1:]

    return inputs, targets, vocab_to_int, int_to_vocab

@tf.function
def predict(inputs, model):
    # Make a prediction on all the batch
    predictions = model(inputs)
    return predictions