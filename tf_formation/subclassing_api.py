import tensorflow as tf
import numpy as np

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
tf.keras.backend.clear_session()

fashion_mnist = tf.keras.datasets.fashion_mnist
(images, targets), (_, _) = fashion_mnist.load_data()
images = images[:10000]
targets = targets[:10000]

targets_name = ["T-shirt/top", "Trouser", "Pullover", "Dress", 
"Coat", "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]

images = images.astype(np.float32).reshape(-1, 28*28)
scaler = MinMaxScaler()
images = scaler.fit_transform(images)

class CustomModel(tf.keras.Model):
    
    def __init__(self):
        super(CustomModel, self).__init__()

        # First in the init method you need to instanciate the layers you will use
        self.first_layer = tf.keras.layers.Dense(64, activation='relu', name="first_layer")
        # WARNING: DO NOT CALL ONE OF YOUR CLASS VARIABLE: output
        self.output_layer = tf.keras.layers.Dense(10, activation='softmax', name="output_layer")

    def call(self, x):
        # Then in the call method you can execute your operations
        _layer_first = self.first_layer(x)
        _layer_first = tf.nn.sigmoid(_layer_first)
        out = self.output_layer(_layer_first)
        return out

try:
    model = CustomModel()
    model.compile(
        loss = tf.keras.losses.SparseCategoricalCrossentropy(),
        optimizer = tf.keras.optimizers.RMSprop(),
        metrics = [tf.keras.metrics.SparseCategoricalAccuracy()]
    )
    model.fit(images, targets, epochs=5)
    loss = model.predict(images[0:1])
    print(np.max(loss), targets[0:1])
except Exception as e:
    print("e=", e)