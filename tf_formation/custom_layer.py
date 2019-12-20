import tensorflow as tf
import numpy as np

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
tf.keras.backend.clear_session()

class MlpLayer(tf.keras.layers.Layer):

    def __init__(self, units, activations, **kwargs):
        super(MlpLayer, self).__init__(**kwargs)
        # Set the property to the layer
        self.units = units
        self.activations_list = activations
        self.weights_list = []

    # The build method will be called once
    # we know the shape of the previous Layer: input_shape
    def build(self, input_shape):
        # Create trainable weights variables for this layer.
        # We create matrix of weights for each layer
        # Each weight have this shape: (previous_layer_size, layer_size)
        i = 0
        for units in self.units:
            weights = self.add_weight(
                name='weights-%s' % i,
                shape=(input_shape[1], units),
                initializer='uniform',
                trainable=True
            )
            i += 1
            self.weights_list.append(weights)
            input_shape = (None, units)
        super(MlpLayer, self).build(input_shape)
        
    def call(self, x):
        output = x

        # We go through each weight to compute the dot product between the previous
        # activation and the weight of the layer.
        # At the first pass, the previous activation is just the variable "x": The input vector
        for weights, activation in zip(self.weights_list, self.activations_list):
            # We can still used low level operations as tf.matmul, tf.nn.relu... 
            output = tf.matmul(output, weights)

            if activation == "relu":
                output = tf.nn.relu(output)
            elif activation == "sigmoid":
                output = tf.nn.sigmoid(output)
            elif activation == "softmax":
                output = tf.nn.softmax(output)
        
        return output

    # By adding the get_config method you can then save your model with the custom layer
    # and retrieve the model with the same parameters
    def get_config(self):
        config = {
            'units': self.units,
            "activations": self.activations_list
        }
        # Retrieve the config from the parent layer
        base_config = super(MlpLayer, self).get_config()
        # Return the final config
        return dict(list(base_config.items()) + list(config.items()))

# Flatten
model = tf.keras.models.Sequential()

# Add the layers
model.add(MlpLayer([4 , 2], ["relu", "softmax"]))
model.predict(np.zeros((5, 10)))