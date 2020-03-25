from OneHot import *

class RnnModel(tf.keras.Model):

    def __init__(self, vocab):
        super(RnnModel, self).__init__()
        # Convolutions
        self.one_hot = OneHot(len(vocab))

    def call(self, inputs):
        output = self.one_hot(inputs)
        return output