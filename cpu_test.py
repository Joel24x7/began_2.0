import tensorflow as tf
from model import Began

if __name__ == '__main__':

    model = Began()
    with tf.Graph().asDefault():
        model.train(1)
