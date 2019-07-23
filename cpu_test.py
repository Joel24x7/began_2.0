import tensorflow as tf
from model import Began
from train import test

if __name__ == '__main__':

    model = Began()
    with tf.Graph().as_default():
        test(model)
