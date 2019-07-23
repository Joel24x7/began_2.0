import tensorflow as tf
from model import Began
from train import test, train

if __name__ == '__main__':

    model = Began()
    with tf.device('/gpu:0'):
        config = tf.ConfigProto(allow_soft_placement=True)
        with tf.Graph().as_default():
            train(model, 10)
