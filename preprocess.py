import glob
import matplotlib.pyplot as plt
from PIL import Image, ImageOps
import numpy as np
import h5py
import tensorflow as tf

def preprocess_image(image, size=64):
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, [size, size])
    image = image / (255.0*0.5) - 1.0
    print(image.shape)
    # print(image.numpy().min())
    # print(image.numpy().max())
    return image

def load_paths(path):
    image = tf.read_file(path)
    return preprocess_image(image)

tf.enable_eager_execution()
file_names='celeb/*.jpg'
files = glob.glob(file_names)
plt.imshow(load_paths(files[0]))
plt.show()
