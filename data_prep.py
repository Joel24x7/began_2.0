import os

import h5py
import matplotlib.pyplot as plt
import numpy as np
import PIL.Image as PILImage
import scipy
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

image_size=64

def load_data():
    with h5py.File('mnist_data.h5') as file:
        data = file['mnist_data']
        data = np.array(data, dtype=np.float16)
        return data

def prep_mnist_color(change_colors=True):

    scale_factor=image_size/28.0
    x_train = input_data.read_data_sets("mnist", one_hot=True).train.images
    x_train = x_train.reshape(-1, 28, 28, 1).astype(np.float32)
    
    path = os.path.abspath('color_img.png') 
    color_img = PILImage.open(path)

    resized = np.asarray([scipy.ndimage.zoom(image, (scale_factor, scale_factor, 1), order=1) for image in x_train])
    color = np.concatenate([resized, resized, resized], axis=3)    
    binary = (color > 0.5)
    
    dataset = np.zeros((x_train.shape[0], image_size, image_size, 3))
    for i in range(x_train.shape[0]):

        x_c = np.random.randint(0, color_img.size[0] - image_size)
        y_c = np.random.randint(0, color_img.size[1] - image_size)
        image = color_img.crop((x_c, y_c, x_c + image_size, y_c + image_size))

        image = np.asarray(image) / (255*0.5)
        image -= 1.0

        if change_colors:
            for j in range(3):
                image[:, :, j] = (image[:, :, j] + np.random.uniform(0, 1)) / 2.0

        image[binary[i]] = 1 - image[binary[i]]
        
        dataset[i] = image
    return dataset

if __name__ == "__main__":
    data = prep_mnist_color()
    with h5py.File('mnist_data.h5', 'w') as file:
        file.create_dataset('mnist_data', data=data)

    # plt.figure(figsize=(15,3))
    count = 8
    for i in range(count):
        plt.subplot(2, count // 2, i+1)
        plt.imshow(data[i])
        plt.axis('off')
    plt.tight_layout()
    plt.show()