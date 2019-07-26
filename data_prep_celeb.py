import glob
import h5py
import numpy as np
import scipy
from PIL import Image, ImageOps
import matplotlib.pyplot as plt

def prepare_images(root='celeb/*.jpg', size=64):

    file_names = glob.glob(root)
    dataset = np.zeros((len(file_names), size, size, 3))
    index = 0
    for file in file_names:
        image = Image.open(file)
        image = ImageOps.fit(image, (size, size))
        image_arr = np.asarray(image, dtype=float)
        image_arr = (image_arr/(255*0.5)) - 1.0
        dataset[index] = image_arr
        index += 1

    with h5py.File('celeb.h5', 'w') as hf:
            hf.create_dataset('celeb', data=dataset)

def load_data(data_name='celeb'):
    with h5py.File('{}.h5'.format(data_name)) as file:
        data = file[data_name]
        data = np.array(data, dtype=np.float32)
        return data

if __name__ == '__main__':
    prepare_images()
    data = load_data()
    
    count = 8
    for i in range(count):
        plt.subplot(2, count // 2, i+1)
        plt.imshow(data[i])
        plt.axis('off')
    plt.tight_layout()
    plt.show()
