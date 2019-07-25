import glob
from matplotlib import pyplot, image
from PIL import Image, ImageOps
import numpy as np
import h5py

def prep_data(data_name, size=64, file_names='celeb/*.jpg'):

    count = 0
    files = glob.glob(file_names)
    dataset = np.zeros((len(files), size, size, 3))

    for file in files:
        img = Image.open(file)
        img_resized = ImageOps.fit(img, (size, size), Image.ANTIALIAS)
        img_data = np.asarray(img_resized)
        # img_data = (img_data/(255.0 * 0.5)) - 1.0
        dataset[count] = img_data
        count += 1

    with h5py.File('{}.h5'.format(data_name), 'w') as file:
        file.create_dataset(data_name, data=dataset)

def load_data(data_name):
    with h5py.File('{}.h5'.format(data_name)) as file:
        data = file[data_name]
        data_set = np.array(data[:,:,:])
        return data_set

if __name__ == '__main__':

    prep_data('celeb_data')
    data = load_data('celeb_data')
    img = data[0]
    # img = (img + 1.0)/2.0
    pyplot.imshow(img)
    pyplot.show()

    # print(data.shape)

    # count = 1
    # for i in range(count):
    #     img = data[0]
    #     # img = (img + 1.0)/2.0
    #     pyplot.imshow(img)
    #     pyplot.show()