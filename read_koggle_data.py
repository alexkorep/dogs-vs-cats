import os
import cv2
import numpy as np
from keras import backend as K

img_rows = 100
img_cols = 100

def _read_file(filename):
    with open(filename, mode='rb') as file:  # b is important -> binary
        fileContent = file.read()
        return fileContent


def _read_images(path='data/train'):
    x = []
    y = []
    filenames = []
    files = os.listdir(path)
    total_files = len(files)
    i = 0
    for file in files:
        # if i > 100:
        #     break

        fullname = path + '/' + file
        im = cv2.imread(fullname)
        im = cv2.resize(im, (img_rows, img_cols))
        im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        x.append(im)
        is_dog = 1 if 'dog' in file else 0
        y.append([is_dog, 1 - is_dog])
        filenames.append(file)
        i += 1
        if not i%100:
            print("Read file {} out of {}".format(i, total_files))

    return x, y, filenames


def save_dump_to_disk(x, y, filenames):
    np.save('data/x.npy', x)
    np.save('data/y.npy', y)
    np.save('data/filenames.npy', filenames)

def reshape_for_backend(x, img_rows, img_cols):
    if K.image_data_format() == 'channels_first':
        x = x.reshape(x.shape[0], 1, img_rows, img_cols)
    else:
        x = x.reshape(x.shape[0], img_rows, img_cols, 1)
    return x

def read_train_data():
    try:
        x = np.load('data/x.npy')
        y = np.load('data/y.npy')
        filenames = np.load('data/filenames.npy')
    except:
        x, y, filenames = _read_images()
        save_dump_to_disk(x, y, filenames)

    x = reshape_for_backend(x, img_rows, img_cols)

    return x, y, filenames

def prepare_train_test_data(x_source, y_source, x_size, y_size):
    x_train = x_source[:x_size]
    y_train = y_source[:x_size]
    x_test = x_source[x_size : x_size + y_size]
    y_test = y_source[x_size : x_size + y_size]
    return (x_train, y_train), (x_test, y_test)
