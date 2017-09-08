import os
import cv2
import numpy as np


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
        #if i > 1000:
        #    break

        fullname = path + '/' + file
        im = cv2.imread(fullname)
        im = cv2.resize(im, (100, 100))
        x.append(im)
        is_dog = 'dog' in file
        y.append(is_dog)
        filenames.append(file)
        i += 1
        if not i%100:
            print("Read file {} out of {}".format(i, total_files))

    return x, y, filenames


def save_dump_to_disk(x, y, filenames):
    np.save('data/x.npy', x)
    np.save('data/y.npy', y)
    np.save('data/filenames.npy', filenames)


def read_train_data():
    try:
        x = np.load('data/x.npy')
        y = np.load('data/y.npy')
        filenames = np.load('data/filenames.npy')
    except:
        x, y, filenames = _read_images()
        save_dump_to_disk(x, y, filenames)

    return x, y, filenames

