from keras.layers import Flatten, Dense, Conv2D, MaxPooling2D, Dropout, ZeroPadding2D, Convolution2D
from keras.models import Sequential
from keras.optimizers import SGD

from read_koggle_data import read_train_data, prepare_train_test_data,\
    img_rows, img_cols

x_source, y_source, filenames = read_train_data()

(x_train, y_train), (x_test, y_test) = prepare_train_test_data(
    x_source, y_source, 1000, 1000)

num_classes = 2

# TODO get from prepare_train_test_data because it depends on the backend
input_shape = (img_rows, img_cols, 1)

print('x_train shape:', x_train.shape)
print('y_train shape:', y_train.shape)
print('input_shape shape:', input_shape)

model = Sequential()
# Add convolution here

##################################
model.add(ZeroPadding2D((1,1),input_shape=input_shape))
model.add(Convolution2D(64, 3, 3, activation='relu'))
model.add(ZeroPadding2D((1,1)))
model.add(Convolution2D(64, 3, 3, activation='relu'))
model.add(MaxPooling2D((2,2), strides=(2,2)))

model.add(ZeroPadding2D((1,1)))
model.add(Convolution2D(128, 3, 3, activation='relu'))
model.add(ZeroPadding2D((1,1)))
model.add(Convolution2D(128, 3, 3, activation='relu'))
model.add(MaxPooling2D((2,2), strides=(2,2)))

model.add(ZeroPadding2D((1,1)))
model.add(Convolution2D(256, 3, 3, activation='relu'))
model.add(ZeroPadding2D((1,1)))
model.add(Convolution2D(256, 3, 3, activation='relu'))
model.add(ZeroPadding2D((1,1)))
model.add(Convolution2D(256, 3, 3, activation='relu'))
model.add(MaxPooling2D((2,2), strides=(2,2)))

model.add(ZeroPadding2D((1,1)))
model.add(Convolution2D(512, 3, 3, activation='relu'))
model.add(ZeroPadding2D((1,1)))
model.add(Convolution2D(512, 3, 3, activation='relu'))
model.add(ZeroPadding2D((1,1)))
model.add(Convolution2D(512, 3, 3, activation='relu'))
model.add(MaxPooling2D((2,2), strides=(2,2)))

model.add(ZeroPadding2D((1,1)))
model.add(Convolution2D(512, 3, 3, activation='relu'))
model.add(ZeroPadding2D((1,1)))
model.add(Convolution2D(512, 3, 3, activation='relu'))
model.add(ZeroPadding2D((1,1)))
model.add(Convolution2D(512, 3, 3, activation='relu'))
model.add(MaxPooling2D((2,2), strides=(2,2)))

model.add(Flatten())
model.add(Dense(4096, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(4096, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(1000, activation='softmax'))

sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(optimizer=sgd, loss='categorical_crossentropy')
model.fit(x_train, y_train, epochs=30, batch_size=32)


#loss_and_metrics = model.evaluate(x_test, y_test, batch_size=16)

##################################

#print x_train, y_train