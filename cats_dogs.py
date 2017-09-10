from keras.layers import Flatten, Dense, Conv2D, MaxPooling2D
from keras.models import Sequential

from read_koggle_data import read_train_data, prepare_train_test_data,\
    img_rows, img_cols

x_source, y_source, filenames = read_train_data()

(x_train, y_train), (x_test, y_test) = prepare_train_test_data(
    x_source, y_source, 10000, 1000)

num_classes = 2

# TODO get from prepare_train_test_data because it depends on the backend
input_shape = (img_rows, img_cols, 1)

print('x_train shape:', x_train.shape)
print('y_train shape:', y_train.shape)
print('input_shape shape:', input_shape)

model = Sequential()
# Add convolution here
model.add(Conv2D(32, kernel_size=(8, 8),
                 activation='relu',
                 input_shape=input_shape))

model.add(MaxPooling2D())

model.add(Conv2D(32, kernel_size=(8, 8), activation='relu'))

model.add(Flatten(input_shape=input_shape))
model.add(Dense(num_classes, activation='softmax'))
model.compile(optimizer='rmsprop',
              loss='mse',
              metrics=['accuracy'])

model.fit(x_train, y_train, epochs=30, batch_size=32)
loss_and_metrics = model.evaluate(x_test, y_test, batch_size=16)

#print x_train, y_train