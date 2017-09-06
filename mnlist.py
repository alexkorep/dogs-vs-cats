import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense
from keras import backend as K

num_classes = 10
# input image dimensions
img_rows, img_cols = 28, 28


#########################################################
# Prepare data
# the data, shuffled and split between train and test sets
(x_train, y_train), (x_test, y_test) = mnist.load_data()
if K.image_data_format() == 'channels_first':
    x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
    x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
    input_shape = (1, img_rows, img_cols)
else:
    x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
    x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
    input_shape = (img_rows, img_cols, 1)
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255

# convert class vectors to binary class matrices
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)
#########################################################

print('x_train shape:', x_train.shape)
print('y_train shape:', y_train.shape)
print('input_shape shape:', input_shape)

model = Sequential()
model.add(Dense(10, input_shape=input_shape))
model.add(Dense(1))
#model.add(Dense(1, input_dim=4))
model.compile(optimizer='rmsprop',
              loss='mse',
              metrics=['accuracy'])

model.fit(x_train, y_train, epochs=30, batch_size=32)

loss_and_metrics = model.evaluate(x_test, y_test, batch_size=16)
#print(loss_and_metrics)

pred = model.predict(x_test, batch_size=32, verbose=0)

print("expected:")
print(y_test)
print("actual:")
print(pred)

