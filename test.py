import random
from keras.models import Sequential
from keras.layers import Dense, Activation
import numpy as np

def generate_data(size):
    """"
    x_train = []
    y_train = []
    for i in range(size):
        x = random.randint(0, 100)
        y = 2*x
        x_train.append(x)
        y_train.append(y)
    return np.array(x_train), np.array(y_train)
    """
    import numpy as np
    #data = np.random.random((10000, 100))
    #labels = np.random.randint(2, size=(10000, 1))
    data = np.random.random((10000, 1))
    labels = data*2
    return data, labels


model = Sequential()
model.add(Dense(1, input_dim=1))
model.compile(optimizer='rmsprop',
              loss='binary_crossentropy',
              metrics=['accuracy'])

x_train, y_train = generate_data(1000)
x_test, y_test = generate_data(10)

model.fit(x_train, y_train, epochs=1000, batch_size=32)

loss_and_metrics = model.evaluate(x_test, y_test, batch_size=128)

print(loss_and_metrics)
