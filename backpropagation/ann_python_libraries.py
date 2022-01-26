import tensorflow as tf
import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
from timeit import default_timer as timer
from keras import backend as k


data = pd.read_csv("robots.csv")
x_train = data.iloc[:, :-3]
y_train = data.iloc[:, -3:]
print(x_train.head())
print(y_train.head())
input_shape = (x_train.shape[1],)

model = Sequential()
# layers
l1 = Dense(1, input_shape=input_shape, activation='sigmoid', dtype=np.double)
l2 = Dense(3, activation='sigmoid', dtype=np.double)
model.add(l1)
model.add(l2)


l1.set_weights([np.array([[0.84],
                         [0.39],
                         [0.78],
                         [0.79],
                         [0.91],
                         [0.19]], dtype=np.double),
               np.array([0.33], dtype=np.double)])


l2.set_weights([np.array([[0.76, 0.27, 0.55]], dtype=np.double),
               np.array([0.47, 0.62, 0.36], dtype=np.double)])

print(l1.get_weights())
print(l2.get_weights())
model.compile(loss='mean_squared_error',
              optimizer=tf.keras.optimizers.SGD(learning_rate=0.1, momentum=0.0),
              metrics=['mse'])

start = timer()
historico = model.fit(x_train,
                      y_train,
                      epochs=100,
                      batch_size=1, # to get gradient descent
                      verbose=1,
                      validation_split=0.0,
                      shuffle=False,
                      use_multiprocessing=False)
end = timer()
elapsed_time = (end - start)

MIN_MSE_TRAIN = min(historico.history['loss'])
print(f"Error minimo: {MIN_MSE_TRAIN}.")
print(f"Tiempo necesario para ejecución en nanosegundos: {elapsed_time * (10 ** 9)}.")
print(f"Tiempo necesario para ejecución en segundos: {elapsed_time}.")
print(l1.get_weights())
print(l2.get_weights())