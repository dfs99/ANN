import tensorflow as tf
import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
from timeit import default_timer as timer


data = pd.read_csv("robots.csv")
x_train = data.iloc[:, :-3]
y_train = data.iloc[:, -3:]
input_shape = (x_train.shape[1],)

model = Sequential()
model.add(Dense(6, input_shape=input_shape, activation='sigmoid'))
model.add(Dense(3, activation='sigmoid'))

model.compile(loss='mean_squared_error',
              optimizer=tf.keras.optimizers.SGD(learning_rate=0.001, momentum=0),
              metrics=['mse'])
start = timer()
historico = model.fit(x_train,
                      y_train,
                      epochs=10,
                      batch_size=1, # to get plain gradient descent
                      verbose=0,
                      validation_split=0.0,
                      shuffle=False,
                      use_multiprocessing=False)
end = timer()
elapsed_time = (end - start)

MIN_MSE_TRAIN = min(historico.history['loss'])
print(f"Error minimo: {MIN_MSE_TRAIN}.")
print(f"Tiempo necesario para ejecución en nanosegundos: {elapsed_time * (10 ** 9)}.")
print(f"Tiempo necesario para ejecución en segundos: {elapsed_time}.")
