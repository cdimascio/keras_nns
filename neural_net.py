from keras.models import Sequential
from keras.layers import Dense
import numpy as np

model = Sequential()
# 10.5,5.,9.5,12 => 18.5
# large hidden nodes more complex, longer traing, but potentially better, possible overfit
model.add(Dense(8, activation='relu', input_dim=4))
model.add(Dense(16, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(1, activation='linear'))

model.compile(
    optimizer='adam',
    loss='mean_squared_error'
)

x_train = np.array([
    [1,2,3,4],
    [4,6,1,2],
    [10,9,10,11],
    [10,12,9,13],
    [99,100,101,102],
    [105,111,109,102]
])

y_train = np.array([
    [2.5],
    [3.25],
    [10.0],
    [11.0],
    [100.5],
    [106.75]
])

model.fit(
    x_train,
    y_train,
    batch_size=2, # 16, 32, 64, 256,
    epochs=100 # number iteration over set (more epochs better train, but takes longer)
    verbose=1,
    validation_split=0.2
)