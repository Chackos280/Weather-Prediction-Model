import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

data = pd.read_csv("../data/sample_weather.csv")
vals = data[["temp", "humidity", "rainfall"]].values

X = []
Y = []
for i in range(len(vals)-7):
    X.append(vals[i:i+7])
    Y.append(vals[i+7])
X = np.array(X)
Y = np.array(Y)

model = Sequential()
model.add(LSTM(32, input_shape=(7,3)))
model.add(Dense(16, activation="relu"))
model.add(Dense(3))

model.compile(optimizer="adam", loss="mse")
model.fit(X, Y, epochs=5, batch_size=8)

model.save("weather_model.h5")