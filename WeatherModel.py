import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

df = pd.read_csv("../data/weather.csv")
vals = df[["temp","humidity","rainfall"]].values

X, y = [], []
for i in range(len(vals)-7):
    X.append(vals[i:i+7])
    y.append(vals[i+7])
X = np.array(X)
y = np.array(y)

model = Sequential()
model.add(LSTM(32, input_shape=(7,3)))
model.add(Dense(16, activation="relu"))
model.add(Dense(3))

model.compile(optimizer="adam", loss="mse", metrics=["mae"])
model.fit(X, y, epochs=7, batch_size=8)

loss, mae = model.evaluate(X, y)
print("done training, MAE:", mae)

model.save("weather_model.h5")
