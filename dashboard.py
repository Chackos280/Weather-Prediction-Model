from flask import Flask, render_template
import pandas as pd
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

app = Flask(__name__)
model = tf.keras.models.load_model("../model/weather_model.h5")

@app.route("/")
def index():
    df = pd.read_csv("../data/weather.csv")
    actual = df["temp"].tolist()

    feats = df[["temp","humidity","rainfall"]].values
    X = []
    for i in range(len(feats)-7):
        X.append(feats[i:i+7])
    X = np.array(X)
    preds = model.predict(X)[:,0].tolist()

    pd.DataFrame({
        "day": range(len(preds)),
        "actual": actual[7:len(preds)+7],
        "pred": preds
    }).plot(x="day", y=["actual","pred"], marker="o")

    plt.savefig("static/weather_plot.png")
    return render_template("index.html")

if __name__ == "__main__":
    app.run()