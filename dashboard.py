from flask import Flask, render_template
import pandas as pd
import matplotlib.pyplot as plt

app = Flask(__name__)

@app.route("/")
def index():
    actual = [22,23,24,21,20,19,25]
    pred = [21,22,23,21,19,20,24]

    df = pd.DataFrame({
        "Day": range(len(actual)),
        "Actual": actual,
        "Predicted": pred
    })
    df.plot(x="Day", y=["Actual","Predicted"])
    plt.savefig("static/weather_plot.png")
    return render_template("index.html")

if __name__ == "__main__":
    app.run()