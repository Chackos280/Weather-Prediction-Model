from flask import Flask, request, jsonify
import tensorflow as tf
import numpy as np

app = Flask(__name__)
model = tf.keras.models.load_model("../model/weather_model.h5")

@app.route("/predict", methods=["POST"])
def predict():
    j = request.get_json()
    arr = np.array(j["features"]).reshape(1,7,3)
    out = model.predict(arr)
    return jsonify({
        "temp": float(out[0][0]),
        "humidity": float(out[0][1]),
        "rainfall": float(out[0][2])
    })

if __name__ == "__main__":
    app.run()