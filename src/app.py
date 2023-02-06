import numpy as np
from flask import Flask, request, jsonify, render_template
from markupsafe import escape
from gunicorn.app.base import Application
from gunicorn import util
import pickle
import fcntl

# Create flask app
app = Flask(__name__)
model = pickle.load(open("model.pkl", "rb"))

@app.route("/")
def Home():
    return render_template("index.html")

@app.route("/predict", methods = ["POST"])
def predict():
    float_features = [float(x) for x in request.form.values()]
    features = [np.array(float_features)]
    prediction = model.predict(features)
    return render_template("formulario.html", prediction_text = "Predicted Quality of Wine is: Good" if prediction >0 else "Predicted Quality of Wine is: Bad" )





if __name__ == "__main__":
    app.run(debug=True)
