from flask import Flask, request, render_template
import pickle
import numpy as np

app = Flask(__name__)

with open("model.pkl","rb") as f:
    model = pickle.load(f)

@app.route('/')
def home():
    return render_template("index.html")

@app.route('/predict',methods=['POST'])
def predict():

    features = [float(x) for x in request.form.values()]
    final = np.array(features).reshape(1,-1)

    prediction = model.predict(final)

    return render_template("index.html", prediction=prediction[0])

if __name__ == "__main__":
    app.run(debug=True)